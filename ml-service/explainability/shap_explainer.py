from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.config import EXPLANATION_EXPORT_DIR

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None


FEATURE_LABELS = {
    "Age": "age",
    "AvgMonthlyGBDownload": "monthly data usage",
    "AvgMonthlyLongDistanceCharges": "long-distance charges",
    "AvgRevenue": "average revenue",
    "Contract": "contract",
    "Dependents": "dependents",
    "EngagementScore": "engagement score",
    "Gender": "gender",
    "HighValueCustomer": "customer value tier",
    "InternetService": "internet service",
    "InternetType": "internet plan",
    "Married": "marital status",
    "MonthlyCharge": "monthly charges",
    "MultipleLines": "multiple lines",
    "NumberofDependents": "number of dependents",
    "NumberofReferrals": "referrals",
    "OnlineBackup": "online backup",
    "OnlineSecurity": "online security",
    "PaperlessBilling": "paperless billing",
    "PaymentMethod": "payment method",
    "PhoneService": "phone service",
    "PremiumTechSupport": "tech support",
    "ReferredaFriend": "referral activity",
    "SeniorCitizen": "senior-citizen status",
    "ServiceCount": "service count",
    "StreamingMovies": "streaming movies",
    "StreamingMusic": "streaming music",
    "StreamingTV": "streaming TV",
    "TenureinMonths": "tenure",
    "Under30": "under-30 status",
    "UnlimitedData": "data plan",
}

VALUE_LABELS = {
    "Contract": {0: "Month-to-Month", 1: "One Year", 2: "Two Year"},
    "Gender": {0: "Female", 1: "Male"},
    "InternetService": {0: "No internet service", 1: "Internet service active"},
    "InternetType": {0: "DSL", 1: "Fiber Optic", 2: "Cable"},
    "PaymentMethod": {0: "Bank Transfer", 1: "Credit Card", 2: "Mailed Check"},
    "PaperlessBilling": {0: "Paper billing", 1: "Paperless billing"},
    "PhoneService": {0: "No phone service", 1: "Phone service active"},
    "MultipleLines": {0: "No multiple lines", 1: "Multiple lines active"},
    "OnlineSecurity": {0: "No online security", 1: "Online security active"},
    "OnlineBackup": {0: "No online backup", 1: "Online backup active"},
    "DeviceProtectionPlan": {0: "No device protection", 1: "Device protection active"},
    "PremiumTechSupport": {0: "No tech support", 1: "Tech support active"},
    "StreamingTV": {0: "No streaming TV", 1: "Streaming TV active"},
    "StreamingMovies": {0: "No streaming movies", 1: "Streaming movies active"},
    "StreamingMusic": {0: "No streaming music", 1: "Streaming music active"},
    "UnlimitedData": {0: "Limited data", 1: "Unlimited data"},
    "Married": {0: "Not married", 1: "Married"},
    "Dependents": {0: "No dependents", 1: "Has dependents"},
    "ReferredaFriend": {0: "No referral", 1: "Referred a friend"},
    "SeniorCitizen": {0: "Not a senior citizen", 1: "Senior citizen"},
    "Under30": {0: "Age 30 or above", 1: "Under 30"},
    "HighValueCustomer": {0: "Standard value", 1: "High value customer"},
}

NUMERIC_UNITS = {
    "Age": "years",
    "AvgMonthlyGBDownload": "GB",
    "AvgMonthlyLongDistanceCharges": "₹",
    "AvgRevenue": "₹",
    "EngagementScore": "out of 5",
    "MonthlyCharge": "₹",
    "NumberofDependents": "dependents",
    "NumberofReferrals": "referrals",
    "ServiceCount": "services",
    "TenureinMonths": "months",
    "Latitude": "degrees",
    "Longitude": "degrees",
}

IMPACT_FEATURES = {
    "TenureinMonths",
    "Contract",
    "MonthlyCharge",
    "PaymentMethod",
    "EngagementScore",
    "ServiceCount",
    "InternetService",
    "InternetType",
    "AvgMonthlyGBDownload",
    "OnlineSecurity",
    "PremiumTechSupport",
    "OnlineBackup",
    "DeviceProtectionPlan",
    "PaperlessBilling",
    "PhoneService",
    "MultipleLines",
    "UnlimitedData",
}


def _ensure_shap_available() -> None:
    if shap is None:
        raise RuntimeError("SHAP is not installed. Install with: pip install shap")


def _build_tree_explainer(model):
    _ensure_shap_available()
    return shap.TreeExplainer(model)


def _transform_input(pipeline, raw_df: pd.DataFrame):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    X_t = preprocessor.transform(raw_df)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return model, X_t, feature_names


def _normalize_binary_shap_values(shap_values):
    if hasattr(shap_values, "values"):
        arr = np.asarray(shap_values.values)
    elif isinstance(shap_values, list):
        if len(shap_values) == 0:
            raise ValueError("SHAP returned an empty list of values")
        if len(shap_values) == 1:
            arr = np.asarray(shap_values[0])
        else:
            arr = np.asarray(shap_values[1])
    else:
        arr = np.asarray(shap_values)

    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            return arr[:, :, 1]
        if arr.shape[0] == 2:
            return arr[1, :, :]
    raise ValueError(f"Unsupported SHAP values shape: {arr.shape}")


def _compute_binary_shap_values(explainer, X):
    try:
        raw = explainer(X)
    except Exception:
        raw = explainer.shap_values(X)
    return _normalize_binary_shap_values(raw)


def _build_ordered_legacy_row(
    raw_record: dict[str, Any],
    feature_names: list[str],
) -> pd.DataFrame:
    missing = [name for name in feature_names if name not in raw_record]
    if missing:
        raise ValueError(f"Missing features for SHAP explanation: {missing}")
    row = {name: raw_record[name] for name in feature_names}
    return pd.DataFrame([row], columns=feature_names)


def _to_python_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_feature_name(feature_name: str) -> tuple[str, str | None]:
    if feature_name.startswith("num__"):
        return feature_name.removeprefix("num__"), None
    if feature_name.startswith("cat__"):
        body = feature_name.removeprefix("cat__")
        if "_" in body:
            base, category = body.rsplit("_", 1)
            return base, category
        return body, None
    return feature_name, None


def _humanize_feature_name(feature_name: str) -> str:
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]
    cleaned = re.sub(r"(?<!^)(?=[A-Z])", " ", feature_name).replace("_", " ")
    return cleaned.strip().lower()


def _safe_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number):
        return None
    return number


def _format_numeric_value(feature_name: str, value: Any) -> str:
    number = _safe_number(value)
    if number is None:
        return "missing"
    if feature_name in {"Age", "EngagementScore", "NumberofDependents", "NumberofReferrals", "ServiceCount"}:
        return str(int(round(number)))
    if feature_name in {"Latitude", "Longitude"}:
        return f"{number:.2f}"
    if feature_name in NUMERIC_UNITS:
        unit = NUMERIC_UNITS[feature_name]
        if unit == "₹":
            return f"₹{number:,.0f}" if abs(number) >= 100 else f"₹{number:,.2f}"
        return f"{number:.0f} {unit}" if unit not in {"out of 5", "degrees"} else f"{number:.1f} {unit}"
    return f"{number:.2f}" if abs(number) < 100 else f"{number:,.0f}"


def _lookup_input_value(input_row: dict[str, Any], feature_name: str, base_feature: str) -> Any:
    if feature_name in input_row:
        return _to_python_value(input_row[feature_name])
    if base_feature in input_row:
        return _to_python_value(input_row[base_feature])
    return None


def _normalize_category(base_feature: str, category: str | None, input_value: Any) -> str:
    if base_feature in VALUE_LABELS and input_value is not None:
        try:
            key = int(round(float(input_value)))
            if key in VALUE_LABELS[base_feature]:
                return VALUE_LABELS[base_feature][key]
        except (TypeError, ValueError):
            pass
    if category is not None and base_feature in VALUE_LABELS:
        try:
            key = int(round(float(category)))
            if key in VALUE_LABELS[base_feature]:
                return VALUE_LABELS[base_feature][key]
        except (TypeError, ValueError):
            pass
    if category is not None:
        return str(category).replace("_", " ")
    if input_value is not None:
        return str(input_value)
    return "unknown"


def _render_feature_value(feature_name: str, base_feature: str, category: str | None, input_value: Any) -> str:
    if base_feature in VALUE_LABELS:
        label = _normalize_category(base_feature, category, input_value)
        if label != "unknown":
            return label
    if input_value is not None:
        return _format_numeric_value(feature_name, input_value)
    if category is not None:
        return str(category).replace("_", " ")
    return "missing"


def _aggregate_contributions(
    input_row: dict[str, Any],
    shap_values: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    if len(shap_values) != len(feature_names):
        raise ValueError(
            "Feature count mismatch between SHAP values and feature names "
            f"({len(shap_values)} vs {len(feature_names)})"
        )

    grouped: dict[str, dict[str, Any]] = {}

    for index, feature_name in enumerate(feature_names):
        base_feature, category = _normalize_feature_name(feature_name)
        shap_value = float(shap_values[index])
        entry = grouped.setdefault(
            base_feature,
            {
                "feature": base_feature,
                "feature_name": feature_name,
                "category": category,
                "shap_value": 0.0,
                "abs_shap_value": 0.0,
                "members": [],
            },
        )
        entry["shap_value"] += shap_value
        entry["abs_shap_value"] = abs(entry["shap_value"])
        entry["members"].append(
            {
                "feature_name": feature_name,
                "category": category,
                "shap_value": shap_value,
                "input_value": _lookup_input_value(input_row, feature_name, base_feature),
            }
        )

    ranked: list[dict[str, Any]] = []
    for base_feature, entry in grouped.items():
        members = entry["members"]
        representative = max(members, key=lambda item: abs(item["shap_value"]))
        raw_value = _lookup_input_value(input_row, representative["feature_name"], base_feature)
        value_text = _render_feature_value(
            representative["feature_name"],
            base_feature,
            representative["category"],
            raw_value,
        )
        display_name = _humanize_feature_name(base_feature)
        ranked.append(
            {
                "feature": base_feature,
                "display_name": display_name,
                "feature_name": representative["feature_name"],
                "category": _normalize_category(base_feature, representative["category"], raw_value),
                "input_value": _to_python_value(raw_value),
                "value_text": value_text,
                "shap_value": float(entry["shap_value"]),
                "direction": "increases churn" if entry["shap_value"] > 0 else "reduces churn",
                "abs_shap_value": float(abs(entry["shap_value"])),
            }
        )

    ranked.sort(key=lambda item: item["abs_shap_value"], reverse=True)
    return ranked


def _feature_reason(feature: dict[str, Any]) -> str:
    name = feature["feature"]
    value_text = feature["value_text"]
    shap_value = feature["shap_value"]
    positive = shap_value > 0

    if name == "TenureinMonths":
        months = _safe_number(feature["input_value"])
        if months is None:
            return ""
        if months <= 6:
            return (
                f"A very short tenure of {int(round(months))} months suggests limited loyalty and weak switching friction."
                if positive
                else f"Even with only {int(round(months))} months of tenure, this customer is showing retention strength."
            )
        if months <= 24:
            return (
                f"A tenure of {int(round(months))} months is still relatively early in the relationship, so it can make churn easier."
                if positive
                else f"A tenure of {int(round(months))} months is long enough to show some established loyalty."
            )
        return (
            f"Although tenure is already {int(round(months))} months, other signals are overpowering the usual loyalty effect."
            if positive
            else f"A long tenure of {int(round(months))} months is acting as a strong retention anchor."
        )

    if name == "MonthlyCharge":
        amount = _safe_number(feature["input_value"])
        if amount is None:
            return ""
        if amount >= 100:
            return (
                f"High monthly charges of {value_text} are likely creating price pressure and increasing churn risk."
                if positive
                else f"Despite monthly charges of {value_text}, this customer is not showing the usual price-driven churn signal."
            )
        if amount >= 70:
            return (
                f"Monthly charges of {value_text} are high enough to matter, so the model sees price sensitivity here."
                if positive
                else f"Monthly charges of {value_text} are manageable and are helping keep the account stable."
            )
        return (
            f"Lower monthly charges of {value_text} are not creating much churn pressure."
            if positive
            else f"Lower monthly charges of {value_text} are acting as a small retention buffer."
        )

    if name == "Contract":
        contract_text = value_text
        if contract_text == "Month-to-Month":
            return (
                f"A month-to-month contract gives the customer very little lock-in, which makes switching easier."
                if positive
                else f"A month-to-month contract leaves little switching friction, so it only helps when tenure and support are strong."
            )
        if contract_text == "One Year":
            return (
                f"A one-year contract usually improves retention, but the model is still seeing other pressures."
                if positive
                else f"A one-year contract is helping anchor the customer and reduce churn risk."
            )
        if contract_text == "Two Year":
            return (
                f"A two-year contract is usually sticky, so churn pressure here is coming from price, engagement, or service friction."
                if positive
                else f"A two-year contract is providing strong retention protection."
            )
        return ""

    if name in {"OnlineSecurity", "PremiumTechSupport", "OnlineBackup", "DeviceProtectionPlan", "PaperlessBilling", "PhoneService", "MultipleLines", "UnlimitedData", "InternetService"}:
        if value_text.startswith("No") or value_text.startswith("Not") or value_text.startswith("Paper billing"):
            if positive:
                return f"{value_text} removes a retention buffer and makes churn more likely."
            return f"{value_text} is not creating churn pressure, so it is helping stability."
        return (
            f"{value_text} is adding a retention cushion and lowering churn risk."
            if not positive
            else f"{value_text} should help retention, but other signals are still pushing churn up."
        )

    if name == "EngagementScore":
        score = _safe_number(feature["input_value"])
        if score is None:
            return ""
        if score <= 2:
            return (
                f"A low engagement score of {int(round(score))}/5 points to weaker product usage and a higher churn chance."
                if positive
                else f"A low engagement score of {int(round(score))}/5 is not severe enough to overpower the retention buffer."
            )
        if score >= 4:
            return (
                f"A strong engagement score of {int(round(score))}/5 usually protects the account, but not enough to fully offset other issues."
                if positive
                else f"A strong engagement score of {int(round(score))}/5 is reinforcing customer retention."
            )
        return (
            f"An average engagement score of {int(round(score))}/5 provides only limited protection against churn."
            if positive
            else f"An average engagement score of {int(round(score))}/5 is helping maintain a stable relationship."
        )

    if name == "ServiceCount":
        count = _safe_number(feature["input_value"])
        if count is None:
            return ""
        if count <= 2:
            return (
                f"A low service count of {int(round(count))} means the customer is not deeply embedded across the product suite."
                if positive
                else f"A low service count of {int(round(count))} is still manageable and not causing much churn pressure."
            )
        if count >= 4:
            return (
                f"A broad service footprint of {int(round(count))} services usually supports retention, but friction elsewhere is still showing up."
                if positive
                else f"A broad service footprint of {int(round(count))} services is strengthening stickiness."
            )
        return (
            f"A mid-sized service bundle of {int(round(count))} services gives only moderate retention strength."
            if positive
            else f"A mid-sized service bundle of {int(round(count))} services is providing some lock-in."
        )

    if name == "PaymentMethod":
        if value_text == "Mailed Check":
            return (
                f"Mailed check payments often indicate lower digital stickiness and can make churn more likely."
                if positive
                else f"Mailed check payments are unusual here, but they are not driving churn pressure."
            )
        if value_text in {"Bank Transfer", "Credit Card"}:
            return (
                f"{value_text} payments usually support retention by making billing smoother and more automatic."
                if not positive
                else f"{value_text} payments reduce billing friction, but they are not enough to counter the churn pressure elsewhere."
            )
        return ""

    if name in {"Age", "Under30", "SeniorCitizen"}:
        if name == "Age":
            age = _safe_number(feature["input_value"])
            if age is None:
                return ""
            if age < 30:
                return (
                    f"A younger customer profile at {int(round(age))} years old often behaves more like a switch-ready segment."
                    if positive
                    else f"A younger customer profile at {int(round(age))} years old is still showing stable behavior."
                )
            if age >= 65:
                return (
                    f"An older customer profile at {int(round(age))} years old is bringing a different churn pattern into the model."
                    if positive
                    else f"An older customer profile at {int(round(age))} years old is helping stability."
                )
            return (
                f"An age of {int(round(age))} years is not a major stabilizer, so other signals matter more."
                if positive
                else f"An age of {int(round(age))} years is neutral to mildly stabilizing here."
            )
        if name == "Under30":
            return (
                f"Being under 30 is often associated with higher switching frequency and price sensitivity."
                if positive
                else f"Being under 30 is not creating churn pressure in this case."
            )
        return (
            f"Senior-citizen status can shift service expectations and affect retention patterns."
            if positive
            else f"Senior-citizen status is helping the model see a more stable customer profile."
        )

    if name in {"AvgMonthlyGBDownload", "AvgMonthlyLongDistanceCharges", "AvgRevenue", "NumberofReferrals", "NumberofDependents", "Latitude", "Longitude"}:
        pretty_value = value_text
        if positive:
            return f"{_humanize_feature_name(name).capitalize()} at {pretty_value} is contributing to churn pressure."
        return f"{_humanize_feature_name(name).capitalize()} at {pretty_value} is helping stabilize the account."
    return ""


def _join_phrases(phrases: list[str]) -> str:
    phrases = [phrase.strip().rstrip(".") for phrase in phrases if phrase and phrase.strip()]
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    return "; ".join(phrases)


def _group_feature(feature: dict[str, Any]) -> str:
    name = feature["feature"]
    if name in {"TenureinMonths", "Contract"}:
        return "Commitment"
    if name in {"MonthlyCharge", "PaymentMethod"}:
        return "Financial pressure"
    if name in {"EngagementScore", "ServiceCount", "InternetService", "InternetType", "AvgMonthlyGBDownload"}:
        return "Engagement"
    if name in {"OnlineSecurity", "PremiumTechSupport", "OnlineBackup", "DeviceProtectionPlan", "PaperlessBilling", "PhoneService", "MultipleLines", "UnlimitedData"}:
        return "Support / retention signals"
    return ""


def _feature_phrase(feature: dict[str, Any]) -> str:
    name = feature["feature"]
    value_text = feature["value_text"]
    if name == "TenureinMonths":
        return f"tenure at {value_text}"
    if name == "Contract":
        return f"a {value_text.lower()} contract"
    if name == "MonthlyCharge":
        return f"monthly charges of {value_text}"
    if name == "PaymentMethod":
        return f"{value_text.lower()} payment behavior"
    if name == "EngagementScore":
        return f"engagement at {value_text}/5"
    if name == "ServiceCount":
        return f"service depth of {value_text}"
    if name == "InternetService":
        return f"{value_text.lower()}"
    if name == "InternetType":
        return f"{value_text.lower()} access"
    if name == "AvgMonthlyGBDownload":
        return f"data usage at {value_text}"
    if name == "OnlineSecurity":
        return f"{value_text.lower()}"
    if name == "PremiumTechSupport":
        return f"{value_text.lower()}"
    if name == "OnlineBackup":
        return f"{value_text.lower()}"
    if name == "DeviceProtectionPlan":
        return f"{value_text.lower()}"
    if name == "PaperlessBilling":
        return f"{value_text.lower()}"
    if name == "PhoneService":
        return f"{value_text.lower()}"
    if name == "MultipleLines":
        return f"{value_text.lower()}"
    if name == "UnlimitedData":
        return f"{value_text.lower()}"
    return f"{_humanize_feature_name(name)} at {value_text}"


def generate_explanation(
    input_data: dict[str, Any] | pd.Series | pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: list[str],
    top_k: int = 5,
) -> dict[str, Any]:
    if isinstance(input_data, pd.DataFrame):
        input_row = input_data.iloc[0].to_dict()
    elif isinstance(input_data, pd.Series):
        input_row = input_data.to_dict()
    else:
        input_row = dict(input_data)

    shap_array = np.asarray(shap_values)
    if shap_array.ndim != 1:
        shap_array = shap_array.reshape(-1)

    contributions = _aggregate_contributions(input_row, shap_array, feature_names)
    top_features = [feature for feature in contributions if feature["feature"] in IMPACT_FEATURES][:top_k]

    positive_drivers = [feature for feature in top_features if feature["shap_value"] > 0]
    protective_drivers = [feature for feature in top_features if feature["shap_value"] < 0]
    net_shap = float(sum(feature["shap_value"] for feature in top_features))
    if net_shap >= 0.25:
        classification = "High risk"
    elif net_shap <= -0.25:
        classification = "Stable"
    else:
        classification = "Moderate risk"

    category_buckets: dict[str, list[dict[str, Any]]] = {
        "Engagement": [],
        "Commitment": [],
        "Financial pressure": [],
        "Support / retention signals": [],
    }
    for feature in top_features:
        bucket = _group_feature(feature)
        if bucket in category_buckets:
            category_buckets[bucket].append(feature)

    def _build_category_line(label: str, items: list[dict[str, Any]]) -> str:
        if not items:
            return ""
        ordered = sorted(items, key=lambda item: abs(item["shap_value"]), reverse=True)
        snippets = []
        for feature in ordered[:2]:
            reason = _feature_reason(feature)
            if reason:
                snippets.append(reason)
        if not snippets:
            return ""
        return f"{label}: {_join_phrases(snippets)}."

    explanation_lines: list[str] = []
    positive_summary = _join_phrases([_feature_phrase(feature) for feature in positive_drivers[:2] if _feature_phrase(feature)])
    protective_summary = _join_phrases([_feature_phrase(feature) for feature in protective_drivers[:2] if _feature_phrase(feature)])

    if classification == "High risk":
        if positive_summary and protective_summary:
            explanation_lines.append(f"High risk: {positive_summary} outweigh the protection from {protective_summary}.")
        elif positive_summary:
            explanation_lines.append(f"High risk: {positive_summary} is driving the churn call.")
        else:
            explanation_lines.append("High risk: the strongest signals point toward churn.")
    elif classification == "Stable":
        if protective_summary and positive_summary:
            explanation_lines.append(f"Stable: {protective_summary} is holding the account steady, while {positive_summary} stays secondary.")
        elif protective_summary:
            explanation_lines.append(f"Stable: {protective_summary} is the main retention anchor.")
        else:
            explanation_lines.append("Stable: retention signals keep churn pressure low.")
    else:
        if positive_summary and protective_summary:
            explanation_lines.append(f"Moderate risk: {positive_summary} is only partly offset by {protective_summary}.")
        elif positive_summary:
            explanation_lines.append(f"Moderate risk: {positive_summary} is creating pressure, but not enough for a hard churn call.")
        else:
            explanation_lines.append("Moderate risk: the pressure is real, but not yet decisive.")

    for category in ["Engagement", "Commitment", "Financial pressure", "Support / retention signals"]:
        line = _build_category_line(category, category_buckets[category])
        if line:
            explanation_lines.append(line)

    explanation = " ".join(explanation_lines[:5])

    top_feature_payload = [
        {
            "feature": feature["feature"],
            "display_name": feature["display_name"],
            "category": _group_feature(feature),
            "value": feature["input_value"],
            "value_text": feature["value_text"],
            "shap_value": feature["shap_value"],
            "direction": feature["direction"],
            "abs_shap_value": feature["abs_shap_value"],
            "reason": _feature_reason(feature),
        }
        for feature in top_features
    ]

    return {
        "top_features": top_feature_payload,
        "top_contributors": [
            {"feature": item["feature"], "shap_value": item["shap_value"]}
            for item in top_feature_payload
        ],
        "explanation": explanation,
    }


def explain_local_prediction(pipeline, raw_record: dict[str, Any]) -> dict[str, Any]:
    raw_df = pd.DataFrame([raw_record])
    model, X_t, feature_names = _transform_input(pipeline, raw_df)
    explainer = _build_tree_explainer(model)
    shap_values = _compute_binary_shap_values(explainer, X_t)
    instance_values = shap_values[0]
    return generate_explanation(raw_df.iloc[0].to_dict(), instance_values, feature_names)


def explain_local_prediction_legacy(
    model,
    feature_names: list[str],
    raw_record: dict[str, Any],
) -> dict[str, Any]:
    X = _build_ordered_legacy_row(raw_record, feature_names)
    explainer = _build_tree_explainer(model)
    shap_values = _compute_binary_shap_values(explainer, X)
    instance_values = shap_values[0]
    return generate_explanation(raw_record, instance_values, feature_names)


def explain_global_importance(pipeline, raw_df: pd.DataFrame) -> list[dict[str, Any]]:
    model, X_t, feature_names = _transform_input(pipeline, raw_df)
    explainer = _build_tree_explainer(model)
    shap_values = _compute_binary_shap_values(explainer, X_t)
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(
        [
            {"feature": feature_names[i], "importance": float(mean_abs[i])}
            for i in range(len(feature_names))
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )
    return ranked


def export_local_explanation(
    pipeline,
    raw_record: dict[str, Any],
    output_dir: str | Path | None = None,
    file_stem: str = "local_explanation",
) -> dict[str, str]:
    out_dir = Path(output_dir) if output_dir else EXPLANATION_EXPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = explain_local_prediction(pipeline, raw_record)
    json_path = out_dir / f"{file_stem}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    raw_df = pd.DataFrame([raw_record])
    model, X_t, feature_names = _transform_input(pipeline, raw_df)
    explainer = _build_tree_explainer(model)
    shap_values = _compute_binary_shap_values(explainer, X_t)
    plot_path = out_dir / f"{file_stem}.png"
    shap.summary_plot(
        shap_values,
        features=X_t,
        feature_names=feature_names,
        show=False,
    )
    import matplotlib.pyplot as plt

    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return {"json_path": str(json_path), "plot_path": str(plot_path)}

