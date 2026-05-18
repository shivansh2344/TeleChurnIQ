from __future__ import annotations

from typing import Any

import pandas as pd


CITY_COORDINATES = {
    "mumbai": (19.0760, 72.8777),
    "delhi": (28.6139, 77.2090),
    "bengaluru": (12.9716, 77.5946),
    "bangalore": (12.9716, 77.5946),
    "hyderabad": (17.3850, 78.4867),
    "chennai": (13.0827, 80.2707),
    "kolkata": (22.5726, 88.3639),
    "pune": (18.5204, 73.8567),
    "ahmedabad": (23.0225, 72.5714),
}


DEFAULT_FEATURES = {
    "Gender": 1,
    "Age": 40,
    "Married": 0,
    "Dependents": 0,
    "NumberofDependents": 0,
    "Latitude": 20.5937,
    "Longitude": 78.9629,
    "ReferredaFriend": 0,
    "NumberofReferrals": 0,
    "TenureinMonths": 12,
    "Offer": 0,
    "PhoneService": 1,
    "AvgMonthlyLongDistanceCharges": 20.0,
    "MultipleLines": 0,
    "InternetService": 1,
    "InternetType": 1,
    "AvgMonthlyGBDownload": 12.0,
    "OnlineSecurity": 0,
    "OnlineBackup": 0,
    "DeviceProtectionPlan": 0,
    "PremiumTechSupport": 0,
    "StreamingTV": 0,
    "StreamingMovies": 0,
    "StreamingMusic": 0,
    "UnlimitedData": 1,
    "Contract": 0,
    "PaperlessBilling": 1,
    "PaymentMethod": 1,
    "MonthlyCharge": 70.0,
    "StatusID": 1,
    "EngagementScore": 3,
    "ServiceCount": 3,
}


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_model_feature_row(
    input_data: dict[str, Any],
    required_features: list[str],
) -> dict[str, Any]:
    row = dict(DEFAULT_FEATURES)
    row.update(input_data)

    row["Age"] = _clamp(_as_float(row.get("Age"), DEFAULT_FEATURES["Age"]), 18, 100)
    row["TenureinMonths"] = _clamp(_as_float(row.get("TenureinMonths"), DEFAULT_FEATURES["TenureinMonths"]), 0, 120)
    row["MonthlyCharge"] = _clamp(_as_float(row.get("MonthlyCharge"), DEFAULT_FEATURES["MonthlyCharge"]), 0, 10000)
    row["EngagementScore"] = _clamp(_as_float(row.get("EngagementScore"), DEFAULT_FEATURES["EngagementScore"]), 1, 5)

    row["Gender"] = _as_int(row.get("Gender"), DEFAULT_FEATURES["Gender"])
    row["Contract"] = _as_int(row.get("Contract"), DEFAULT_FEATURES["Contract"])
    row["InternetService"] = _as_int(row.get("InternetService"), DEFAULT_FEATURES["InternetService"])
    row["PaymentMethod"] = _as_int(row.get("PaymentMethod"), DEFAULT_FEATURES["PaymentMethod"])
    row["OnlineSecurity"] = _as_int(row.get("OnlineSecurity"), DEFAULT_FEATURES["OnlineSecurity"])
    row["PremiumTechSupport"] = _as_int(row.get("PremiumTechSupport"), DEFAULT_FEATURES["PremiumTechSupport"])

    if "ServiceCount" in row and row["ServiceCount"] is not None:
        row["ServiceCount"] = _clamp(_as_float(row["ServiceCount"], DEFAULT_FEATURES["ServiceCount"]), 1, 8)
    else:
        base_services = 1
        base_services += int(row["InternetService"] == 1)
        base_services += int(row["OnlineSecurity"] == 1)
        base_services += int(row["PremiumTechSupport"] == 1)
        row["ServiceCount"] = float(_clamp(base_services, 1, 8))

    if row.get("City"):
        city_key = str(row["City"]).strip().lower()
        if city_key in CITY_COORDINATES:
            lat, lon = CITY_COORDINATES[city_key]
            row["Latitude"] = lat
            row["Longitude"] = lon

    row["NumberofDependents"] = _as_int(row.get("NumberofDependents"), DEFAULT_FEATURES["NumberofDependents"])
    row["Dependents"] = 1 if row["NumberofDependents"] > 0 else _as_int(row.get("Dependents"), DEFAULT_FEATURES["Dependents"])

    row["NumberofReferrals"] = _as_int(row.get("NumberofReferrals"), DEFAULT_FEATURES["NumberofReferrals"])
    row["ReferredaFriend"] = 1 if row["NumberofReferrals"] > 0 else _as_int(row.get("ReferredaFriend"), DEFAULT_FEATURES["ReferredaFriend"])

    row["Under30"] = 1 if row["Age"] < 30 else 0
    row["SeniorCitizen"] = 1 if row["Age"] > 65 else 0
    row["Married"] = _as_int(row.get("Married"), DEFAULT_FEATURES["Married"])
    row["Offer"] = _as_int(row.get("Offer"), DEFAULT_FEATURES["Offer"])
    row["PhoneService"] = _as_int(row.get("PhoneService"), DEFAULT_FEATURES["PhoneService"])
    row["InternetType"] = _as_int(row.get("InternetType"), DEFAULT_FEATURES["InternetType"])

    if row["InternetService"] == 0:
        row["InternetType"] = 0
        row["AvgMonthlyGBDownload"] = 0.0
        row["OnlineSecurity"] = 0
        row["OnlineBackup"] = 0
        row["DeviceProtectionPlan"] = 0
        row["PremiumTechSupport"] = 0
        row["StreamingTV"] = 0
        row["StreamingMovies"] = 0
        row["StreamingMusic"] = 0
        row["UnlimitedData"] = 0
    else:
        row["AvgMonthlyGBDownload"] = _clamp(
            _as_float(row.get("AvgMonthlyGBDownload"), row["EngagementScore"] * 8.0),
            1,
            200,
        )
        row["OnlineBackup"] = _as_int(row.get("OnlineBackup"), row["OnlineSecurity"])
        row["DeviceProtectionPlan"] = _as_int(row.get("DeviceProtectionPlan"), row["PremiumTechSupport"])
        stream_default = 1 if row["ServiceCount"] >= 4 else 0
        row["StreamingTV"] = _as_int(row.get("StreamingTV"), stream_default)
        row["StreamingMovies"] = _as_int(row.get("StreamingMovies"), stream_default)
        row["StreamingMusic"] = _as_int(row.get("StreamingMusic"), stream_default)
        row["UnlimitedData"] = _as_int(row.get("UnlimitedData"), 1)

    row["AvgMonthlyLongDistanceCharges"] = _clamp(
        _as_float(
            row.get("AvgMonthlyLongDistanceCharges"),
            row["MonthlyCharge"] * (0.15 if row["PhoneService"] else 0.0),
        ),
        0,
        1000,
    )
    row["MultipleLines"] = _as_int(row.get("MultipleLines"), 1 if row["ServiceCount"] >= 5 else 0)
    row["PaperlessBilling"] = _as_int(row.get("PaperlessBilling"), 0 if row["PaymentMethod"] == 2 else 1)
    row["StatusID"] = _as_int(row.get("StatusID"), DEFAULT_FEATURES["StatusID"])

    total_charges = row["MonthlyCharge"] * max(row["TenureinMonths"], 1)
    row["TotalCharges"] = float(total_charges)
    row["AvgRevenue"] = float(total_charges)
    row["HighValueCustomer"] = 1 if (row["MonthlyCharge"] >= 90 or total_charges >= 1500) else 0

    feature_row: dict[str, Any] = {}
    for feature in required_features:
        feature_row[feature] = row.get(feature, 0)
    return feature_row


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Age" in out.columns:
        out["Under30"] = (out["Age"] < 30).astype(int)
        out["SeniorCitizen"] = (out["Age"] > 65).astype(int)
    return out

