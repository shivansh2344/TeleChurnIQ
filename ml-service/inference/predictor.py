from __future__ import annotations

from typing import Any

import joblib
import pandas as pd
import pickle

from explainability.shap_explainer import (
    explain_local_prediction,
    explain_local_prediction_legacy,
)
from features.engineering import build_model_feature_row
from utils.config import FEATURE_LIST_PATH, MODEL_PATH, PIPELINE_BUNDLE_PATH


class ChurnPredictor:
    def __init__(self) -> None:
        self.bundle = None
        self.legacy_model = None
        self.legacy_features = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        if PIPELINE_BUNDLE_PATH.exists():
            self.bundle = joblib.load(PIPELINE_BUNDLE_PATH)
            return

        if not MODEL_PATH.exists() or not FEATURE_LIST_PATH.exists():
            raise FileNotFoundError(
                "No model artifacts found. Expected pipeline_bundle.pkl or legacy churn_model.pkl + feature_list.pkl"
            )
        with MODEL_PATH.open("rb") as model_file:
            self.legacy_model = pickle.load(model_file)
        with FEATURE_LIST_PATH.open("rb") as feature_file:
            self.legacy_features = pickle.load(feature_file)

    def _predict_legacy(self, data: dict[str, Any]) -> dict[str, Any]:
        model_row = build_model_feature_row(data, self.legacy_features)

        X = pd.DataFrame(
            [{feature: model_row[feature] for feature in self.legacy_features}],
            columns=self.legacy_features,
        )
        pred = int(self.legacy_model.predict(X)[0])
        try:
            prob = float(self.legacy_model.predict_proba(X)[0][1])
        except (AttributeError, IndexError):
            prob = float(pred)
        explanation = explain_local_prediction_legacy(
            self.legacy_model,
            self.legacy_features,
            model_row,
        )
        return {
            "prediction": pred,
            "probability": prob,
            "top_features": explanation["top_features"],
            "top_contributors": explanation["top_contributors"],
            "explanation": explanation["explanation"],
        }

    def _predict_pipeline(self, data: dict[str, Any]) -> dict[str, Any]:
        pipeline = self.bundle["pipeline"]
        required_features = pipeline.named_steps["preprocessor"].feature_names_in_.tolist()
        model_row = build_model_feature_row(data, required_features)
        row = pd.DataFrame([model_row], columns=required_features)
        pred = int(pipeline.predict(row)[0])
        if hasattr(pipeline, "predict_proba"):
            prob = float(pipeline.predict_proba(row)[0][1])
        else:
            prob = float(pred)
        explanation = explain_local_prediction(pipeline, model_row)
        return {
            "prediction": pred,
            "probability": prob,
            "top_features": explanation["top_features"],
            "top_contributors": explanation["top_contributors"],
            "explanation": explanation["explanation"],
        }

    def predict_customer(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.bundle is not None:
            result = self._predict_pipeline(data)
        else:
            result = self._predict_legacy(data)
        result["status"] = "success"
        return result

