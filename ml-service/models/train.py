from __future__ import annotations

from dataclasses import asdict, dataclass

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from evaluation.metrics import classification_metrics
from features.engineering import add_derived_features
from models.registry import get_model_registry
from preprocessing.transformers import build_preprocessor, infer_feature_schema
from utils.config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TEST_SIZE,
    PIPELINE_BUNDLE_PATH,
)


@dataclass
class TrainResult:
    selected_model: str
    metrics_by_model: dict[str, dict[str, float]]
    feature_names: list[str]
    bundle_path: str


def train_pipeline(
    df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    model_name: str = DEFAULT_MODEL_NAME,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_size: float = DEFAULT_TEST_SIZE,
    export_path: str | None = None,
) -> TrainResult:
    data = add_derived_features(df)
    if target_column not in data.columns:
        raise ValueError(f"Target column not found: {target_column}")

    y = data[target_column]
    X = data.drop(columns=[target_column])
    schema = infer_feature_schema(data, target_column)
    preprocessor = build_preprocessor(schema)
    models = get_model_registry(random_state=random_state)
    if model_name not in models:
        raise ValueError(f"Unsupported model_name: {model_name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(set(y)) > 1 else None,
    )

    metrics_by_model: dict[str, dict[str, float]] = {}
    trained_pipelines: dict[str, Pipeline] = {}
    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics_by_model[name] = classification_metrics(y_test, y_pred)
        trained_pipelines[name] = pipeline

    selected_name = model_name
    selected_pipeline = trained_pipelines[selected_name]

    fitted_preprocessor = selected_pipeline.named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out().tolist()

    bundle = {
        "pipeline": selected_pipeline,
        "selected_model": selected_name,
        "metrics_by_model": metrics_by_model,
        "feature_names": feature_names,
        "schema": asdict(schema),
        "target_column": target_column,
        "random_state": random_state,
    }

    output_path = export_path or str(PIPELINE_BUNDLE_PATH)
    joblib.dump(bundle, output_path)

    return TrainResult(
        selected_model=selected_name,
        metrics_by_model=metrics_by_model,
        feature_names=feature_names,
        bundle_path=output_path,
    )

