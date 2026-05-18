from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureSchema:
    numeric_features: list[str]
    categorical_features: list[str]


def infer_feature_schema(df: pd.DataFrame, target_column: str) -> FeatureSchema:
    X = df.drop(columns=[target_column])
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns.tolist() if c not in numeric_features]
    return FeatureSchema(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def build_preprocessor(schema: FeatureSchema) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, schema.numeric_features),
            ("cat", categorical_pipeline, schema.categorical_features),
        ]
    )

