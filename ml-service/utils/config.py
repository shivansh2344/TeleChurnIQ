from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR
MODEL_PATH = ARTIFACT_DIR / "churn_model.pkl"
FEATURE_LIST_PATH = ARTIFACT_DIR / "feature_list.pkl"
PIPELINE_BUNDLE_PATH = ARTIFACT_DIR / "pipeline_bundle.pkl"
EXPLANATION_EXPORT_DIR = ARTIFACT_DIR / "explanations"

DEFAULT_TARGET_COLUMN = "Churn"
DEFAULT_TEST_SIZE = 0.3
DEFAULT_RANDOM_STATE = 42
DEFAULT_MODEL_NAME = "random_forest"

