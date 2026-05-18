from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None


def get_model_registry(random_state: int) -> dict[str, object]:
    registry: dict[str, object] = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state
        ),
    }
    if XGBClassifier is not None:
        registry["xgboost"] = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
        )
    return registry

