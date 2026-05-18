from __future__ import annotations

from typing import Any

from inference import ChurnPredictor

_predictor: ChurnPredictor | None = None


def _get_predictor() -> ChurnPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor()
    return _predictor


def predict_customer(data: dict[str, Any]) -> dict[str, Any]:
    return _get_predictor().predict_customer(data)

