from __future__ import annotations

import json
import sys

from inference import ChurnPredictor


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input provided"}))
        sys.exit(1)

    try:
        payload = json.loads(sys.argv[1])
    except Exception as exc:
        print(json.dumps({"error": f"Failed to parse input: {exc}"}))
        sys.exit(1)

    try:
        predictor = ChurnPredictor()
        result = predictor.predict_customer(payload)
        print(json.dumps(result))
    except Exception as exc:
        print(json.dumps({"error": f"Prediction error: {exc}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
