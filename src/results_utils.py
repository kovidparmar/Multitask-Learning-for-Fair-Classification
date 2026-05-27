import json
from pathlib import Path

from .config import RESULTS_DIR


def save_result(name, payload):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "results.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    data[name] = payload
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_results():
    path = RESULTS_DIR / "results.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
