import joblib
from pathlib import Path


MODEL_PATH = Path("model.joblib")


def save_model(pipe):
    joblib.dump(pipe, MODEL_PATH)


def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None
