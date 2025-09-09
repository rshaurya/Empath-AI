from app.schemas import PredictBatchRequest, PredictRequest
from app.store import save_model, load_model

from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from typing import List
import numpy as np


app = FastAPI(title="Classic Sentiment predictions API", version="1.0")


@app.get("/health")
def health():
    return {"status": "Ok"}


@app.post("/train")
async def train(file: UploadFile = File(...), use_logreg: bool = False):
    df = pd.read_csv(file.file)
    if not {"text", "label"}.issubset(df.columns):
        raise HTTPException(400, detail="CSV must have 'text' and 'label' columns")

    X, y = df["text"].astype(str).tolist(), df["label"].astype(str).tolist()

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)
    if use_logreg:
        base = LogisticRegression(max_iter=200)
        pipe = Pipeline([("vec", vec), ("clf", base)])
    else:
        base = LinearSVC()
        pipe = Pipeline([("vec", vec), ("clf", CalibratedClassifierCV(base, cv=3))])

    pipe.fit(X, y)
    save_model(pipe)
    labels = sorted(set(y))
    return {"status": "trained", "classes": labels}


@app.post("/predict")
def predict(req: PredictRequest):
    pipe = load_model()
    if pipe is None:
        raise HTTPException(400, detail="Model not trained. Upload CSV via /train.")
    proba = _predict_proba(pipe, [req.text])[0]
    return _topk(proba, pipe.classes_, req.top_k)


@app.post("/predict/batch")
def predict_batch(req: PredictBatchRequest):
    pipe = load_model()
    if pipe is None:
        raise HTTPException(400, detail="Model not trained. Upload CSV via /train.")
    probas = _predict_proba(pipe, req.texts)
    return [_topk(p, pipe.classes_, req.top_k) for p in probas]


# helper functions
def _predict_proba(pipe, texts: List[str]):
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        return pipe.predict_proba(texts)
    dec = pipe.decision_function(texts)
    if dec.ndim == 1:
        dec = np.vstack([-dec, dec]).T
    t = 1.5
    ex = np.exp(dec / t)
    return ex / ex.sum(axis=1, keepdims=True)


def _topk(probs, labels, k):
    idx = np.argsort(-probs)[:k]
    return {
        "labels": [{"label": labels[i], "score": float(probs[i])} for i in idx],
        "primary": labels[idx[0]],
    }
