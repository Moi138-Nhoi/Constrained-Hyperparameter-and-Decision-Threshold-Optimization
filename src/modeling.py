"""Fold-local preprocessing, metrics, exact threshold search and timing."""

from __future__ import annotations

import io
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import FEATURES


class FeatureMaker:
    """Deterministic, observation-time features; fitted state is empty."""

    def __init__(self, engineered: bool = False):
        self.engineered = engineered

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        z = x.loc[:, FEATURES].copy()
        if self.engineered:
            z["Temperature difference [K]"] = z["Process temperature [K]"] - z["Air temperature [K]"]
            z["Mechanical power [W]"] = z["Torque [Nm]"] * z["Rotational speed [rpm]"] * (2 * np.pi / 60)
            z["Torque x wear"] = z["Torque [Nm]"] * z["Tool wear [min]"]
        return z


def make_model(params: dict | None, seed: int, *, dummy=False, engineered=False):
    numeric = FEATURES[1:]
    if engineered:
        numeric = numeric + ["Temperature difference [K]", "Mechanical power [W]", "Torque x wear"]
    prep = ColumnTransformer(
        [
            ("type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Type"]),
            ("numeric", "passthrough", numeric),
        ],
        sparse_threshold=0,
    )
    if dummy:
        estimator = DummyClassifier(strategy="most_frequent")
    else:
        kwargs = {} if params is None else params.copy()
        estimator = RandomForestClassifier(**kwargs, random_state=seed, n_jobs=1)
    return Pipeline([("features", FeatureMaker(engineered)), ("preprocess", prep), ("classifier", estimator)])


def model_size_mb(model) -> float:
    stream = io.BytesIO()
    joblib.dump(model, stream, compress=0)
    return stream.tell() / 1_000_000


def latency_ms(model, batch: pd.DataFrame, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        model.predict_proba(batch)
    measurements = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict_proba(batch)
        measurements.append((time.perf_counter() - start) * 1000)
    return float(np.median(measurements))


def classification_metrics(y, prob, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)),
        "average_precision": float(average_precision_score(y, prob)),
        "accuracy": float(accuracy_score(y, pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_threshold(y, prob, fpr_limit: float) -> dict:
    """Search all distinct scores for max F2 under an empirical FPR cap."""
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    order = np.argsort(-prob, kind="stable")
    scores, labels = prob[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(scores) != 0), len(scores) - 1]
    tp = np.cumsum(labels)[ends]
    fp = (ends + 1) - tp
    positives = int(y.sum())
    negatives = len(y) - positives
    fn = positives - tp
    fpr = fp / negatives
    f2 = 5 * tp / np.maximum(5 * tp + 4 * fn + fp, 1)
    allowed = np.flatnonzero(fpr <= fpr_limit + 1e-12)
    candidates = [(float(f2[i]), -float(fpr[i]), float(scores[ends[i]])) for i in allowed]
    # A threshold in the gap above the maximum score predicts no alarms. This
    # matters for constant-score models and tight caps; unique scores alone miss it.
    if scores[0] < 1.0:
        candidates.append((0.0, 0.0, 1.0))
    if not candidates:
        # There is no feasible threshold in [0, 1] when probability one
        # occurs on too many negatives. Keep complete metrics for the log.
        return classification_metrics(y, prob, 1.0)
    return classification_metrics(y, prob, max(candidates)[2])
