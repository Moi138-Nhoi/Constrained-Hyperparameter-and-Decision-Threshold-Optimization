"""Paired, order-balanced inference benchmarks for the resource constraint."""

from __future__ import annotations

import time

import numpy as np


def summarize_pairs(samples: list[dict]) -> dict:
    """Use all paired rounds; one slow call cannot determine feasibility."""
    if not samples:
        raise ValueError("At least one paired timing round is required")
    ratios = np.asarray([s["ratio"] for s in samples], dtype=float)
    candidate = np.asarray([s["candidate_ms"] for s in samples], dtype=float)
    reference = np.asarray([s["reference_ms"] for s in samples], dtype=float)
    median = float(np.median(ratios))
    return {
        "latency_ratio": median,
        "latency_ratio_q25": float(np.quantile(ratios, 0.25)),
        "latency_ratio_q75": float(np.quantile(ratios, 0.75)),
        "latency_ratio_mad": float(np.median(np.abs(ratios - median))),
        "latency_candidate_ms": float(np.median(candidate)),
        "latency_reference_ms": float(np.median(reference)),
        "latency_rounds": len(samples),
    }


def paired_latency(candidate, reference, batch, *, warmup: int, rounds: int) -> list[dict]:
    """Time A-B-B-A / B-A-A-B rounds on the same batch and CPU settings.

    Each model is called twice per round. Averaging its two calls removes a
    first-order order effect; the candidate/reference ratio is paired within
    the same short time window. Callers aggregate all fold-round ratios.
    """
    if warmup < 0 or rounds < 1:
        raise ValueError("warmup must be nonnegative and rounds must be positive")
    for _ in range(warmup):
        candidate.predict_proba(batch)
        reference.predict_proba(batch)
    samples = []
    for round_number in range(rounds):
        order = (("candidate", "reference", "reference", "candidate")
                 if round_number % 2 == 0 else
                 ("reference", "candidate", "candidate", "reference"))
        measured = {"candidate": [], "reference": []}
        for name in order:
            model = candidate if name == "candidate" else reference
            start = time.perf_counter_ns()
            model.predict_proba(batch)
            measured[name].append((time.perf_counter_ns() - start) / 1_000_000)
        candidate_ms = float(np.mean(measured["candidate"]))
        reference_ms = float(np.mean(measured["reference"]))
        samples.append({
            "round": round_number,
            "candidate_ms": candidate_ms,
            "reference_ms": reference_ms,
            "ratio": candidate_ms / reference_ms,
        })
    return samples
