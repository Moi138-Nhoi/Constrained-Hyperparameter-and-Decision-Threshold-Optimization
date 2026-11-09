"""Reproduce baselines, constrained search, final evaluation and figures."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from .data import DEFAULT_DATA, FEATURES, TARGET, load_data, split_data
from .latency import paired_latency, summarize_pairs
from .modeling import best_threshold, classification_metrics, make_model, model_size_mb


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def make_folds(y: np.ndarray, seed: int, n_splits: int):
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y))


def evaluate_oof(x: pd.DataFrame, y: np.ndarray, folds, params, cfg, *, reference_models=None,
                 dummy=False, engineered=False, model_seed=None):
    probs = np.empty(len(y), dtype=float)
    fit_seconds, predict_seconds, sizes, models = 0.0, 0.0, [], []
    seed = cfg["seed"] if model_seed is None else model_seed
    for train, valid in folds:
        model = make_model(params, seed, dummy=dummy, engineered=engineered)
        start = time.perf_counter()
        model.fit(x.iloc[train], y[train])
        fit_seconds += time.perf_counter() - start
        start = time.perf_counter()
        probs[valid] = model.predict_proba(x.iloc[valid])[:, 1]
        predict_seconds += time.perf_counter() - start
        sizes.append(model_size_mb(model))
        models.append(model)
    measurements = {
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "model_size_mb": float(max(sizes)),
        "fold_sizes_mb": [float(v) for v in sizes],
    }
    if reference_models is not None:
        measurements.update(measure_models(models, reference_models, x, folds, cfg))
    return probs, measurements, models


def measure_models(models, references, x, folds, cfg) -> dict:
    if len(models) != len(references) or len(models) != len(folds):
        raise ValueError("Candidate, reference and fold counts must agree")
    all_samples, fold_medians = [], []
    for fold_number, ((_, valid), model, reference) in enumerate(zip(folds, models, references)):
        batch = x.iloc[valid[: cfg["latency_batch_size"]]]
        samples = paired_latency(model, reference, batch,
                                 warmup=cfg["latency_warmup"], rounds=cfg["latency_repeats"])
        all_samples.extend({"fold": fold_number, **sample} for sample in samples)
        fold_medians.append(float(np.median([sample["ratio"] for sample in samples])))
    return {**summarize_pairs(all_samples),
            "fold_ratio_medians": fold_medians,
            "latency_samples": all_samples}


def record_candidate(method, trial, params, probs, y, measurements, cfg, *, fixed_threshold=None):
    limit = cfg["primary_fpr_limit"]
    primary = (best_threshold(y, probs, limit) if fixed_threshold is None
               else classification_metrics(y, probs, fixed_threshold))
    ratio = measurements["latency_ratio"]
    size_ok = measurements["model_size_mb"] <= cfg["max_model_mb"]
    latency_ok = ratio <= cfg["max_latency_ratio"]
    resources_ok = size_ok and latency_ok
    threshold_ok = primary["threshold"] is not None and primary["fpr"] <= limit + 1e-12
    row = {
        "method": method, "trial": trial, "params_json": json.dumps(params, sort_keys=True),
        **primary,
        "fit_seconds": measurements["fit_seconds"],
        "predict_seconds": measurements["predict_seconds"],
        "latency_ms_per_batch": measurements["latency_candidate_ms"],
        "baseline_ms_per_batch": measurements["latency_reference_ms"],
        "latency_ratio": float(ratio),
        "latency_ratio_q25": measurements["latency_ratio_q25"],
        "latency_ratio_q75": measurements["latency_ratio_q75"],
        "latency_ratio_mad": measurements["latency_ratio_mad"],
        "latency_rounds": measurements["latency_rounds"],
        "model_size_mb": measurements["model_size_mb"],
        "fold_ratio_medians_json": json.dumps(measurements["fold_ratio_medians"]),
        "latency_samples_json": json.dumps(measurements["latency_samples"]),
        "fold_sizes_mb_json": json.dumps(measurements["fold_sizes_mb"]),
        "fpr_ok": bool(threshold_ok),
        "size_ok": bool(size_ok),
        "latency_ok": bool(latency_ok),
        "feasible": bool(resources_ok and threshold_ok),
        "status": "feasible" if resources_ok and threshold_ok else "infeasible",
    }
    sensitivity = []
    for cap in cfg["fpr_limits"]:
        m = best_threshold(y, probs, cap)
        sensitivity.append({"method": method, "trial": trial, "fpr_cap": cap, **m,
                            "feasible": bool(resources_ok and m["fpr"] <= cap + 1e-12)})
    return row, sensitivity


def random_params(space: dict, rng: np.random.Generator) -> dict:
    return {name: choices[int(rng.integers(len(choices)))] for name, choices in space.items()}


def optuna_params(trial, space: dict) -> dict:
    return {name: trial.suggest_categorical(name, choices) for name, choices in space.items()}


def ranked_candidates(rows: list[dict]) -> list[dict]:
    """Predeclared development-only order; refit each unique decision once."""
    eligible = [r for r in rows if r["feasible"] and r["method"] != "dummy"]
    ordered = sorted(eligible, key=lambda r: (-r["f2"], r["fpr"], r["model_size_mb"],
                                             r["latency_ratio"], r["method"], r["trial"]))
    unique, seen = [], set()
    for row in ordered:
        key = (row["params_json"], row["threshold"])
        if key not in seen:
            unique.append(row)
            seen.add(key)
    return unique


def params_for_row(row: dict) -> dict | None:
    if row["method"] in {"rf_default", "rf_default_oof_threshold"}:
        return None
    if row["method"] in {"rf_balanced", "rf_balanced_oof_threshold"}:
        return {"class_weight": "balanced"}
    return json.loads(row["params_json"])


def try_ranked_candidates(ranked: list[dict], evaluator, record):
    """Try refits in the locked development order; never inspect test metrics."""
    attempts = []
    for rank, row in enumerate(ranked, start=1):
        attempt, model, measure, size = evaluator(rank, row)
        attempts.append(attempt)
        record(attempt, attempts)
        if attempt["feasible"]:
            return row, model, measure, size, attempts
    return None, None, None, None, attempts


def plot_results(rows: list[dict], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in ("random_search", "optuna_tpe"):
        history = [r for r in rows if r["method"] == method]
        best = -np.inf
        values = []
        for row in history:
            if row["feasible"]:
                best = max(best, row["f2"])
            values.append(np.nan if best == -np.inf else best)
        ax.plot(range(1, len(values) + 1), values, marker=".", label=method)
    ax.set(xlabel="Trial", ylabel="Best feasible OOF F₂", title="Optimization history")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "optimization_history.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for method, marker in (("random_search", "o"), ("optuna_tpe", "s")):
        subset = [r for r in rows if r["method"] == method and r["f2"] is not None]
        for feasible, alpha in ((False, 0.35), (True, 0.9)):
            group = [r for r in subset if r["feasible"] == feasible]
            if not group:
                continue
            ax.scatter([r["fpr"] for r in group], [r["f2"] for r in group],
                       marker=marker, alpha=alpha, label=f"{method}, {'feasible' if feasible else 'infeasible'}")
    ax.axvline(0.05, color="black", linestyle="--", linewidth=1, label="FPR cap 5%")
    ax.set(xlabel="OOF false positive rate", ylabel="OOF F₂", title="Quality and false alarms")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "f2_fpr.png", dpi=160)
    plt.close(fig)


def run(args) -> None:
    started = time.perf_counter()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.trials is not None:
        cfg["trials_per_method"] = args.trials
    if cfg["trials_per_method"] < 1:
        raise ValueError("trials_per_method must be positive")
    default_name = ("pilot" if args.pilot else "full") + "_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or Path("results/runs") / default_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Choose a fresh --output-dir.")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output.resolve()}", flush=True)
    figures = output / "figures"
    df, audit = load_data(args.data)
    dev_idx, test_idx = split_data(df, cfg["seed"])
    dev = df.loc[dev_idx]
    x, y = dev[FEATURES], dev[TARGET].to_numpy()
    folds = make_folds(y, cfg["seed"], cfg["folds"])
    fold_rows = [{"row_index": int(dev.index[pos]), "fold": int(k)}
                 for k, (_, valid) in enumerate(folds) for pos in valid]
    save_rows(output / "fold_assignments.csv", sorted(fold_rows, key=lambda r: r["row_index"]))
    save_rows(output / "split_assignments.csv",
              [{"row_index": int(i), "set": "development"} for i in dev_idx] +
              [{"row_index": int(i), "set": "test"} for i in test_idx])
    manifest = {
        "run_type": "pilot" if args.pilot else "full",
        "config": cfg, "dataset_audit": audit,
        "split": {"method": "stratified train_test_split", "test_fraction": 0.2,
                  "development_rows": int(len(dev_idx)), "test_rows": int(len(test_idx)),
                  "development_positives": int(y.sum()),
                  "fold_method": "StratifiedKFold(shuffle=True)", "folds": cfg["folds"]},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "processor": platform.processor(),
                    "packages": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "optuna", "matplotlib", "joblib", "threadpoolctl"]},
                    "rf_n_jobs": 1, "threadpool_limit": 1},
        "measurement": (f"Per fold: {cfg['latency_warmup']} candidate/reference warmups, then "
                        f"{cfg['latency_repeats']} paired ABBA/B A A B rounds, each with two calls per model "
                        f"on the same first {cfg['latency_batch_size']} validation rows. One CPU thread. "
                        "Constraint uses median of all fold-round candidate/reference ratios; q25, q75, "
                        "MAD, fold medians and raw rounds are logged. Size is maximum uncompressed "
                        "joblib bytes across folds."),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    save_json(output / "manifest.json", manifest)
    default_probs, default_measure, reference_models = evaluate_oof(x, y, folds, None, cfg)
    default_measure.update(measure_models(reference_models, reference_models, x, folds, cfg))
    default_measure["latency_ratio"] = 1.0  # The reference compared with itself.
    dummy_probs, dummy_measure, _ = evaluate_oof(x, y, folds, None, cfg,
                                                   reference_models=reference_models, dummy=True)
    balanced_probs, balanced_measure, _ = evaluate_oof(x, y, folds, {"class_weight": "balanced"}, cfg,
                                                        reference_models=reference_models)
    base_results = {
        "dummy": (dummy_probs, dummy_measure),
        "rf_default": (default_probs, default_measure),
        "rf_balanced": (balanced_probs, balanced_measure),
    }
    for name, (_, measure) in base_results.items():
        print(f"Baseline {name}: fit {measure['fit_seconds']:.1f}s", flush=True)
    baseline_rows, sensitivity_rows = [], []
    baseline_specs = [
        ("dummy", "dummy", {}, 0.5),
        ("rf_default", "rf_default", {}, 0.5),
        ("rf_balanced", "rf_balanced", {"class_weight": "balanced"}, 0.5),
        ("rf_default_oof_threshold", "rf_default", {}, None),
        ("rf_balanced_oof_threshold", "rf_balanced", {"class_weight": "balanced"}, None),
    ]
    for name, source, params, threshold in baseline_specs:
        probs, measure = base_results[source]
        row, sens = record_candidate(name, -1, params, probs, y, measure, cfg,
                                     fixed_threshold=threshold)
        baseline_rows.append(row)
        sensitivity_rows.extend(sens)
    save_rows(output / "baselines.csv", baseline_rows)
    trial_rows = []

    def evaluate_trial(method, number, params):
        probs, measure, _ = evaluate_oof(x, y, folds, params, cfg, reference_models=reference_models)
        row, sens = record_candidate(method, number, params, probs, y, measure, cfg)
        trial_rows.append(row)
        sensitivity_rows.extend(sens)
        save_rows(output / "trials.csv", trial_rows)
        save_rows(output / "sensitivity.csv", sensitivity_rows)
        print(f"{method} {number + 1}/{cfg['trials_per_method']}: F2={row['f2']}, "
              f"FPR={row['fpr']}, feasible={row['feasible']}", flush=True)
        return row

    rng = np.random.default_rng(cfg["seed"])
    for number in range(cfg["trials_per_method"]):
        evaluate_trial("random_search", number, random_params(cfg["search_space"], rng))

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"], n_startup_trials=10))

    def objective(trial):
        params = optuna_params(trial, cfg["search_space"])
        row = evaluate_trial("optuna_tpe", trial.number, params)
        return row["f2"] if row["feasible"] else -1.0

    study.optimize(objective, n_trials=cfg["trials_per_method"], n_jobs=1)
    plot_results(trial_rows, figures)
    eligible = [r for r in baseline_rows + trial_rows if r["feasible"] and r["method"] != "dummy"]
    ranked = ranked_candidates(baseline_rows + trial_rows)
    save_rows(output / "candidate_ranking.csv",
              [{"rank": i + 1, **row} for i, row in enumerate(ranked)])
    summary = {
        "run_type": manifest["run_type"],
        "feasible_oof_rows": len(eligible),
        "unique_ranked_decisions": len(ranked),
        "baseline_results": baseline_rows,
        "search_counts": {m: sum(r["method"] == m and r["feasible"] for r in trial_rows)
                          for m in ("random_search", "optuna_tpe")},
        "ranking_rule": "OOF feasible only; F2 descending, FPR ascending, model MB ascending, paired latency ratio ascending, method and trial ascending; duplicate parameter-plus-threshold decisions refit once.",
    }

    def finish(status: str) -> None:
        summary["status"] = status
        summary["wall_seconds"] = float(time.perf_counter() - started)
        manifest["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest["wall_seconds"] = summary["wall_seconds"]
        save_json(output / "manifest.json", manifest)
        save_json(output / "summary.json", summary)

    if not ranked:
        finish("no_feasible_oof_configuration")
        print("No feasible OOF Random Forest; test was not evaluated.", flush=True)
        return
    summary["top_development_candidate"] = ranked[0]

    def development_diagnostics(selected: dict) -> None:
        params = params_for_row(selected)
        threshold = selected["threshold"]
        stability = []
        for seed in cfg["stability_seeds"]:
            seed_folds = make_folds(y, seed, cfg["folds"])
            probs, _, _ = evaluate_oof(x, y, seed_folds, params, cfg, model_seed=seed)
            stability.append({"seed": seed, **classification_metrics(y, probs, threshold)})
        save_rows(output / "stability.csv", stability)
        summary["stability"] = stability
        if cfg["run_ablation"]:
            ablation = []
            for engineered in (False, True):
                probs, measure, _ = evaluate_oof(x, y, folds, params, cfg,
                                                  reference_models=reference_models,
                                                  engineered=engineered)
                ablation.append({"engineered": engineered, **classification_metrics(y, probs, threshold),
                                 "fit_seconds": measure["fit_seconds"],
                                 "latency_ratio": measure["latency_ratio"],
                                 "latency_ratio_q25": measure["latency_ratio_q25"],
                                 "latency_ratio_q75": measure["latency_ratio_q75"],
                                 "latency_samples_json": json.dumps(measure["latency_samples"]),
                                 "model_size_mb": measure["model_size_mb"]})
            save_rows(output / "ablation.csv", ablation)
            summary["ablation"] = [{k: v for k, v in row.items() if k != "latency_samples_json"}
                                   for row in ablation]

    if args.pilot:
        development_diagnostics(ranked[0])
        finish("pilot_complete_no_test_evaluation")
        print("Pilot complete; held-out test was not evaluated.", flush=True)
        return

    # Lock ranking before refit. Inspect only fit size and paired latency here;
    # no test features, probabilities or labels are accessed until a feasible
    # final model has been selected by this predeclared fallback rule.
    baseline_full = make_model(None, cfg["seed"])
    start = time.perf_counter()
    baseline_full.fit(x, y)
    baseline_full_fit_seconds = time.perf_counter() - start
    batch = x.iloc[: cfg["latency_batch_size"]]
    def evaluate_refit(rank, row):
        params = params_for_row(row)
        if params is None:
            candidate = baseline_full
            fit_seconds = baseline_full_fit_seconds
        else:
            candidate = make_model(params, cfg["seed"])
            start = time.perf_counter()
            candidate.fit(x, y)
            fit_seconds = time.perf_counter() - start
        size = model_size_mb(candidate)
        samples = paired_latency(candidate, baseline_full, batch,
                                 warmup=cfg["latency_warmup"], rounds=cfg["latency_repeats"])
        measure = summarize_pairs(samples)
        size_ok = size <= cfg["max_model_mb"]
        latency_ok = measure["latency_ratio"] <= cfg["max_latency_ratio"]
        attempt = {"rank": rank, "method": row["method"], "trial": row["trial"],
                   "params_json": row["params_json"], "threshold": row["threshold"],
                   "oof_f2": row["f2"], "oof_fpr": row["fpr"],
                   "fit_seconds": fit_seconds, "model_size_mb": size,
                   **measure, "latency_samples_json": json.dumps(samples),
                   "size_ok": bool(size_ok), "latency_ok": bool(latency_ok),
                   "feasible": bool(size_ok and latency_ok)}
        return attempt, candidate, measure, size

    def record_refit(attempt, attempts):
        save_rows(output / "refit_attempts.csv", attempts)
        print(f"Refit rank {attempt['rank']}: size={attempt['model_size_mb']:.3f} MB, "
              f"paired ratio={attempt['latency_ratio']:.3f}, "
              f"feasible={attempt['feasible']}", flush=True)

    selected, final_model, final_measure, final_size, attempts = try_ranked_candidates(
        ranked, evaluate_refit, record_refit)
    summary["refit_attempts"] = [{k: v for k, v in attempt.items() if k != "latency_samples_json"}
                                  for attempt in attempts]
    if selected is None:
        finish("no_feasible_refit_configuration")
        print("No feasible full-development refit; test was not evaluated.", flush=True)
        return
    summary["selected_final"] = selected
    summary["fallback_used"] = bool(attempts[-1]["rank"] > 1)
    development_diagnostics(selected)
    joblib.dump(final_model, output / "final_model.joblib", compress=0)

    test = df.loc[test_idx]
    start = time.perf_counter()
    test_prob = final_model.predict_proba(test[FEATURES])[:, 1]
    test_predict_seconds = time.perf_counter() - start
    test_metrics = classification_metrics(test[TARGET].to_numpy(), test_prob, selected["threshold"])
    test_metrics.update({"latency_ms_per_batch": final_measure["latency_candidate_ms"],
                         "latency_baseline_ms_per_batch": final_measure["latency_reference_ms"],
                         "latency_ratio": final_measure["latency_ratio"],
                         "latency_ratio_q25": final_measure["latency_ratio_q25"],
                         "latency_ratio_q75": final_measure["latency_ratio_q75"],
                         "latency_ratio_mad": final_measure["latency_ratio_mad"],
                         "model_size_mb": final_size,
                         "predict_test_seconds": test_predict_seconds,
                         "final_refit_resource_feasible": True})
    manifest["split"]["test_positives"] = int(test[TARGET].sum())
    summary["test"] = test_metrics
    finish("complete")
    print(f"Selected {selected['method']} trial {selected['trial']}; test F2={test_metrics['f2']:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--trials", type=int, help="Override budget for each search method")
    parser.add_argument("--pilot", action="store_true", help="Label outputs as a pilot run")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
