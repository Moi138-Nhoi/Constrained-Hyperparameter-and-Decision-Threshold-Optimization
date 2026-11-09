"""Reproduce baselines, constrained search, final evaluation and figures."""

from __future__ import annotations

import argparse
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
from .modeling import best_threshold, classification_metrics, latency_ms, make_model, model_size_mb


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def make_folds(y: np.ndarray, seed: int, n_splits: int):
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y))


def evaluate_oof(x: pd.DataFrame, y: np.ndarray, folds, params, cfg, *, dummy=False, engineered=False, model_seed=None):
    probs = np.empty(len(y), dtype=float)
    fit_seconds, predict_seconds, latencies, sizes = 0.0, 0.0, [], []
    seed = cfg["seed"] if model_seed is None else model_seed
    for train, valid in folds:
        model = make_model(params, seed, dummy=dummy, engineered=engineered)
        start = time.perf_counter()
        model.fit(x.iloc[train], y[train])
        fit_seconds += time.perf_counter() - start
        start = time.perf_counter()
        probs[valid] = model.predict_proba(x.iloc[valid])[:, 1]
        predict_seconds += time.perf_counter() - start
        batch = x.iloc[valid[: cfg["latency_batch_size"]]]
        latencies.append(latency_ms(model, batch, warmup=cfg["latency_warmup"], repeats=cfg["latency_repeats"]))
        sizes.append(model_size_mb(model))
    return probs, {
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "latency_ms_per_batch": float(np.median(latencies)),
        "model_size_mb": float(max(sizes)),
        "fold_latencies_ms": [float(v) for v in latencies],
        "fold_sizes_mb": [float(v) for v in sizes],
    }


def record_candidate(method, trial, params, probs, y, measurements, baseline_latency, cfg, *, fixed_threshold=None):
    limit = cfg["primary_fpr_limit"]
    primary = (best_threshold(y, probs, limit) if fixed_threshold is None
               else classification_metrics(y, probs, fixed_threshold))
    ratio = measurements["latency_ms_per_batch"] / baseline_latency
    resources_ok = measurements["model_size_mb"] <= cfg["max_model_mb"] and ratio <= cfg["max_latency_ratio"]
    threshold_ok = primary["threshold"] is not None and primary["fpr"] <= limit + 1e-12
    row = {
        "method": method, "trial": trial, "params_json": json.dumps(params, sort_keys=True),
        **primary,
        "fit_seconds": measurements["fit_seconds"],
        "predict_seconds": measurements["predict_seconds"],
        "latency_ms_per_batch": measurements["latency_ms_per_batch"],
        "latency_ratio": float(ratio),
        "model_size_mb": measurements["model_size_mb"],
        "fold_latencies_ms_json": json.dumps(measurements["fold_latencies_ms"]),
        "fold_sizes_mb_json": json.dumps(measurements["fold_sizes_mb"]),
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
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.trials is not None:
        cfg["trials_per_method"] = args.trials
    if cfg["trials_per_method"] < 1:
        raise ValueError("trials_per_method must be positive")
    output = args.output_dir or Path("results/pilot" if args.pilot else "results/full")
    output.mkdir(parents=True, exist_ok=True)
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
                  "test_positives": int(df.loc[test_idx, TARGET].sum()),
                  "fold_method": "StratifiedKFold(shuffle=True)", "folds": cfg["folds"]},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "processor": platform.processor(),
                    "packages": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "optuna", "matplotlib", "joblib", "threadpoolctl"]},
                    "rf_n_jobs": 1, "threadpool_limit": 1},
        "measurement": (f"Each fold: {cfg['latency_warmup']} warmups and {cfg['latency_repeats']} timed "
                        f"predict_proba calls on the first {cfg['latency_batch_size']} validation rows; "
                        "median batch milliseconds. Trial latency is median of fold medians; model size is "
                        "maximum uncompressed joblib bytes across folds. Same protocol for baselines and search."),
    }
    save_json(output / "manifest.json", manifest)
    baseline_specs = [
        ("dummy", None, True),
        ("rf_default", None, False),
        ("rf_balanced", {"class_weight": "balanced"}, False),
    ]
    base_results = {}
    for name, params, dummy in baseline_specs:
        probs, measure = evaluate_oof(x, y, folds, params, cfg, dummy=dummy)
        base_results[name] = (probs, measure)
        print(f"Baseline {name}: fit {measure['fit_seconds']:.1f}s", flush=True)
    reference_latency = base_results["rf_default"][1]["latency_ms_per_batch"]
    if reference_latency <= 0:
        raise RuntimeError("Invalid baseline latency")
    baseline_rows, sensitivity_rows = [], []
    for name, params, _ in baseline_specs:
        probs, measure = base_results[name]
        row, sens = record_candidate(name, -1, params or {}, probs, y, measure, reference_latency, cfg,
                                     fixed_threshold=0.5)
        baseline_rows.append(row)
        sensitivity_rows.extend(sens)
    save_rows(output / "baselines.csv", baseline_rows)
    trial_rows = []

    def evaluate_trial(method, number, params):
        probs, measure = evaluate_oof(x, y, folds, params, cfg)
        row, sens = record_candidate(method, number, params, probs, y, measure, reference_latency, cfg)
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
    # Dummy is a diagnostic baseline, not a Random Forest solution.
    feasible = [r for r in baseline_rows + trial_rows if r["feasible"] and r["method"] != "dummy"]
    summary = {"run_type": manifest["run_type"], "feasible_count": len(feasible),
               "baseline_results": baseline_rows,
               "search_counts": {m: sum(r["method"] == m and r["feasible"] for r in trial_rows)
                                 for m in ("random_search", "optuna_tpe")}}
    if not feasible:
        summary["status"] = "no_feasible_configuration"
        save_json(output / "summary.json", summary)
        print("No feasible configuration; test set was not evaluated.", flush=True)
        return
    selected = max(feasible, key=lambda r: (r["f2"], -r["fpr"], -r["model_size_mb"], -r["latency_ms_per_batch"]))
    params = json.loads(selected["params_json"])
    dummy = selected["method"] == "dummy"
    if selected["method"] == "rf_default":
        params = None
    elif selected["method"] == "rf_balanced":
        params = {"class_weight": "balanced"}
    threshold = selected["threshold"]
    summary["selected_from_development"] = selected

    stability = []
    for seed in cfg["stability_seeds"]:
        seed_folds = make_folds(y, seed, cfg["folds"])
        probs, _ = evaluate_oof(x, y, seed_folds, params, cfg, dummy=dummy, model_seed=seed)
        stability.append({"seed": seed, **classification_metrics(y, probs, threshold)})
    save_rows(output / "stability.csv", stability)
    summary["stability"] = stability

    if cfg["run_ablation"]:
        ablation = []
        for engineered in (False, True):
            probs, measure = evaluate_oof(x, y, folds, params, cfg, dummy=dummy, engineered=engineered)
            ablation.append({"engineered": engineered, **classification_metrics(y, probs, threshold),
                             "fit_seconds": measure["fit_seconds"],
                             "latency_ms_per_batch": measure["latency_ms_per_batch"],
                             "model_size_mb": measure["model_size_mb"]})
        save_rows(output / "ablation.csv", ablation)
        summary["ablation"] = ablation

    if args.pilot:
        summary["status"] = "pilot_complete_test_untouched"
        save_json(output / "summary.json", summary)
        print("Pilot complete; held-out test was not evaluated.", flush=True)
        return

    # All choices above are frozen before the held-out test is accessed.
    baseline_full = make_model(None, cfg["seed"])
    baseline_full.fit(x, y)
    batch = x.iloc[: cfg["latency_batch_size"]]
    baseline_full_latency = latency_ms(baseline_full, batch, warmup=cfg["latency_warmup"], repeats=cfg["latency_repeats"])
    final_model = make_model(params, cfg["seed"], dummy=dummy)
    final_model.fit(x, y)
    final_latency = latency_ms(final_model, batch, warmup=cfg["latency_warmup"], repeats=cfg["latency_repeats"])
    final_size = model_size_mb(final_model)
    joblib.dump(final_model, output / "final_model.joblib", compress=0)
    test = df.loc[test_idx]
    start = time.perf_counter()
    test_prob = final_model.predict_proba(test[FEATURES])[:, 1]
    test_predict_seconds = time.perf_counter() - start
    test_metrics = classification_metrics(test[TARGET].to_numpy(), test_prob, threshold)
    test_metrics.update({"latency_ms_per_batch": final_latency,
                         "latency_baseline_ms_per_batch": baseline_full_latency,
                         "latency_ratio": final_latency / baseline_full_latency,
                         "model_size_mb": final_size,
                         "predict_2000_seconds": test_predict_seconds,
                         "final_refit_resource_feasible": bool(final_size <= cfg["max_model_mb"] and
                                                               final_latency / baseline_full_latency <= cfg["max_latency_ratio"])})
    summary["status"] = "complete" if test_metrics["final_refit_resource_feasible"] else "final_refit_resource_violation"
    summary["test"] = test_metrics
    save_json(output / "summary.json", summary)
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
