"""Check run artifacts and render the README results block from real logs.

Usage:
    python scripts/verify_results.py --results-dir results/full --write-readme
    python scripts/verify_results.py --results-dir results/full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import FEATURES, TARGET, load_data, split_data  # noqa: E402
from src.modeling import classification_metrics  # noqa: E402

BEGIN = "<!-- GENERATED_RESULTS_BEGIN -->"
END = "<!-- GENERATED_RESULTS_END -->"


def near(actual, expected, label, tolerance=1e-10):
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{label}: {actual} != {expected}")


def render_results(summary: dict, manifest: dict, baselines: pd.DataFrame,
                   trials: pd.DataFrame) -> str:
    cfg = manifest["config"]
    if summary["status"] != "complete":
        return f"Run status: **{summary['status']}**. No valid final test result is claimed."
    selected, test = summary["selected_final"], summary["test"]
    lines = [
        f"**Official run:** {cfg['trials_per_method']} Random Search + {cfg['trials_per_method']} "
        "Optuna TPE trials, three fixed baselines and two OOF-threshold baselines. "
        f"Feasible search trials: Random Search {summary['search_counts']['random_search']}, "
        f"TPE {summary['search_counts']['optuna_tpe']}.",
        "",
        "| Development OOF candidate | F₂ | Recall | Precision | FPR | Threshold |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in baselines.iterrows():
        lines.append(f"| {row['method']} | {row['f2']:.4f} | {row['recall']:.4f} | "
                     f"{row['precision']:.4f} | {row['fpr']:.4f} | {row['threshold']:.4f} |")
    best_by_method = {}
    for method in ("random_search", "optuna_tpe"):
        group = trials[(trials["method"] == method) & (trials["feasible"])]
        if len(group):
            best = group.sort_values(["f2", "fpr", "model_size_mb", "latency_ratio"],
                                     ascending=[False, True, True, True]).iloc[0]
            best_by_method[method] = best
            lines.append(f"| Best {method} (trial {int(best['trial'])}) | {best['f2']:.4f} | "
                         f"{best['recall']:.4f} | {best['precision']:.4f} | "
                         f"{best['fpr']:.4f} | {best['threshold']:.4f} |")
        else:
            lines.append(f"| Best {method} | no feasible trial | — | — | — | — |")
    lines.extend(["", f"The locked final decision came from **{selected['method']}**, trial "
                  f"**{selected['trial']}**, with `params_json={selected['params_json']}` and "
                  f"threshold **{selected['threshold']:.6f}**. "
                  f"Development OOF F₂ was **{selected['f2']:.4f}**, FPR "
                  f"**{selected['fpr']:.4f}**. "
                  f"Refit fallback used: **{summary['fallback_used']}**."])
    if len(best_by_method) == 2:
        left, right = best_by_method.values()
        if (left["params_json"] == right["params_json"] and
                math.isclose(left["threshold"], right["threshold"], abs_tol=1e-12)):
            lines.append("Both methods found the same best parameter and threshold decision; "
                         "this run does not establish superiority of either optimizer.")
        else:
            lines.append("The methods found different best candidates in this run; "
                         "one budget and one split do not establish general superiority.")
    lines.extend([
        "",
        "**Held-out test, after development-only selection:** confusion matrix "
        f"`[[{test['tn']}, {test['fp']}], [{test['fn']}, {test['tp']}]]` "
        "(rows actual 0/1, columns predicted 0/1). "
        f"F₂ **{test['f2']:.4f}**, recall **{test['recall']:.4f}**, "
        f"precision **{test['precision']:.4f}**, FPR **{test['fpr']:.4f}**, "
        f"average precision/PR-AUC **{test['average_precision']:.4f}**, "
        f"accuracy **{test['accuracy']:.4f}**. "
        f"Final uncompressed size **{test['model_size_mb']:.3f} MB** "
        f"(limit {cfg['max_model_mb']} MB); paired latency ratio "
        f"**{test['latency_ratio']:.3f}×** (limit {cfg['max_latency_ratio']}×), "
        f"with q25–q75 **{test['latency_ratio_q25']:.3f}–{test['latency_ratio_q75']:.3f}×**.",
        "",
        "![Optimization history](results/full/figures/optimization_history.png)",
        "",
        "![F₂ versus FPR](results/full/figures/f2_fpr.png)",
        "",
        "See [sensitivity](results/full/sensitivity.csv), "
        "[stability](results/full/stability.csv), "
        "[ablation](results/full/ablation.csv), and "
        "[refit attempts](results/full/refit_attempts.csv).",
    ])
    return "\n".join(lines)


def replace_readme_block(readme: Path, block: str, *, write: bool) -> None:
    text = readme.read_text(encoding="utf-8")
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise AssertionError("README must contain exactly one results marker pair")
    before, rest = text.split(BEGIN, 1)
    current, after = rest.split(END, 1)
    expected = "\n" + block + "\n"
    if write:
        readme.write_text(before + BEGIN + expected + END + after, encoding="utf-8")
    elif current != expected:
        raise AssertionError("README result block differs from run logs; use --write-readme")


def verify(root: Path, readme: Path, *, write_readme: bool = False) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if manifest["run_type"] != "full" or summary["run_type"] != "full":
        raise AssertionError("Expected a full run, not a pilot")
    cfg = manifest["config"]
    baselines = pd.read_csv(root / "baselines.csv")
    trials = pd.read_csv(root / "trials.csv")
    sensitivity = pd.read_csv(root / "sensitivity.csv")
    ranking = pd.read_csv(root / "candidate_ranking.csv")
    attempts = pd.read_csv(root / "refit_attempts.csv")
    if len(baselines) != 5:
        raise AssertionError("Five baseline rows required")
    for method in ("random_search", "optuna_tpe"):
        method_rows = trials[trials["method"] == method]
        if len(method_rows) != cfg["trials_per_method"]:
            raise AssertionError(f"Wrong {method} budget")
        if method_rows["trial"].tolist() != list(range(cfg["trials_per_method"])):
            raise AssertionError(f"Wrong {method} trial indices")
        if int(method_rows["feasible"].sum()) != summary["search_counts"][method]:
            raise AssertionError(f"Wrong {method} feasible count")
    if len(sensitivity) != (len(baselines) + len(trials)) * len(cfg["fpr_limits"]):
        raise AssertionError("Wrong sensitivity row count")
    if len(ranking) != summary["unique_ranked_decisions"]:
        raise AssertionError("Wrong candidate ranking length")
    if len(attempts) < 1 or not bool(attempts.iloc[-1]["feasible"]):
        raise AssertionError("No feasible final refit")
    if len(attempts) > 1 and bool(attempts.iloc[:-1]["feasible"].any()):
        raise AssertionError("Refit continued after a feasible candidate")
    if bool(summary["fallback_used"]) != (len(attempts) > 1):
        raise AssertionError("Fallback flag does not match attempts")
    selected = summary["selected_final"]
    final_attempt = attempts.iloc[-1]
    if (selected["method"] != final_attempt["method"] or
            selected["trial"] != int(final_attempt["trial"]) or
            selected["params_json"] != final_attempt["params_json"]):
        raise AssertionError("Selected final row differs from refit log")
    test = summary["test"]
    near(test["f2"], 5 * test["tp"] / (5 * test["tp"] + 4 * test["fn"] + test["fp"]), "test F2")
    near(test["fpr"], test["fp"] / (test["fp"] + test["tn"]), "test FPR")
    near(test["latency_ratio"], final_attempt["latency_ratio"], "test latency ratio")
    near(test["model_size_mb"], final_attempt["model_size_mb"], "test model size")
    if test["latency_ratio"] > cfg["max_latency_ratio"] or test["model_size_mb"] > cfg["max_model_mb"]:
        raise AssertionError("Final refit violates a resource constraint")
    data_path = PROJECT_ROOT / "ai4i+2020+predictive+maintenance+dataset" / "ai4i2020.csv"
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != manifest["dataset_audit"]["sha256"]:
        raise AssertionError("Dataset hash differs from manifest")
    assignments = pd.read_csv(root / "split_assignments.csv")
    fold_assignments = pd.read_csv(root / "fold_assignments.csv")
    if len(assignments) != 10_000 or assignments["row_index"].nunique() != 10_000:
        raise AssertionError("Split assignments do not cover each dataset row once")
    if len(fold_assignments) != 8_000 or fold_assignments["row_index"].nunique() != 8_000:
        raise AssertionError("Fold assignments do not cover each development row once")
    for relative in ("figures/optimization_history.png", "figures/f2_fpr.png", "final_model.joblib"):
        if not (root / relative).is_file() or (root / relative).stat().st_size == 0:
            raise AssertionError(f"Missing artifact: {relative}")
    df, _ = load_data(data_path)
    _, test_idx = split_data(df, cfg["seed"])
    saved_model = joblib.load(root / "final_model.joblib")
    test_df = df.loc[test_idx]
    probabilities = saved_model.predict_proba(test_df[FEATURES])[:, 1]
    recomputed = classification_metrics(test_df[TARGET].to_numpy(), probabilities, test["threshold"])
    for key in ("tn", "fp", "fn", "tp", "f2", "fpr", "average_precision", "accuracy"):
        near(recomputed[key], test[key], f"saved model {key}")
    report_metadata = PROJECT_ROOT / "report/build_metadata.json"
    if report_metadata.is_file():
        metadata = json.loads(report_metadata.read_text(encoding="utf-8"))
        actual_hash = hashlib.sha256((root / "summary.json").read_bytes()).hexdigest()
        if metadata["source_summary_sha256"] != actual_hash:
            raise AssertionError("Report was built from another summary.json")
        for name in (metadata["report_pdf"], metadata["defense_summary_pdf"]):
            if not (report_metadata.parent / name).is_file():
                raise AssertionError(f"Missing report artifact: {name}")
    block = render_results(summary, manifest, baselines, trials)
    replace_readme_block(readme, block, write=write_readme)
    return {"status": "verified", "trials": len(trials), "baselines": len(baselines),
            "selected_method": selected["method"], "test_f2": test["f2"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results/full")
    parser.add_argument("--readme", type=Path, default=PROJECT_ROOT / "README.md")
    parser.add_argument("--write-readme", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.results_dir, args.readme, write_readme=args.write_readme), indent=2))


if __name__ == "__main__":
    main()
