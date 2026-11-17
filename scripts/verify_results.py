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
import re
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import FEATURES, TARGET, load_data, split_data  # noqa: E402
from src.modeling import classification_metrics  # noqa: E402
from src.experiment import make_folds, ranked_candidates, params_for_row  # noqa: E402
from src.latency import summarize_pairs  # noqa: E402

BEGIN = "<!-- GENERATED_RESULTS_BEGIN -->"
END = "<!-- GENERATED_RESULTS_END -->"


def near(actual, expected, label, tolerance=1e-10):
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{label}: {actual} != {expected}")


def render_results(summary: dict, manifest: dict, baselines: pd.DataFrame,
                   trials: pd.DataFrame) -> str:
    cfg = manifest["config"]
    if summary["status"] != "complete":
        return f"Trạng thái lần chạy: **{summary['status']}**. Chưa có kết quả đánh giá cuối hợp lệ trên tập kiểm tra."
    selected, test = summary["selected_final"], summary["test"]
    lines = [
        f"**Thực nghiệm chính thức:** {cfg['trials_per_method']} lượt Random Search + {cfg['trials_per_method']} "
        "lượt Optuna TPE, ba mô hình cơ sở có ngưỡng cố định và hai mô hình cơ sở chọn ngưỡng từ OOF. "
        f"Số lượt tìm kiếm khả thi: Random Search {summary['search_counts']['random_search']}, "
        f"TPE {summary['search_counts']['optuna_tpe']}.",
        "",
        "| Cấu hình trên OOF của tập phát triển | F₂ | Recall | Precision | FPR | Ngưỡng |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "dummy": "Dummy: luôn chọn lớp phổ biến nhất",
        "rf_default": "RF mặc định, ngưỡng 0.5",
        "rf_balanced": "RF cân bằng lớp, ngưỡng 0.5",
        "rf_default_oof_threshold": "RF mặc định, ngưỡng OOF",
        "rf_balanced_oof_threshold": "RF cân bằng lớp, ngưỡng OOF",
        "random_search": "Random Search",
        "optuna_tpe": "Optuna TPE",
    }
    for _, row in baselines.iterrows():
        lines.append(f"| {labels.get(row['method'], row['method'])} | {row['f2']:.4f} | {row['recall']:.4f} | "
                     f"{row['precision']:.4f} | {row['fpr']:.4f} | {row['threshold']:.4f} |")
    best_by_method = {}
    for method in ("random_search", "optuna_tpe"):
        group = trials[(trials["method"] == method) & (trials["feasible"])]
        if len(group):
            best = group.sort_values(["f2", "fpr", "model_size_mb", "latency_ratio"],
                                     ascending=[False, True, True, True]).iloc[0]
            best_by_method[method] = best
            lines.append(f"| Tốt nhất của {labels[method]} (lượt #{int(best['trial'])}) | {best['f2']:.4f} | "
                         f"{best['recall']:.4f} | {best['precision']:.4f} | "
                         f"{best['fpr']:.4f} | {best['threshold']:.4f} |")
        else:
            lines.append(f"| Tốt nhất của {labels[method]} | Không có lượt khả thi | — | — | — | — |")
    lines.extend(["", "Chỉ số lượt thử được đánh số từ 0, giống cột `trial` trong log.", "",
                  f"Nghiệm cuối đã chốt thuộc **{labels.get(selected['method'], selected['method'])}**, lượt "
                  f"**#{selected['trial']}**, với `params_json={selected['params_json']}` và "
                  f"ngưỡng **{selected['threshold']:.6f}**. "
                  f"Trên OOF của tập phát triển, F₂ đạt **{selected['f2']:.4f}**, FPR "
                  f"**{selected['fpr']:.4f}**. "
                  f"Phải chuyển sang nghiệm dự phòng sau khi huấn luyện lại: **{'Có' if summary['fallback_used'] else 'Không'}**."])
    if len(best_by_method) == 2:
        left, right = best_by_method.values()
        if (left["params_json"] == right["params_json"] and
                math.isclose(left["threshold"], right["threshold"], abs_tol=1e-12)):
            lines.append("Hai phương pháp tìm được cùng bộ siêu tham số và ngưỡng tốt nhất; "
                         "lần chạy này chưa chứng minh phương pháp nào ưu việt hơn.")
        else:
            lines.append("Hai phương pháp tìm được các nghiệm tốt nhất khác nhau trong lần chạy này; "
                         "một ngân sách và một cách chia dữ liệu chưa chứng minh ưu thế tổng quát của phương pháp nào.")
    lines.extend([
        "",
        "**Đánh giá trên tập kiểm tra sau khi chọn nghiệm chỉ bằng tập phát triển:** ma trận nhầm lẫn "
        f"`[[{test['tn']}, {test['fp']}], [{test['fn']}, {test['tp']}]]` "
        "(hàng là nhãn thực 0/1, cột là nhãn dự đoán 0/1). "
        f"F₂ **{test['f2']:.4f}**, độ bao phủ (Recall) **{test['recall']:.4f}**, "
        f"độ chính xác cảnh báo (Precision) **{test['precision']:.4f}**, FPR **{test['fpr']:.4f}**, "
        f"AP/PR-AUC **{test['average_precision']:.4f}**, "
        f"tỷ lệ dự đoán đúng (Accuracy) **{test['accuracy']:.4f}**. "
        f"Kích thước mô hình cuối không nén **{test['model_size_mb']:.3f} MB** "
        f"(giới hạn {cfg['max_model_mb']} MB); tỷ số độ trễ đo ghép cặp "
        f"**{test['latency_ratio']:.3f}×** (giới hạn {cfg['max_latency_ratio']}×), "
        f"khoảng phân vị q25–q75 **{test['latency_ratio_q25']:.3f}–{test['latency_ratio_q75']:.3f}×**.",
        "",
        "![Diễn biến tìm kiếm nghiệm khả thi tốt nhất](results/full/figures/optimization_history.png)",
        "",
        "![Quan hệ giữa F₂ và tỷ lệ cảnh báo sai FPR](results/full/figures/f2_fpr.png)",
        "",
        "Xem [phân tích độ nhạy](results/full/sensitivity.csv), "
        "[độ ổn định](results/full/stability.csv), "
        "[thí nghiệm bổ sung đặc trưng](results/full/ablation.csv) và "
        "[các lần huấn luyện lại](results/full/refit_attempts.csv).",
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
        readme.write_text(before + BEGIN + expected + END + after, encoding="utf-8", newline="\n")
    elif current != expected:
        raise AssertionError("README result block differs from run logs; use --write-readme")


def verify_relative_links(readme: Path) -> int:
    from urllib.parse import unquote
    links = re.findall(r"!?\[[^\]]*\]\(([^\s)]+)\)", readme.read_text(encoding="utf-8"))
    local = [unquote(link.split("#")[0]) for link in links
             if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", link) and not link.startswith("#")]
    for link in local:
        if not (readme.parent / link).exists():
            raise AssertionError(f"Broken README link: {link}")
    return len(local)


def verify_hashes(base: Path, hashes: dict) -> None:
    for relative, expected in hashes.items():
        path = (base / relative).resolve()
        if not path.is_relative_to(base.resolve()) or not path.is_file():
            raise AssertionError(f"Missing/unsafe artifact: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise AssertionError(f"Artifact hash differs: {relative}")


def verify_candidate_rows(rows: pd.DataFrame, cfg: dict) -> None:
    for row in rows.to_dict("records"):
        label = f"{row['method']}:{row['trial']}"
        tp, fp, fn, tn = (row[k] for k in ("tp", "fp", "fn", "tn"))
        near(tp + fn, 271, label + " positives")
        near(tn + fp, 7729, label + " negatives")
        near(row["f2"], 5*tp / max(5*tp + 4*fn + fp, 1), label + " F2")
        near(row["recall"], tp / (tp + fn), label + " recall")
        near(row["precision"], tp / max(tp + fp, 1), label + " precision")
        near(row["fpr"], fp / (fp + tn), label + " FPR")
        near(row["accuracy"], (tp+tn) / (tp+tn+fp+fn), label + " accuracy")
        samples = json.loads(row["latency_samples_json"])
        for s in samples:
            near(s["ratio"], s["candidate_ms"] / s["reference_ms"], label + " raw ratio")
        recomputed = summarize_pairs(samples)
        for key in ("latency_ratio", "latency_ratio_q25", "latency_ratio_q75", "latency_ratio_mad", "latency_rounds"):
            # The default reference's median ratio is explicitly defined as 1.
            expected = 1.0 if key == "latency_ratio" and row["method"] in {"rf_default", "rf_default_oof_threshold"} else recomputed[key]
            near(row[key], expected, label + " " + key)
        near(row["model_size_mb"], max(json.loads(row["fold_sizes_mb_json"])), label + " size")
        flags = {"fpr_ok": row["fpr"] <= cfg["primary_fpr_limit"] + 1e-12,
                 "size_ok": row["model_size_mb"] <= cfg["max_model_mb"],
                 "latency_ok": row["latency_ratio"] <= cfg["max_latency_ratio"]}
        flags["feasible"] = all(flags.values())
        for key, expected in flags.items():
            if bool(row[key]) != expected:
                raise AssertionError(label + " inconsistent " + key)


def verify(root: Path, readme: Path, *, write_readme: bool = False) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if manifest["run_type"] != "full" or summary["run_type"] != "full":
        raise AssertionError("Expected a full run, not a pilot")
    cfg = manifest["config"]
    baselines = pd.read_csv(root / "baselines.csv")
    trials = pd.read_csv(root / "trials.csv")
    all_rows = pd.concat([baselines, trials], ignore_index=True)
    verify_candidate_rows(all_rows, cfg)
    sensitivity = pd.read_csv(root / "sensitivity.csv")
    ranking = pd.read_csv(root / "candidate_ranking.csv")
    attempts = pd.read_csv(root / "refit_attempts.csv")
    if len(baselines) != 5:
        raise AssertionError("Five baseline rows required")
    expected_baselines = {"dummy", "rf_default", "rf_balanced", "rf_default_oof_threshold", "rf_balanced_oof_threshold"}
    if set(baselines["method"]) != expected_baselines:
        raise AssertionError("Unexpected baseline identities")
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
    expected_ranking = ranked_candidates(all_rows.to_dict("records"))
    for expected, actual in zip(expected_ranking, ranking.to_dict("records"), strict=True):
        for key in ("method", "trial", "params_json"):
            if expected[key] != actual[key]:
                raise AssertionError(f"Candidate ranking differs: {key}")
        near(expected["threshold"], actual["threshold"], "ranking threshold")
    if len(attempts) < 1 or not bool(attempts.iloc[-1]["feasible"]):
        raise AssertionError("No feasible final refit")
    if len(attempts) > 1 and bool(attempts.iloc[:-1]["feasible"].any()):
        raise AssertionError("Refit continued after a feasible candidate")
    if bool(summary["fallback_used"]) != (len(attempts) > 1):
        raise AssertionError("Fallback flag does not match attempts")
    selected = summary["selected_final"]
    selected_rows = all_rows[(all_rows.method == selected["method"]) & (all_rows.trial == selected["trial"])]
    if len(selected_rows) != 1:
        raise AssertionError("Selected candidate missing or duplicated")
    for key in ("f2", "fpr", "threshold", "recall", "precision"):
        near(selected_rows.iloc[0][key], selected[key], "selected " + key)
    for attempt, candidate in zip(attempts.to_dict("records"), expected_ranking):
        for key in ("method", "trial", "params_json"):
            if attempt[key] != candidate[key]:
                raise AssertionError("Refit did not follow development ranking")
        raw = json.loads(attempt["latency_samples_json"])
        for sample in raw:
            near(sample["ratio"], sample["candidate_ms"] / sample["reference_ms"], "refit raw ratio")
        for key, value in summarize_pairs(raw).items():
            near(attempt[key], value, "refit " + key)
        expected_feasible = (attempt["model_size_mb"] <= cfg["max_model_mb"] and
                             attempt["latency_ratio"] <= cfg["max_latency_ratio"])
        if bool(attempt["feasible"]) != expected_feasible:
            raise AssertionError("Refit feasibility differs from resources")
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
    dev_idx, test_idx = split_data(df, cfg["seed"])
    expected_sets = {int(i): "development" for i in dev_idx} | {int(i): "test" for i in test_idx}
    if dict(zip(assignments.row_index, assignments["set"])) != expected_sets:
        raise AssertionError("Saved split assignments differ from seeded split")
    folds = make_folds(df.loc[dev_idx, TARGET].to_numpy(), cfg["seed"], cfg["folds"])
    expected_folds = {int(dev_idx[pos]): k for k, (_, valid) in enumerate(folds) for pos in valid}
    if dict(zip(fold_assignments.row_index, fold_assignments.fold)) != expected_folds:
        raise AssertionError("Saved fold assignments differ from seeded folds")
    saved_model = joblib.load(root / "final_model.joblib")
    actual_params = saved_model.named_steps["classifier"].get_params()
    for key, expected in (params_for_row(selected) or {}).items():
        if actual_params[key] != expected:
            raise AssertionError(f"Saved model parameter differs: {key}")
    near(actual_params["random_state"], cfg["seed"], "model seed")
    near(actual_params["n_jobs"], 1, "model CPU setting")
    near((root / "final_model.joblib").stat().st_size / 1_000_000, test["model_size_mb"], "saved bytes")
    near(selected["threshold"], test["threshold"], "test threshold")
    near(selected["threshold"], final_attempt["threshold"], "refit threshold")
    test_df = df.loc[test_idx]
    probabilities = saved_model.predict_proba(test_df[FEATURES])[:, 1]
    recomputed = classification_metrics(test_df[TARGET].to_numpy(), probabilities, test["threshold"])
    for key in ("tn", "fp", "fn", "tp", "f2", "recall", "precision", "fpr", "average_precision", "accuracy"):
        near(recomputed[key], test[key], f"saved model {key}")
    for name in ("stability", "ablation"):
        frame = pd.read_csv(root / f"{name}.csv")
        for logged, recorded in zip(frame.to_dict("records"), summary[name], strict=True):
            for key in recorded:
                near(logged[key], recorded[key], f"{name} {key}")
    report_metadata = PROJECT_ROOT / "report/build_metadata.json"
    if root.resolve() == (PROJECT_ROOT / "results/full").resolve():
        if json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8")) != cfg:
            raise AssertionError("Official manifest config differs from config.json")
        if not report_metadata.is_file():
            raise AssertionError("Missing report/build_metadata.json")
        metadata = json.loads(report_metadata.read_text(encoding="utf-8"))
        actual_hash = hashlib.sha256((root / "summary.json").read_bytes()).hexdigest()
        if metadata["source_summary_sha256"] != actual_hash:
            raise AssertionError("Report was built from another summary.json")
        verify_hashes(PROJECT_ROOT, metadata["source_hashes"])
        verify_hashes(report_metadata.parent, metadata["output_hashes"])
        for name in (metadata["report_pdf"], metadata["report_docx"], metadata["defense_summary_pdf"], metadata["defense_summary_docx"]):
            if not (report_metadata.parent / name).is_file():
                raise AssertionError(f"Missing report artifact: {name}")
    block = render_results(summary, manifest, baselines, trials)
    replace_readme_block(readme, block, write=write_readme)
    link_count = verify_relative_links(readme)
    return {"status": "verified", "trials": len(trials), "baselines": len(baselines),
            "selected_method": selected["method"], "test_f2": test["f2"], "relative_links": link_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results/full")
    parser.add_argument("--readme", type=Path, default=PROJECT_ROOT / "README.md")
    parser.add_argument("--write-readme", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.results_dir, args.readme, write_readme=args.write_readme), indent=2))


if __name__ == "__main__":
    main()
