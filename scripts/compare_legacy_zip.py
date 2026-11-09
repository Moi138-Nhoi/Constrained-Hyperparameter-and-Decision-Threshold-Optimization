"""Document the supplied ZIP's inconsistent run against the verified run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path


def compare(archive: Path, official: Path, output: Path) -> dict:
    with zipfile.ZipFile(archive) as zipped:
        old = json.loads(zipped.read("BT_Lon/results/full/summary.json"))
        old_readme = zipped.read("BT_Lon/README.md").decode("utf-8")
        old_manifest = json.loads(zipped.read("BT_Lon/results/full/manifest.json"))
    new = json.loads((official / "summary.json").read_text(encoding="utf-8"))
    new_manifest = json.loads((official / "manifest.json").read_text(encoding="utf-8"))
    match = re.search(r"(\d+) Random Search and (\d+) TPE trials met", old_readme)
    if not match:
        raise ValueError("Could not locate the legacy README's feasible counts")
    old_claim = {"random_search": int(match.group(1)), "optuna_tpe": int(match.group(2))}
    old_selected = old["selected_from_development"]
    new_selected = new["selected_final"]
    old_test, new_test = old["test"], new["test"]
    result = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "archive_name": archive.name,
        "legacy_readme_claimed_search_counts": old_claim,
        "legacy_summary_search_counts": old["search_counts"],
        "legacy_readme_summary_mismatch": old_claim != old["search_counts"],
        "legacy": {"status": old["status"], "selected_method": old_selected["method"],
                   "selected_trial": old_selected["trial"], "oof_f2": old_selected["f2"],
                   "test_f2": old_test["f2"], "final_latency_ratio": old_test["latency_ratio"],
                   "final_model_size_mb": old_test["model_size_mb"],
                   "latency_cap": old_manifest["config"]["max_latency_ratio"]},
        "official": {"status": new["status"], "selected_method": new_selected["method"],
                     "selected_trial": new_selected["trial"], "oof_f2": new_selected["f2"],
                     "test_f2": new_test["f2"], "final_latency_ratio": new_test["latency_ratio"],
                     "final_model_size_mb": new_test["model_size_mb"],
                     "latency_cap": new_manifest["config"]["max_latency_ratio"],
                     "search_counts": new["search_counts"],
                     "wall_seconds": new["wall_seconds"]},
        "note": "Legacy README and legacy summary are different run histories. The legacy final refit violates the 1.25 latency cap and is not a feasible result. The new test was evaluated only after development selection and a feasible refit. The older pilot had already viewed this test, so it is not strictly blind.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "archive_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
    lines = ["# Supplied ZIP versus verified official run", "",
             f"Source ZIP SHA-256: `{result['archive_sha256']}`. The ZIP is not tracked in Git.", "",
             "| Check | Supplied ZIP summary | Verified official run |", "|---|---:|---:|",
             f"| Status | {old['status']} | {new['status']} |",
             f"| Feasible Random Search trials | {old['search_counts']['random_search']} | {new['search_counts']['random_search']} |",
             f"| Feasible TPE trials | {old['search_counts']['optuna_tpe']} | {new['search_counts']['optuna_tpe']} |",
             f"| Selected decision | {old_selected['method']} #{old_selected['trial']} | {new_selected['method']} #{new_selected['trial']} |",
             f"| OOF F₂ | {old_selected['f2']:.4f} | {new_selected['f2']:.4f} |",
             f"| Test F₂ | {old_test['f2']:.4f} | {new_test['f2']:.4f} |",
             f"| Final latency ratio (cap 1.25×) | {old_test['latency_ratio']:.3f}× | {new_test['latency_ratio']:.3f}× |",
             f"| Final model size | {old_test['model_size_mb']:.3f} MB | {new_test['model_size_mb']:.3f} MB |",
             "", f"The ZIP README claimed {old_claim['random_search']} feasible Random Search and "
             f"{old_claim['optuna_tpe']} feasible TPE trials, which disagreed with its summary JSON. "
             "The ZIP's final model violated the latency cap and is not described as feasible.", "",
             "The verified run has a different latency protocol and a fresh output directory. "
             "Its test set is the same original holdout, which an older pilot had already viewed; "
             "the new optimization did not use pilot test metrics for selection."]
    (output / "archive_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path("../BT_Lon(1).zip"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/full"))
    parser.add_argument("--output-dir", type=Path, default=Path("report"))
    args = parser.parse_args()
    result = compare(args.zip, args.results_dir, args.output_dir)
    print(json.dumps({"mismatch": result["legacy_readme_summary_mismatch"],
                      "legacy_status": result["legacy"]["status"],
                      "official_status": result["official"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
