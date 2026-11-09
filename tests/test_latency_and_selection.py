import unittest

import numpy as np

from src.experiment import ranked_candidates, try_ranked_candidates
from src.latency import paired_latency, summarize_pairs
from src.modeling import classification_metrics


class FakeModel:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def predict_proba(self, batch):
        self.calls.append(self.name)
        return np.zeros((len(batch), 2))


class LatencyAndSelectionTests(unittest.TestCase):
    def test_f2_and_fpr_match_confusion_matrix(self):
        y = np.array([0, 0, 0, 0, 1, 1])
        p = np.array([0.9, 0.1, 0.1, 0.1, 0.8, 0.2])
        m = classification_metrics(y, p, 0.5)
        self.assertEqual((m["tn"], m["fp"], m["fn"], m["tp"]), (3, 1, 1, 1))
        self.assertAlmostEqual(m["f2"], 5 / (5 + 4 + 1))
        self.assertAlmostEqual(m["fpr"], 1 / 4)

    def test_paired_order_and_robust_median(self):
        calls = []
        candidate, reference = FakeModel("A", calls), FakeModel("B", calls)
        samples = paired_latency(candidate, reference, [0], warmup=0, rounds=2)
        self.assertEqual(calls, ["A", "B", "B", "A", "B", "A", "A", "B"])
        self.assertEqual(len(samples), 2)
        synthetic = [{"ratio": 1.0, "candidate_ms": 10.0, "reference_ms": 10.0}] * 10
        synthetic.append({"ratio": 100.0, "candidate_ms": 1000.0, "reference_ms": 10.0})
        self.assertEqual(summarize_pairs(synthetic)["latency_ratio"], 1.0)

    def test_rank_ignores_dummy_and_deduplicates_decisions(self):
        base = {"feasible": True, "f2": 0.7, "fpr": 0.01,
                "model_size_mb": 1.0, "latency_ratio": 1.1,
                "params_json": '{"n_estimators": 50}', "threshold": 0.2, "trial": 0}
        rows = [
            {**base, "method": "dummy", "f2": 1.0},
            {**base, "method": "random_search", "latency_ratio": 1.2},
            {**base, "method": "optuna_tpe", "latency_ratio": 1.1},
            {**base, "method": "random_search", "trial": 1,
             "params_json": '{"n_estimators": 100}', "f2": 0.6},
        ]
        ranked = ranked_candidates(rows)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["method"], "optuna_tpe")
        self.assertEqual(ranked[1]["trial"], 1)

    def test_refit_fallback_uses_second_candidate_without_test(self):
        ranked = [{"trial": 0}, {"trial": 1}, {"trial": 2}]
        evaluated, logged = [], []

        def evaluator(rank, row):
            evaluated.append(row["trial"])
            feasible = rank == 2
            return ({"rank": rank, "feasible": feasible}, f"model-{rank}",
                    {"latency_ratio": 1.1 if feasible else 1.4}, 1.0)

        def record(attempt, attempts):
            logged.append((attempt["rank"], len(attempts)))

        row, model, measure, size, attempts = try_ranked_candidates(ranked, evaluator, record)
        self.assertEqual(evaluated, [0, 1])
        self.assertEqual(logged, [(1, 1), (2, 2)])
        self.assertEqual(row["trial"], 1)
        self.assertEqual(model, "model-2")
        self.assertEqual(len(attempts), 2)


if __name__ == "__main__":
    unittest.main()
