import unittest

import numpy as np
import pandas as pd

from src.data import FEATURES, TARGET, DEFAULT_DATA, load_data, split_data
from src.modeling import FeatureMaker, best_threshold, make_model


class DataAndThresholdTests(unittest.TestCase):
    def test_supplied_data_contract_and_split(self):
        frame, audit = load_data(DEFAULT_DATA)
        self.assertEqual(audit["class_counts"], {"0": 9661, "1": 339})
        self.assertEqual(audit["label_flag_or_mismatches"], 27)
        self.assertEqual(audit["sequential_70_15_15_hdf_counts_diagnostic_only"]["validation"], 0)
        self.assertEqual(audit["sequential_70_15_15_hdf_counts_diagnostic_only"]["test"], 0)
        self.assertEqual(len(FEATURES), 6)
        self.assertFalse(set(FEATURES) & {"UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF", TARGET})
        dev, test = split_data(frame, 42)
        self.assertEqual((len(dev), len(test)), (8000, 2000))
        self.assertEqual(len(set(dev) & set(test)), 0)
        self.assertEqual(int(frame.loc[test, TARGET].sum()), 68)

    def test_threshold_can_choose_no_alarm(self):
        y = np.array([0, 0, 0, 1])
        prob = np.zeros(4)
        chosen = best_threshold(y, prob, 0.05)
        self.assertEqual(chosen["threshold"], 1.0)
        self.assertEqual(chosen["fpr"], 0.0)
        self.assertEqual(chosen["f2"], 0.0)
        impossible = best_threshold(y, np.ones(4), 0.05)
        self.assertEqual(impossible["threshold"], 1.0)
        self.assertGreater(impossible["fpr"], 0.05)

    def test_fold_preprocessor_ignores_unseen_type(self):
        train = pd.DataFrame({
            "Type": ["L", "L", "M", "M"],
            "Air temperature [K]": [300, 301, 302, 303],
            "Process temperature [K]": [310, 311, 312, 313],
            "Rotational speed [rpm]": [1000, 1100, 1200, 1300],
            "Torque [Nm]": [30, 31, 32, 33],
            "Tool wear [min]": [1, 2, 3, 4],
        })
        model = make_model({"n_estimators": 5}, 42)
        model.fit(train, [0, 0, 1, 1])
        self.assertEqual(set(model.named_steps["preprocess"].named_transformers_["type"].categories_[0]), {"L", "M"})
        unseen = train.iloc[[0]].copy()
        unseen["Type"] = "H"
        self.assertEqual(len(model.predict_proba(unseen)), 1)
        made = FeatureMaker(engineered=True).transform(unseen)
        self.assertEqual(len(made.columns), 9)
        self.assertEqual(made["Temperature difference [K]"].iloc[0], 10)


if __name__ == "__main__":
    unittest.main()
