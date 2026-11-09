"""Data contract and the locked development/test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Machine failure"
FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FAILURE_FLAGS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
EXPECTED_COLUMNS = ["UDI", "Product ID", *FEATURES, TARGET, *FAILURE_FLAGS]
DEFAULT_DATA = Path("ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv")


def load_data(path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required CSV: {path.resolve()}")
    raw = path.read_bytes()
    df = pd.read_csv(path)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns/order: {list(df.columns)}")
    if len(df) != 10_000 or df.shape[1] != 14:
        raise ValueError(f"Unexpected dataset shape: {df.shape}")
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values")
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must be binary")
    mismatch = df[TARGET].ne(df[FAILURE_FLAGS].any(axis=1).astype(int))
    audit = {
        "csv_path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "missing_cells": int(df.isna().sum().sum()),
        "label_flag_or_mismatches": int(mismatch.sum()),
        "sequential_70_15_15_hdf_counts_diagnostic_only": {
            "train": int(df.iloc[:7000]["HDF"].sum()),
            "validation": int(df.iloc[7000:8500]["HDF"].sum()),
            "test": int(df.iloc[8500:]["HDF"].sum()),
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "type_counts": {str(k): int(v) for k, v in df["Type"].value_counts().sort_index().items()},
        "features": FEATURES,
    }
    return df, audit


def split_data(df: pd.DataFrame, seed: int):
    indices = df.index.to_numpy()
    dev_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=df[TARGET], random_state=seed
    )
    return dev_idx, test_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the supplied AI4I CSV")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df, audit = load_data(args.data)
    dev, test = split_data(df, args.seed)
    audit["split"] = {
        "method": "stratified train_test_split",
        "seed": args.seed,
        "development_rows": len(dev),
        "test_rows": len(test),
        "development_positives": int(df.loc[dev, TARGET].sum()),
        "test_positives": int(df.loc[test, TARGET].sum()),
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
