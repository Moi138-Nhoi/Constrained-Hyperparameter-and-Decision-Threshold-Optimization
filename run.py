"""Run the complete AI4I experiment from any working directory.

Default: 40 Random Search and 40 Optuna TPE trials. Command-line options
such as --pilot, --trials and --config are forwarded to src.experiment.
"""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    from src.experiment import main as experiment_main

    experiment_main()


if __name__ == "__main__":
    main()
