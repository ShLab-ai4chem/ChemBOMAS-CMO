"""CLI for FD_wqp condition design.

The original implementation is retained in ``legacy_run_realexp.py`` while
this thin adapter supplies repository-relative paths and explicit arguments.
Model prediction tensors are intentionally external artifacts.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ...paths import data_root, results_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Design the next FD_wqp wet-lab batch")
    parser.add_argument("--round", default="round_7", help="round directory, e.g. round_7")
    parser.add_argument("--first-round", action="store_true", help="use round-0 diversity sampling")
    parser.add_argument("--num-sample", type=int, default=8)
    parser.add_argument("--kappa", type=float, default=None, help="single kappa; omit to use the legacy comparison list")
    parser.add_argument("--predictions", type=Path, help="external LLM_predict.pt for the selected round")
    parser.add_argument("--output-dir", type=Path, help="override output directory")
    args = parser.parse_args()

    wet_data = data_root("wet")
    os.environ["CHEMBOMAS_DATA_DIR"] = str(wet_data)

    from . import legacy_run_realexp as legacy

    legacy.ROUND_NAME = args.round
    legacy.IS_FIRST_ROUND = args.first_round
    legacy.NUM_SAMPLE = args.num_sample
    legacy.KAPPA = 1.0 if args.kappa is None else args.kappa
    legacy.BASIC_DIR = str(wet_data / "00-basic" / legacy.PROJECT_NAME)
    legacy.CLUSTER_DIR = str(wet_data / "01-cluster" / legacy.PROJECT_NAME)
    legacy.REG_DIR = str(wet_data / "02-regression" / legacy.PROJECT_NAME / args.round)
    input_bo_dir = wet_data / "03-bo" / legacy.PROJECT_NAME / args.round
    legacy.BO_DIR = args.output_dir or (results_root("wet") / legacy.PROJECT_NAME / args.round)
    Path(legacy.BO_DIR).mkdir(parents=True, exist_ok=True)
    legacy.Pred = str(args.predictions or (Path(legacy.REG_DIR) / legacy.pred_file))
    legacy.Wet_Result = str(input_bo_dir / legacy.wet_exp_result_file)
    legacy.Uncompleted = str(input_bo_dir / legacy.uncompleted_exp_file)
    legacy.Partition = str(Path(legacy.CLUSTER_DIR) / legacy.partition_file)
    legacy.Order = str(Path(legacy.CLUSTER_DIR) / legacy.order_file)
    legacy.Search_Space = str(Path(legacy.BASIC_DIR) / legacy.search_space_file)
    legacy.Design_Exp = str(Path(legacy.BO_DIR) / legacy.designed_exp_file)

    if not args.first_round and not Path(legacy.Pred).exists():
        raise FileNotFoundError(
            f"Prediction tensor not found: {legacy.Pred}. Download the paper model artifact "
            "or pass --predictions /path/to/LLM_predict.pt."
        )
    if not args.first_round:
        for required in (legacy.Wet_Result, legacy.Uncompleted):
            if not Path(required).exists():
                raise FileNotFoundError(f"Required FD_wqp result file not found: {required}")

    legacy.main()


if __name__ == "__main__":
    main()
