"""CLI for the EDBO dry-experiment benchmark."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ...paths import data_root, results_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EDBO ChemBOMAS benchmark")
    parser.add_argument("--iteration", type=int, default=20)
    parser.add_argument("--repeat-time", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--partition-method", default="expert")
    parser.add_argument("--pseudo-method", default="SFT_5.0")
    parser.add_argument("--pseudo-predictions", type=Path, help="external prediction tensor from the HF model release")
    parser.add_argument("--activated-module", choices=["full", "wo_data", "wo_knowledge", "wo_both"], default="full")
    parser.add_argument("--job-name", default="public_run")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    dry_data = data_root("dry")
    os.environ["CHEMBOMAS_DATA_DIR"] = str(dry_data)
    os.environ["CHEMBOMAS_RESULTS_DIR"] = str(args.output_dir or (results_root("dry") / "EDBO" / args.job_name))

    from . import legacy_run_dryexp as legacy

    if args.pseudo_predictions:
        legacy.PSEUDO_METHOD_MAP[args.pseudo_method] = str(args.pseudo_predictions)

    namespace = argparse.Namespace(
        repeat_time=args.repeat_time,
        iteration=args.iteration,
        dataset_name="EDBO",
        partition_method=args.partition_method,
        pseudo_method=args.pseudo_method,
        activated_module=args.activated_module,
        job_name=args.job_name,
    )
    legacy.main(namespace, args.seed)


if __name__ == "__main__":
    main()
