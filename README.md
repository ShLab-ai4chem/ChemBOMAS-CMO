# ChemBOMAS-CMO

Public, paper-linked release of the ChemBOMAS code used for the two application tracks:

- **FD_wqp** — wet-lab reaction-condition design (from `ChemBOMAS-prod`)
- **EDBO** — dry-experiment benchmark and ablations (from `ChemBOMAS-V1`)

Only these two named tasks are included. Other datasets, exploratory runs, private
clustering services, credentials, logs, checkpoints, and model weights are excluded.
The language-model checkpoints are released separately; pass a downloaded prediction
tensor with `--predictions` or `--pseudo-predictions`.

## Repository layout

```text
src/chembomas/       reusable BO/MCTS code and task entry points
data/wet/            FD_wqp inputs and compact CSV result tables
data/dry/            EDBO inputs and compact regression/BO tables
results/             paper-facing summaries and figures
configs/             documented experiment settings
scripts/             shell entry points
docs/                provenance, data and model-release notes
tests/               import and layout smoke tests
```

The `00-basic`, `01-cluster`, `02-regression`, and `03-bo` directories under each
scope mirror the stages used by the original experiments. The copied `legacy_*.py`
files are kept as an audit trail; the public `run.py` adapters provide repository-
relative paths and explicit command-line arguments.

## Installation

```bash
python -m pip install -e .
```

For a light import/layout check:

```bash
pytest -q
```

## Running the public entry points

FD_wqp round design (requires the separately released `LLM_predict.pt` for rounds
after round 0):

```bash
PYTHONPATH=src python -m chembomas.wet.fd_wqp.run \
  --round round_7 --predictions /path/to/LLM_predict.pt --kappa 0.1
```

EDBO benchmark (the prediction tensor is optional; without it the code records a
deterministic observed-value fallback for smoke testing):

```bash
PYTHONPATH=src python -m chembomas.dry.edbo.run \
  --iteration 20 --pseudo-predictions /path/to/LLM_predict.pt
```

See [docs/data-manifest.md](docs/data-manifest.md) for the exact inclusion/exclusion
policy and [docs/model-release.md](docs/model-release.md) for the external model
artifact contract.

## Citation and license

Please cite the accompanying ChemBOMAS-CMO paper using `CITATION.cff`. Code is
released under the MIT License; third-party dependencies retain their own licenses.
