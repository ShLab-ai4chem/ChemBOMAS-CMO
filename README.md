# ChemBOMAS-CMO

ChemBOMAS-CMO is the accompanying code and data release for the ChemBOMAS paper.
It provides reproducible examples of Bayesian optimization for chemical reaction
condition discovery in two settings:

- **FD_wqp**: wet-laboratory reaction-condition design;
- **EDBO**: dry-experiment benchmark and ablation studies.

The language-model checkpoints and prediction tensors are distributed separately.
After downloading the corresponding artifact, pass its local path with
`--predictions` or `--pseudo-predictions`.

## Repository layout

```text
src/chembomas/       reusable BO/MCTS code and task entry points
data/wet/            FD_wqp inputs and compact CSV result tables
data/dry/            EDBO inputs and compact regression tables
results/             paper-facing summaries and figures
configs/             experiment settings
scripts/             command-line launchers
docs/                data and model-release notes
tests/               import and layout smoke tests
```

The numbered data directories describe the preparation, clustering, regression, and
optimization stages. The public `run.py` modules provide repository-relative paths
and explicit command-line arguments.

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

See [docs/data-manifest.md](docs/data-manifest.md) for the data contents and
[docs/model-release.md](docs/model-release.md) for the external model artifact
contract.

## Citation and license

Please cite the accompanying ChemBOMAS-CMO paper using `CITATION.cff`. Code is
released under the MIT License; third-party dependencies retain their own licenses.
