# ChemBOMAS-CMO

ChemBOMAS-CMO is the accompanying code and data release for the ChemBOMAS paper.
It provides reproducible examples of Bayesian optimization for chemical reaction
condition discovery in two settings:

- **FD_wqp**: wet-laboratory reaction-condition design;
- **EDBO**: dry-experiment benchmark and ablation studies.

The prediction tensors used by the paper are included with the corresponding data
directories. Large language-model checkpoints are distributed separately; external
prediction files can still be supplied through the command-line options.

## Repository layout

```text
src/chembomas/       reusable BO/MCTS code and task entry points
Rag-Cluster/         optional API-backed literature ranking and clustering scripts
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

FD_wqp round design:

```bash
PYTHONPATH=src python -m chembomas.wet.fd_wqp.run \
  --round round_7 --kappa 0.1
```

The command uses the bundled round-specific prediction tensor by default. Use
`--predictions /path/to/LLM_predict.pt` to override it.

EDBO benchmark:

```bash
PYTHONPATH=src python -m chembomas.dry.edbo.run \
  --iteration 5
```

The bundled `data_volume_5` prediction tensor is used by default. Use
`--pseudo-predictions /path/to/LLM_predict.pt` to override it.

See [docs/data-manifest.md](docs/data-manifest.md) for the data contents and
[docs/model-release.md](docs/model-release.md) for the external model artifact
contract. `Rag-Cluster/` contains the optional literature-assisted workflow;
users must provide their own API key and local document/model paths.

## Citation and license

Please cite the accompanying ChemBOMAS-CMO paper using `CITATION.cff`. Code is
released under the MIT License; third-party dependencies retain their own licenses.
