# Data manifest and publication boundary

This manifest describes the datasets and artifacts included in the ChemBOMAS-CMO
release.

| Public area | Included |
| --- | --- |
| `data/wet/**` | FD_wqp search-space and inference-space files, frozen order/partition, round-level train/test/all CSVs, paper prediction tensors, and compact experiment tables |
| `data/dry/**` | EDBO search-space files, observed split, expert order/partition, data-volume-5 regression tables and prediction tensor |
| `src/chembomas/bo_core/**` | Reusable MCTS and Bayesian-optimization implementation |
| `src/chembomas/wet/fd_wqp/**` | FD_wqp design implementation and command-line adapter |
| `Rag-Cluster/**` | Optional literature-assisted retrieval, ranking, and clustering scripts; API credentials are user-supplied |
| `src/chembomas/regression/**` | Prompt-generation and model-adapter components for the released tasks |

The following are intentionally excluded: unpublished task variants, private
service endpoints, API keys, private absolute paths, raw run directories, logs,
model weights, non-paper prediction variants, and exploratory `.pt` outputs. The
release is self-contained and does not depend on any private source tree.

The small split tensor is retained because it is a frozen EDBO input rather than a
model checkpoint. The paper prediction tensors are bundled in this repository;
the separately released model weights are only needed when regenerating those
predictions from scratch.
