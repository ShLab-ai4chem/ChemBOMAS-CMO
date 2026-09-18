# Data manifest and publication boundary

This repository is a fresh, allowlisted release. It is not a git merge of either
source repository.

| Public area | Source | Included |
| --- | --- | --- |
| `data/wet/**` | `ChemBOMAS-prod` | FD_wqp basic/search-space files, frozen order/partition, round-level train/test/all CSVs, compact design/result tables |
| `data/dry/**` | `ChemBOMAS-V1` | EDBO basic/search-space files, `new_b30_50.pt` observed split, expert order/partition, compact regression tables and aggregate comparison results |
| `src/chembomas/bo_core/**` | `ChemBOMAS-V1` | EDBO-capable MCTS/BO implementation with package-relative imports |
| `src/chembomas/wet/fd_wqp/**` | `ChemBOMAS-prod` | FD_wqp design implementation and path-safe adapter |
| `src/chembomas/regression/**` | both | only prompt/model-adapter components needed for FD_wqp/EDBO |

The following are intentionally excluded: `Rag-Cluster`, other task names, private
service clients, API keys, private absolute paths, raw run directories, logs, model
weights, per-round prediction tensors, and exploratory `.pt` outputs. The original
repositories remain untouched and are not dependencies of this release.

The small split tensor is retained because it is a frozen EDBO input rather than a
model checkpoint. Prediction tensors are external artifacts and must be obtained
from the model release described in `model-release.md`.
