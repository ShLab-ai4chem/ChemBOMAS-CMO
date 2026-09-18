# Migration status

This is the first public-release checkpoint.

Completed:

- fresh Git repository with only FD_wqp and EDBO task namespaces;
- package-relative BO/MCTS imports and repository-relative wet/dry adapters;
- compact allowlisted inputs, training tables, result CSVs, and paper figures;
- removal of private absolute paths, `Rag-Cluster`, model checkpoints, logs, and other tasks;
- smoke-tested EDBO tree initialization and FD_wqp round-0 design.

Before external publication:

1. replace the Hugging Face placeholder and record model revision/checksums;
2. run `pytest -q` after installing the optional development dependency;
3. run a secret scanner on the final checkout;
4. decide whether to publish the optional SFT training path (`regression` extra) or
   keep it as source-only reference code;
5. add the paper DOI/authors to `CITATION.cff`.
