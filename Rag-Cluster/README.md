# RAG-cluster utilities

This directory contains the original literature-assisted ranking and clustering
workflow. It is an optional component and is not required to run the published
FD_wqp or EDBO BO examples.

The scripts use an OpenAI-compatible API and may optionally use a local
HuggingFace model or a Serper web-search tool. No API key, private endpoint,
private filesystem path, or model checkpoint is included in this repository.

Install the optional dependencies in your environment, then configure the
provider before running a script:

```bash
python -m pip install -e ".[rag]"
export OPENAI_API_KEY="<your-key>"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # or another compatible endpoint
export OPENAI_MODEL="gpt-4o-mini"
export RAG_OPTIONS_JSON="/path/to/options.json"
export RAG_DOCS_DIR="/path/to/local/literature"
export RAG_OUTPUT_DIR="/path/to/output"
```

For agent scripts, set `RAG_LOCAL_MODEL` and `RAG_EMBEDDING_MODEL` if the local
model or embedding model is not available by its default HuggingFace name. Set
`RAG_ENABLE_WEB_SEARCH=1` and provide `SERPER_API_KEY` only when web search is
intentionally enabled.

The `core/settings.py` module reads these environment variables and contains no
credential defaults. The scripts can therefore be adapted to a user's own API
provider without modifying the public repository.

The checked-in JSON files under `json_files/` are small example inputs and
previous clustering outputs. They are not model checkpoints.


```
**Objective:**
Classify the provided list of candidate chemical substances into THREE groups according to the [Specified_physicochemical_Properties]. Your primary method for classification must be the utilization of quantitative data that would typically be found in a comprehensive physicochemical property database.

**Crucial Instructions:**
**Prioritize Quantitative Data: **For each substance and property, you should first attempt to classify it based on specific, measurable, quantitative values (e.g. pKa for basicity/acidity, dielectric constant for polarity, boiling point for volatility, specific functional group counts).
**Minimize General Knowledge/Intuition:** Avoid relying on your general, unquantified chemical knowledge or intuition. If a quantitative value from the "database" directly supports a classification, state that. If a direct value isn't typically used for a category but strong structural indicators (which could be quantified, e.g., number of H-bond donors) point to it, explain this as an inference based on data-like principles.
**Adhere to Provided Categories:** Classify substances strictly into the categories provided for each property. If a substance doesn't clearly fit or straddles categories based on (assumed) data, note this ambiguity.

**Candidate Substances to Classify:**
"base": ["KOH","nothing","Et3N","K3PO4","LiOtBu","CsF","NaOH","NaHCO3"]
```
