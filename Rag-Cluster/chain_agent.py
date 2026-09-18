import os, json
from core.RagAgent_Chain import RagAgent
from loguru import logger
from core.settings import get_settings
from core.paths import DEFAULT_DOCS_DIR, DEFAULT_EMBEDDING_MODEL, DEFAULT_INDEX_PATH, DEFAULT_OPTIONS_JSON, configured_path

settings = get_settings()

# 配置参数
DOCS_DIR = configured_path("RAG_DOCS_DIR", DEFAULT_DOCS_DIR)
MODEL_PATH = configured_path("RAG_LOCAL_MODEL", "")
EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
FAISS_INDEX_PATH = configured_path("RAG_INDEX_PATH", DEFAULT_INDEX_PATH)
ENABLE_WEB_SEARCH = os.environ.get("RAG_ENABLE_WEB_SEARCH", "0") == "1"

if __name__ == "__main__":
    # 初始化RAG系统
    # rag_system = RagAgent(DOCS_DIR, MODEL_PATH, EMBEDDING_MODEL, FAISS_INDEX_PATH, ENABLE_WEB_SEARCH)
    options_path = configured_path("RAG_OPTIONS_JSON", DEFAULT_OPTIONS_JSON)
    json_data = json.load(open(options_path, "r", encoding='utf-8'))
    reaction_name = "Suzuki"

    # targets = ['base', 'ligand', 'solvent']
    targets = ['solvent']

    cluster = True

    if cluster:

        for target_name in targets:
            json_string = json_data[target_name]

    # ===================== 聚类 =====================


#             queries =f"""
# # Task 1
# Based strictly on scientific literature, identify:
# 1.  The most crucial influencing factor for {target_name} in Suzuki reactions
# 2.  A quantifiable metric (e.g., pKa, pH, Temperature) that measures this factor
# 3.  The mechanistic impact on reaction pathway

# Requirements:
# - Explicitly state which property/condition most significantly affects yield/selectivity
# - Explain precisely how this factor mechanistically influences the reaction pathway
# - Provide specific experimental evidence from literature
# - Output ONLY: "factor: [name], metric: [name], mechanism: [1-sentence explanation]"

# # Task 2
# Using EXCLUSIVELY:
# 1.  The metric '[metric_name]' from Query 1
# 2.  This JSON data: {json_string}

# Extract numerical values for ALL substances:
# - Output: "values: {'{sub1: X, sub2: Y, nothing: null, ...}'}"
# - Use null for 'nothing'
# - Report exact values from JSON (no calculations)

# # Task 3
# Based on the searches in the literature and **the answers you have just given**, divide {json_string} into 3 categories according to the value of the factor:
# - 'nothing' must be included if present (treat as 'absence of substance')
# - Output format (strictly enforced): "{target_name}": [[],[],[]]   # 3 lists, no additional text
# """

            queries = f"""
**Objective:**
Classify the provided list of candidate chemical substances into THREE groups according to the [Specified_physicochemical_Properties]. Your primary method for classification must be the utilization of quantitative data that would typically be found in a comprehensive physicochemical property database.

**Crucial Instructions:**
**Prioritize Quantitative Data: **For each substance and property, you should first attempt to classify it based on specific, measurable, quantitative values (e.g. pKa for basicity/acidity, dielectric constant for polarity, boiling point for volatility, specific functional group counts).
**Minimize General Knowledge/Intuition:** Avoid relying on your general, unquantified chemical knowledge or intuition. If a quantitative value from the "database" directly supports a classification, state that. If a direct value isn't typically used for a category but strong structural indicators (which could be quantified, e.g., number of H-bond donors) point to it, explain this as an inference based on data-like principles.
**Adhere to Provided Categories:** Classify substances strictly into the categories provided for each property. If a substance doesn't clearly fit or straddles categories based on (assumed) data, note this ambiguity.

**Candidate Substances to Classify:**
"{target_name}": {json_string}
"""

            print(queries)
            exit()
            # 执行查询
            queries = "Please tell me the pKa of ['KOH', 'nothing', 'Et3N', 'K3PO4', 'LiOtBu', 'CsF', 'NaOH', 'NaHCO3']"
            results, history = rag_system.run_stratified_query(queries)
            with open("log.txt", "a") as f:
                f.write(f"Queries: {queries}\n\n")
                f.write("="*50)
                f.write(f"Results: {results}\n\n")
                f.write("="*50)
                f.write(f"History: {history}\n\n")
