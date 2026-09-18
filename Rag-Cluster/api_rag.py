import os, json
from core.RagSys import RAGSystem
from loguru import logger
from core.settings import get_settings
from core.paths import DEFAULT_DOCS_DIR, DEFAULT_EMBEDDING_MODEL, DEFAULT_INDEX_PATH, DEFAULT_LOCAL_MODEL, DEFAULT_OPTIONS_JSON, configured_path

settings = get_settings()

# 配置参数
DOCS_DIR = configured_path("RAG_DOCS_DIR", DEFAULT_DOCS_DIR)
MODEL_PATH = configured_path("RAG_LOCAL_MODEL", DEFAULT_LOCAL_MODEL)
EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
FAISS_INDEX_PATH = configured_path("RAG_INDEX_PATH", DEFAULT_INDEX_PATH)

if __name__ == "__main__":
    # 初始化RAG系统
    rag = RAGSystem(
        docs_dir=DOCS_DIR,
        model_path=MODEL_PATH,
        embedding_model=EMBEDDING_MODEL,
        faiss_index_path=FAISS_INDEX_PATH,
    )

    targets = ['base', 'ligand', 'solvent']
    # targets = ['base']

    cluster = True

    if cluster:

        for target_name in targets:

            # 定义查询
            options_path = configured_path("RAG_OPTIONS_JSON", DEFAULT_OPTIONS_JSON)
            json_string = json.load(open(options_path, "r", encoding='utf-8'))[target_name]

    # ===================== 聚类 =====================


            queries = [
                f"""Based strictly on scientific literature, identify:
    1.  The most crucial influencing factor for {target_name} in Suzuki reactions
    2.  A quantifiable metric (e.g., pKa, pH, Temperature) that measures this factor
    3.  The mechanistic impact on reaction pathway

    Requirements:
    - Explicitly state which property/condition most significantly affects yield/selectivity
    - Explain precisely how this factor mechanistically influences the reaction pathway
    - Provide specific experimental evidence from literature
    - Output ONLY: "factor: [name], metric: [name], mechanism: [1-sentence explanation]"
    """,
                f"""Using EXCLUSIVELY:
    1.  The metric '[metric_name]' from Query 1
    2.  This JSON data: {json_string}

    Extract numerical values for ALL substances:
    - Output: "values: {'{sub1: X, sub2: Y, nothing: null, ...}'}"
    - Use null for 'nothing'
    - Report exact values from JSON (no calculations)
    """,
                f"""Based on the searches in the literature and **the answers you have just given**, divide {json_string} into 3 categories according to the value of the factor:
    - 'nothing' must be included if present (treat as 'absence of substance')
    - Output format (strictly enforced): "{target_name}": [[],[],[]]   # 3 lists, no additional text

    """
            ]

            final_answer = rag.run_queries(queries)

            logger.info("\n🎯 最终答案:")
            print(final_answer)

            print(settings.model_name)


    else:
    # ===================== 排序 =====================

        queries = [
            f"""You are a professional chemistry researcher who needs to analyze the importance of substances in the chemical literature. Please strictly follow the following procedure:
    1. Rank the importance based on reaction mechanism, experimental data and literature description
    2. all analyses must be based on objective evidence from the literature, avoiding subjective speculation


    Please based on the literature, analyze the role of the following substances in the reaction
    **{",".join(targets)}**

    Perform the task:
    1. list the specific role of the substance in the reaction
    """,
            f"""Perform an in-depth analysis based on the list of substances extracted in the previous round and the content of the literature:
    1. Assess importance in three dimensions:
    - Necessity of the reaction (0-10 points): whether it is indispensable or not
    - Strength of impact (0-10 points): impact on yield/rate/selectivity
    - Literature Emphasis (0-10 points): number and extent of special mentions
    2. Provide evidence of scoring for each substance (cite key sentences from the original article)
    """,
            f"""Combine the first two rounds of analysis to perform the final ranking:
    1. by total score = necessity x 0.5 + intensity of impact x 0.3 + emphasis x 0.2
    2. if total scores are the same, the highest score for necessity is preferred
    3. must include three key evidence statements:
    - Why the TOP1 substance is the most important (citing the original sentence from the literature)
    - Substances most likely to be replaced and why
    - Any anomalous findings (e.g. substances not mentioned in the literature but critical)
    4. Final output format:
    ### Explanation
    [description of the analysis in 200 words]
    ### Results
    ["Substance A", "Substance B", "Substance C", ...]
    """
        ]

        final_answer = rag.run_queries(queries)

        logger.info("\n🎯 最终答案:")
        print(final_answer)

        print(settings.model_name)
