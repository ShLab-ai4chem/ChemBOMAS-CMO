from core.settings import get_settings
from core.openai_used_attrs import rounds
from openai import OpenAI
import base64, json
# import pubchempy as pcp
from loguru import logger
import os
import glob
from core.paths import DEFAULT_DOCS_DIR, DEFAULT_OPTIONS_JSON, configured_path

settings = get_settings()

client = OpenAI(
    api_key=settings.api_key,
    base_url=settings.base_url,
)

# ============== Read File ==============

dataset = os.environ.get("RAG_DATASET", "FD")
json_path = configured_path("RAG_OPTIONS_JSON", DEFAULT_OPTIONS_JSON)
save_path = configured_path("RAG_OUTPUT_DIR", str(DEFAULT_OPTIONS_JSON.parent))


pdf_dir = configured_path("RAG_DOCS_DIR", DEFAULT_DOCS_DIR)
pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))



templates = {
    "role": "user",
    "content": []
}

for pdf_path in pdf_files:
    with open(pdf_path, "rb") as f:
        data = f.read()
    base64_string = base64.b64encode(data).decode("utf-8")
    templates["content"].append({
        "type": "input_file",
        "filename": os.path.basename(pdf_path),
        "file_data": f"data:application/pdf;base64,{base64_string}",
    })

# ============== Read Targets ==============
with open(json_path, "r") as f:
    data = json.load(f)

targets = list(data.keys())

queries = f"""
As a professional chemistry researcher, analyze the roles and importance of the following substances in the first step reaction of synthesis of oseltamivir based on objective evidence from the chemical literatures: **{",".join(targets)}**. Strictly rank their importance using reaction mechanisms, experimental data, and literature descriptions, avoiding subjective speculation.

Procedure:
1. List the specific role of each substance in the reaction, citing key evidence from the literature.
2. Assess and rank importance based on three dimensions: Necessity (indispensability), Impact (on yield/rate/selectivity)
3. Final ranking: Prioritize by overall importance

Output Format:
### Explanation
[description of the analysis]
### Results
["Substance A", "Substance B", "Substance C", ...]
"""

logger.info(f"Rank : {",".join(targets)}")


templates["content"].append({
    "type": "input_text",
    "text": queries,
})

response = client.responses.create(
    model=settings.model_name,
    tools=[{"type": "web_search_preview"}] if "gpt" in settings.model_name else None,
    input=[templates],
)
try:
    reply = response.output[1].content[0].text
except:
    import pdb;pdb.set_trace()

print(reply)

# 保存结果
with open(f"openai_{dataset}_rank.md", "w") as f:
    f.write(reply)
