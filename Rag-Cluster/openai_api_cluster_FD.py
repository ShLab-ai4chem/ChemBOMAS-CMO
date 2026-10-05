from core.settings import get_settings
from openai import OpenAI
import base64, json
import pubchempy as pcp
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

dataset = "FD"
json_path = configured_path("RAG_OPTIONS_JSON", DEFAULT_OPTIONS_JSON)
save_path = configured_path("RAG_OUTPUT_DIR", str(DEFAULT_OPTIONS_JSON.parent))

pdf_dir = configured_path("RAG_DOCS_DIR", DEFAULT_DOCS_DIR)
pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

user_input = """
**Objective:**
Classify the provided list of candidate chemical substances into NO MORE THAN FIVE groups according to the [Specified_physicochemical_Properties], or place them all in ONE class if justified.. Your primary method for classification must be the utilization of quantitative data that would typically be found in a comprehensive physicochemical property database.

**Crucial Instructions:**
**Prioritize Quantitative Data: **For each substance and property, you should first attempt to classify it based on specific, measurable, quantitative values (e.g. pKa for basicity/acidity, dielectric constant for polarity, boiling point for volatility, specific functional group counts).
**Minimize General Knowledge/Intuition:** Avoid relying on your general, unquantified chemical knowledge or intuition. If a quantitative value from the "database" directly supports a classification, state that. If a direct value isn't typically used for a category but strong structural indicators (which could be quantified, e.g., number of H-bond donors) point to it, explain this as an inference based on data-like principles.
**Adhere to Provided Categories:** Classify substances strictly into the categories provided for each property. If a substance doesn't clearly fit or straddles categories based on (assumed) data, note this ambiguity.

**Candidate Substances to Classify:**
{target} : {substances}
"""

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


# ============== Chem Database ==============
def get_pubchem_props(compound_name: str) -> dict:
    compounds = pcp.get_compounds(compound_name, 'name')
    if not compounds:
        return {"error": f"No compound found for {compound_name}"}
    cmpd = compounds[0].to_dict()
    # ['atom_stereo_count', 'atoms', 'bond_stereo_count', 'bonds', 'cactvs_fingerprint', 'canonical_smiles', 'charge', 'cid', 'complexity', 'conformer_id_3d', 'conformer_rmsd_3d', 'coordinate_type', 'covalent_unit_count', 'defined_atom_stereo_count', 'defined_bond_stereo_count', 'effective_rotor_count_3d', 'elements', 'exact_mass', 'feature_selfoverlap_3d', 'fingerprint', 'h_bond_acceptor_count', 'h_bond_donor_count', 'heavy_atom_count', 'inchi', 'inchikey', 'isomeric_smiles', 'isotope_atom_count', 'iupac_name', 'mmff94_energy_3d', 'mmff94_partial_charges_3d', 'molecular_formula', 'molecular_weight', 'monoisotopic_mass', 'multipoles_3d', 'pharmacophore_features_3d', 'record', 'rotatable_bond_count', 'shape_fingerprint_3d', 'shape_selfoverlap_3d', 'tpsa', 'undefined_atom_stereo_count', 'undefined_bond_stereo_count', 'volume_3d', 'xlogp']
    no_keep_keys = ['atoms', 'bonds', 'cactvs_fingerprint', 'coordinate_type', 'elements', 'fingerprint', 'record']
    for key in no_keep_keys:
        if key in cmpd:
            del cmpd[key]
    return cmpd

# ============== Read Targets ==============

with open(json_path, "r") as f:
    data = json.load(f)

for target, variables in data.items():

    if target != "Additive":
        continue

    logger.info(f"Start {target}")
    logger.info(f"Variables: {variables}")

    props = {}
    messages = []

    json_string = str(variables)
    for item in variables:
        cmpd = get_pubchem_props(item)
        props[item] = cmpd

    user_input = user_input.format(target=target, substances=json_string)

    # print(user_input)

    # import pdb;pdb.set_trace()

    temp = templates.copy()
    temp["content"].append({
        "type": "input_text",
        "text": user_input,
    })
    messages.append(temp)

    response = client.responses.create(
        model=settings.model_name,
        tools=[{"type": "web_search_preview"}] if "gpt" in settings.model_name else None,
        input=messages,
    )
    try:
        reply = response.output[1].content[0].text
    except:
        import pdb;pdb.set_trace()

    print(reply)

    # save message
    with open(os.path.join(save_path, f"openai_{target}.md"), "w") as f:
        f.write(reply)
