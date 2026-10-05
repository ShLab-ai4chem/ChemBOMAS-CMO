from core.settings import get_settings
from core.openai_used_attrs import rounds
from openai import OpenAI
import base64, json
import pubchempy as pcp
from loguru import logger
from core.paths import DEFAULT_OPTIONS_JSON, configured_path

settings = get_settings()

client = OpenAI(
    api_key=settings.api_key,
    base_url=settings.base_url,
)

# ============== Read File ==============

with open("docs/science.aap9112.pdf", "rb") as f:
    data_1 = f.read()

with open("docs/Si_aap9112_perera_sm.pdf", "rb") as f:
    data_2 = f.read()

base64_string_1 = base64.b64encode(data_1).decode("utf-8")
base64_string_2 = base64.b64encode(data_2).decode("utf-8")

templates = {
    "role": "user",
    "content": [
        {
            "type": "input_file",
            "filename": "science.aap9112.pdf",
            "file_data": f"data:application/pdf;base64,{base64_string_1}",
        },
        {
            "type": "input_file",
            "filename": "Si_aap9112_perera_sm.pdf",
            "file_data": f"data:application/pdf;base64,{base64_string_2}",
        },
    ],
}

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
json_path = configured_path("RAG_OPTIONS_JSON", DEFAULT_OPTIONS_JSON)

with open(json_path, "r") as f:
    data = json.load(f)

for target, variables in data.items():
    logger.info(f"Start {target}")
    logger.info(f"Variables: {variables}")

    props = {}
    messages = []

    json_string = str(variables)
    for item in variables:
        cmpd = get_pubchem_props(item)
        props[item] = cmpd

    for idx, user_input in enumerate(rounds):
        user_input = user_input.format(target=target, json_string=json_string, props_string=str(props))

        print(user_input)

        if idx == 0:
            temp_template = templates.copy()
            temp_template["content"].append({
                "type": "input_text",
                "text": user_input,
            })
            messages.append(temp_template)
        else:
            messages.append({
                "role": "user",
                "content": user_input,
            })

        response = client.responses.create(
            model=settings.model_name,
            tools=[{"type": "web_search_preview"}],
            input=messages,
        )
        try:
            reply = response.output[0].content[0].text
        except:
            import pdb;pdb.set_trace()

        print(reply)
        messages.append({
            "role": "assistant",
            "content": reply,
        })

    # save message
    with open(f"openai_{target}.json", "w") as f:
        f.write(reply)
