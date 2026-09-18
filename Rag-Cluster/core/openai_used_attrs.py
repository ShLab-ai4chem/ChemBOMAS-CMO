rounds = [
    """
You are an expert in chemistry.
Please list **1-3 key properties** of [{target}] that critically influence their effectiveness in chemical reactions (e.g., pKa for stronger basicity, steric hindrance, solubility, metal counter‑ion effects, etc.).
- A concise list (1-3 items) of properties.
- Justify each property’s relevance (e.g., "pKa determines proton affinity").
""",
"""
For the following compounds:
{json_string}

For each reagent, please collect documented information and literature data on the ALL properties identified in Step 1.
- If the property has numeric values (e.g. pKa, solubility in specific solvent), provide precise figures with citation.
- If the property can’t be quantified, describe it in detail (e.g. “strongly soluble in water, insoluble in organic solvent”, or “bulky” vs. “compact”).
Format: for each reagent name, list all attributes with their value and text description
**Special Cases:**
- "nothing" : Conceptualize as neutral substance with weak properties (primarily H₂O).
- Retain provided `IUPAC` from database.
- Unavailable data: Mark as "unknown".

The following information was collected from the PubChem database:
{props_string}
""",

"""
Convert data into this JSON structure and remove all the citations:
```json
[
  {{
    "name": "compound_name",
    "IUPAC": "iupac_name",
    "property_1": "value+text",
    "property_2": "value+text",
    ...
  }},
  ...
]
```
"""
]