import pubchempy as pcp

def get_pubchem_props(compound_name: str, keys: list) -> dict:
    """查询 PubChem，返回指定化合物的若干常用属性字典。"""
    compounds = pcp.get_compounds(compound_name, 'name')
    if not compounds:
        return {"error": f"No compound found for '{compound_name}'"}

    cmpd = compounds[0]
    props = {}
    for key in keys:
        # 对不存在属性或 None 做容错
        value = getattr(cmpd, key, None)
        # 对于 synonyms，可能是列表，保留原样
        props[key] = value if value is not None else "N/A"

    return props

# 调用示例
if __name__ == '__main__':
    keys = [
        'cid',
        'iupac_name',
        'molecular_formula',
        'molecular_weight',
        'canonical_smiles',
        'isomeric_smiles',
        'xlogp',
        'hb_donor_count',
        'hb_acceptor_count',
        'rotatable_bond_count',
        'tpsa',
        'complexity',
        'charge',
        'synonyms',
        "pKa"
    ]
    result = get_pubchem_props("KOH", keys)
    for k, v in result.items():
        print(f"{k}: {v}")
