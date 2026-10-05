from typing import Any, Dict, List


def _eval_condition(combo_dict: Dict[str, Any], cond: Dict[str, Any]) -> bool:
    """
    cond 示例:
    {"col":"Additive", "op":"==", "value":"Nothing"}
    """
    col = cond["col"]
    op = cond["op"]
    val = cond.get("value")
    cur = combo_dict.get(col)

    if op == "==":
        return cur == val
    if op == "!=":
        return cur != val
    if op == "in":
        if not isinstance(val, list):
            raise ValueError(f"op='in' requires list value, got: {type(val)}")
        return cur in val
    if op == "not_in":
        if not isinstance(val, list):
            raise ValueError(f"op='not_in' requires list value, got: {type(val)}")
        return cur not in val

    raise ValueError(f"Unsupported op: {op}")


def check_constraints(combo_dict: Dict[str, Any], constraints: List[Dict[str, Any]]) -> bool:
    """
    支持规则:
      - if_then: if 条件全部满足时，then 条件必须全部满足
      - all: conditions 全部满足
      - any: conditions 任一满足
    """
    if not constraints:
        return True

    for rule in constraints:
        rtype = rule.get("type", "if_then")

        if rtype == "if_then":
            if_conds = rule.get("if", [])
            then_conds = rule.get("then", [])
            if_ok = all(_eval_condition(combo_dict, c) for c in if_conds)
            if if_ok:
                then_ok = all(_eval_condition(combo_dict, c) for c in then_conds)
                if not then_ok:
                    return False

        elif rtype == "all":
            conds = rule.get("conditions", [])
            if not all(_eval_condition(combo_dict, c) for c in conds):
                return False

        elif rtype == "any":
            conds = rule.get("conditions", [])
            if not any(_eval_condition(combo_dict, c) for c in conds):
                return False

        else:
            raise ValueError(f"Unsupported constraint type: {rtype}")

    return True