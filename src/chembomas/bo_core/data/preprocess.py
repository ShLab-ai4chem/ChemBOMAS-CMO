import numpy as np


def default_processor(O1, O2):
    """
    默认处理方式，直接返回两个指标
    """
    return [O1, O2], 2

def num_logic_processor(O_num, O_logic, p=1.0, q=1.0, a=0.05, b=80):
    """
    计算双指标合并公式: Ot = (p * On)^q / (1 + exp(-a * (Ol - b)))
    
    参数:
    O_num: 数值指标 (标量或数组)，若无真实值，则使用-1替代
    O_logic: 逻辑指标 (标量或数组)，若无真实值，则使用-1替代
    p, q, a, b: 超参数
    
    返回:
    obj: 合并后的指标，若输入指标均为-1，同样返回-1
    obj_len: 指标数目
    """
    obj_len = 1

    # 设定无实验结果的目标为-1
    if O_num == O_logic == -1:
        return -1, obj_len
    else:
        obj = (p * O_num) ** q / (1 + np.exp(-a * (O_logic - b)))
    
        return obj, obj_len

def add_processor(O1, O2):
    """
    将两个指标直接加和后返回
    """
    return O1 + O2, 1

PREPROCESS_FUNC = {
    'default': default_processor,
    'num_logic': num_logic_processor,
    'add': add_processor,
}