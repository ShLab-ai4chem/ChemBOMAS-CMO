import sys
from ...bo_core.data.bodata import BoData, Sample
from ...bo_core.sampler.bo import BOSampler
from ...bo_core.mcts.tree import MCTS
from ...bo_core.utils.plot import plot_resultses
import torch
import time
import json
import argparse
import pandas as pd
import copy
import numpy as np
import os, random

_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
BASE_DIR = os.environ.get("CHEMBOMAS_DATA_DIR", os.path.join(_PACKAGE_ROOT, "data"))

# 定义实验信息
PROJECT_NAME = 'FD_wqp'  # 项目名称，需与文件目录中一致
IS_FIRST_ROUND = False  # True: 无湿实验结果，无LLM预测结果; False: 有湿实验结果，有基于湿实验微调LLM预测结果
ROUND_NAME = 'round_7'  # 轮次名称，需与文件目录中一致
NUM_SAMPLE = 8  # 生成样本的数量
RANDOM_SEED = 100
RETURN_ACQ_VALUES = False  # 是否返回acquisition values，仅在使用BO时有效

# ===============================================================================================
# 定义目录路径 （无需修改）
BASIC_DIR = f'{BASE_DIR}/00-basic/{PROJECT_NAME}'  # 基本信息目录
CLUSTER_DIR = f'{BASE_DIR}/01-cluster/{PROJECT_NAME}'  # 聚类结果目录
REG_DIR = f'{BASE_DIR}/02-regression/{PROJECT_NAME}/{ROUND_NAME}'  # 保存LLM推理结果目录
BO_DIR = f'{BASE_DIR}/03-bo/{PROJECT_NAME}/{ROUND_NAME}'  # BO目录
# ===============================================================================================

# 定义文件名
## 随轮次变化（第0轮无需）
pred_file = "LLM_predict.pt"  # SFT后LLM输出预测结果文件名
wet_exp_result_file = "pre_wet_exp_result.csv"  # 此前所有湿实验结果
uncompleted_exp_file = "uncompleted_exp.csv"  # 未完成实验结果
## 不随轮次变化
partition_file = "partition.json"  # 聚类结果
order_file = "order.json"  # 重要性排序结果
search_space_file = "search_space.csv"  # 反应条件空间
designed_exp_file = "ChemBOMAS_design.csv"  # 输出推荐实验结果路径
# ===============================================================================================
# 获取完整路径（无需修改）
Pred = f'{REG_DIR}/{pred_file}'
Wet_Result = f'{BO_DIR}/{wet_exp_result_file}'
Uncompleted = f'{BO_DIR}/{uncompleted_exp_file}'
Partition = f'{CLUSTER_DIR}/{partition_file}'
Order = f'{CLUSTER_DIR}/{order_file}'
Search_Space = f'{BASIC_DIR}/{search_space_file}'
Design_Exp = f'{BO_DIR}/{designed_exp_file}'
# ===============================================================================================

# 定义优化相关信息
objective = ['Yield', 'Selectivity']   # 反应优化目标
preprocess = 'num_logic'  # 反应优化目标预处理方式，'default'表示不进行预处理，其他预处理方式见data/bodata.py中的preprocess_target函数
preprocess_kwargs = {"p": 1.0, "q": 1.0, "a": 0.3, "b": 80}  # 反应优化目标预处理参数

category = ['Palladium_Source', 'Chiral_Ligand', 'Solvent', 'Temperature', 'Additive', 'Base']  # 反应条件空间中的可变条件 TODO: 改为从文件中读取

# BO相关超参设置
## KAPPA数值越大越倾向于探索，数值越小越倾向于利用，通常范围：[0.01, 1.0]，可以通过向列表中添加多个KAPPA值来进行不同探索利用平衡策略的对比实验
KAPPA = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 1.0, 2.0, 4.0]

## 由于我们设计了前半轮使用EI、后半轮使用UCB的策略，因此需要分别设置EI和UCB的初始值和增长率
## EI = EI_INIT + EI_INCRE * (BOround - 1)，UCB = UCB_INIT + UCB_INCRE * (BOround - totalround/2)
## 其中BOround表示当前是第几轮BO，totalround表示总共的BO轮数。通过调整EI_INIT、EI_INCRE、UCB_INIT、UCB_INCRE这四个参数，可以控制前半轮和后半轮的探索利用平衡策略。
## EI数值越大越倾向于探索，数值越小越倾向于利用，通常范围：[0.01, 1.0]，可以通过调整EI_INIT和EI_INCRE来实现前半轮从更偏向利用到更偏向探索的平滑过渡。
EI_INIT = 0.01
EI_INCRE = 0.14
## 同样地，UCB数值越大越倾向于探索，数值越小越倾向于利用，通常范围：[0.2, 2.0]，可以通过调整UCB_INIT和UCB_INCRE来实现后半轮从更偏向利用到更偏向探索的平滑过渡。
UCB_INIT = 0.10
UCB_INCRE = 0.28


def design_exp(
    project: str = PROJECT_NAME,
    round_name=ROUND_NAME,
    is_first_round: bool = False,
    kappa: float = 1.0,
    kappa_by_depth: bool = False,
    depth_kappa_mapping: dict = None,
    return_acq_values: bool = False,
):
    '''运行推荐算法'''
    print(f"Designing {round_name} experiment for project {project}")

    # 第0轮推荐，使用多样性采样
    if is_first_round:
        print("Round 0 design, no wet experiment data or LLM predicted data is used, diversity aware random selection!")
        # 从文件中构建数据集
        dataset = BoData.read_round_0_data(
            Search_Space,
            objective=objective,
            category=category,
            preprocess=preprocess,
            preprocess_kwargs=preprocess_kwargs
        )
        # 从文件中读取排序和分类信息
        partition = json.load(open(Partition))
        order = json.load(open(Order))
        # 创建树
        mcts = MCTS(
            dataset=dataset,
            sampler=BOSampler(dataset),
            num_init_samples=NUM_SAMPLE,  # 第0轮生成样本数量通过num_init_samples控制
            variable_nums=len(order),
        )
        # 推荐下一轮反应
        next_batch = mcts.real_exp_random(partition, order)
        return next_batch

    # 第n轮推荐
    else:
        print(f"{round_name} design, wet experiment data and LLM predicted data avaliable!")
        # 从文件中构建数据集,根据已有湿实验数据和未完成的实验mask部分条件
        dataset, mask = BoData.read_round_n_data(
            Search_Space,
            Wet_Result,
            Uncompleted,
            objective=objective,
            category=category,
            preprocess=preprocess,
            preprocess_kwargs=preprocess_kwargs
        )
        # 加载LLM预测值
        dataset.load_data_prediction(
            pred_val_path=Pred,
            preprocess=preprocess,
        )
        # 从文件中读取排序和分类信息
        partition = json.load(open(Partition))
        order = json.load(open(Order))
        # 创建树
        mcts = MCTS(
            dataset=dataset,
            sampler=BOSampler(dataset),
            batch_size=NUM_SAMPLE,  # 第n轮生成样本数量通过batch_size控制
            variable_nums=len(order),
        )
        # 推荐下一轮反应
        if return_acq_values:
            next_batch, acq_values = mcts.real_exp(
                partition,
                order,
                mask,
                kappa=kappa,
                kappa_by_depth=kappa_by_depth,
                depth_kappa_mapping=depth_kappa_mapping,
                ei_init=EI_INIT,
                ei_incre=EI_INCRE,
                ucb_init=UCB_INIT,
                ucb_incre=UCB_INCRE,
                return_acq_values=return_acq_values
            )
            return next_batch, acq_values
        else:
            next_batch = mcts.real_exp(
                partition,
                order,
                mask,
                kappa=kappa,
                kappa_by_depth=kappa_by_depth,
                depth_kappa_mapping=depth_kappa_mapping,
                ei_init=EI_INIT,
                ei_incre=EI_INCRE,
                ucb_init=UCB_INIT,
                ucb_incre=UCB_INCRE,
                return_acq_values=return_acq_values
            )
            return next_batch


def setup_seed(seed):
    print(f'Random seed is set to {seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_batch_with_optional_acq(next_batch, acq_values, desired_order, csv_file, with_acq=False):
    """
    保存推荐结果：
    - with_acq=False: 普通去重并按 desired_order 输出
    - with_acq=True : 保持 sample-acq 对应，去重时保留 acq 最大项，并附加 acq_value 列
    """
    if with_acq:
        paired = []
        for s, a in zip(next_batch, acq_values):
            paired.append((frozenset(s.items()), dict(s), float(a)))

        # 根据样本键集合去重，保留 acq 更大的一项
        dedup = {}
        for keyset, sample_dict, acq in paired:
            if keyset not in dedup or acq > dedup[keyset][1]:
                dedup[keyset] = (sample_dict, acq)

        unique_batch = [v[0] for v in dedup.values()]
        unique_acqs = [v[1] for v in dedup.values()]

        output_rows = []
        for sample_dict, acq in zip(unique_batch, unique_acqs):
            ordered = {key: sample_dict.get(key) for key in desired_order}
            ordered['acq_value'] = acq
            output_rows.append(ordered)

        if len(output_rows) == 0:
            print(f"[Warning] No samples generated for {csv_file}, skip saving.")
            return

        column_names = list(output_rows[0].keys())
        df = pd.DataFrame(output_rows, columns=column_names)
        df.to_csv(csv_file, index=False)
        print(f"Designed experiments (with acq) saved to {csv_file}")
    else:
        # 去重
        unique_batch = [dict(t) for t in {frozenset(sample.items()) for sample in next_batch}]
        sorted_unique_batch = [{key: sample[key] for key in desired_order} for sample in unique_batch]

        if len(sorted_unique_batch) == 0:
            print(f"[Warning] No samples generated for {csv_file}, skip saving.")
            return

        column_names = sorted_unique_batch[0].keys()
        df = pd.DataFrame(sorted_unique_batch, columns=column_names)
        df.to_csv(csv_file, index=False)
        print(f"Designed experiments saved to {csv_file}")


def main():
    '''完整流程'''
    # 设定随机种子
    if RANDOM_SEED:
        setup_seed(RANDOM_SEED)

    # 运行推荐算法
    if isinstance(KAPPA, list):
        for kappa in KAPPA:
            # 生成输出文件名
            if isinstance(kappa, dict):
                kappa_str = '_'.join([f"{key}-{value}" for key, value in kappa.items()])
                csv_file = Design_Exp.replace('.csv', f'_kappa_by_depth_{kappa_str}.csv')
            else:
                csv_file = Design_Exp.replace('.csv', f'_k{kappa}.csv')

            # 运行推荐
            if isinstance(kappa, dict):
                if RETURN_ACQ_VALUES:
                    next_batch, acq_values = design_exp(
                        PROJECT_NAME,
                        ROUND_NAME,
                        IS_FIRST_ROUND,
                        kappa=None,
                        kappa_by_depth=True,
                        depth_kappa_mapping=kappa,
                        return_acq_values=RETURN_ACQ_VALUES
                    )
                else:
                    next_batch = design_exp(
                        PROJECT_NAME,
                        ROUND_NAME,
                        IS_FIRST_ROUND,
                        kappa=None,
                        kappa_by_depth=True,
                        depth_kappa_mapping=kappa,
                        return_acq_values=RETURN_ACQ_VALUES
                    )
            else:
                if RETURN_ACQ_VALUES:
                    next_batch, acq_values = design_exp(
                        PROJECT_NAME,
                        ROUND_NAME,
                        IS_FIRST_ROUND,
                        kappa=kappa,
                        return_acq_values=RETURN_ACQ_VALUES
                    )
                else:
                    next_batch = design_exp(
                        PROJECT_NAME,
                        ROUND_NAME,
                        IS_FIRST_ROUND,
                        kappa=kappa,
                        return_acq_values=RETURN_ACQ_VALUES
                    )

            # 保存输出
            if RETURN_ACQ_VALUES:
                save_batch_with_optional_acq(
                    next_batch=next_batch,
                    acq_values=acq_values,
                    desired_order=category,
                    csv_file=csv_file,
                    with_acq=True
                )
            else:
                save_batch_with_optional_acq(
                    next_batch=next_batch,
                    acq_values=None,
                    desired_order=category,
                    csv_file=csv_file,
                    with_acq=False
                )

    else:
        # 单个 KAPPA（可能是 dict 或 float）
        if isinstance(KAPPA, dict):
            if RETURN_ACQ_VALUES:
                next_batch, acq_values = design_exp(
                    PROJECT_NAME,
                    ROUND_NAME,
                    IS_FIRST_ROUND,
                    kappa=None,
                    kappa_by_depth=True,
                    depth_kappa_mapping=KAPPA,
                    return_acq_values=RETURN_ACQ_VALUES
                )
            else:
                next_batch = design_exp(
                    PROJECT_NAME,
                    ROUND_NAME,
                    IS_FIRST_ROUND,
                    kappa=None,
                    kappa_by_depth=True,
                    depth_kappa_mapping=KAPPA,
                    return_acq_values=RETURN_ACQ_VALUES
                )
            kappa_str = '_'.join([f"{key}-{value}" for key, value in KAPPA.items()])
            csv_file = Design_Exp.replace('.csv', f'_kappa_by_depth_{kappa_str}.csv')
        else:
            if RETURN_ACQ_VALUES:
                next_batch, acq_values = design_exp(
                    PROJECT_NAME,
                    ROUND_NAME,
                    IS_FIRST_ROUND,
                    kappa=KAPPA,
                    return_acq_values=RETURN_ACQ_VALUES
                )
            else:
                next_batch = design_exp(
                    PROJECT_NAME,
                    ROUND_NAME,
                    IS_FIRST_ROUND,
                    kappa=KAPPA,
                    return_acq_values=RETURN_ACQ_VALUES
                )
            csv_file = Design_Exp.replace('.csv', f'_k{KAPPA}.csv')

        if RETURN_ACQ_VALUES:
            save_batch_with_optional_acq(
                next_batch=next_batch,
                acq_values=acq_values,
                desired_order=category,
                csv_file=csv_file,
                with_acq=True
            )
        else:
            save_batch_with_optional_acq(
                next_batch=next_batch,
                acq_values=None,
                desired_order=category,
                csv_file=csv_file,
                with_acq=False
            )


if __name__ == '__main__':
    main()
