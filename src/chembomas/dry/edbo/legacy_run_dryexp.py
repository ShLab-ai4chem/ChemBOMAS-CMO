from ...bo_core.data.bodata import BoData, Sample
from ...bo_core.sampler.bo import BOSampler
from ...bo_core.mcts.tree import MCTS
from ...bo_core.utils.plot import plot_resultses
import torch
import time
import json
import argparse 
import os, random
from loguru import logger
from tqdm import tqdm
import numpy as np
import sys


os.environ["USE_WEIGHT_RANDOM_SAMPLE"] = "YES"  # BO丢弃伪数据时的策略采用反向权重采样 详见BOSampler pseudo_label_sample
RUN_DH = False  # use data harder or not
USE_DIVERSE = False  # use diverse sample before BO or not

_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
BASE_DIR = os.environ.get("CHEMBOMAS_DATA_DIR", os.path.join(_PACKAGE_ROOT, "data"))

# dataset info and hyperparameters
DATASET_HP_MAP = {
    'EDBO': {'num_init_sample': 18,  # num of known data points for BO initialization and regression model training
             'batch_size': 2,  # num of candidates selected in each BO round
             'num_diverse_sample': 6,  # num of candidates selected in diverse sampling before BO
             'category': ['Ligand','Base','Solvent','Concentration','Temperature'],
             'objective': ['Yield', 'Cost'],  # multi-objective optimization
            #  'preprocess': 'none',  # dual target preprocess method; none表示多标签不合并
            #  'preprocess_kwargs': {},
             'preprocess': 'num_logic',
             'preprocess_kwargs': {'p': 1.0, 'q': 1.0, 'a': -60.0, 'b': 0.1},
             'kappa': 0.05,  # 用于计算节点UCB打分的kappa值（可选）
             # 'per_batch_kappas': [0.05, 1.0],  # 每轮batch内按顺序使用的kappa值列表（可选）
             'observed_idx_file': 'new_b30_50.pt',  # 预先划分好的观测数据索引文件（可选）
             'save_reaction_indices': True,   # 新增: 是否将每轮选中的反应index保存到pt
             # constrained BO params
            #  'use_constraint_bo': True,       # 是否启用约束BO
            #  'target_output_idx': 0,          # 目标列索引（默认Yield）
            #  'constraint_output_idx': 1,      # 约束列索引（默认第2列）
            #  'constraint_operator': '<=',     # 可选 '>=' 或 '<='
            #  'constraint_threshold': 0.1,    # 约束阈值：constraint列 >= threshold
            #  'infeasible_cost': -1.0,         # 不可行惩罚
            },
    }

PARTITION_METHOD_MAP = {
    'expert': 'partition_expert.json',
    }

PSEUDO_METHOD_MAP = {
    'SFT_5.0': 'data_volume_5/LLM_predict.pt',
    }

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def set_parameters(dataset_name: str, partition_method: str, pseudo_method: str):
    '''return all paths and hyperparameters based on dataset name.
    '''
    parameters = {'dataset_name': dataset_name, 'partition_method': partition_method}
    # dataset info and hyperparameters
    parameters.update(DATASET_HP_MAP[dataset_name])
    parameters['build_tree'] = False if partition_method == 'wo-tree' else True
    # paths
    parameters['search_space'] = f"{BASE_DIR}/00-basic/{dataset_name}/search_space.csv"
    if 'observed_idx_file' in parameters:
        parameters['observed_idx'] = torch.load(f"{BASE_DIR}/00-basic/{dataset_name}/{parameters['observed_idx_file']}")['train']
    else:
        parameters['observed_idx'] = torch.load(f"{BASE_DIR}/00-basic/{dataset_name}/split_idx.pt")['train']
    parameters['order'] = json.load(open(f"{BASE_DIR}/01-cluster/{dataset_name}/order.json"))
    parameters['cluster'] = json.load(open(f"{BASE_DIR}/01-cluster/{dataset_name}/{PARTITION_METHOD_MAP[partition_method]}"))
    parameters['pseudo_data'] = f"{BASE_DIR}/02-regression/{dataset_name}/{PSEUDO_METHOD_MAP[pseudo_method]}"
    parameters['handled_idx'] = f"{BASE_DIR}/02-regression/{dataset_name}/handled_idx.npy"  # data harder modification

    pretty_params = "\n".join([f"  - {k}: {parameters[k]}" for k in sorted(parameters.keys())])
    logger.info(f"loaded parameters:\n{pretty_params}")

    return parameters

def return_dataset(parameters: dict):
    ''''return dataset based on dataset name, and do data harder if needed.
    '''
    dataset = BoData.read_dryexp_data(
        parameters['search_space'],
        parameters['objective'],
        parameters['category'],
        parameters['preprocess'],
        parameters['preprocess_kwargs'],
        parameters.get('target_output_idx', 0),
    )
    pseudo_path = parameters.get('pseudo_data')
    if pseudo_path and os.path.exists(pseudo_path):
        dataset.load_data_prediction(
            pseudo_path,
            parameters['preprocess'],
            parameters['preprocess_kwargs'],
            parameters.get('target_output_idx', 0),
        )
    else:
        logger.warning("No pseudo-label file supplied; using observed dry-experiment values as a reproducible fallback.")
        for sample in dataset._samples.values():
            sample.predict_value = sample.observed_value.clone()
    logger.info(f"loaded dataset, length = {len(dataset)}")
        
    if RUN_DH:
        logger.info("="*50+"Running with DH"+"="*50)
        parameters['observed_idx'] = dataset.make_data_harder_from_npy(
            parameters['handled_idx'], parameters['observed_idx'])
        logger.info(f"dataset length after data harder: {len(dataset)}")

    return dataset

def build_sampler(dataset, parameters):
    """统一创建BOSampler，避免在各模块重复传参"""
    return BOSampler(
        dataset=dataset,
        use_constraint=parameters.get('use_constraint_bo', False),
        target_output_idx=parameters.get('target_output_idx', 0),
        constraint_output_idx=parameters.get('constraint_output_idx', 1),
        constraint_operator=parameters.get('constraint_operator', '>='),
        constraint_threshold=parameters.get('constraint_threshold', 80.0),
        infeasible_cost=parameters.get('infeasible_cost', -1.0),
    )

def run_all(repeat_time: int,  # default=1
            iteration: int,  # default=40, total rounds of dry experiments
            parameters: dict,
            ):
    '''Full ChemBOMAS'''
    results = []
    save_reaction_indices = parameters.get('save_reaction_indices', False)  # 是否保存每轮选中的反应index
    for _ in range(repeat_time):
        result = []
        result_idx = []
        dataset = return_dataset(parameters)
        # build tree
        mcts = MCTS(
            dataset=dataset,
            sampler=build_sampler(dataset, parameters),
            batch_size=parameters['batch_size'],
            variable_nums=len(parameters['order']),
            n_candidates=20,
            num_init_samples=parameters['num_init_sample']
            )
        # use observed value and pseudo data to init tree
        obs = mcts.observed_init_tree(
            parameters['cluster'],
            parameters['order'],
            parameters['observed_idx'],
            pseudo_label=True,
            dont_build_tree=False,
            kappa=parameters['kappa'] if 'kappa' in parameters else 1.0,
            )
        # do diverse sampling
        if USE_DIVERSE:
            obs = mcts.diverse_sample(
                pseudo_label=True,
                is_tree_exist=True,
                num_diverse_sample=parameters['num_diverse_sample'],
                )
            result.extend(obs)
            # BO rounds = iteration-1, because the first round is already used for diverse sampling
            for i in tqdm(range(1, iteration)):
                if save_reaction_indices:
                    obs, idxes = mcts.search(
                        pseudo_label=True,
                        iteration_index=i,
                        per_batch_kappas=parameters.get('per_batch_kappas', None),
                        return_indices=True,
                    )
                    result.extend(obs)
                    result_idx.extend(idxes)
                else:
                    obs = mcts.search(
                        pseudo_label=True,
                        iteration_index=i,
                        per_batch_kappas=parameters.get('per_batch_kappas', None),
                    )
                    result.extend(obs)
        else:
            # BO rounds = iteration
            for i in tqdm(range(iteration)):
                # use pseudo data and tree to search
                if save_reaction_indices:
                    obs, idxes = mcts.search(
                        pseudo_label=True,
                        iteration_index=i,
                        per_batch_kappas=parameters.get('per_batch_kappas', None),
                        return_indices=True,
                    )
                    result.extend(obs)
                    result_idx.extend(idxes)
                else:
                    obs = mcts.search(
                        pseudo_label=True,
                        iteration_index=i,
                        per_batch_kappas=parameters.get('per_batch_kappas', None),
                    )
                    result.extend(obs)

        if save_reaction_indices:
            results.append({'observed_values': result, 'selected_indices': result_idx})
        else:
            results.append(result)

    return results

def run_wo_pseudo_data(repeat_time: int,  # default=1
                       iteration: int,  # default=40, total rounds of dry experiments
                       parameters: dict,
                       ):
    '''w/o data module'''
    results = []
    save_reaction_indices = parameters.get('save_reaction_indices', False)
    for _ in range(repeat_time):
        result = []
        result_idx = []
        dataset = return_dataset(parameters)
        # build tree
        mcts = MCTS(
            dataset=dataset,
            sampler=build_sampler(dataset, parameters),
            batch_size=parameters['batch_size'],
            variable_nums=len(parameters['order']),
            n_candidates=20,
            num_init_samples=parameters['num_init_sample']
            )
        # use observed value to init tree
        obs = mcts.observed_init_tree(
            parameters['cluster'],
            parameters['order'],
            parameters['observed_idx'],
            pseudo_label=False,
            dont_build_tree=False,
            kappa=parameters['kappa'] if 'kappa' in parameters else 1.0,
            )
        # do diverse sampling
        if USE_DIVERSE:
            obs = mcts.diverse_sample(
                pseudo_label=False,
                is_tree_exist=True,
                num_diverse_sample=parameters['num_diverse_sample'],
                )
            result.extend(obs)
            # BO rounds = iteration-1, because the first round is already used for diverse sampling
            for i in tqdm(range(1, iteration)):
                # no pseudo data and tree to search
                if save_reaction_indices:
                    obs, idxes = mcts.search(
                        pseudo_label=False,
                        iteration_index=i,
                        per_batch_kappas=parameters.get('per_batch_kappas', None),
                        return_indices=True,
                    )
                    result.extend(obs)
                    result_idx.extend(idxes)
                else:
                    obs = mcts.search(
                        pseudo_label=False,
                        iteration_index=i,
                        per_batch_kappas=parameters.get('per_batch_kappas', None),
                    )
                    result.extend(obs)
        else:
            # BO rounds = iteration
            for i in tqdm(range(iteration)):
                # no pseudo data and tree to search
                if save_reaction_indices:
                    obs, idxes = mcts.search(
                        pseudo_label=False,
                        iteration_index=i,
                        per_batch_kappas=parameters.get('per_batch_kappas', None),
                        return_indices=True,
                    )
                    result.extend(obs)
                    result_idx.extend(idxes)
                else:
                    obs = mcts.search(
                        pseudo_label=False,
                        iteration_index=i,
                        per_batch_kappas=parameters.get('per_batch_kappas', None),
                    )
                    result.extend(obs)

        if save_reaction_indices:
            results.append({'observed_values': result, 'selected_indices': result_idx})
        else:
            results.append(result)

    return results

def run_wo_tree(repeat_time: int,  # default=1
                iteration: int,  # default=40, total rounds of dry experiments
                parameters: dict,
                ):
    '''w/o knowledge module'''
    results = []
    save_reaction_indices = parameters.get('save_reaction_indices', False)
    for _ in range(repeat_time):
        result = []
        result_idx = []
        dataset = return_dataset(parameters)
        # build tree
        mcts = MCTS(
            dataset=dataset,
            sampler=build_sampler(dataset, parameters),
            batch_size=parameters['batch_size'],
            variable_nums=len(parameters['order']),
            n_candidates=20,
            # use_diverse_sample=DIVERSE,
            num_init_samples=parameters['num_init_sample']
            )
        # use observed value and pseudo data to init tree, do not build tree
        obs = mcts.observed_init_tree(
            parameters['cluster'],
            parameters['order'],
            parameters['observed_idx'],
            pseudo_label=True,
            dont_build_tree=True,
            )
        # do diverse sampling
        if USE_DIVERSE:
            obs = mcts.diverse_sample(
                pseudo_label=True,
                is_tree_exist=False,
                num_diverse_sample=parameters['num_diverse_sample'],
                )
            result.extend(obs)
            # BO rounds = iteration-1, because the first round is already used for diverse sampling
            for i in tqdm(range(1,iteration)):
                # use pseudo data and no tree to search
                if save_reaction_indices:
                    obs, idxes = mcts.baseline_search(pseudo_label=True, return_indices=True)
                    result.extend(obs)
                    result_idx.extend(idxes)
                else:
                    obs = mcts.baseline_search(pseudo_label=True)
                    result.extend(obs)
        else:
            # BO rounds = iteration
            for i in tqdm(range(iteration)):
                # use pseudo data and no tree to search
                if save_reaction_indices:
                    obs, idxes = mcts.baseline_search(pseudo_label=True, return_indices=True)
                    result.extend(obs)
                    result_idx.extend(idxes)
                else:
                    obs = mcts.baseline_search(pseudo_label=True)
                    result.extend(obs)

        if save_reaction_indices:
            results.append({'observed_values': result, 'selected_indices': result_idx})
        else:
            results.append(result)

    return results

def run_wo_both(repeat_time: int,  # default=1
            iteration: int,  # default=40, total rounds of dry experiments
            parameters: dict,
            ):
    '''w/o both module'''
    results = []
    save_reaction_indices = parameters.get('save_reaction_indices', False)
    for _ in range(repeat_time):
        result = []
        result_idx = []
        dataset = return_dataset(parameters)
        # build tree
        mcts = MCTS(
            dataset=dataset,
            sampler=build_sampler(dataset, parameters),
            batch_size=parameters['batch_size'],
            variable_nums=len(parameters['order']),
            n_candidates=20,
            # use_diverse_sample=DIVERSE,
            num_init_samples=parameters['num_init_sample']
            )
        # use observed value to init tree, do not build tree
        obs = mcts.observed_init_tree(
            parameters['cluster'],
            parameters['order'],
            parameters['observed_idx'],
            pseudo_label=False,
            dont_build_tree=True,
            )
        # do diverse sampling
        if USE_DIVERSE:
            obs = mcts.diverse_sample(
                pseudo_label=False,
                is_tree_exist=False,
                num_diverse_sample=parameters['num_diverse_sample'],
                )
            result.extend(obs)
            # BO rounds = iteration-1, because the first round is already used for diverse sampling
            for i in tqdm(range(1,iteration)):
                # no pseudo data and no tree to search
                if save_reaction_indices:
                    obs, idxes = mcts.baseline_search(pseudo_label=False, return_indices=True)
                    result.extend(obs)
                    result_idx.extend(idxes)
                else:
                    obs = mcts.baseline_search(pseudo_label=False)
                    result.extend(obs)
        else:
            # BO rounds = iteration
            for i in tqdm(range(iteration)):
                # no pseudo data and no tree to search
                if save_reaction_indices:
                    obs, idxes = mcts.baseline_search(pseudo_label=False, return_indices=True)
                    result.extend(obs)
                    result_idx.extend(idxes)
                else:
                    obs = mcts.baseline_search(pseudo_label=False)
                    result.extend(obs)

        if save_reaction_indices:
            results.append({'observed_values': result, 'selected_indices': result_idx})
        else:
            results.append(result)

    return results


MODULE_ABLATION_MAP = {
    'full': run_all,
    'wo_data': run_wo_pseudo_data,
    'wo_knowledge': run_wo_tree,
    'wo_both': run_wo_both,
}

def main(args, random_seed):

    # read basic parameters
    repeat_time = args.repeat_time
    iteration = args.iteration
    dataset_name = args.dataset_name
    partition_method = args.partition_method
    pseudo_method = args.pseudo_method
    activated_module = args.activated_module
    job_name = args.job_name
    setup_seed(random_seed)
    # read parameters based on dataset name used
    parameters = set_parameters(dataset_name, partition_method, pseudo_method)
    # make results save path
    results_path = os.environ.get(
        'CHEMBOMAS_RESULTS_DIR',
        f'{BASE_DIR}/03-bo/{dataset_name}/{job_name}/{activated_module}/{partition_method}/{pseudo_method}',
    )
    os.makedirs(results_path, exist_ok=True)

    # start running
    logger.info("="*50+f"Running ChemBOMAS"+"="*50)
    resultses = []
    resultses.append(MODULE_ABLATION_MAP[activated_module](repeat_time, iteration, parameters))
    # save raw results    
    now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    pt_path = os.path.join(results_path, rf'{random_seed}_{repeat_time}+{iteration}_raw_results_'+now_time+'.pt')
    torch.save(resultses, pt_path)
    logger.info(f"raw results saved to {pt_path}")


if __name__ == '__main__':
    # set seed
    random_seed = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # random_seed = [100]

    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat_time', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=40)
    parser.add_argument('--dataset_name', type=str, default='EDBO')
    parser.add_argument('--partition_method', type=str, default='expert')
    parser.add_argument('--pseudo_method', type=str, default='SFT_5.0')
    parser.add_argument('--activated_module', type=str, default='full')
    parser.add_argument('--job_name', type=str, default='data_volume')
    args = parser.parse_args()

    start_time = time.time()
    for seed in random_seed:
        main(args, seed)
    end_time = time.time()
    logger.success(f"total time: {end_time-start_time}")
