import os
import pandas as pd
import argparse
import json
import math
import numpy as np
from sklearn.model_selection import KFold
from ..paths import data_root
from ..prep.name_to_smiles_converter import convert_names_to_smiles
from .prompt_generator import get_generator, list_available_generators

BASE_DIR = str(data_root("wet"))

def load_project_info(project_name):
    """
    加载项目配置信息
    
    Args:
        project_name: 项目名称
        
    Returns:
        tuple: (project_info, column_mapping, target_columns)
    """
    project_dir = f'{BASE_DIR}/00-basic/{project_name}'
    
    project_info_path = f'{project_dir}/project_info.json'
    
    if not os.path.exists(project_info_path):
        print(f"提示: 未找到project_info.json: {project_info_path}")
        return None, {}, []
    
    try:
        with open(project_info_path, 'r', encoding='utf-8') as f:
            project_info = json.load(f)
        
        # 获取目标列
        target_columns = project_info.get('target', ['Yield'])
        if isinstance(target_columns, str):
            target_columns = [target_columns]
        
        # 获取列映射
        column_mapping = project_info.get('column_mapping', {})
        
        return project_info, column_mapping, target_columns
        
    except Exception as e:
        print(f"警告: 读取project_info.json失败: {e}")
        return None, {}, []


def load_name_to_smiles_mapping(project_name):
    """
    加载名称到SMILES的映射数据
    
    Args:
        project_name: 项目名称
        
    Returns:
        DataFrame: 映射数据，如果不存在则返回None
    """
    project_dir = f'{BASE_DIR}/00-basic/{project_name}'

    mapping_path = f'{project_dir}/name_to_smiles.csv'
    
    if os.path.exists(mapping_path):
        try:
            df_mapping = pd.read_csv(mapping_path)
            print(f"加载名称到SMILES映射: {mapping_path}")
            return df_mapping
        except Exception as e:
            print(f"警告: 读取name_to_smiles.csv失败: {e}")
            return None
    else:
        print(f"提示: 未找到name_to_smiles.csv: {mapping_path}")
        return None

def merge_with_existing_data(df_new: pd.DataFrame,
                             extra_csv: str = None,
                             dedup: bool = True) -> pd.DataFrame:
    """
    将历史训练数据csv与当前新数据合并。
    - extra_csv为空时直接返回df_new
    - dedup=True时按 instruction/input/output 去重
    """
    if not extra_csv:
        return df_new

    if not os.path.exists(extra_csv):
        print(f"警告: extra_csv不存在，跳过合并: {extra_csv}")
        return df_new

    try:
        df_old = pd.read_csv(extra_csv)
        print(f"加载历史训练数据: {extra_csv} (count={len(df_old)})")
    except Exception as e:
        print(f"警告: 读取历史训练数据失败，跳过合并: {e}")
        return df_new

    # 对齐列（缺失列补空）
    for c in df_new.columns:
        if c not in df_old.columns:
            df_old[c] = ""
    df_old = df_old[df_new.columns]

    df_merged = pd.concat([df_old, df_new], ignore_index=True)
    before = len(df_merged)

    if dedup:
        dedup_cols = [c for c in ["instruction", "input", "output"] if c in df_merged.columns]
        if dedup_cols:
            df_merged = df_merged.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)

    print(f"合并完成: new={len(df_new)}, merged={len(df_merged)}, 去重前={before}")
    return df_merged

def create_sft_data(input_csv: str,
                    project_name: str = 'FD_wqp',
                    round_name: str = None,
                    convert_to_smiles: bool = True,
                    ):
    """
    创建SFT训练数据
    
    Args:
        input_csv: 输入CSV文件路径
        project_name: 项目名称，对应指令生成器类型
        round_name: 可为具体轮次如round_1或searchspace
        convert_to_smiles: 是否需要转换为smiles
    """
    # 读取数据
    print(f"读取数据: {input_csv}")
    df_raw = pd.read_csv(input_csv)

    # 加载项目配置
    _, column_mapping, target_columns = load_project_info(project_name)

    # 如果需要转换名称到SMILES
    if convert_to_smiles:
        # 加载映射数据
        df_mapping = load_name_to_smiles_mapping(project_name)
            
        if df_mapping is not None and column_mapping:
            print("正在进行名称到SMILES的转换...")
            # 备份原始数据
            df_raw_original = df_raw.copy()
            
            # 执行转换
            try:
                df_raw = convert_names_to_smiles(
                    search_space_df=df_raw,
                    name_to_smiles_df=df_mapping,
                    column_mapping=column_mapping,
                    target_columns=target_columns
                )
                print("名称到SMILES转换完成")
                
                # 检查转换结果，如果转换失败则恢复原始数据
                if df_raw.empty or df_raw.shape[0] != df_raw_original.shape[0]:
                    print("警告: 转换失败或数据丢失，使用原始数据")
                    df_raw = df_raw_original
                    
            except Exception as e:
                print(f"警告: 名称到SMILES转换失败: {e}")
                print("使用原始数据继续处理")
                df_raw = df_raw_original
        else:
            print("提示: 缺少映射数据或列映射配置，跳过名称到SMILES转换")
    else:
        print("提示: 跳过名称到SMILES转换")

    # 获取指令生成器
    generator = get_generator(project_name)
    
    # 创建输出DataFrame
    df_new = pd.DataFrame()
    
    # 生成指令
    print("生成指令...")
    df_new['instruction'] = df_raw.apply(
        lambda row: generator.generate_instruction(row.to_dict()), 
        axis=1
    )
    
    # 设置输入字段
    df_new['input'] = ""
    
    # 设置输出字段
    if len(target_columns) == 1:  # 单一目标
        if round_name == 'searchspace':  # 推理空间生成
            df_new['output'] = "-1"
        else:  # 微调数据生成
            df_new['output'] = df_raw[target_columns[0]].astype(str)

    else:  # 多目标，用逗号分隔
        if round_name == 'searchspace':
            df_new['output'] = ','.join(['-1' for _ in range(len(target_columns))])
        else:
            df_new['output'] = df_raw[target_columns].astype(str).agg(','.join, axis=1)
    
    # 设置历史字段
    df_new['history'] = [[] for _ in range(len(df_new))]
    
    print(f"完成！生成 {len(df_new)} 条数据")
    return df_new


def save_split_data(df, output_dir, round_name, kfold=1, shuffle=True, random_state=42, train_prop=None):
    """
    保存划分的训练集和测试集数据
    
    Args:
        df: 完整的数据DataFrame
        output_dir: 输出目录
        round_name: 轮次名称
        kfold: 折数，如果为1则只生成train.csv和test.csv
        shuffle: 是否打乱数据
        random_state: 随机种子
    """
    n_samples = len(df)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果是searchspace或kfold=1，使用默认划分
    if round_name == 'searchspace':
        # 保存完整的all.csv
        all_path = os.path.join(output_dir, 'inference_space.csv')
        df.to_csv(all_path, index=False, quoting=1)
        print(f"保存完整数据到: {all_path}")

    elif kfold == 1:
        # 保存完整的all.csv
        all_path = os.path.join(output_dir, 'all.csv')
        df.to_csv(all_path, index=False, quoting=1)
        print(f"保存完整数据到: {all_path}")

        # 如果提供了 train_prop 且在 (0,1) 范围内，则按比例随机抽样训练集（向上取整），剩余为测试集
        if train_prop is not None and (0 < float(train_prop) < 1) and round_name != 'searchspace':
            n_samples = len(df)
            n_train = int(math.ceil(n_samples * float(train_prop)))
            print(f"按比例划分训练集: 总样本={n_samples}, 训练集大小(向上取整)={n_train}")
            if shuffle:
                train_df = df.sample(n=n_train, random_state=random_state)
                test_df = df.drop(train_df.index).reset_index(drop=True)
                train_df = train_df.reset_index(drop=True)
            else:
                train_df = df.iloc[:n_train].reset_index(drop=True)
                test_df = df.iloc[n_train:].reset_index(drop=True)

            train_path = os.path.join(output_dir, 'train.csv')
            test_path = os.path.join(output_dir, 'test.csv')
            train_df.to_csv(train_path, index=False, quoting=1)
            test_df.to_csv(test_path, index=False, quoting=1)
            print(f"保存训练数据到: {train_path} (count={len(train_df)})")
            print(f"保存测试数据到: {test_path} (count={len(test_df)})")
        else:
            # 对于 searchspace 或未指定 train_prop，保留原有行为：全部作为训练，测试集为空
            train_path = os.path.join(output_dir, 'train.csv')
            df.to_csv(train_path, index=False, quoting=1)
            print(f"保存训练数据到: {train_path}")

            # 创建空的测试集
            test_path = os.path.join(output_dir, 'test.csv')
            empty_df = pd.DataFrame(columns=df.columns)
            empty_df.to_csv(test_path, index=False, quoting=1)
            print(f"创建空的测试数据: {test_path}")
        
    else:
        # 进行k-fold划分
        print(f"开始 {kfold}-fold 数据划分...")
        kf = KFold(n_splits=kfold, shuffle=shuffle, random_state=random_state)
        
        indices = np.arange(n_samples)
        
        # 保存完整的all.csv
        all_path = os.path.join(output_dir, 'all.csv')
        df.to_csv(all_path, index=False, quoting=1)
        print(f"保存完整数据到: {all_path}")
        
        # 为每一折创建子目录
        for fold in range(kfold):
            fold_dir = os.path.join(output_dir, f'fold_{fold+1}')
            os.makedirs(fold_dir, exist_ok=True)
            
            # 获取训练和测试索引
            train_idx = None
            test_idx = None
            
            for i, (train_index, test_index) in enumerate(kf.split(indices)):
                if i == fold:
                    train_idx = train_index
                    test_idx = test_index
                    break
            
            # 划分数据
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            
            # 保存训练集和测试集
            train_path = os.path.join(fold_dir, 'train.csv')
            test_path = os.path.join(fold_dir, 'test.csv')
            
            train_df.to_csv(train_path, index=False, quoting=1)
            test_df.to_csv(test_path, index=False, quoting=1)
            
            print(f"Fold {fold+1}: 训练集 {len(train_df)} 条, 测试集 {len(test_df)} 条")
            print(f"  训练集保存到: {train_path}")
            print(f"  测试集保存到: {test_path}")


def main(project_name, round_name, convert_to_smiles=True, kfold=1, train_prop=None,
         extra_csv=None, merge_dedup=True):
    """
    生成SFT训练数据或推理空间的CSV文件
    
    Args:
        project_name: 项目名称
        round_name: 轮次名称，或为'searchspace'时则生成推理空间
        convert_to_smiles: 是否将名称转换为SMILES
    """
    # 输入文件路径
    if round_name == 'searchspace':
        # 搜索空间数据
        input_dir = f'{BASE_DIR}/Data/00-basic/{project_name}'
        input_csv = f'{input_dir}/search_space.csv'
    else:
        # 此前实验结果数据
        input_dir = f'{BASE_DIR}/Data/03-bo/{project_name}/{round_name}'
        input_csv = f'{input_dir}/pre_wet_exp_result.csv'

    # 输出文件路径
    if round_name == 'searchspace':
        # 推理空间数据
        output_dir = f'{BASE_DIR}/Data/02-regression/{project_name}'
        output_csv = f'{output_dir}/inference_space.csv'
    else:
        # 回归训练数据
        output_dir = f'{BASE_DIR}/Data/02-regression/{project_name}/{round_name}'
        output_csv = f'{output_dir}/all.csv'

    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 执行数据生成
    print(f"项目: {project_name}")
    print(f"轮次: {round_name}")
    print(f"输入文件: {input_csv}")
    print(f"输出文件: {output_csv}")
    print(f"是否转换名称到SMILES: {convert_to_smiles}")
    print("-" * 50)

    # 生成SFT数据
    df_sft = create_sft_data(
        input_csv=input_csv,
        project_name=project_name,
        round_name=round_name,
        convert_to_smiles=convert_to_smiles,
    )

    # 新增：与历史all.csv合并
    df_sft = merge_with_existing_data(
        df_new=df_sft,
        extra_csv=extra_csv,
        dedup=merge_dedup,
    )

    # 保存划分的数据
    save_split_data(
        df=df_sft,
        output_dir=output_dir,
        round_name=round_name,
        kfold=kfold,
        train_prop=train_prop,
    )

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='生成SFT训练数据',
    )
    
    # 必需参数
    parser.add_argument(
        '--project', '-p',
        type=str,
        required=True,
        help='项目名称（对应Data/02-regression/下的子目录名）'
    )
    
    parser.add_argument(
        '--round', '-r',
        type=str,
        required=True,
        help='轮次名称，或可为searchspace'
    )
    
    # 可选参数
    parser.add_argument(
        '--convert-smiles',
        action='store_true',
        default=True,
        help='是否将名称转换为SMILES'
    )

    parser.add_argument(
        '--train-prop',
        type=float,
        default=None,
        help='训练集比例（0-1），例如 0.01 表示随机抽取全部样本的 1%（向上取整）作为训练集，剩余为测试集。'
    )

    parser.add_argument(
        '--kfold', '-k',
        type=int,
        default=1,
        help='k-fold交叉验证的折数，默认为1（不划分）'
    )

    parser.add_argument(
        '--extra-csv',
        type=str,
        default=None,
        help='可选：历史训练数据csv路径。提供后会与当前新数据合并再用于训练。'
    )

    parser.add_argument(
        '--no-merge-dedup',
        action='store_true',
        help='合并历史数据时不去重（默认会按instruction/input/output去重）。'
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(
        project_name=args.project,
        round_name=args.round,
        convert_to_smiles=args.convert_smiles,
        kfold=args.kfold,
        train_prop=args.train_prop,
        extra_csv=args.extra_csv,
        merge_dedup=(not args.no_merge_dedup),
    )
