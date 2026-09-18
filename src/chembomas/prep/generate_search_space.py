import json
import itertools
import csv
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from name_to_smiles_converter import convert_names_to_smiles_from_files
from check_condition_constraints import check_constraints

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ChemBOMAS_Wetlab

def main(project_name):
    """
    生成实验搜索空间的CSV文件
    
    Args:
        project_name: 项目名称，对应Data/00-basic/下的子目录
    """
    proj_dir = f'{BASE_DIR}/Data/00-basic/{project_name}'

    # 检查项目目录是否存在
    if not os.path.exists(proj_dir):
        print(f"错误：项目目录不存在: {proj_dir}")
        return
    
    # 检查JSON文件是否存在
    json_path = f'{proj_dir}/options.json'
    project_info_path = f'{proj_dir}/project_info.json'
    
    if not os.path.exists(json_path):
        print(f"错误：options.json文件不存在: {json_path}")
        return
    
    if not os.path.exists(project_info_path):
        print(f"错误：project_info.json文件不存在: {project_info_path}")
        return

    try:
        # 读取 options.json 文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            print("错误：options.json文件为空或格式不正确")
            return
        
        # 读取 project_info.json 文件
        with open(project_info_path, 'r', encoding='utf-8') as f:
            project_info = json.load(f)
            
        if not project_info:
            print("错误：project_info.json文件为空或格式不正确")
            return
            
        # 获取目标列
        target_columns = project_info.get('target', ['Yield'])
        if isinstance(target_columns, str):
            target_columns = [target_columns]
        elif isinstance(target_columns, list):
            pass
        else:
            print("错误：project_info.json中的target字段格式不正确")
            return
            
        # 获取列映射
        column_mapping = project_info.get('column_mapping', {})
        if not isinstance(column_mapping, dict):
            print("错误：project_info.json中的column_mapping字段格式不正确")
            return

        constraints = project_info.get('constraints', [])
        if not isinstance(constraints, list):
            print("错误：project_info.json中的constraints字段格式不正确（应为list）")
            return

    except json.JSONDecodeError as e:
        print(f"错误：JSON文件解析失败: {e}")
        return
    except Exception as e:
        print(f"错误：读取文件时发生异常: {e}")
        return

    # 获取原始键（列名），并添加 target 列
    param_keys = list(data.keys())
    keys = param_keys + target_columns
    values = list(data.values())  # 转为list，避免迭代器被消耗

    # 生成CSV文件路径
    search_space_path = f'{proj_dir}/search_space.csv'
    search_space_smiles_path = f'{proj_dir}/search_space_smiles.csv'
    name_to_smiles_path = f'{proj_dir}/name_to_smiles.csv'
    
    try:
        # 计算组合总数
        total_combinations = 1
        for v in values:
            total_combinations *= len(v)
        
        print(f"生成搜索空间...")
        print(f"参数列: {list(data.keys())}")
        print(f"目标列: {target_columns}")
        print(f"总组合数: {total_combinations}")

        valid_count = 0
        filtered_count = 0

        # 生成组合并写入 CSV
        with open(search_space_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            
            for i, combo in enumerate(itertools.product(*values), 1):
                combo_dict = dict(zip(param_keys, combo))

                # 新增：约束过滤
                if not check_constraints(combo_dict, constraints):
                    filtered_count += 1
                    continue

                # 添加空字符串作为目标列的值
                row = combo + ('',) * len(target_columns)
                writer.writerow(row)
                valid_count += 1
                
                # 显示进度（每10%显示一次）
                if i % max(1, total_combinations // 10) == 0 or i == total_combinations:
                    progress = (i / total_combinations) * 100
                    print(f"进度: {progress:.1f}% ({i}/{total_combinations})")
        
        print(f"CSV文件已生成: {search_space_path}")
        print(f"列头: {keys}")
        print(f"过滤非法组合数: {filtered_count}")
        print(f"最终有效组合数: {valid_count}")

        # 如果存在name_to_smiles.csv且convert_names_to_smiles_from_files可用，则生成SMILES版本
        if os.path.exists(name_to_smiles_path) and convert_names_to_smiles_from_files:
            print(f"\n检测到name_to_smiles.csv文件，正在生成SMILES版本...")
            try:
                df_converted = convert_names_to_smiles_from_files(
                    search_space_file=search_space_path,
                    name_to_smiles_file=name_to_smiles_path,
                    output_file=search_space_smiles_path,
                    column_mapping=column_mapping,
                    target_columns=target_columns
                )
                print(f"SMILES版本已生成: {search_space_smiles_path}")
            except Exception as e:
                print(f"警告：生成SMILES版本时出错: {e}")
                print("将继续使用原始版本")
        else:
            if not os.path.exists(name_to_smiles_path):
                print(f"\n提示：未找到name_to_smiles.csv文件，跳过生成SMILES版本")
            else:
                print(f"\n提示：name_to_smiles_converter模块不可用，跳过生成SMILES版本")

    except Exception as e:
        print(f"错误：生成CSV文件时发生异常: {e}")
        return

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='生成实验搜索空间的CSV文件',
    )
    
    # 必需参数
    parser.add_argument(
        '--project', '-p',
        type=str,
        required=True,
        help='项目名称（对应Data/00-basic/下的子目录名）'
    )
    
    return parser.parse_args()

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_arguments()
    
    # 运行主函数
    main(project_name=args.project)
