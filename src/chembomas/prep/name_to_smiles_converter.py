
import pandas as pd

def convert_names_to_smiles(search_space_df, name_to_smiles_df, column_mapping, target_columns=None):
    """
    将名称列转换为SMILES列
    
    Args:
        search_space_df: 搜索空间的DataFrame
        name_to_smiles_df: 名称到SMILES映射的DataFrame
        column_mapping: 列映射字典，格式为 {'原始列名': 'SMILES列名'}
        target_columns: 目标列列表，这些列不会被转换
        
    Returns:
        DataFrame: 转换后的DataFrame
    """
    if target_columns is None:
        target_columns = []
    
    # 创建输出DataFrame的副本
    df_output = search_space_df.copy()
    
    # 为每个需要转换的列构建映射字典
    for orig_col, smiles_col in column_mapping.items():
        if orig_col not in df_output.columns:
            print(f"警告: 原始列 '{orig_col}' 不在搜索空间DataFrame中")
            continue
            
        # 构建映射字典
        mapping = {}
        # 查找名称列和SMILES列
        name_col = None
        actual_smiles_col = None
        
        # 在name_to_smiles_df中查找对应的列
        for col in name_to_smiles_df.columns:
            if col == orig_col:
                name_col = col
            elif col == smiles_col:
                actual_smiles_col = col
                
        if name_col is None or actual_smiles_col is None:
            print(f"警告: 在name_to_smiles.csv中找不到列 '{orig_col}' 或 '{smiles_col}'")
            continue
            
        # 构建映射
        for _, row in name_to_smiles_df.iterrows():
            name = row[name_col]
            smiles = row[actual_smiles_col]
            
            # 如果名称不是NaN且不是"nothing"
            if pd.notna(name):
                name_str = str(name).strip()
                if name_str and name_str.lower() != "nothing":
                    # 如果SMILES是NaN，使用"nothing"
                    mapping[name_str] = smiles if pd.notna(smiles) else "nothing"
        
        # 定义一个函数来安全地获取SMILES
        def get_smiles(name, map_dict):
            if pd.isna(name):
                return "nothing"
                
            name_str = str(name).strip()
            if not name_str or name_str.lower() == "nothing":
                return "nothing"
                
            return map_dict.get(name_str, f"UNKNOWN:{name_str}")
        
        # 应用转换
        df_output[orig_col] = df_output[orig_col].apply(lambda x: get_smiles(x, mapping))
        
        # 重命名列（如果需要）
        if orig_col != smiles_col:
            df_output = df_output.rename(columns={orig_col: smiles_col})
    
    # 保持目标列不变
    for target_col in target_columns:
        if target_col in df_output.columns:
            # 确保目标列保留原始值
            df_output[target_col] = search_space_df[target_col] if target_col in search_space_df.columns else ""
    
    return df_output


def convert_names_to_smiles_from_files(search_space_file, name_to_smiles_file, output_file, 
                                      column_mapping, target_columns=None):
    """
    从文件读取并转换名称到SMILES
    
    Args:
        search_space_file: 搜索空间CSV文件路径
        name_to_smiles_file: 名称到SMILES映射CSV文件路径
        output_file: 输出文件路径
        column_mapping: 列映射字典
        target_columns: 目标列列表
    """
    # 读取CSV文件
    df_space = pd.read_csv(search_space_file)
    df_name_to_smiles = pd.read_csv(name_to_smiles_file)
    
    # 转换
    df_converted = convert_names_to_smiles(df_space, df_name_to_smiles, column_mapping, target_columns)
    
    # 保存到文件
    df_converted.to_csv(output_file, index=False)
    
    print(f"已成功生成 {output_file}")
    return df_converted
