import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ===== 路径配置（参考 run_realexp.py）=====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ChemBOMAS-prod
PROJECT_NAME = "FD_wqp"
ROUND_NAME = "round_7"

BO_DIR = f"{BASE_DIR}/Data/03-bo/{PROJECT_NAME}/{ROUND_NAME}"
INPUT_DIR = BO_DIR
DESIGNED_EXP_FILE_TEMPLATE = "ChemBOMAS_design.csv"  # 输出推荐实验结果路径模板
BASIC_OUTPUT_FILE = f"{BO_DIR}/{DESIGNED_EXP_FILE_TEMPLATE.split('.')[0]}_selected_basic.csv"
DIVERSE_OUTPUT_FILE = f"{BO_DIR}/{DESIGNED_EXP_FILE_TEMPLATE.split('.')[0]}_selected_diverse.csv"
# =========================================

# ===== 反应条件列配置（按项目修改）=====
CATEGORY = ['Palladium_Source', 'Chiral_Ligand', 'Solvent', 'Temperature', 'Additive', 'Base']

# 其他项目示例：
# CATEGORY = ['Catalyst', 'Aux_catalyst', 'Solvent', 'Acid', 'Additive', 'Reaction Temp&Time']
# CATEGORY = ['Catalyst', 'Ligand', 'Quinone', 'Solvent']
# ======================================

# ===== 选择参数 =====
N_SELECT = 8
BASIC_REPETITION_WEIGHT = 0.6
BASIC_DIVERSITY_WEIGHT = 0.4
DIVERSE_REPETITION_WEIGHT = 0.4
DIVERSE_DIVERSITY_WEIGHT = 0.6
# ===================

# ===== 输出参数 =====
CATEGORY_ONLY = True  # 是否在输出中仅保留类别列（True）还是保留所有列（False）
# ===================


class ReactionConditionSelector:
    def __init__(self, data_dir=".", category=None):
        self.data = None
        self.data_dir = data_dir
        self.category = category if category is not None else CATEGORY
        self.unique_conditions = []

    # ---------- 通用辅助 ----------
    def _validate_columns(self, df, file_path=""):
        missing = [c for c in self.category if c not in df.columns]
        if missing:
            print(f"文件缺少必要列，跳过: {os.path.basename(file_path)}")
            print(f"  缺少列: {missing}")
            return False
        return True

    def _key_from_row(self, row):
        return "|".join(str(row[c]) for c in self.category)

    def _key_from_cond(self, cond):
        return "|".join(str(cond.get(c, "")) for c in self.category)

    def _feature_from_cond(self, cond):
        return " ".join(str(cond.get(c, "")) for c in self.category)

    def _preview_from_key(self, key, max_items=4):
        parts = key.split("|")
        show_n = min(max_items, len(parts))
        return " | ".join(parts[:show_n])

    # ---------- 数据加载 ----------
    def load_data(self, file_list=None):
        """加载CSV文件"""
        if file_list is None:
            if not os.path.exists(self.data_dir):
                print(f"目录不存在: {self.data_dir}")
                return False

            csv_files = sorted(
                [f for f in os.listdir(self.data_dir) if f.startswith(f"{DESIGNED_EXP_FILE_TEMPLATE.split('.')[0]}_k") and f.endswith(".csv")]
            )
            file_list = [os.path.join(self.data_dir, f) for f in csv_files]

        if not file_list:
            print(f"未找到输入文件（{DESIGNED_EXP_FILE_TEMPLATE.split('.')[0]}_k*.csv）")
            return False

        all_data = []
        valid_file_count = 0
        for file_path in file_list:
            try:
                df = pd.read_csv(file_path)
                if not self._validate_columns(df, file_path):
                    continue

                df = df.copy()
                df["source_file"] = os.path.basename(file_path)
                all_data.append(df)
                valid_file_count += 1
                print(f"已加载: {os.path.basename(file_path)}, 条件数: {len(df)}")
            except Exception as e:
                print(f"加载文件失败: {file_path}, 错误: {e}")

        if not all_data:
            print("没有可用数据（文件为空/列不匹配/读取失败）")
            return False

        self.data = pd.concat(all_data, ignore_index=True)
        print(f"\n总共加载 {len(self.data)} 个反应条件")
        print(f"有效文件数: {valid_file_count}")

        self._show_duplicates()
        return True

    def _show_duplicates(self):
        if self.data is None or len(self.data) == 0:
            return

        condition_strings = [self._key_from_row(row) for _, row in self.data.iterrows()]
        counter = Counter(condition_strings)

        print(f"唯一条件数: {len(counter)}")
        print("重复条件统计（Top 10）:")
        for cond_str, count in counter.most_common(10):
            if count > 1:
                print(f"  出现{count}次: {self._preview_from_key(cond_str)}")

    # ---------- 唯一条件 ----------
    def extract_unique_conditions(self):
        if self.data is None or len(self.data) == 0:
            print("请先加载数据")
            return

        unique_dict = {}
        for _, row in self.data.iterrows():
            key = self._key_from_row(row)
            if key not in unique_dict:
                item = {c: row[c] for c in self.category}
                item["occurrence_count"] = 1
                item["source_files"] = [row.get("source_file", "unknown")]
                unique_dict[key] = item
            else:
                unique_dict[key]["occurrence_count"] += 1
                sf = row.get("source_file", "unknown")
                if sf not in unique_dict[key]["source_files"]:
                    unique_dict[key]["source_files"].append(sf)

        self.unique_conditions = list(unique_dict.values())
        self.unique_conditions.sort(key=lambda x: x["occurrence_count"], reverse=True)
        print(f"\n提取完成：唯一条件 {len(self.unique_conditions)} 个")

    # ---------- 打分 ----------
    def calculate_repetition_scores(self):
        if not self.unique_conditions:
            print("请先提取唯一反应条件")
            return None

        max_count = max(cond["occurrence_count"] for cond in self.unique_conditions)
        max_count = max(max_count, 1)

        for cond in self.unique_conditions:
            cond["repetition_score"] = cond["occurrence_count"] / max_count

        return [cond["repetition_score"] for cond in self.unique_conditions]

    def calculate_diversity_scores(self):
        if not self.unique_conditions:
            print("请先提取唯一反应条件")
            return None

        features = [self._feature_from_cond(cond) for cond in self.unique_conditions]

        try:
            vectorizer = TfidfVectorizer()
            feature_vectors = vectorizer.fit_transform(features)
            similarity_matrix = cosine_similarity(feature_vectors)

            for i in range(len(self.unique_conditions)):
                other_similarities = [similarity_matrix[i, j] for j in range(len(self.unique_conditions)) if j != i]
                avg_similarity = float(np.mean(other_similarities)) if other_similarities else 0.0
                self.unique_conditions[i]["diversity_score"] = 1.0 - avg_similarity

            div_scores = [cond["diversity_score"] for cond in self.unique_conditions]
            max_div = max(div_scores) if max(div_scores) > 0 else 1.0
            for cond in self.unique_conditions:
                cond["diversity_score"] = cond["diversity_score"] / max_div

        except Exception:
            print("TF-IDF失败，使用回退多样性分数")
            n = len(self.unique_conditions)
            for i, cond in enumerate(self.unique_conditions):
                cond["diversity_score"] = 1.0 - (i / max(n, 1)) * 0.5

        return [cond["diversity_score"] for cond in self.unique_conditions]

    # ---------- 选择策略 ----------
    def _compute_combined_scores(self, repetition_weight=0.6, diversity_weight=0.4):
        self.calculate_repetition_scores()
        self.calculate_diversity_scores()

        for cond in self.unique_conditions:
            cond["combined_score"] = (
                repetition_weight * cond["repetition_score"] + diversity_weight * cond["diversity_score"]
            )

    def select_unique_optimal_conditions(self, n_select=15, repetition_weight=0.6, diversity_weight=0.4):
        """按综合分数直接排序选择"""
        if not self.unique_conditions:
            print("请先提取唯一反应条件")
            return []

        self._compute_combined_scores(repetition_weight, diversity_weight)

        sorted_conditions = sorted(self.unique_conditions, key=lambda x: x["combined_score"], reverse=True)
        selected = [cond.copy() for cond in sorted_conditions[: min(n_select, len(sorted_conditions))]]

        for i, cond in enumerate(selected, 1):
            cond["rank"] = i
        return selected

    def select_diverse_conditions(self, n_select=15, repetition_weight=0.5, diversity_weight=0.5):
        """贪婪算法：综合分数 + 与已选集合的差异性"""
        if not self.unique_conditions:
            print("请先提取唯一反应条件")
            return []

        self._compute_combined_scores(repetition_weight, diversity_weight)

        features = [self._feature_from_cond(cond) for cond in self.unique_conditions]
        vectorizer = TfidfVectorizer()
        feature_vectors = vectorizer.fit_transform(features)

        selected_indices = []
        remaining_indices = list(range(len(self.unique_conditions)))

        initial_idx = int(np.argmax([cond["combined_score"] for cond in self.unique_conditions]))
        selected_indices.append(initial_idx)
        remaining_indices.remove(initial_idx)

        for _ in range(min(n_select - 1, len(remaining_indices))):
            best_score = -1.0
            best_idx = -1

            for idx in remaining_indices:
                similarities = []
                for sel_idx in selected_indices:
                    sim = cosine_similarity(feature_vectors[idx], feature_vectors[sel_idx])[0][0]
                    similarities.append(sim)

                avg_similarity = float(np.mean(similarities)) if similarities else 0.0
                diversity_bonus = 1.0 - avg_similarity

                score = self.unique_conditions[idx]["combined_score"] * 0.7 + diversity_bonus * 0.3

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx != -1:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        selected_conditions = []
        for rank, idx in enumerate(selected_indices, 1):
            cond = self.unique_conditions[idx].copy()
            cond["rank"] = rank
            selected_conditions.append(cond)

        return selected_conditions

    # ---------- 输出 ----------
    def print_selected_conditions(self, selected_conditions, method="basic"):
        print("\n" + "=" * 100)
        print(f"选中的 {len(selected_conditions)} 个唯一反应条件（{method}）:")
        print("=" * 100)

        for cond in selected_conditions:
            print(f"\n#{cond['rank']} 综合分数: {cond['combined_score']:.3f}")
            print(f"  出现次数: {cond['occurrence_count']} 次")
            print(f"  重复性分数: {cond['repetition_score']:.3f}")
            print(f"  多样性分数: {cond['diversity_score']:.3f}")

            for c in self.category:
                val = str(cond.get(c, ""))
                if len(val) > 120:
                    val = val[:117] + "..."
                print(f"  {c}: {val}")

            print(f"  来源文件数: {len(cond.get('source_files', []))}")
            print("-" * 100)

    def save_selected_conditions(self, selected_conditions, output_file="selected_unique_conditions.csv"):
        df_selected = pd.DataFrame(selected_conditions)

        if "source_files" in df_selected.columns:
            df_selected["source_files"] = df_selected["source_files"].apply(
                lambda x: ";".join(x) if isinstance(x, list) else x
            )

        # CATEGORY_ONLY=True: 仅保留类别列；False: 保留完整输出信息
        if CATEGORY_ONLY:
            columns_order = ["rank"] + self.category
        else:
            columns_order = (
                ["rank"]
                + self.category
                + ["occurrence_count", "repetition_score", "diversity_score", "combined_score", "source_files"]
            )

        columns_order = [col for col in columns_order if col in df_selected.columns]
        df_selected = df_selected[columns_order]
        df_selected.to_csv(output_file, index=False)
        print(f"\n选中条件已保存: {output_file} (CATEGORY_ONLY={CATEGORY_ONLY})")
        return df_selected
        return df_selected

    def analyze_statistics(self, selected_conditions):
        if not selected_conditions:
            print("没有可分析的条件")
            return

        print("\n选中条件组分分布:")
        n = len(selected_conditions)
        for comp in self.category:
            values = [cond.get(comp, "") for cond in selected_conditions]
            vc = pd.Series(values).value_counts(dropna=False)

            print(f"\n{comp}:")
            for k, v in vc.head(10).items():
                print(f"  {k}: {v} ({v / n * 100:.1f}%)")


def main():
    print("唯一反应条件选择器")
    print("=" * 60)

    os.makedirs(BO_DIR, exist_ok=True)

    selector = ReactionConditionSelector(INPUT_DIR, category=CATEGORY)

    if not selector.load_data():
        print("无法加载数据，请检查路径和输入文件")
        return

    selector.extract_unique_conditions()

    # 方法1：基础排序
    print("\n" + "=" * 60)
    print("方法1: 基本选择（综合分数排序）")
    print("=" * 60)
    basic_selected = selector.select_unique_optimal_conditions(
        n_select=N_SELECT,
        repetition_weight=BASIC_REPETITION_WEIGHT,
        diversity_weight=BASIC_DIVERSITY_WEIGHT,
    )
    selector.print_selected_conditions(basic_selected, method="基本")
    selector.save_selected_conditions(basic_selected, BASIC_OUTPUT_FILE)
    selector.analyze_statistics(basic_selected)

    # 方法2：多样性优先
    print("\n" + "=" * 60)
    print("方法2: 多样性优先（贪婪算法）")
    print("=" * 60)
    diverse_selected = selector.select_diverse_conditions(
        n_select=N_SELECT,
        repetition_weight=DIVERSE_REPETITION_WEIGHT,
        diversity_weight=DIVERSE_DIVERSITY_WEIGHT,
    )
    selector.print_selected_conditions(diverse_selected, method="多样性优先")
    selector.save_selected_conditions(diverse_selected, DIVERSE_OUTPUT_FILE)
    selector.analyze_statistics(diverse_selected)

    # 两种方法比较
    print("\n" + "=" * 60)
    print("两种方法结果比较")
    print("=" * 60)

    basic_keys = set(selector._key_from_cond(cond) for cond in basic_selected)
    diverse_keys = set(selector._key_from_cond(cond) for cond in diverse_selected)
    common_keys = basic_keys.intersection(diverse_keys)

    print(f"基本方法选择: {len(basic_selected)}")
    print(f"多样性方法选择: {len(diverse_selected)}")
    print(f"共同选择条件数: {len(common_keys)}")
    overlap = (len(common_keys) / len(basic_selected) * 100) if basic_selected else 0
    print(f"重叠比例: {overlap:.1f}%")

    if common_keys:
        print("\n共同选择的条件（预览）:")
        for key in sorted(common_keys):
            print(f"  {selector._preview_from_key(key)}")


if __name__ == "__main__":
    main()