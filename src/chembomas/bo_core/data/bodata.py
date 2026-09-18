from typing import Dict,NamedTuple,Optional,Union,List
import torch
from collections import namedtuple
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from loguru import logger
from .preprocess import PREPROCESS_FUNC


class Sample:
    '''样本'''
    def __init__(self,feat:Optional[torch.Tensor],observed_value:Optional[torch.Tensor],
                 category_value:Optional[Dict],predict_value:Optional[torch.Tensor]=None,
                 additional_feat:Dict[str,torch.Tensor]={})->None:
        self._feat:torch.Tensor = feat
        self.additional_feat:Dict[str,torch.Tensor] = additional_feat
        self.predict_value:torch.Tensor = predict_value
        self._observed_value:torch.Tensor = observed_value
        self.category_value:Dict[str,str] = category_value
        self._is_observed:bool = False
    
    
    def __str__(self):
        return f"{self.category_value} -> observed_value:{self.observed_value},predict_value:{self.predict_value}"
    @property
    def is_observed(self)->bool:
        return self._is_observed
    @property
    def embedding(self)->torch.Tensor:
        return self.additional_feat.get('embedding')
    @property
    def feat(self)->torch.Tensor:
        return self._feat
    @feat.setter
    def feat(self,value:torch.Tensor)->None:
        self._feat = value
    @property
    def observed_value(self)->torch.Tensor:
        self._is_observed = True
        return self._observed_value

    
class BoData():
    '''数据集'''
    def __init__(self, samples: Optional[Dict[int, Sample]]) -> None:
        self._samples: Dict[int,Sample] = samples

    @classmethod
    def read_round_0_data(cls,
                          search_space_path: str,
                          objective: List[str],
                          category: List[str],
                          preprocess: str='default',
                          preprocess_kwargs: dict={},
                          target_output_idx: int = 0,
                          ):
        '''读取第0轮生成所需的数据。
        Args:
            search_space_path: 待搜索的变量空间文件路径
            objective: csv文件中反应优化目标列名
            category: csv文件中反应条件（变量空间）列名，若无则默认除objective外的所有列
            preprocess: 存在多个指标时如何进行预处理
            preprocess_kwargs: 预处理参数
            target_output_idx: 当存在多个优化目标时，指定用于BO优化的目标索引（默认0，即第1个目标）
        '''
        # 读取变量空间
        search_space = pd.read_csv(search_space_path, na_values=[])
        search_space[objective] = search_space[objective].fillna(-1)
        search_space = search_space.astype(str)

        # 进行独热编码并转化为tensor
        one_hot = pd.get_dummies(search_space[category])
        feats = torch.tensor(one_hot.values, dtype=torch.float64)

        samples = {}
        # 遍历变量空间，为每种组合创建一个sample
        for i, (index, row) in tqdm(enumerate(search_space.iterrows()), desc="create sample for all combinations"):
            category_value = {}
            objective_value = []
            for col in category:
                category_value[col] = row[col]

            # 额外保留双指标原始值，用于约束BO
            raw_objective_value = None
            if len(objective) == 2:
                raw_objective_value = torch.tensor(
                    [float(row[objective[0]]), float(row[objective[1]])],
                    dtype=torch.float64
                )

            # 根据objective数量和preprocess方法设置进行数据处理
            if len(objective) == 1:  # 单指标直接使用
                objective_value.append(float(row[objective[0]]))
                obj_len = 1
            elif len(objective) == 2:
                if preprocess == 'none':
                    # none: 多标签不合并；树/MCTS仍使用第1个目标（通常Yield）作为单值
                    if target_output_idx < 0 or target_output_idx >= len(objective):
                        raise ValueError(f"target_output_idx {target_output_idx} out of range for objective={objective}")
                    objective_value.append(float(row[objective[target_output_idx]]))
                    obj_len = 1
                else:
                    proc_obj, obj_len = PREPROCESS_FUNC[preprocess](
                        float(row[objective[0]]), float(row[objective[1]]), **preprocess_kwargs)
                    objective_value.append(proc_obj)
            else:
                raise NotImplementedError(f'objective number {len(objective)} exceed 2, unable to handle now')

            objective_value = torch.tensor(objective_value, dtype=torch.float64).view(-1, obj_len).clone().detach()

            # 创建sample
            sample = Sample(
                feat=feats[i],
                observed_value=objective_value,
                category_value=category_value,
                )
            if raw_objective_value is not None:
                sample.additional_feat['raw_objective'] = raw_objective_value.clone().detach()

            samples[i] = sample

        return cls(samples)

    @classmethod
    def read_round_n_data(cls,
                          search_space_path: str,
                          wet_exp_result_path: str,
                          uncompleted_exp_path: str,
                          objective: List[str],
                          category: List[str],
                          preprocess: str='default',
                          preprocess_kwargs: dict={},
                          target_output_idx: int = 0,
                          ):
        '''读取第n轮生成所需的数据。
        Args:
            search_space_path: 待搜索的变量空间文件路径
            wet_exp_result_path: 湿实验结果文件路径
            uncompleted_path: 未完成实验文件路径
            objective: csv文件中反应优化目标列名
            category: csv文件中反应条件（变量空间）列名，若无则默认除objective外的所有列
            preprocess: 存在多个指标时如何进行预处理
            preprocess_kwargs: 预处理参数
            target_output_idx: 当存在多个优化目标时，指定用于BO优化的目标索引（默认0，即第1个目标）
        '''
        # 读取变量空间
        search_space = pd.read_csv(search_space_path, na_values=[]).astype(str)
        wet_exp_result = pd.read_csv(wet_exp_result_path, na_values=[]).astype(str)
        uncompleted_exp = pd.read_csv(uncompleted_exp_path, na_values=[]).astype(str)
        print(f'searchspace size: {search_space.shape}')
        print(f'exp_results: {wet_exp_result[category + objective]}')

        # 将实验结果合并到变量空间中，并且保留变量空间所有组合
        merged_data = pd.merge(
            search_space[category],
            wet_exp_result[category + objective],  # 带Yield的实验数据
            on=category,
            how='left'  # 保留所有变量空间的组合
        )
        print(f'merged data size: {merged_data.shape}')
        # 设定无实验结果或未完成的yield为-1
        merged_data[objective] = merged_data[objective].fillna(-1)
        uncompleted_exp[objective] = -1

        # 对条件进行独热编码并转化为tensor
        one_hot = pd.get_dummies(merged_data[category])
        feats = torch.tensor(one_hot.values, dtype=torch.float64)

        samples = {}
        mask_uncompleted = []
        # 遍历变量空间，为每种组合创建一个sample
        for i, (index, row) in tqdm(enumerate(merged_data.iterrows()), desc="create sample for all combinations"):
            category_value = {}
            objective_value = []
            for col in category:
                category_value[col] = row[col]

            # 额外保留双指标原始值，用于约束BO
            raw_objective_value = None
            if len(objective) == 2:
                raw_objective_value = torch.tensor(
                    [float(row[objective[0]]), float(row[objective[1]])],
                    dtype=torch.float64
                )

            # 根据objective数量和preprocess方法设置进行数据处理
            if len(objective) == 1:  # 单指标直接使用
                objective_value.append(float(row[objective[0]]))
                obj_len = 1
            elif len(objective) == 2:
                if preprocess == 'none':
                    # none: 多标签不合并；树/MCTS仍使用第1个目标（通常Yield）作为单值
                    if target_output_idx < 0 or target_output_idx >= len(objective):
                        raise ValueError(f"target_output_idx {target_output_idx} out of range for objective={objective}")
                    objective_value.append(float(row[objective[target_output_idx]]))
                    obj_len = 1
                else:
                    proc_obj, obj_len = PREPROCESS_FUNC[preprocess](
                        float(row[objective[0]]), float(row[objective[1]]), **preprocess_kwargs)
                    objective_value.append(proc_obj)
            else:
                raise NotImplementedError(f'objective number {len(objective)} exceed 2, unable to handle now')

            objective_value = torch.tensor(objective_value, dtype=torch.float64).view(-1, obj_len).clone().detach()

            # 将未完成的实验mask掉
            if (row[category] == uncompleted_exp[category]).all(axis=1).any():
                mask_uncompleted.append(i)
            # 创建sample
            sample = Sample(
                feat=feats[i],
                observed_value=objective_value,
                category_value=category_value,
                )
            if raw_objective_value is not None:
                sample.additional_feat['raw_objective'] = raw_objective_value.clone().detach()
            # 有湿实验结果
            if objective_value[0] != -1:
                sample._is_observed = True

            samples[i] = sample

        return cls(samples), mask_uncompleted

    @classmethod
    def read_dryexp_data(cls,
                         search_space_with_obj_path: str,
                         objective: List[str],
                         category: List[str],
                         preprocess: str='default',
                         preprocess_kwargs: dict={},
                         target_output_idx: int = 0,
                         ):
        '''读取干实验结果数据
        Args:
            search_space_with_obj_path: 待搜索的变量空间文件路径，需包含所有数据点objective_value
            objective: csv文件中反应优化目标列名
            category: csv文件中反应条件（变量空间）列名
            preprocess: 存在多个指标时如何进行预处理
            preprocess_kwargs: 预处理参数
            target_output_idx: 当存在多个优化目标时，指定用于BO优化的目标索引（默认0，即第1个目标）
        '''
        # 读取变量空间和实验结果
        search_space_with_obj = pd.read_csv(search_space_with_obj_path, na_values=[]).astype(str)
        logger.info(f'dry experiment search space size: {search_space_with_obj.shape}')

        # 对条件进行独热编码并转化为tensor
        one_hot = pd.get_dummies(search_space_with_obj[category])
        feats = torch.tensor(one_hot.values, dtype=torch.float64)

        samples = {}
        best_objective_value = -float('inf')
        # 遍历变量空间，为每种组合创建一个sample
        for i, (index,row) in tqdm(enumerate(search_space_with_obj.iterrows()),
                                   desc="create sample for all combinations"):
            category_value = {}
            objective_value = []
            for col in category:
                category_value[col] = row[col]

            # 额外保留双指标原始值，用于约束BO
            raw_objective_value = None
            if len(objective) == 2:
                raw_objective_value = torch.tensor(
                    [float(row[objective[0]]), float(row[objective[1]])],
                    dtype=torch.float64
                )

            # 根据objective数量和preprocess方法设置进行数据处理
            if len(objective) == 1:  # 单指标直接使用
                objective_value.append(float(row[objective[0]]))
                obj_len = 1
            elif len(objective) == 2:
                if preprocess == 'none':
                    # none: 多标签不合并；树/MCTS仍使用第1个目标（通常Yield）作为单值
                    if target_output_idx < 0 or target_output_idx >= len(objective):
                        raise ValueError(f"target_output_idx {target_output_idx} out of range for objective={objective}")
                    objective_value.append(float(row[objective[target_output_idx]]))
                    obj_len = 1
                else:
                    proc_obj, obj_len = PREPROCESS_FUNC[preprocess](
                        float(row[objective[0]]), float(row[objective[1]]), **preprocess_kwargs)
                    objective_value.append(proc_obj)
            else:
                raise NotImplementedError(f'objective number {len(objective)} exceed 2, unable to handle now')

            # objective_value = torch.tensor(objective_value, dtype=torch.float64).view(-1,len(objective)).clone().detach()
            objective_value = torch.tensor(objective_value, dtype=torch.float64).clone().detach()
            best_objective_value = max(best_objective_value, objective_value[0].item())
            
            # 创建sample
            sample = Sample(
                feat=feats[i],
                observed_value=objective_value,
                category_value=category_value,
                )
            if raw_objective_value is not None:
                sample.additional_feat['raw_objective'] = raw_objective_value.clone().detach()

            samples[i] = sample

        logger.info(f'best objective value in dry experiment: {best_objective_value}')

        return cls(samples)

    def load_data_prediction(self,
                             pred_val_path,
                            #  group_index=None,
                             preprocess: str='default',
                             preprocess_kwargs: dict={},
                             target_output_idx: int = 0,
                             ) -> np.ndarray:
        '''将LLM预测的反应结果加载到数据集中，作为sample的predict_value'''
        # 这里读取逻辑可能要修改
        prediction = torch.load(pred_val_path, weights_only=False, map_location=torch.device('cpu'))['pred_yields_by_rxn']
        if prediction.dim() == 1:
            prediction = prediction.unsqueeze(1)
        logger.info(f'loaded predicted value shape: {prediction.shape}')

        # if group_index is not None:  # ?
        #     prediction = prediction[int(group_index[0]): int(group_index[1]) + 1]

        # 设置pred_value
        # 在这里增加一个可选的数据处理逻辑，将两个预测值合并
        if len(prediction[0]) == 1:
            processed_prediction = prediction.to(torch.float64)
        elif len(prediction[0]) == 2:
            if preprocess == 'none':
                # none: 保留双标签预测给约束BO，同时给树保留单值预测（默认第1个目标）
                if target_output_idx < 0 or target_output_idx >= prediction.shape[1]:
                    raise ValueError(f"target_output_idx {target_output_idx} out of range for prediction shape {prediction.shape}")
                processed_prediction = prediction[:, target_output_idx].view(-1, 1).to(torch.float64)
            else:
                processed_prediction = []
                for pred in prediction:
                    processed_pred, _ = PREPROCESS_FUNC[preprocess](pred[0], pred[1], **preprocess_kwargs)
                    processed_prediction.append(processed_pred)
                processed_prediction = torch.tensor(processed_prediction, dtype=torch.float64).view(-1,1)
        else:
            raise NotImplementedError(f'objective number {len(prediction[0])} exceed 2, unable to handle now')

        # 在约束BO模式下，保存双标签预测到additional_feat['raw_prediction']
        if prediction.shape[1] == 2:
            for i in range(prediction.shape[0]):
                self._samples[i].additional_feat['raw_prediction'] = prediction[i].clone().detach().to(torch.float64)

        for i, pred in enumerate(processed_prediction):
            self._samples[i].predict_value = pred.clone().detach().view(-1,1).to(torch.float64)
        
        return processed_prediction

    def load_data_embedding(self,path,n_pca:Optional[int])->None:
        embedding = torch.load(path,weights_only=False,map_location=torch.device('cpu'))['cls_embs']
        if n_pca is not None:
            pca = PCA(n_components=n_pca)
            embedding = pca.fit_transform(embedding)
            embedding = torch.tensor(embedding,dtype=torch.float64)
        for i, emb in enumerate(embedding):
            self._samples[i].additional_feat['embedding'] = emb.clone().detach().to(torch.float64)

    def make_data_harder_from_npy(self, npy_path: str, observed_idx = None) -> None:
        """从预先生成的长尾分布结果 npy 文件中加载数据"""
        kept_indices = np.load(npy_path)          # 一维 int64 数组
        logger.info(f'initial kept_indices length {len(kept_indices)}')

        # if observed_idx:  # 处理与先验数据冲突的问题
        #     # 如果有在先验数据中出现的idx不在kept_indices的情况，应该将其加入
        #     add_indices = [idx for idx in observed_idx if idx not in kept_indices]
        #     logger.debug(f'added indices: {add_indices}')
        #     kept_indices = np.append(kept_indices, add_indices)
        #     logger.info(f'modified kept_indices length {len(kept_indices)}')

        # 过滤：只保留 kept_indices 里出现的样本
        self._samples = {idx: self._samples[idx]
            for idx in kept_indices
            if idx in self._samples}

        if observed_idx:  # 处理与先验数据冲突的问题
            # 如果有在先验数据中出现的idx不在kept_indices的情况，应该将其从先验数据中删除
            modified_observed_indices = [idx for idx in observed_idx if idx in kept_indices]
            logger.info(f'modified observed indices: {modified_observed_indices}')
            return modified_observed_indices
    
    def make_data_harder(self,factor,keep_idx = True,random_seed=None)->None:
        """Make the data distribution more challenging by dropping samples to create a long-tail effect"""
        
        sorted_indices = sorted(self._samples.keys(), 
                            key=lambda x: self._samples[x]._observed_value)
        
        n_samples = len(sorted_indices)
        samples_to_keep = {}
        samples_to_keep_idx = {}
        new_idx = 0
        # Keep samples with probability decreasing exponentially
        for i, idx in enumerate(sorted_indices):
            # Calculate retention probability based on position
            keep_prob = np.exp(-0.1 * i * factor/ n_samples)
            
            # Randomly decide whether to keep sample
            if np.random.random() < keep_prob:
                samples_to_keep[new_idx] = self._samples[idx]
                samples_to_keep_idx[idx] = self._samples[idx]
                new_idx += 1
        if keep_idx:
            self._samples = samples_to_keep_idx
        else:
            self._samples = samples_to_keep
        
    def use_test_set(self, path, group_index=None)->None:
        # import pdb;pdb.set_trace()
        test_idx = torch.load(path, weights_only=False,map_location=torch.device('cpu'))['val_idx']
        if group_index is not None:
            test_idx = [i for i in test_idx if i >= int(group_index[0]) and i <= int(group_index[1])]
        self._samples = {i:self._samples[i] for i in test_idx}
        
    def __len__(self):
        return len(self._samples)
    def __getitem__(self,index:Union[int,List[int]])->Sample:
        if type(index) == list:
            return [self._samples.get(i) for i in index]
        return self._samples.get(index)
    def __iter__(self):
        # Convert samples dict to list of values for iteration
        self._iter_samples = list(self._samples.values())
        self._iter_index = 0
        return self
    
    def __next__(self):
        if self._iter_index >= len(self._iter_samples):
            raise StopIteration
        sample = self._iter_samples[self._iter_index]
        self._iter_index += 1
        return sample
    
    @property
    def all_idxes(self)->List[int]:
        return list(self._samples.keys())
    
    @property
    def max_value(self)->float:
        return max([sample._observed_value for sample in self._samples.values()])