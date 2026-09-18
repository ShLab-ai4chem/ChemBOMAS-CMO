from ..type.sampler import Sampler
from ..data.bodata import BoData
from typing import List, Dict, Optional
import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition import (
    qLogExpectedImprovement,
    ExpectedImprovement,
    UpperConfidenceBound,
    qUpperConfidenceBound,
)
from botorch.acquisition.objective import GenericMCObjective
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from ..utils.acq import optimize_acqf_discrete_idx, optimize_acqf_discrete_weighted_idx


class BOSampler(Sampler):
    def __init__(
        self,
        dataset: BoData,
        init_x=None,
        init_y=None,
        use_constraint: bool = False,
        target_output_idx: int = 0,
        constraint_output_idx: int = 1,
        constraint_threshold: float = 80.0,
        constraint_operator: str = ">=",
        infeasible_cost: float = -1.0,
    ):
        super().__init__(dataset, init_x, init_y)

        # 约束BO相关配置（默认关闭，保持历史行为）
        self.use_constraint = use_constraint
        self.target_output_idx = target_output_idx
        self.constraint_output_idx = constraint_output_idx
        self.constraint_threshold = float(constraint_threshold)
        self.constraint_operator = constraint_operator.strip()
        if self.constraint_operator not in [">=", "<="]:
            raise ValueError(f"constraint_operator must be '>=' or '<=', got {self.constraint_operator}")
        self.infeasible_cost = float(infeasible_cost)

        if init_x is not None:

            self.train_x = init_x
            self.train_y = init_y
            self.init_model(self.train_x, self.train_y)
        else:
            self.train_x = torch.Tensor([])
            self.train_y = torch.Tensor([])

    def _is_feasible(self, values: torch.Tensor) -> torch.Tensor:
        """判断约束可行性: >= 或 <="""
        if self.constraint_operator == ">=":
            return values >= self.constraint_threshold
        else:
            return values <= self.constraint_threshold

    def _constraint_violation(self, z: torch.Tensor) -> torch.Tensor:
        """
        BoTorch约束函数要求 c(z) <= 0 为可行
        - 若是 >= 阈值: c = threshold - value
        - 若是 <= 阈值: c = value - threshold
        """
        v = z[..., self.constraint_output_idx]
        if self.constraint_operator == ">=":
            return self.constraint_threshold - v
        else:
            return v - self.constraint_threshold

    def _get_target_train_y(self) -> torch.Tensor:
        """返回用于EI比较的目标列（默认第0列）"""
        if self.train_y.dim() == 1:
            return self.train_y
        if self.train_y.shape[-1] == 1:
            return self.train_y.squeeze(-1)
        return self.train_y[:, self.target_output_idx]

    def _get_raw_objective_tensor(self, idxes: List[int]) -> torch.Tensor:
        """从dataset中读取双标签真实值（用于约束BO训练）"""
        rows = []
        for idx in idxes:
            raw_obj = self.dataset[idx].additional_feat.get("raw_objective", None)
            if raw_obj is None:
                raise ValueError(
                    f"sample idx={idx} has no raw_objective. "
                    f"Please ensure preprocess='none' and raw objective is stored in BoData."
                )
            rows.append(raw_obj.view(1, -1).to(torch.float64))
        return torch.vstack(rows)

    def _get_raw_prediction_tensor(self, idxes: List[int]) -> torch.Tensor:
        """从dataset中读取双标签伪值（用于约束BO伪初始化）"""
        rows = []
        for idx in idxes:
            raw_pred = self.dataset[idx].additional_feat.get("raw_prediction", None)
            if raw_pred is None:
                raise ValueError(
                    f"sample idx={idx} has no raw_prediction. "
                    f"Please ensure prediction file has 2 columns and load_data_prediction stores raw_prediction."
                )
            rows.append(raw_pred.view(1, -1).to(torch.float64))
        return torch.vstack(rows)

    def _build_constrained_acq(self):
        """构造约束LogEI: 最大化目标列，同时满足约束列与阈值关系"""
        if self.train_y.dim() == 1 or self.train_y.shape[-1] < 2:
            raise ValueError("Constrained BO requires train_y with at least 2 outputs.")

        feasible_mask = self._is_feasible(self.train_y[:, self.constraint_output_idx])
        if feasible_mask.any():
            best_f = self.train_y[feasible_mask, self.target_output_idx].max().item()
        else:
            best_f = self.infeasible_cost

        # qLogEI 不能配 ConstrainedMCObjective，需 objective + constraints 分开传
        objective = GenericMCObjective(
            objective=lambda z, X=None: z[..., self.target_output_idx]
        )

        acq = qLogExpectedImprovement(
            model=self.model,
            best_f=best_f,
            objective=objective,
            constraints=[lambda z: self._constraint_violation(z)],
        )
        return acq

    def _build_default_acq(self):
        """保持历史逻辑：qLogEI"""
        acq = qLogExpectedImprovement(self.model, best_f=self._get_target_train_y().max())
        return acq

    def _build_acq(self):
        if self.use_constraint:
            return self._build_constrained_acq()
        return self._build_default_acq()

    def _append_selected_train_data(self, candidates: List[int]) -> None:
        feats = torch.vstack([self.dataset[idx].feat for idx in candidates])
        self.train_x = torch.cat([self.train_x, feats])

        if self.use_constraint:
            # 约束BO训练用双标签真实值
            obj = self._get_raw_objective_tensor(candidates)
        else:
            obj = torch.vstack([self.dataset[idx]._observed_value for idx in candidates])

        self.train_y = torch.cat([self.train_y, obj])

    def update_train(self, idx: List[int], obj: torch.Tensor) -> None:
        feats = torch.vstack([self.dataset[idx].feat for idx in idx])
        self.train_x = torch.cat([self.train_x, feats])

        if self.use_constraint:
            # 约束BO模式下优先读取raw_objective，忽略外部传入obj（兼容现有MCTS调用）
            raw_obj = self._get_raw_objective_tensor(idx)
            self.train_y = torch.cat([self.train_y, raw_obj])
        else:
            self.train_y = torch.cat([self.train_y, obj])

    def acq_score(self, sub_space_idx: List[int]) -> torch.Tensor:
        acq = self._build_acq()
        search_space = torch.vstack([self.dataset[idx].feat for idx in sub_space_idx])
        search_space = search_space.view(
            search_space.shape[0], 1, search_space.shape[-1]
        )
        acq_score = acq(search_space)
        return acq_score

    def pseudo_label_sample(
        self, 
        sub_space_idx: List[int], 
        n_sample: int, 
        weight: torch.Tensor = None,
        pseudo_lable_drop_strategy: str = "weight_random",
    ) -> List[int]:
        
        factor = min(0.5, 1 - self._get_target_train_y().max() / 100)
        print(f'drop pseduo factor: {factor}, drop strategy: {pseudo_lable_drop_strategy}')
        # factor = min(0.3, 1 - self.train_y.max() / 100)
        # if self.train_y.max() < 50:
        #     factor = 0

        # 伪数据读取：约束BO模式读取双标签raw_prediction，否则走原逻辑
        if self.use_constraint:
            predict_value = self._get_raw_prediction_tensor(sub_space_idx)
        else:
            predict_value = torch.vstack(
                [self.dataset[idx].predict_value for idx in sub_space_idx]
            )

        # 完整子空间
        search_space = torch.vstack([self.dataset[idx].feat for idx in sub_space_idx])

        #####################################
        pseudo_point = list(zip(search_space, predict_value))

        if pseudo_lable_drop_strategy == "fixed":
            # 约束BO排序：先满足约束，再按目标列排序
            if self.use_constraint:
                pseudo_point = sorted(
                    pseudo_point,
                    key=lambda x: (
                        float(self._is_feasible(x[1][self.constraint_output_idx])),
                        float(x[1][self.target_output_idx]),
                    ),
                    reverse=True
                )
            else:
                pseudo_point = sorted(pseudo_point, key=lambda x: x[1], reverse=True)

            # 丢部分伪数据
            pseudo_point = pseudo_point[int(len(pseudo_point) * factor):]

        elif pseudo_lable_drop_strategy == "weight_random":
            # 基于预测值反向权重随机保留
            if self.use_constraint:
                # 用目标列作为权重基础
                yields = predict_value[:, self.target_output_idx].detach().numpy()
            else:
                yields = predict_value.detach().numpy()

            inv_weights = 1.0 / (yields - yields.min() + 1e-6)
            inv_weights = inv_weights.flatten()
            inv_weights /= inv_weights.sum()

            keep_size = int(inv_weights.shape[0] * (1 - factor))
            if keep_size <= 0:
                keep_size = 1

            keep_indices = np.random.choice(
                inv_weights.shape[0], size=keep_size, replace=False, p=inv_weights
            )

            # 按原始顺序返回保留的样本
            temp_predict_value = [predict_value[i] for i in sorted(keep_indices)]
            temp_search_space = [search_space[i] for i in sorted(keep_indices)]
            pseudo_point = [(temp_search_space[idx], v) for idx, v in enumerate(temp_predict_value)]

        else:
            raise ValueError(f"Unsupported pseudo_lable_drop_strategy: {pseudo_lable_drop_strategy}")
        #####################################
        
        #random_pseudo_point = torch.randperm(len(pseudo_point))
        pseudo_x = torch.vstack([x[0] for x in pseudo_point])
        pseudo_y = torch.vstack([x[1] for x in pseudo_point])
        # pseudo_x = torch.vstack([pseudo_point[i][0] for i in random_pseudo_point])
        # pseudo_y = torch.vstack([pseudo_point[i][1] for i in random_pseudo_point])
        pseudo_x = torch.concat([pseudo_x, self.train_x])
        pseudo_y = torch.concat([pseudo_y, self.train_y])
        # 用没被丢掉的点初始化bo模型
        self.pseudo_init_model(pseudo_x, pseudo_y, self.model.state_dict())
        acq = self._build_acq()
        # acq = qUpperConfidenceBound()     # tofix
        #####################################
        if weight is not None:
            # 完整子空间都有可能推荐
            candidates = optimize_acqf_discrete_weighted_idx(
                acq, q=n_sample, choices=search_space, weights=weight
            ).tolist()
        else:
            candidates = optimize_acqf_discrete_idx(
                acq, q=n_sample, choices=search_space
            ).tolist()
        candidates = (
            [sub_space_idx[idx] for idx in candidates]
            if isinstance(candidates, list)
            else [sub_space_idx[candidates]]
        )
        self._append_selected_train_data(candidates)
        self.init_model(self.train_x, self.train_y, self.model.state_dict())
        return candidates
    
    
    def real_exp_pseudo_label_sample(
        self, sub_space_idx: List[int], n_sample: int, cur_time:int, weight: torch.Tensor = None,
        eta = 0.001, beta = 0.01, n_times = 20, mask:List[int] = None, return_acq_values:bool = False
    ) -> List[int]:
        
        factor = min(0.3, 1 - self._get_target_train_y().max() / 100)
        # if self.train_y.max() < 50:
        #     factor = 0

        if self.use_constraint:
            predict_value_all = self._get_raw_prediction_tensor(sub_space_idx)
        else:
            predict_value_all = torch.vstack(
                [self.dataset[idx].predict_value for idx in sub_space_idx]
            )

        # print(f'mask: {mask} and {mask is None}')
        # print(f'sub_space_idx: {sub_space_idx}')
        valid_idx = [idx for idx in sub_space_idx if idx not in mask]
        search_space = torch.vstack([self.dataset[idx].feat for idx in valid_idx])

        # 让predict_value和search_space对齐
        idx2pos = {idx: i for i, idx in enumerate(sub_space_idx)}
        predict_value = torch.vstack([predict_value_all[idx2pos[idx]] for idx in valid_idx])

        #####################################
        pseudo_point = list(zip(search_space, predict_value))
        
        if self.use_constraint:
            pseudo_point = sorted(
                pseudo_point,
                key=lambda x: (
                    float(self._is_feasible(x[1][self.constraint_output_idx])),
                    float(x[1][self.target_output_idx]),
                ),
                reverse=True
            )
        else:
            pseudo_point = sorted(pseudo_point, key=lambda x: x[1],reverse=True)

        pseudo_point = pseudo_point[int(len(pseudo_point) * factor) :]
        
        random_pseudo_point = torch.randperm(len(pseudo_point))
        #  random_pseudo_point = torch.randperm(factor * len(search_space))
        
        pseudo_x = torch.vstack([x[0] for x in pseudo_point])
        pseudo_y = torch.vstack([x[1] for x in pseudo_point])
        pseudo_x = torch.vstack([pseudo_point[i][0] for i in random_pseudo_point])
        pseudo_y = torch.vstack([pseudo_point[i][1] for i in random_pseudo_point])
        pseudo_x = torch.concat([pseudo_x, self.train_x])
        # print(f'pseudo_x device: {pseudo_x.device}')
        # print(f'train_x device: {self.train_x.device}')
        # print(f'pseudo_y device: {pseudo_y.device}')
        # print(f'train_y device: {self.train_y.device}')
        pseudo_y = torch.concat([pseudo_y.to(self.train_y.device), self.train_y])

        self.pseudo_init_model(pseudo_x, pseudo_y, self.model.state_dict())
        if self.use_constraint:
            acq = self._build_constrained_acq()
        else:
            if cur_time<(n_times/2):
                acq = qLogExpectedImprovement(self.model, best_f=self._get_target_train_y().max(),eta = eta)
            elif cur_time>=(n_times/2):
                acq = qUpperConfidenceBound(self.model, beta=beta)
        #####################################
        if weight is not None:
            candidates = optimize_acqf_discrete_weighted_idx(
                acq, q=n_sample, choices=search_space, weights=weight
            ).tolist()
        else:
            if return_acq_values:
                candidates, acq_val = optimize_acqf_discrete_idx(
                    acq, q=n_sample, choices=search_space, return_acq_values=True
                )
            else:
                candidates = optimize_acqf_discrete_idx(
                    acq, q=n_sample, choices=search_space
                ).tolist()
        candidates = (
            [valid_idx[idx] for idx in candidates]
            if isinstance(candidates, list)
            else [valid_idx[candidates]]
        )

        if return_acq_values:
            return candidates, acq_val
        else:
            return candidates

    def real_exp( self, sub_space_idx: List[int], n_sample: int, weight: torch.Tensor = None,mask:List[int] = None) -> List[int]:

        
        acq = self._build_acq()
        valid_idx = [idx for idx in sub_space_idx if idx not in mask]
        search_space = torch.vstack([self.dataset[idx].feat for idx in valid_idx])
        print('sub space have {} points'.format(search_space.shape[0]))
        if weight is not None:
            candidates = optimize_acqf_discrete_weighted_idx(
                acq, q=n_sample, choices=search_space, weights=weight
            ).tolist()
        else:
            candidates = optimize_acqf_discrete_idx(
                acq, q=n_sample, choices=search_space
            ).tolist()
        # for i in n_sample:
        #     if i < 10:
        #         acq = qLogExpectedImprovement(self.model, best_f=self.train_y.max(),eta=(i+1)*0.001)
        #     else:
        #         acq = qUpperConfidenceBound(self.model, beta=(i-9)*0.1)
        candidates = (
            [valid_idx[idx] for idx in candidates]
            if isinstance(candidates, list)
            else [valid_idx[candidates]]
        )
        for idx in candidates:
            self.dataset[idx]._is_observed = True

        return candidates
    def sample(
        self, sub_space_idx: List[int], n_sample: int, weight: torch.Tensor = None
    ) -> List[int]:
        acq = self._build_acq()
        search_space = torch.vstack([self.dataset[idx].feat for idx in sub_space_idx])
        # print('sub space have {} points'.format(search_space.shape[0]))
        if weight is not None:
            candidates = optimize_acqf_discrete_weighted_idx(
                acq, q=n_sample, choices=search_space, weights=weight
            ).tolist()
        else:
            candidates = optimize_acqf_discrete_idx(
                acq, q=n_sample, choices=search_space
            ).tolist()
        candidates = (
            [sub_space_idx[idx] for idx in candidates]
            if isinstance(candidates, list)
            else [sub_space_idx[candidates]]
        )
        self._append_selected_train_data(candidates)
        self.init_model(self.train_x, self.train_y, self.model.state_dict())

        return candidates

    def init_model(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        state_dict: Optional[Dict] = None,
    ):
        if train_y.dim() == 1:
            train_y = train_y.view(-1, 1)
        self.model = SingleTaskGP(
            train_x,
            train_y,
            input_transform=Normalize(d=train_x.shape[-1]),
            outcome_transform=Standardize(m=train_y.shape[-1]),
        ).to(train_x)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        if state_dict is not None:
            try:
                self.model.load_state_dict(state_dict)
            except Exception:
                # 输出维度变化时（如从单输出切到双输出）允许忽略旧state_dict
                pass
        fit_gpytorch_mll(self.mll)

    def pseudo_init_model(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        state_dict: Optional[Dict] = None,
    ):
        if train_y.dim() == 1:
            train_y = train_y.view(-1, 1)
        self.model = SingleTaskGP(
            train_x,
            train_y,
            input_transform=Normalize(d=train_x.shape[-1]),
            outcome_transform=Standardize(m=train_y.shape[-1]),
        ).to(train_x)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        if state_dict is not None:
            try:
                self.model.load_state_dict(state_dict)
            except Exception:
                pass
