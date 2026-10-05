import numpy as np
import torch
import random
from typing import List, Dict, Tuple, Optional, ClassVar, Any, Set, Union
from .node import Node, Subspace
from ..data.bodata import BoData
from ..sampler.random import RandomSampler
from collections import defaultdict, deque
from loguru import logger
from tqdm import tqdm


class MCTS:
    '''构建蒙特卡洛树'''
    def __init__(self,
                 dataset: BoData = None,  # 候选数据点集合
                 num_init_samples: int = 5,  # 初始采样数量
                 leaf_size: int = 100,  # leaf node最大数据点数目
                 batch_size: int = 5,  # 每轮迭代选择的新样本数量
                 sampler: Optional[RandomSampler] = None,  # 负责实际采样的模型
                 cp: float = 1.0,  # UCB探索系数
                 variable_nums: int = 0,  # 层数
                 n_candidates: int = 10,  # 初始轮平行重复组数（不影响输出数量）
                 use_diverse_sample: bool = False,  # 是否使用多样性采样
                 ) -> None:

        self._num_init_samples = num_init_samples
        self._cp = cp
        self._dataset = dataset
        self._leaf_size = leaf_size
        self._batch_size = batch_size

        if sampler is None:  # 若未传入采样器则使用随机采样
            self._sampler = RandomSampler(dataset)
        else:
            self._sampler = sampler
        self._root: Optional[Node] = None

        self.variable_nums: int = variable_nums
        self._n_candidates: int = n_candidates
        self.use_diverse_sample: bool = use_diverse_sample

        # 统计访问次数
        self._leaf_visit_counts: Dict[int, int] = defaultdict(int)

    def analyze_combinations(self, order: List[str]) -> Dict[str, List[float]]:
        all_leaves = Node.get_all_leaves()
        combination_results = {}

        if not all_leaves:
            print("警告: 未找到任何叶子节点。树可能尚未构建。")
            return combination_results

        dataset = Subspace._shared_dataset

        for leaf in all_leaves:
            # 如果一个叶子节点是空的，则跳过
            if len(leaf.space.idxes) == 0:
                continue

            # --- 步骤 1: 通过回溯路径来确定组合名称 ---
            path_from_root = leaf.path_from_root()
            combo_parts = []
            
            # 路径从根开始，第一个子节点深度为1，对应order[0]
            # path_from_root[0] 是根节点，我们从 path_from_root[1] 开始
            for node in path_from_root[1:]:
                parent = node.parent
                component_name = order[node._depth - 1] # 节点深度从1开始，对应order索引0

                # 找到当前节点是其父节点的第几个孩子，以确定分组索引
                child_index = -1
                for i, child_id in enumerate(parent._children):
                    if child_id == node.id:
                        child_index = i
                        break
                
                combo_parts.append(f"{component_name}_Group{child_index}")
            
            combination_key = " | ".join(combo_parts)

            # --- 步骤 2: 提取该组合下所有的真实产率 ---
            # .item() 用于将PyTorch张量转换为Python纯数字
            yields = [dataset[idx].observed_value.item() for idx in leaf.space.idxes]
            
            combination_results[combination_key] = yields

        return combination_results

    def find_node_by_data_idx(self, data_idx: int, max_steps: int = 1000) -> Optional[Node]:
        '''根据数据索引找到所属叶节点；异常时抛出错误'''
        if self._root is None:
            raise RuntimeError("find_node_by_data_idx called before tree initialization (root is None).")

        search_node = self._root
        visited: Set[int] = set()
        steps = 0

        while not search_node.is_leaf:
            steps += 1
            if steps > max_steps:
                raise RuntimeError(
                    f"find_node_by_data_idx exceeded max_steps={max_steps}, "
                    f"possible infinite loop. data_idx={data_idx}, current_node={search_node.id}"
                )

            if search_node.id in visited:
                raise RuntimeError(
                    f"Cycle detected in tree traversal. data_idx={data_idx}, node={search_node.id}"
                )
            visited.add(search_node.id)

            found_next = False
            for child in search_node.children:
                if data_idx in child.space.idxes:
                    search_node = child
                    found_next = True
                    break

            if not found_next:
                raise RuntimeError(
                    f"data_idx={data_idx} not found in any child of node={search_node.id}. "
                    "Tree partition may be inconsistent."
                )

        return search_node

    def init_tree(self,
                  expert_partition: Optional[Dict[str, List[List]]] = None,
                  order: Optional[List[str]] = None,
                  dont_build_tree: bool = False,
                  ):
        '''干实验代码
        标准初始化流程：建树 → 随机采样初始点 → 更新模型 → 向全树反向传播 acquisition score。
        '''
        logger.info("Use init_tree")
        # 初始化子空间
        Subspace.init(self._dataset)
        # 初始化节点
        Node.init()
        # 创建根节点，包含所有数据点
        self._root = Node(space=Subspace(self._dataset.all_idxes),
                          leaf_size=self._leaf_size,
                          depth=0)

        # 构建树
        if dont_build_tree:  # 不构建MCT
            pass
        elif expert_partition is None:  # 从上面构建的根节点出发使用默认策略构建新树
            Node.build_tree(self._root)
        else:
            Node.build_tree_by_expert(self._root,  # 从反应物分类和重要性排序构建新树
                                      expert_partition,
                                      order)
        logger.info(f"The tree has {len(Node._all_nodes)} nodes and {len(Node._all_leaves)} leaves")

        # 随机采样初始样本
        random_idxes = torch.randperm(len(range(len(self._dataset.all_idxes))))[:self._num_init_samples].tolist()
        candidates = [self._dataset.all_idxes[idx] for idx in random_idxes]

        # 获取这批初始样本的真实值
        observed_values = [self._dataset[idx].observed_value for idx in candidates]
        # 将初始样本及其真实值加入训练数据集
        self._sampler.update_train(candidates, torch.vstack(observed_values))
        # 使用训练数据集训练采样器
        self._sampler.init_model(self._sampler.train_x, self._sampler.train_y)
        # 对每个数据点计算acq score
        acq_scores = self._sampler.acq_score(self._root.space.idxes)
        # 找到每个数据点所在的叶节点，并反向传播acq score（更新路径上每个节点统计量用于UCB计算）
        for acq, idx in zip(acq_scores, self._root.space.idxes):
            self.find_node_by_data_idx(idx).backprop(acq)

        return observed_values
    
    def diverse_init_tree(self,
                          expert_partition: Optional[Dict[str, List[List]]] = None,
                          order:Optional[List[str]] = None,
                          ):
        '''干实验代码
        冷启动阶段（即没有或仅有少量真实观测数据时）进行结构化多样性采样
        '''
        logger.info("Use diverse_init_tree")
        # 初始化子空间
        Subspace.init(self._dataset)
        # 初始化节点
        Node.init()
        # 创建根节点，包含所有数据点
        self._root = Node(space=Subspace(self._dataset.all_idxes),
                          leaf_size=self._leaf_size,
                          depth=0)

        # 构建树
        if expert_partition is None:  # 从上面构建的根节点出发使用默认策略构建新树
            Node.build_tree(self._root)
        else:
            Node.build_tree_by_expert(self._root,  # 从反应物分类和重要性排序构建新树
                                      expert_partition,
                                      order)

        # 随机生成 n_candidates 组候选叶子集合（每组包含 n_sample 个叶子）
        init_leaves = self.diversity_aware_random_sample(
            n_sample=self._num_init_samples,
            n_candidates=self._n_candidates,
            )
        # 在每个叶子里随机采集一个数据点
        candidates = []
        for leaf in init_leaves:
            idx = random.choice(leaf.space.idxes)
            candidates.append(idx)

        # 获取这批初始样本的真实值
        observed_values = [self._dataset[idx].observed_value for idx in candidates]
        # 将初始样本及其真实值加入训练数据集
        self._sampler.update_train(candidates, torch.vstack(observed_values))
        # 使用训练数据集训练采样器
        self._sampler.init_model(self._sampler.train_x, self._sampler.train_y)

        # 对所有数据点使用预测值进行伪反向传播
        for idx in self._dataset.all_idxes:
            leaf = self.find_node_by_data_idx(idx)
            alpha = 1 / (1 + len(leaf.space.idxes))  # 衰减系数，叶子越大当前操作影响越小
            self.find_node_by_data_idx(idx).pseudo_backprop(self._dataset[idx].predict_value.squeeze(),
                                                            alpha=alpha,
                                                            )
        
        # 对已采样数据点基于真实值进行反向传播
        for candidate, value in zip(candidates, observed_values):
            node = self.find_node_by_data_idx(candidate)
            node.inc_incomplete_count()  # 进行中实验+1
            node.backprop(value.squeeze())  # 进行中实验-1

        return observed_values

    def observed_init_tree(self,
                           expert_partition: Dict[str,List[List]],
                           order: List[str],
                           observed_idx: List,
                           pseudo_label: bool = False,
                           dont_build_tree: bool = False,
                           kappa: float = 0.01,  # 用于计算节点UCB打分的kappa值
                           kappa_by_depth: bool = False,  # 是否根据节点深度调整kappa值
                           depth_kappa_mapping: Optional[Dict[str, float]] = None, # 若按深度调整kappa，则传入深度-对应kappa值映射
                           ):
        '''干实验代码
        在已有部分真实观测数据的基础上进行树的初始化
        '''
        Subspace.init(self._dataset)
        Node.init()
        self._root = Node(space = Subspace(self._dataset.all_idxes), leaf_size=self._leaf_size, depth=0)
        logger.info("Use observed_init_tree")

        if dont_build_tree:
            logger.info("no tree is built")
            pass
        elif expert_partition is None:
            Node.build_tree(self._root)
            logger.info("build tree with no partition")
        else:
            Node.build_tree_by_expert(self._root, expert_partition, order)
            logger.info("build tree with expert partition")

        logger.debug(f"Observed idx: {observed_idx}")
        logger.info(f"The tree has {len(Node._all_nodes)} nodes and {len(Node._all_leaves)} leaves")
        ################################################################################

        # 基于 index 获取观测值，过滤掉不在数据集中的index
        observed_idx = [x for x in observed_idx if x in self._dataset.all_idxes]
        logger.debug(f"Num of processed observed values: {len(observed_idx)}")
        # 在此处调用observed_value参数时已经隐性将数据点设置为is_observed=True(参见bodata class Sample observed_value)
        # 在后续的search过程中起到了mask的效果
        observed_value = [self._dataset[idx].observed_value for idx in observed_idx]

        # 使用观测值更新BOsampler的训练数据，并训练模型
        logger.info(f"Loading {len(observed_idx)} train data as observed values to BOsampler")
        self._sampler.update_train(observed_idx, torch.vstack(observed_value))
        logger.debug(f'Length of train_x: {len(self._sampler.train_x)}, Length of train_y: {len(self._sampler.train_y)}')
        self._sampler.init_model(self._sampler.train_x, self._sampler.train_y)
        logger.debug('init_model completed')

        # 使用伪数据进行伪反向传播（UCB值更新）
        if pseudo_label:
            logger.info('start pseudo backprop')
            for idx in self._dataset.all_idxes:
                leaf = self.find_node_by_data_idx(idx)
                alpha = 1 / (1 + len(leaf.space.idxes))
                #alpha = 1
                self.find_node_by_data_idx(idx).pseudo_backprop(self._dataset[idx].predict_value.squeeze(),
                                                                alpha=alpha)
            logger.info('finish pseudo backprop')

        # 使用真实观测值进行反向传播（UCB值更新）
        logger.info('start real backprop')
        for idx, value in zip(observed_idx, observed_value):
            logger.debug('start find_node')
            node = self.find_node_by_data_idx(idx)
            logger.debug('start inc_incomplete')
            node.inc_incomplete_count()
            logger.debug('start backprop')
            node.backprop(value.squeeze())
            logger.debug('finish backprop')
        logger.info('finish real backprop')

        if len(observed_idx) != self._num_init_samples:
            logger.warning(f"Have {len(observed_idx)} observed samples, which is != {self._num_init_samples}!")
        
        # 显式设置kappa值
        if kappa_by_depth:
            Node.set_all_kappa_by_depth(order, depth_kappa_mapping)
            logger.info(f"kappa for exploration function calculation is set by depth: {depth_kappa_mapping}")
        else:
            Node.set_all_kappa(kappa)
            logger.info(f"kappa for exploration function calculation is set to {kappa}")

        return observed_value

    def pseudo_init_tree(self,
                         expert_partition: Dict[str, List[List]],
                         order: List[str],
                         pseudo_label: bool = False,  # 是否使用预测数据
                         ):
        '''干实验代码
        在没有任何真实实验数据的情况下，通过伪标签（predict_value）或随机策略，模拟一次“虚拟实验”，完成 MCTS 树的初始化
        '''
        # 初始化
        Subspace.init(self._dataset)
        Node.init()
        self._root = Node(space = Subspace(self._dataset.all_idxes), leaf_size=self._leaf_size, depth=0)
        
        # 基于专家知识建树
        Node.build_tree_by_expert(self._root, expert_partition, order)
        logger.info(f"The tree has {len(Node._all_nodes)} nodes and {len(Node._all_leaves)} leaves")
        ################################################################################

        if pseudo_label:  # 使用预测数据作为伪标签，从平均预测性能最高的子空间中采样
            avg_pred_value_of_leaves: Dict[int, float] = {}
            max_idxes = []  # 选取的数据点index（预测值最高）
            # 计算每个叶子的平均预测性能
            for leaf in Node._all_leaves:
                avg_pred_value_of_leaves[leaf] = np.mean(
                    [self._dataset[idx].predict_value for idx in Node._all_nodes[leaf].space.idxes]
                    )
            # 选取分数最高的k个叶子
            topk_leaves = sorted(avg_pred_value_of_leaves.items(),
                                 key=lambda x: x[1],
                                 reverse=True)[:self._num_init_samples]
            # 在选取的每个叶子中选择预测值最高的样本
            for leaf, _ in topk_leaves:
                idx = np.argmax(
                    [self._dataset[idx].predict_value for idx in Node._all_nodes[leaf].space.idxes]
                    )
                max_idxes.append(Node._all_nodes[leaf].space.idxes[idx])
            # 防止候选不足，随机选取叶子以及数据点补全
            if len(max_idxes) < self._num_init_samples:
                random_leaves = random.sample(list(Node._all_leaves),
                                              self._num_init_samples-len(max_idxes))
                random_idxes = []
                for leaf in random_leaves:
                    random_idxes.append(random.choice(Node._all_nodes[leaf].space.idxes))
                max_idxes += random_idxes
            # 最终候选数据点
            candidates = max_idxes

        else:  # 不使用预测数据作为伪标签，完全随机采样
            # 随机采叶子
            random_leaves = random.sample(list(Node._all_leaves),
                                          self._num_init_samples)
            random_idxes = []
            # 随机采数据点
            for leaf in random_leaves:
                random_idxes.append(random.choice(Node._all_nodes[leaf].space.idxes))
            candidates = random_idxes
        
        # TODO: 是否能够添加一个函数选项使得直接输出candidates，不进行下面操作？
        # 获取候选真实值
        observed_values = [self._dataset[idx].observed_value for idx in candidates]
        self._sampler.update_train(candidates, torch.vstack(observed_values))
        self._sampler.init_model(self._sampler.train_x, self._sampler.train_y)

        # 使用预测值进行伪反向传播
        for idx in self._dataset.all_idxes:
            leaf = self.find_node_by_data_idx(idx)
            alpha = 1 / (1 + len(leaf.space.idxes))
            self.find_node_by_data_idx(idx).pseudo_backprop(self._dataset[idx].predict_value.squeeze(),
                                                            alpha=alpha)

        # 使用真实值进行反向传播
        for candidate, value in zip(candidates, observed_values):
            node = self.find_node_by_data_idx(candidate)
            node.inc_incomplete_count()
            node.backprop(value.squeeze())

        return observed_values


    def real_exp_random(self,
                        expert_partition: Optional[Dict[str, List[List]]] = None,
                        order: Optional[List[str]] = None,
                        ):
        '''湿实验代码
        在无任何观测值的情况下，基于专家树 + 多样性路径选择，返回广泛分布的初始候选点。
        '''
        # 初始化子空间
        Subspace.init(self._dataset)
        # 初始化节点
        Node.init()
        # 创建根节点，包含所有数据点
        self._root = Node(space=Subspace(self._dataset.all_idxes),
                          leaf_size=self._leaf_size,
                          depth=0)

        # 构建树
        if expert_partition is None:  # 从上面构建的根节点出发使用默认策略构建新树
            Node.build_tree(self._root)
        else:
            Node.build_tree_by_expert(self._root,  # 从反应物分类和重要性排序构建新树
                                      expert_partition,
                                      order)
        print(f"The tree has {len(Node._all_nodes)} nodes and {len(Node._all_leaves)} leaves")
        
        # 随机生成 n_candidates 组候选叶子集合（每组包含 n_sample 个叶子）
        init_leaves = self.diversity_aware_random_sample(
            n_sample=self._num_init_samples,
            n_candidates=self._n_candidates,
            )
        # 在每个叶子里随机采集一个数据点
        candidates = []
        for leaf in init_leaves:
            idx = random.choice(leaf.space.idxes)
            candidates.append(idx)
        print(f'Designed experiments index: {candidates}')
        
        # 选取index对应的sample
        next_batch = [self._dataset[idx].category_value for idx in candidates]

        return next_batch

    def real_exp(self,
                 expert_partition: Dict[str, List[List]],
                 order: List[str],
                 mask: List[int],  # 已提交实验或被选中的数据点
                 kappa: float = 0.01,  # 用于计算节点UCB打分的kappa值
                 kappa_by_depth: bool = False,  # 是否根据节点深度调整kappa值
                 depth_kappa_mapping: Optional[Dict[str, float]] = None, # 若按深度调整kappa，则传入深度-对应kappa值映射
                 ei_init: float = 0.01,  # EI初始值（BO获得函数1）
                 ei_incre: float = 0.28,  # EI每轮增加值
                 ucb_init: float = 0.10,  # UCB初始值（BO获得函数2）
                 ucb_incre: float = 0.56,  # UCB每轮增加值
                 return_acq_values: bool = False,  # 是否返回acquisition values
                 ):
        '''湿实验代码
        在已有部分真实观测数据的基础上，执行一轮 MCTS 搜索 + 代理模型采样，推荐下一组实验候选（next_batch）
        '''
        # 初始化
        Subspace.init(self._dataset)
        Node.init()
        self._root = Node(space=Subspace(self._dataset.all_idxes),
                          leaf_size=self._leaf_size,
                          depth=0,
                          )
        # 基于专家知识建树
        Node.build_tree_by_expert(self._root,
                                  expert_partition,
                                  order,
                                  )

        print(f"The tree has {len(Node._all_nodes)} nodes and {len(Node._all_leaves)} leaves")
        
        # 所有已知实验结果的数据点
        observed_candidate = [x for x in self._dataset.all_idxes if self._dataset[x].is_observed]
        observed_values = [self._dataset[idx].observed_value for idx in observed_candidate]
        print(f"Observed_candidate: {observed_candidate}")
        print(f"Observed_values: {observed_values}")
        # 已知实验结果用于训练
        self._sampler.update_train(observed_candidate,torch.vstack(observed_values))
        self._sampler.init_model(self._sampler.train_x,self._sampler.train_y)
        print(f"Sampler trained with observed values")

        # 预测数据用于伪传播
        print(f"Start pseudo_backprop with pseudo data...")
        for idx in tqdm(self._dataset.all_idxes, desc="pseudo backprop"):
            leaf = self.find_node_by_data_idx(idx)
            alpha = 1 / (1 + len(leaf.space.idxes))
            self.find_node_by_data_idx(idx).pseudo_backprop(self._dataset[idx].predict_value.squeeze(),
                                                            alpha=alpha,
                                                            )
        print(f"Pseudo_backprop completed")

        # 真实数据用于传播
        print(f"Start backprop with observed data...")
        for idx in tqdm(observed_candidate, desc="real backprop"):
            self.find_node_by_data_idx(idx).inc_incomplete_count()
            self.find_node_by_data_idx(idx).backprop(self._dataset[idx].observed_value.squeeze())
        
        print("All backprop is completed")  # 基于已有实验和预测数据更新树完成

        print("Start designing")
        # 设定所有节点计算UCB探索参数时的kappa
        if kappa_by_depth:
            Node.set_all_kappa_by_depth(order, depth_kappa_mapping)
            print(f"kappa for exploration function calculation is set by depth")
        else:
            Node.set_all_kappa(kappa)
            print(f"kappa for exploration function calculation is set to {kappa}")

        # 选择需要搜索的节点
        search_nodes = self.sample_path(self._batch_size)
        candidate = []
        eta = ei_init  # 对应EI获得函数
        beta = ucb_init  # 对应UCB获得函数
        if return_acq_values:
            acq_values = []

        # 在每个节点中选取候选数据点
        for i, search_node in enumerate(search_nodes):
            eta = eta + ei_incre * i  # 每选一个点后增加获得函数数值
            if i >= self._batch_size // 2:  # 选择过半后改用UCB获得函数
                beta = beta + ucb_incre * (i - self._batch_size // 2)
            # 选择一个候选实验数据点
            if return_acq_values:
                next_candidate, next_acq_value = self._sampler.real_exp_pseudo_label_sample(
                    search_node.space.unobserved_idxes,
                    n_sample=1,
                    mask=mask,
                    cur_time=i,
                    eta=eta,
                    beta=beta,
                    n_times=self._batch_size,
                    return_acq_values=True,
                    )
                acq_values.append(next_acq_value)
            else:
                next_candidate = self._sampler.real_exp_pseudo_label_sample(
                    search_node.space.unobserved_idxes,
                    n_sample=1,
                    mask=mask,
                    cur_time=i,
                    eta=eta,
                    beta=beta,
                    n_times=self._batch_size,
                    return_acq_values=False,
                    )
            mask.extend(next_candidate)  # 将选中数据加入mask，避免重复选择
            candidate.extend(next_candidate)  # 选中数据加入总列表

        print("Design is completed")
        next_batch = [self._dataset[idx].category_value for idx in candidate]

        if return_acq_values:
            return next_batch, acq_values
        else:
            return next_batch
    
    def sample_path(self, n_sample: int):
        '''使用 UCB（Upper Confidence Bound）策略，从根节点开始，
        沿着“置信上界得分最高”的子节点向下遍历，选择 n_sample 个值得探索的叶子节点。
        '''
        nice_leaves = []
        while len(nice_leaves) < n_sample:
            # 从根节点开始
            search_node = self._root
            while not search_node.is_leaf:
                for node in search_node.children:
                    logger.debug(f"node {node.id} cb: {node.cb}")
                # 进入得分最高子节点
                search_node = max(search_node.children, key=lambda x: x.cb)
                logger.debug(f"selected node {search_node.id} cb: {search_node.cb}")
            # 到达子节点后跳出循环并选择之
            logger.debug(f"leaf node found {search_node.id} cb: {search_node.cb}")
            nice_leaves.append(search_node)
            search_node.inc_incomplete_count()
        return nice_leaves

    def _pick_one_leaf_with_kappa(
        self,
        kappa: float,
        forbidden_leaf_ids: Optional[Set[int]] = None,
        ) -> Optional[Node]:
        """
        在给定 kappa 下，选择一个“可用叶子”。
        
        设计目标：
        1) 优先探索 cb 更高的分支（贪心）
        2) 当高分分支不可用时能自动回退到次优分支（鲁棒）
        3) 支持叶子去重（forbidden_leaf_ids）
        
        为什么是 BFS + 贪心扩展：
        - BFS 使用队列保存“待展开节点”
        - 每次展开某节点时，将其 children 按 cb 从高到低入队
        - 这样高 cb 分支会更早被处理（贪心优先）
        - 但不是只走一条单路径，因此如果首选分支不可用，仍可回退到后续候选分支
        
        返回：
        - 找到合法叶子：返回该叶子
        - 找不到：返回 None
        """
        if self._root is None:
            raise RuntimeError("_pick_one_leaf_with_kappa called before tree initialization.")

        forbidden_leaf_ids = forbidden_leaf_ids or set()

        # 本次“单点选叶子”使用指定 kappa
        Node.set_all_kappa(kappa)

        # 队列：保存待探索节点（BFS框架）
        q = deque([self._root])

        # 防御：避免异常树结构导致重复访问
        visited: Set[int] = set()

        while q:
            node = q.popleft()

            if node.id in visited:
                continue
            visited.add(node.id)

            # 若到达叶子，检查是否合法
            if node.is_leaf:
                # 约束1：叶子不能重复（同一 batch 内）
                if node.id in forbidden_leaf_ids:
                    continue
                # 约束2：叶子必须还有未观测样本
                if len(node.space.unobserved_idxes) == 0:
                    continue
                # 满足条件，立即返回
                return node

            # 非叶子：对子节点按 cb 从高到低排序后入队
            # 含义：高 cb 子节点更早被展开（贪心优先）
            # 处理 NaN：将其视为极小值，避免排序异常
            children_sorted = sorted(
                node.children,
                key=lambda c: c.cb if not np.isnan(c.cb) else -float("inf"),
                reverse=True,
            )

            for child in children_sorted:
                # 轻量剪枝：如果这个子节点已经没有未观测样本，跳过
                # （在当前实现里 unobserved_idxes 会遍历子树索引，可作为可行性判定）
                if len(child.space.unobserved_idxes) == 0:
                    continue
                q.append(child)

        # 所有可达分支都不满足约束
        return None

    def sample_path_with_kappas(self, kappas: List[float]) -> List[Node]:
        """
        每个 batch 位置使用不同 kappa 选叶子。
        
        规则：
        - 叶子不重复（forbidden_leaf_ids）
        - 每次选中后调用 inc_incomplete_count，与 sample_path 行为一致
        - 若某个 kappa 找不到可用叶子，则跳过该 slot
        """
        selected_leaves: List[Node] = []
        selected_leaf_ids: Set[int] = set()

        for k in kappas:
            leaf = self._pick_one_leaf_with_kappa(k, forbidden_leaf_ids=selected_leaf_ids)
            if leaf is None:
                logger.warning(f"No available leaf found for kappa={k}, skip this slot.")
                continue

            logger.debug(f"leaf node found {leaf.id} cb: {leaf.cb} for kappa={k}")
            selected_leaves.append(leaf)
            selected_leaf_ids.add(leaf.id)

            leaf.inc_incomplete_count()

        return selected_leaves

    def random_sample_path(self, n_sample:int):
        """
        在树中完全随机地选择n_sample条路径，直到叶子节点。
        这主要用于MCTS的初始阶段，以保证探索的广度。
        """
        nice_leaves = []
        while len(nice_leaves) < n_sample:
            search_node = self._root
            while not search_node.is_leaf:
                search_node = random.choice(search_node.children)
            nice_leaves.append(search_node)
            search_node.inc_incomplete_count()
        return nice_leaves

    def _get_one_random_path(self):
        """
        辅助函数：从根节点随机走到一个叶子节点并返回，不更新任何计数。
        返回叶子节点
        """
        search_node = self._root
        while not search_node.is_leaf:
            if not search_node.children:
                break
            search_node = random.choice(search_node.children)
        return search_node

    def diversity_aware_random_sample(self,
                                      n_sample: int,
                                      n_candidates: int):
        """
        实现一个带有“多样性”偏好的随机采样。
        它会生成n_candidates个候选路径集，然后根据一个评价函数选择最优的一个。

        Args:
            n_sample (int): 最终需要返回的路径数量 (例如: 5)。
            n_candidates (int): 生成的候选集数量 (例如: 10)。
        """
        # 定义价值函数
        # TODO: 不一定要线性
        step = (0.8 - 0.2) / (self.variable_nums - 1) if self.variable_nums > 1 else 0
        level_weights = {i: round(0.8 - i * step, 2) for i in range(self.variable_nums)}

        candidate_sets = []
        # 生成 n_candidates 个候选集，每个集包含 n_sample 条随机路径（叶子节点）
        for _ in range(n_candidates):
            current_set = [self._get_one_random_path() for _ in range(n_sample)]
            candidate_sets.append(current_set)

        best_set = None
        max_score = -1

        # 遍历所有候选集，为它们打分
        for path_set in candidate_sets:
            # 使用 set 来自动处理路径上的重复节点
            all_nodes_in_paths = set()
            for leaf_node in path_set:
                curr = leaf_node
                while curr is not None:  # 从下至上添加路径上的所有节点
                    all_nodes_in_paths.add(curr)
                    curr = curr.parent

            # 按照层级统计不同节点的数量
            nodes_per_level = defaultdict(int)
            for node in all_nodes_in_paths:
                nodes_per_level[node._depth] += 1
            
            # 计算当前候选集的总分
            current_score = 0
            for level, count in nodes_per_level.items():
                # 使用 .get(level, 0.01) 来处理超出预设权重的深层节点
                weight = level_weights.get(level, 0.01) 
                current_score += count * weight
            
            # 更新最优选择
            if current_score > max_score:
                max_score = current_score
                best_set = path_set
        
        # 只对最终选出的最优路径集进行计数更新
        # if best_set:
        #     for leaf in best_set:
        #         leaf.inc_incomplete_count()
        
        return best_set
            
    def diverse_sample(self, pseudo_label: bool, is_tree_exist: bool, num_diverse_sample: int):
        '''干实验代码
        在使用先验初始化树后，正式开始BO之前，使用多样性采样选取一批数据点
        '''
        candidates = []

        if pseudo_label and is_tree_exist:  # 使用伪数据+树
            logger.info('use pseudo diverse sample')
            avg_pred_value_of_leaves: Dict[int,float] = {}

            for leaf in Node._all_leaves:  # 给每个叶节点基于伪数据打分，并从高到低选择num_diverse_sample个叶节点
                avg_pred_value_of_leaves[leaf] = np.mean(
                    [self._dataset[idx].predict_value for idx in Node._all_nodes[leaf].space.idxes]
                )
            topk_leaves = sorted(avg_pred_value_of_leaves.items(), key=lambda x: x[1], reverse=True)[:num_diverse_sample]
            logger.info(f'topk_leaves idxes are {topk_leaves}')

            for leaf, _ in topk_leaves:  # 在每个选中的叶节点中选择伪数据值最高的条件
                idx = np.argmax([self._dataset[idx].predict_value for idx in Node._all_nodes[leaf].space.idxes])
                candidates.append(Node._all_nodes[leaf].space.idxes[idx])

            if len(candidates) < num_diverse_sample:  # 叶节点总数小于num_diverse_sample，其余使用随机采样补足
                needed_num = num_diverse_sample-len(candidates)
                logger.info(f'need {needed_num} more candidates')
                to_sample = list(Node._all_leaves)
                random_leaves = []
                if len(to_sample) < needed_num:  # 总叶节点数小于所需总数
                    while len(random_leaves) < needed_num:  # 已选节点数
                        random_leaves += to_sample
                    random_leaves = random.sample(random_leaves, needed_num)
                else:
                    random_leaves = random.sample(to_sample, needed_num)
                random_idxes = []
                for leaf in random_leaves:
                    random_idxes.append(random.choice(Node._all_nodes[leaf].space.idxes))
                candidates += random_idxes

        elif not is_tree_exist:  # 无树结构，直接随机采样
            logger.info('no tree structure, use random sample')
            random_idxes = torch.randperm(len(range(len(self._dataset.all_idxes))))[:num_diverse_sample].tolist()
            candidates = [self._dataset.all_idxes[idx] for idx in random_idxes]

        else:  # 不使用伪数据
            logger.info('use non-pseudo diverse sample')
            diverse_leaves = self.diversity_aware_random_sample(  # 多样性采样num_diverse_sample个叶节点
                n_sample=num_diverse_sample, n_candidates=self._n_candidates)

            for leaf in diverse_leaves:  # 在叶节点中随机采条件
                idx = random.choice(leaf.space.idxes)
                candidates.append(idx)

        # 使用上述采到的数据点更新BOsampler
        observed_values = [self._dataset[idx].observed_value for idx in candidates]
        self._sampler.update_train(candidates, torch.vstack(observed_values))
        self._sampler.init_model(self._sampler.train_x, self._sampler.train_y)

        # 使用上述采到的数据点更新树                        
        for candidate, value in zip(candidates, observed_values):
            node = self.find_node_by_data_idx(candidate)
            logger.info(f'selected datapoint {candidate} in node {node.id} with obs value {value}')
            node.inc_incomplete_count()
            node.backprop(value.squeeze())
        
        logger.info(f'number of diverse sampled points: {len(observed_values)}')
        return observed_values

    def search(
        self,
        pseudo_label: bool = False,
        iteration_index: int = 1,
        per_batch_kappas: Optional[List[float]] = None,
        return_indices: bool = False,
    ):
        '''干实验代码
        search with tree
        注意：为保持与历史实验可比性，本函数不做candidate去重与自动补全。
        '''
        # 1) 选择搜索叶子：支持每个 batch 位点不同 kappa
        if per_batch_kappas is not None and len(per_batch_kappas) > 0:
            if len(per_batch_kappas) < self._batch_size:
                logger.warning(
                    f"len(per_batch_kappas)={len(per_batch_kappas)} < batch_size={self._batch_size}, "
                    f"pad with last kappa={per_batch_kappas[-1]}"
                )
                per_batch_kappas = per_batch_kappas + [per_batch_kappas[-1]] * (self._batch_size - len(per_batch_kappas))
            elif len(per_batch_kappas) > self._batch_size:
                logger.warning(
                    f"len(per_batch_kappas)={len(per_batch_kappas)} > batch_size={self._batch_size}, truncate."
                )
                per_batch_kappas = per_batch_kappas[:self._batch_size]

            logger.info(f"Using per-batch kappas: {per_batch_kappas}")
            search_nodes = self.sample_path_with_kappas(per_batch_kappas)
            if len(search_nodes) == 0:
                logger.warning("sample_path_with_kappas returns empty, fallback to default sample_path.")
                search_nodes = self.sample_path(self._batch_size)
        else:
            search_nodes = self.sample_path(self._batch_size)

        # 2) 从每个叶子里按原有逻辑选候选（不去重）
        candidate = []
        observed_values = []
        for search_node in search_nodes:
            # 统计搜索次数
            self._leaf_visit_counts[search_node.id] += 1

            if pseudo_label:
                one = self._sampler.pseudo_label_sample(
                    search_node.space.unobserved_idxes, n_sample=1
                )[0]
            else:
                one = self._sampler.sample(
                    search_node.space.unobserved_idxes, n_sample=1
                )[0]
            candidate.append(one)
            observed_values.append(self._dataset[one].observed_value)

        # observed_values = [self._dataset[idx].observed_value for idx in candidate]

        # 3) 按 observed value 更新节点（沿用原逻辑）
        for idx, v in zip(candidate, observed_values):
            node = self.find_node_by_data_idx(idx)
            logger.info(f'selected datapoint {idx} in node {node.id} with obs value {v}')
            node.inc_incomplete_count()
            node.backprop(v.squeeze())

        if return_indices:
            return observed_values, candidate
        return observed_values

    def get_search_statistics(self, order: List[str]) -> Dict[str, int]:
        """
        Retrieves the historical search counts for each combination/group (leaf node).

        Args:
            order (List[str]): The order of components used to build the tree,
                               needed to generate readable combination names.

        Returns:
            Dict[str, int]: A dictionary where keys are combination names and
                            values are their total visit counts.
        """
        # If no searches have happened, return empty dict
        if not self._leaf_visit_counts:
            return {}

        all_leaves = Node.get_all_leaves()
        # Create a mapping from node ID to its human-readable combination name
        id_to_combo_name = {}
        for leaf in all_leaves:
            path = leaf.path_from_root()
            combo_parts = []
            for node in path[1:]: # Skip root node
                parent = node.parent
                component_name = order[node._depth - 1]
                child_index = parent._children.index(node.id)
                combo_parts.append(f"{component_name}_Group{child_index}")
            id_to_combo_name[leaf.id] = " | ".join(combo_parts)

        # Build the final statistics dictionary
        statistics = {}
        for node_id, count in self._leaf_visit_counts.items():
            # It's possible a node was visited but is no longer a leaf (if the tree can be re-split).
            # We only report stats for current leaves.
            if node_id in id_to_combo_name:
                combo_name = id_to_combo_name[node_id]
                statistics[combo_name] = count
        
        return statistics

    def search_weighted(self):
        search_node = self._root
        leaf_nodes = Node.get_all_leaves()
        cbs = np.array([node.cb for node in leaf_nodes])
        cbs[cbs == float('inf')] = np.mean(cbs[cbs != float('inf')])
        tau = 0.7
        cbs_logits = torch.softmax(torch.tensor(cbs)/tau,dim=0)
        print(cbs_logits)
        #根据idx，给每个idx一个权重
        search_space_idx = search_node.space.unobserved_idxes
        search_space_weight = torch.zeros(len(search_space_idx))
        for i, idx in enumerate(search_space_idx):
            for node, cb in zip(leaf_nodes,cbs_logits):
                if idx in node.space.idxes:
                    search_space_weight[i] = cb
                    break
        cb_top5_node_id = [node.id for node in sorted(leaf_nodes,key=lambda x: x.cb,reverse=True)[:5]]
        
        candidate = self._sampler.pseudo_label_sample(search_space_idx, n_sample=self._batch_size,weight=search_space_weight)
        candidate_node_id = [self.find_node_by_data_idx(idx).id for idx in candidate]
        print(f"    ucb_top5_node_id: {cb_top5_node_id}")
        print(f"    ucb_top5: {[cb for cb in sorted(cbs_logits ,reverse=True)[:5]]}")
        print(f"    candidate_node_id: {candidate_node_id}")
        print(f"    candidate: {candidate}")
        observed_values = [self._dataset[idx].observed_value for idx in candidate]
        # 按照observed value更新节点
        for node, v in zip(candidate, observed_values):
            self.find_node_by_data_idx(node).backprop(v.squeeze())
            
        return observed_values
    
    def baseline_search(self, pseudo_label: bool = False, return_indices: bool = False):
        
        search_node = self._root
        # print(f"    unobserved length: {len(search_node.space.unobserved_idxes)}")
        if pseudo_label:
            logger.debug("start pseudo_label_sample")
            candidate = self._sampler.pseudo_label_sample(search_node.space.unobserved_idxes, n_sample=self._batch_size)
            logger.debug("finish pseudo_label_sample")
        else:
            candidate = self._sampler.sample(search_node.space.unobserved_idxes, n_sample=self._batch_size)
        observed_values = [self._dataset[idx].observed_value for idx in candidate]
        candidate_node_id = [self.find_node_by_data_idx(idx).id for idx in candidate]
        # print(f"    candidate_node_id: {candidate_node_id}")
        if return_indices:
            return observed_values, candidate
        return observed_values
    
    def get_max_idx_info(self):
        max_value = 0.0
        max_idx = -1
        for idx in self._dataset.all_idxes:
            value = self._dataset[idx].observed_value
            if value > max_value:
                max_value = value
                max_idx = idx

        max_node = self.find_node_by_data_idx(max_idx)
        if max_node.is_leaf:
            logger.info(f'max node {max_node.id} is leaf node')
        else:
            logger.warning(f'max node {max_node.id} is not leaf node')
        logger.info(f'max obs value datapoint {max_idx} in node {max_node.id} with obs value {max_value}')
        logger.warning('No further calculations should be done with this tree! Since all the obs value bave been revealed!')
