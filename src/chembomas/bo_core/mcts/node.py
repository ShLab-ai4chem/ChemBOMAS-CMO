import numpy as np
import random
from typing import List, Dict, Tuple, Optional,ClassVar,Any,Set
from itertools import count
from collections import deque
from ..data.bodata import BoData
import torch
from ..cluster.kmeans import KMeansCluster
from loguru import logger

class Subspace:
    _shared_dataset:ClassVar[BoData] = None
    @classmethod
    def init(cls,dataset:BoData)->None:
        cls._shared_dataset = dataset
        
    
    def __init__(self, idxes: List[int], value: np.float32 = 0):
        self._idxes = idxes
        self._sum = 0  # Track sum
        self._count = 0    # Track count
        self._incomplete_count = 0 
        self.avg_value = value if value is not None else 0
    
    
    
    @property
    def unobserved_idxes(self)->List[int]:
        return [idx for idx in self.idxes if not Subspace._shared_dataset[idx].is_observed]
    
        
    @property
    def feats(self)->torch.Tensor:
        return torch.vstack([Subspace._shared_dataset[idx].feat for idx in self.idxes])
    @property
    def idxes(self)->np.ndarray:
        return self._idxes
    
    @property
    def count(self)->int:
        return self._count
    @property
    def sum(self)->np.float32:
        return self._sum
    @property
    def incomplete_count(self)->int:
        return self._incomplete_count
    @count.setter
    def count(self,value:int)->None:
        self._count = value
    @sum.setter
    def sum(self,value:np.float32)->None:
        self._sum = value
    @incomplete_count.setter
    def incomplete_count(self,value:int)->None:
        self._incomplete_count = value
    def __len__(self):
        return len(self.idxes)
    
    def get_additional_feat(self, key:str)->torch.Tensor:

        if key not in Subspace._shared_dataset[0].additional_feat:
            return None
        return torch.vstack([Subspace._shared_dataset[idx].additional_feat[key] for idx in self.idxes])
    
    
class Node:
    _next_id:ClassVar = count()
    _all_nodes: ClassVar[Dict[int, Any]] = {}
    _all_leaves: ClassVar[Set[int]] = set()
    
    
    def init():
        Node._next_id = count()
        Node._all_nodes.clear()
        Node._all_leaves.clear()
        
        
    @staticmethod
    def get_node(id:int)->Optional['Node']:
        return Node._all_nodes.get(id)
    
    @staticmethod
    def delete_node(id:int)->None:
        Node._all_nodes.pop(id)
    
    @staticmethod
    def get_all_leaves()->List['Node']:
        return [Node.get_node(leaf) for leaf in Node._all_leaves]
    
    @staticmethod
    def cleanup():
        Node._all_nodes.clear()
        Node._all_leaves.clear()
        
    
    @staticmethod
    def build_tree(root:'Node') -> Any:
        assert Subspace._shared_dataset is not None
        queue = deque()
        queue.append(root)
        # 首先将树析构，清空所有节点，保证树的唯一性
        while len(queue) > 0:
            node = queue.popleft()
            for child in node.children:
                if child.is_leaf:
                    Node._all_leaves.remove(child.id)
                else:
                    queue.append(child)
                del Node._all_nodes[child.id]
            node.clear_children()
        # 重新构建树
        queue = deque()
        queue.append(root)
        while len(queue) > 0:
            node = queue.popleft()
            Node._all_nodes[node.id] = node
            children = node.split()
            if len(children) == 0:
                Node._all_leaves.add(node.id)
            else:
                for child in children:
                    queue.append(child)
                    
    @staticmethod
    def build_tree_by_expert(root:'Node',
                             expert_partition: Dict[str, List[List]],  # 反应物分类
                             order: List[str],  # 反应物重要性排序
                             ) -> Any:
        """Build the tree by external expert partition
            order: the order of the partition,e.g. ['Ligand','Base','Solvent']
            expert_partition: the partition of the search space by expert
            e.g.
            {
                'Ligand':[['Ligand1','Ligand2'],['Ligand3','Ligand4']],
                'Base':[['Base1','Base2'],['Base3','Base4']],
                'Solvent':[['Solvent1','Solvent2'],['Solvent3','Solvent4']]
            }
        """
        queue = deque() # 创建一个双端队列，用于广度优先遍历树
        queue.append(root) # 将根节点加入队列
        # print(f'all nodes: {Node._all_nodes}')
        # 首先将树析构，清空所有节点，保证树的唯一性
        while len(queue) > 0:
            node = queue.popleft()
            for child in node.children:
                if child.is_leaf:
                    Node._all_leaves.remove(child.id) #?
                else:
                    queue.append(child)
                del Node._all_nodes[child.id]  #?
                # print(f'all nodes afer del: {Node._all_nodes}')
            node.clear_children()
        # 重新构建树
        queue = deque()
        depth = -1
        queue.append((root,0))
        while len(queue) > 0:
            node,depth = queue.popleft()
            Node._all_nodes[node.id] = node
            # print(f'expert_partition_order_depth: {expert_partition[order[depth]]}\norder_depth: {order[depth]}')

            sub_space_idxes = node.generate_sub_space_idxes_by_expert_partition(expert_partition[order[depth]],order[depth])
            # print(f"node {node.id} at depth {depth} have {len(sub_space_idxes)} sub space")
            # for _,indexes in sub_space_idxes:
            #    print(f"    sub space have {len(indexes)} points")
                
            
            children = node.split(sub_space_idxes)
            non_empty_children = [child for child in children if len(child.space.idxes) > 0]

            # 打印非空子节点的信息
            # for child in non_empty_children:
            #     print(f"    child {child.id} have {len(child.space.idxes)} points")

            # 更新 children 列表
            children[:] = non_empty_children

            if depth == len(order)-1:
                for child in children:
                    Node._all_leaves.add(child.id)
            else:
                for child in children:
                    queue.append((child,depth+1))

    @staticmethod
    def set_all_kappa(kappa: float) -> None:
        for node_id in Node._all_nodes:
            Node.get_node(node_id).kappa = kappa

    @staticmethod
    def set_all_kappa_by_depth(
        order: List[str], 
        depth_kappa_mapping: Dict[str, float]
    ) -> None:
        """
        根据节点层级设置不同的kappa值
        
        Args:
            order: 变量顺序列表，例如 ["Catalyst", "Aux_catalyst", "Acid", "Additive", "Solvent", "Reaction Temp&Time"]
            depth_kappa_mapping: 层级到kappa值的映射，例如 {"Catalyst": 0.05, "Aux_catalyst": 0.05, ...}
        """
        for node_id in Node._all_nodes:
            node = Node.get_node(node_id)
            if node._depth == 0:  # 根节点
                # 根节点的kappa值不影响算法，可以设置为任意值
                node.kappa = 1.0  # 或者设置为默认值
            elif node._depth - 1 < len(order):  # 确保深度在有效范围内
                layer_name = order[node._depth - 1]  # depth-1 因为根节点是depth=0
                node.kappa = depth_kappa_mapping.get(layer_name, 1.0)  # 默认值为1.0
            else:
                node.kappa = 1.0  # 超出预定义层级的节点使用默认值


    def __init__(self,
                 space: str,
                 leaf_size: int,
                 depth: int,
                 parent: Optional[int] = None,
                 kappa: float = 1.0,
                 ) -> None:
        self._children:List[int] = []
        self._leaf_size = leaf_size
        self._space: Subspace = space
        self._parent = parent
        self._id = next(Node._next_id)
        self._cb = float('NaN')
        self._depth = depth
        self.cluster = KMeansCluster(n_clusters=2)
        self.kappa = kappa
        Node._all_nodes[self._id] = self
        

    
    @property
    def id(self)->int:
        return self._id
    
    @property
    def parent(self)->Optional[int]:
        return Node._all_nodes[self._parent] if self._parent is not None else None
    
    @parent.setter
    def parent(self,value:int)->None:
        self._parent = value
        
    @property
    def is_root(self)->bool:
        return self._parent is None
    
    @property
    def is_leaf(self)->bool:
        return len(self._children) == 0

    @property
    def space(self)->Subspace:
        return self._space
    
    

    @property
    def num_leaves(self)->int:
        if self.is_leaf:
            return 1
        return sum(Node._all_nodes[child_id].num_leaves for child_id in self._children)
    @property
    def splittable(self)->bool:
        return len(self.space) > self._leaf_size and self._depth < 5
    
    @property
    def cb(self)->float:
        if self.is_root:
            #return np.float64('inf')
            return 0
        t = self.space.count
        v = self.space.sum
        p_t = self.parent.space.count
        ut = self.space.incomplete_count
        p_ut = self.parent.space.incomplete_count
        if len(self.space.unobserved_idxes) == 0:
            # print(f"Node {self.id} has no unobserved idxes, Node space idxes: {self.space.idxes}")
            return -1
        if t == 0:
            return self.parent.cb
        # print(t,v,p_t,p_ut,ut)
        # kappa = 0.20
        logger.debug(f'kappa: {self.kappa}')
        # print(f'exploitation:{v/(100*t)}\nexploartion:{kappa*np.sqrt(2*np.sqrt(p_t+p_ut)/(t+ut))}')
        logger.debug(f'exploitation:{v/(100*t)}\nexploartion:{self.kappa*np.sqrt(2*np.sqrt(p_t+p_ut)/(t+ut))}')
        # return v/(100*t) + kappa*np.sqrt(2*np.sqrt(p_t+p_ut)/(t+ut))
        return v/(100*t) + self.kappa*np.sqrt(2*np.sqrt(p_t+p_ut)/(t+ut))
    
    
    @property
    def children(self)-> List['Node']:
        return [Node._all_nodes[n] for n in self._children]
    
    def generate_sub_space_idxes_by_expert_partition(self,expert_partition:List[List[str]],name:str)->List[Tuple[int,List[int]]]:
        sub_space_idxes:List[Tuple['label','indexes']] = []
        for label, partition in enumerate(expert_partition):
            # print(f"label: {label}, partition: {partition}")
            indexes = []
            # print(f"sub space idxes: {self.space.idxes}")
            for idx in self.space.idxes:
                # print(f"idx: {idx}, category_value: {Subspace._shared_dataset[idx].category_value[name]}")
                if Subspace._shared_dataset[idx].category_value[name] in partition:
                    indexes.append(idx)
            sub_space_idxes.append((label,indexes))
        return sub_space_idxes
            
            
        
    
    def split(self,sub_space_idxes:Optional[List[Tuple[int,List[int]]]]=None)->None:
        """Split the node into sub_space, generate children nodes

        Args:
            sub_space (List[Tuple[int,int]]): len(sub_space) , the number of sub_space,
            each element is a tuple of (index, space), where index is the index of the sub_space, 
            space is the index of the search space    
        """
        self._children.clear()
        
        children = []
        if not self.splittable and sub_space_idxes is None:
            return children
        if sub_space_idxes is None:
            
            feats = self.space.get_additional_feat('embedding') 
            feats = feats if feats is not None else self.space.feats
            sub_space_idxes = self.cluster(self.space.feats,self.space.idxes)
        for label, indexes in sub_space_idxes:
            
            subspace = Subspace(indexes)
            child = Node(parent=self.id,space=subspace,leaf_size=self._leaf_size,
                            depth=self._depth+1)
            self._children.append(child.id)
            children.append(child)
        return children
    
            
    def path_from_root(self)->List['Node']:
        nodes = [self]
        node = self.parent
        while node is not None:
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        return nodes
            
    def clear_children(self)->None:
        self._children.clear()
        
    def backprop(self, value:np.float32,transform:callable=None)->None:
        if transform is not None:
            value = transform(value)
        node = self
        while node is not None:
            node.space.sum += value
            node.space.count += 1
            node.space.incomplete_count -= 1
            node = node.parent
            
    def pseudo_backprop(self, value:np.float32,alpha:np.float32=0.1,transform:callable=None)->None:
        if transform is not None:
            value = transform(value)
        node = self
        while node is not None:
            node.space.sum += value * alpha
            node.space.count += 1 * alpha
            node = node.parent
            
    def inc_incomplete_count(self)->None:
        node = self
        while node is not None:
            node.space.incomplete_count += 1
            node = node.parent
