import sklearn.cluster
import torch
import sklearn
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Optional,Tuple
from abc import ABC, abstractmethod
from ..data.bodata import BoData, Sample
from torchtyping import TensorType, patch_typeguard
class Cluster(ABC):
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.cluster = None
        
    
    @abstractmethod
    def cluster_data(self, data:TensorType['num','feat'])->None:
        raise NotImplementedError()
        
    
    def __call__(self,data:TensorType['num','feat'],space_idx: List[int])->List[Tuple[int,List[int]]]:
        self.cluster_data(data)
        cluster_dict = self.cluster_label_to_index(space_idx)
        # convert dict to list
        cluster_list = []
        for key in cluster_dict:
            cluster_list.append((key, cluster_dict[key]))
        return cluster_list
    
    def cluster_label_to_index(self, space_idx:List[int])->List[int]:
        cluster_label = self.cluster.labels_
        cluster_dict = {}
        for i, label in enumerate(cluster_label):
            if label not in cluster_dict:
                cluster_dict[label] = []
            cluster_dict[label].append(space_idx[i])
        return cluster_dict
