from typing import List, Dict, Optional
from ..type.clustering import Cluster
import torch
from sklearn.cluster import KMeans

class KMeansCluster(Cluster):
    def __init__(self, n_clusters: int):
        super().__init__(n_clusters)
        
    def cluster_data(self, data: torch.Tensor)->None:
        
        self.cluster = KMeans(n_clusters=self.n_clusters).fit(data)
        
  
