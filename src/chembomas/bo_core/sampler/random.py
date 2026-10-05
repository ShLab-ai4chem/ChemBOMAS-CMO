from ..type.sampler import Sampler
from ..data.bodata import BoData
from typing import List
import torch
class RandomSampler(Sampler):
    def __init__(self, dataset: BoData,init_x,init_y):
        super().__init__(dataset)
        
    def sample(self, sub_space_idx:List[int], n_sample: int)->List[int]:
        return torch.randperm(len(sub_space_idx))[:n_sample].tolist()
    def update_train(self, idx:List[int], value:List[float])->None:
        pass
    def init_model(self):
        pass
