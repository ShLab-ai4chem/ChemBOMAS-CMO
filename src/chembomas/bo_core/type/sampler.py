from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import torch
from ..data.bodata import BoData
class Sampler(ABC):
    def __init__(self, dataset: BoData,init_x,init_y):
        self.dataset = dataset
        
    @abstractmethod
    def sample(self, n_sample: int)->List[int]:
        raise NotImplementedError()
