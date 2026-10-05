import torch
from typing import List, Tuple
import random
import numpy as np

def get_device():
    device = "cpu"
    # if torch.backends.mps.is_available():
        # device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    return device
   

    
def get_init_points(data, init_points: int) -> Tuple[List[torch.Tensor], List[float]]:
    search_space = data.get_search_space()
    init_idx = []
    
    while len(init_idx) < init_points:
        idx = random.sample(range(len(search_space)), 1)[0]

        if idx not in data.evaluated_idx:
        
            init_idx.append(idx)
    
    train_x = data.get_search_space()[init_idx]
    train_y = data.yield_query_from_index(init_idx)
    assert train_x.shape[0] == init_points
    return train_x, train_y, init_idx
def correct_indices(idxes:torch.Tensor,mask_tensor:torch.Tensor)->torch.Tensor:
    ori_idx = torch.arange(mask_tensor.shape[0])
    ori_mask_dict = {}
    _idx = 0
    for i in range(ori_idx.shape[0]):
        if not mask_tensor[i]:
            
            ori_mask_dict[_idx] = i
            _idx += 1
    
    if idxes.dim() == 0:
        corrected_idx = torch.tensor(ori_mask_dict[idxes.item()])
        return corrected_idx
        
    corrected_idx = torch.tensor([ori_mask_dict[idx.item()] for idx in idxes if idx.item() in ori_mask_dict])
    
    return corrected_idx

def nexp(i):
    return np.ceil(np.exp((20-i)*0.0489)*50)