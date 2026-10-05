import torch
import torch.nn.functional as F
import pandas as pd
import os

def get_sim_matrix(data_maps, use_extra=None, eps=1e-8):
    """
    计算可微的相似度矩阵
    data_maps: {name: embedding_tensor} 其中embedding_tensor需要梯度
    use_extra: 特殊嵌入的名称
    """
    names = list(data_maps.keys())
    
    def _flatten(emb: torch.Tensor) -> torch.Tensor:
        if emb.dim() == 2 and emb.size(0) == 1:
            return emb.squeeze(0)
        elif emb.dim() == 1:
            return emb
        else:
            raise ValueError("Embedding must be 1-D or [1, d].")
    
    # 提取所有嵌入并保持梯度
    embeddings = [data_maps[k] for k in names]
    
    if use_extra and use_extra in names:
        # 分离常规嵌入和特殊嵌入
        regular_names = [k for k in names if k != use_extra]
        regular_embs = torch.stack([_flatten(data_maps[k]) for k in regular_names])
        extra_emb = _flatten(data_maps[use_extra])  # [d]
        
        # L2 归一化
        regular_embs_norm = F.normalize(regular_embs + eps, p=2, dim=1)        # [n, d]
        extra_emb_norm    = F.normalize(extra_emb + eps,  p=2, dim=0)          # [d]

        # 相似度计算
        cos_sim   = torch.mm(regular_embs_norm, regular_embs_norm.t())   # [n, n]
        extra_sim = torch.mv(regular_embs_norm, extra_emb_norm)          # [n]

        return cos_sim, extra_sim
    
    else:
        all_embs = torch.stack([_flatten(data_maps[k]) for k in names])  # [n, d]
        all_embs_norm = F.normalize(all_embs + eps, p=2, dim=1)                # [n, d]
        cos_sim = torch.mm(all_embs_norm, all_embs_norm.t())             # [n, n]

        return cos_sim, None