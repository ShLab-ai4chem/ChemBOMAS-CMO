import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json, os
import torch
from collections import defaultdict
import torch.nn.functional as F

def get_format_output(names, labels, file_name):
    temp = {}
    output = {}
    for name, label in zip(names, labels):
        label = int(label)
        if label not in temp:
            temp[label] = []
        temp[label].append(name)
    output[file_name.split("_")[-1]] = list(temp.values())
    return output

def cluster_similarity_virtual_point(
        sims, 
        names, 
        file_name, 
        data_maps, 
        save_path=None, 
        extra_sims=None, 
        extra_name = "nothing", 
        return_raw_distance_list=False,
        use_angle_loss=False
    ):

    use_extra_sims = ""
    if (extra_sims is not None) and (extra_name in names):
        use_extra_sims = "Ex"
        names.remove(extra_name)

    n = sims.shape[0]

    with torch.no_grad():
        # Convert similarities to numpy array
        sims_np = sims.detach().float().cpu().numpy()
        
        # Handle NaN values: Replace NaNs with 0 (indicating no similarity)
        sims_np = np.nan_to_num(sims_np, nan=0.0)
        
        # Ensure non-negative values (set negatives to 0)
        sims_np[sims_np < 0] = 0.0
        
        # Set diagonal to 0 (no self-connections)
        np.fill_diagonal(sims_np, 0.0)

        model = SpectralClustering(
            n_clusters=3,
            affinity="precomputed",
            random_state=42)
        labels = model.fit_predict(sims_np)
        labels = torch.as_tensor(labels, dtype=torch.long)

    if use_extra_sims == "Ex":
        cluster_sims = defaultdict(list)
        for lbl, s in zip(labels.cpu().tolist(), extra_sims.detach()):
            cluster_sims[lbl].append(s.item())
        best_cluster = max(cluster_sims,
                           key=lambda k: np.mean(cluster_sims[k]))
        labels = torch.cat([labels,
                            torch.tensor([best_cluster],
                                         dtype=torch.long)])
        names = names + [extra_name]

    # 2. 计算每个簇的中心点和平均距离
    unique_labels = labels.unique()

    total_loss = []  # 初始化总距离列表
    total_weighted_distance = 0.0

    for label in unique_labels:
        # 获取当前簇的所有点索引
        indices = (labels == label).nonzero().squeeze(1)
        
        # 获取当前簇的所有嵌入向量
        cluster_embs = torch.stack([data_maps[names[i]] for i in indices]).squeeze(1)  # [k, dim]
        
        # 计算簇中心（可微操作）
        centroid = cluster_embs.mean(dim=0, keepdim=True)  # [1, dim]
        
        # 计算每个点到中心的距离（可微操作）
        distances = torch.norm(cluster_embs - centroid, dim=1)  # [k]
        
        # 计算平均距离作为簇内紧密度损失
        avg_distance = distances.mean()
        
        # ====== 余弦相似度损失 ======
        if use_angle_loss:
            cos_sim = F.cosine_similarity(centroid, cluster_embs, dim=1)
            angle_loss = 1.0 - cos_sim.mean()
        
            # 加权累加损失（距离越大表示簇越不紧密）
            total_loss.append(avg_distance + angle_loss)
        
        else:
            total_loss.append(avg_distance)

    loss_tensor = torch.stack(total_loss)
    sum_distance = loss_tensor.sum()
    
    if return_raw_distance_list:
        return total_loss
    
    total_loss = (loss_tensor ** 2) / sum_distance
    total_weighted_distance = total_loss.sum()

    return total_weighted_distance

def cluster_with_distance(
        data_maps, 
        file_name=None, 
        extra_name="None",
        save_path=None,
        algorithm="kmeans",
        return_raw_distance_list=False
    ):
    # 提取名称和嵌入向量
    names = list(data_maps.keys())
    embeds = torch.stack([data_maps[name] for name in names]).squeeze(1).detach().float().cpu().numpy()
    
    algorithm_option = {
        'kmeans': KMeans,
        'hierarchical': AgglomerativeClustering
    }

    with torch.no_grad():
        # 聚类算法
        flag = int(extra_name != "None" and extra_name in names)
        clustering = algorithm_option[algorithm](n_clusters=3+flag)
        labels = clustering.fit_predict(embeds)
        labels = torch.tensor(labels)

    unique_labels = labels.unique()
    total_loss = []
    # calculate distance in cluster
    for label in unique_labels:
        # 获取当前簇的所有点索引
        indices = (labels == label).nonzero().squeeze(1)
        # 获取当前簇的所有嵌入向量
        cluster_embs = torch.stack([data_maps[names[i]] for i in indices])  # [k, dim]
        # 计算簇中心
        centroid = cluster_embs.mean(dim=0, keepdim=True)  # [1, dim]
        # 计算每个点到中心的距离
        distances = torch.norm(cluster_embs - centroid, dim=1)  # [k]
        # 计算平均距离
        avg_distance = distances.mean()
        total_loss.append(avg_distance)

    loss_tensor = torch.stack(total_loss)
    sum_distance = loss_tensor.sum()

    if return_raw_distance_list:
        return total_loss
    
    total_loss = (loss_tensor ** 2) / sum_distance
    total_weighted_distance = total_loss.sum()
    
    return total_weighted_distance