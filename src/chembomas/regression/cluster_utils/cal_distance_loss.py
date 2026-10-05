

def calculate_weights(items, alpha=1.0):
    """
    计算有序列表中各元素的权重（基于排名的反比例函数）
    
    参数:
    items (list): 按重要性排序的有序列表，索引0最重要
    alpha (float): 衰减因子(>0)，值越大权重衰减越快（默认1.0）
    
    返回:
    dict: 元素到权重的映射字典
    """
    n = len(items)
    weights = []
    
    # 计算原始权重 (w = 1 / rank^alpha)
    for rank in range(1, n + 1):
        weight = 1.0 / (rank ** alpha)
        weights.append(weight)
    
    # 归一化处理
    total = sum(weights)
    normalized_weights = [w / total for w in weights]

    return normalized_weights

def calculate_distance_loss(distances, alpha=1.0):
    weights = calculate_weights(list(distances.keys()), alpha)
    loss = 0.0
    for i, name in enumerate(distances.keys()):
        loss += weights[i] * distances[name]
    return loss

if __name__ == "__main__":
    distances = {
        "a": 0.1,
        "b": 0.2,
        "c": 0.3,
        "d": 0.4,
        "e": 0.5
    }
    names = ["a", "b", "c", "d", "e"]
    loss = calculate_distance_loss(distances, alpha=1.0)
    print(loss)