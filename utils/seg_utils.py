import numpy as np
from tqdm import tqdm
import torch
from scipy.spatial.distance import cosine

def num_to_natural(group_ids, void_number=-1):
    if (void_number == -1):
        # [-1,-1,0,3,4,0,6] -> [-1,-1,0,1,2,0,3]
        if np.all(group_ids == -1):
            return group_ids
        array = group_ids.copy()

        unique_values = np.unique(array[array != -1])
        mapping = np.full(np.max(unique_values) + 2, -1)
        # map ith(start from 0) group_id to i
        mapping[unique_values + 1] = np.arange(len(unique_values))
        array = mapping[array + 1]

    elif (void_number == 0):
        # [0,3,4,0,6] -> [0,1,2,0,3]
        if np.all(group_ids == 0):
            return group_ids
        array = group_ids.copy()

        unique_values = np.unique(array[array != 0])
        mapping = np.full(np.max(unique_values) + 2, 0)
        mapping[unique_values] = np.arange(len(unique_values)) + 1
        array = mapping[array]
    else:
        raise Exception("void_number must be -1 or 0")

    return array

class Edge:
    def __init__(self, a, b, w, sem):
        self.a = a
        self.b = b
        self.w = w
        self.sem = sem

# Disjoint-set (union-find) class
class Universe:
    def __init__(self, num_elements, feat):
        self.parent = list(range(num_elements))
        self.rank = [0] * num_elements
        self.size = [1] * num_elements
        self.num = num_elements
        self.feat = feat

    def find(self, u):
        if u != self.parent[u]:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def get_feat_sim(self, u, v):
        feat_u = self.feat[u]
        feat_v = self.feat[v]
        sim = np.dot(feat_u, feat_v) / (np.linalg.norm(feat_u) * np.linalg.norm(feat_v))
        return sim

    def union(self, u, v):
        u_root = self.find(u)
        v_root = self.find(v)

        if u_root == v_root:
            return

        # Union by rank
        if self.rank[u_root] > self.rank[v_root]:
            self.parent[v_root] = u_root
            self.size[u_root] += self.size[v_root]

            # update node feature
            # self.feat[u_root] = (self.size[v_root] * self.feat[v_root] + self.size[u_root] * self.feat[u_root]) / (self.size[v_root] + self.size[u_root])

        else:
            self.parent[u_root] = v_root
            self.size[v_root] += self.size[u_root]

            # update node feature
            # self.feat[v_root] = (self.size[v_root] * self.feat[v_root] + self.size[u_root] * self.feat[u_root]) / (self.size[v_root] + self.size[u_root])

            if self.rank[u_root] == self.rank[v_root]:
                self.rank[v_root] += 1

        self.num -= 1

    def component_size(self, u):
        return self.size[self.find(u)]

    def num_sets(self):
        return self.num

@torch.inference_mode()
def calcu_all_jsd_similar(p_dist, q_dist):
    # p_dist: [n1, d]
    # q_dist: [n2, d]
    assert p_dist.shape[1] == q_dist.shape[1], "dimension should be same."

    def kl_divergence(p, q):
        return torch.sum(p * torch.log(p / q), dim=-1)
    
    epsilon = 1e-10
    p_dist = p_dist + epsilon
    q_dist = q_dist + epsilon
    
    # p_dist -> [n1, 1, d], q_dist -> [1, n2, d]
    p_dist_expanded = p_dist.unsqueeze(1)  # [n1, 1, d]
    q_dist_expanded = q_dist.unsqueeze(0)  # [1, n2, d]
    
    # mean dist. M = 0.5 * (P + Q)
    m_dist = 0.5 * (p_dist_expanded + q_dist_expanded)  # [n1, n2, d]
    
    # cal JSD
    kl_p_m = kl_divergence(p_dist_expanded, m_dist)  # [n1, n2]
    kl_q_m = kl_divergence(q_dist_expanded, m_dist)  # [n1, n2]
    jsd_matrix = 1 - 0.5 * (kl_p_m + kl_q_m)  # [n1, n2]

    return jsd_matrix

def extract_geometric_features(points, ins_members):
    """
    提取实例的几何特征，使用协方差矩阵表示形状特性
    
    Args:
        points: 点云数据，形状为 [n_points, 3]
        ins_members: 字典，键为实例ID，值为该实例包含的点的索引数组
    
    Returns:
        geo_features: 字典，键为实例ID，值为几何特征（协方差矩阵的特征值和特征向量）
    """
    geo_features = {}
    
    for ins_id, member_indices in ins_members.items():
        if len(member_indices) == 0:
            # 空实例，使用零矩阵
            geo_features[ins_id] = np.zeros(6)
            continue
            
        # 获取实例的点云数据
        instance_points = points[member_indices]
        
        # 计算协方差矩阵
        covariance = np.cov(instance_points, rowvar=False)
        
        # 计算协方差矩阵的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
        # 特征值排序（从大到小）
        eigenvalues = eigenvalues[::-1]
        
        # 提取特征向量的主要方向（前三个方向）
        main_directions = eigenvectors[:, ::-1][:, :3].flatten()
        
        # 组合特征：特征值 + 主要方向
        # 使用特征值表示形状的伸展程度，特征向量表示形状的方向
        combined_features = np.concatenate([eigenvalues, main_directions[:3]])
        
        geo_features[ins_id] = combined_features
    
    return geo_features

def extract_semantic_features(feat, ins_members):
    """
    提取超高斯点的语义特征，基于点特征平均
    
    Args:
        feat: 点的特征向量，形状为 [n_points, feat_dim]
        ins_members: 字典，键为实例ID，值为该实例包含的点的索引数组
    
    Returns:
        sem_features: 字典，键为实例ID，值为语义特征向量
    """
    sem_features = {}
    
    for ins_id, member_indices in ins_members.items():
        if len(member_indices) == 0:
            # 空实例，使用零向量
            sem_features[ins_id] = np.zeros(feat.shape[1])
            continue
            
        # 获取实例的点特征
        instance_feats = feat[member_indices]
        
        # 计算平均特征向量作为实例语义特征
        # 也可以考虑使用其他聚合方法，如最大池化、加权平均等
        avg_features = np.mean(instance_feats, axis=0)
        
        # 归一化特征向量
        norm = np.linalg.norm(avg_features)
        if norm > 0:
            avg_features = avg_features / norm
        
        sem_features[ins_id] = avg_features
    
    return sem_features

def extract_context_features(ins_members, points, ins_neighbors, ins_num):
    """
    提取上下文特征，包括邻居数量、邻居距离统计和空间关系
    
    Args:
        ins_members: 字典，键为实例ID，值为该实例包含的点的索引数组
        points: 点云数据，形状为 [n_points, 3]
        ins_neighbors: 邻居关系矩阵，形状为 [ins_num, ins_num]
        ins_num: 实例数量
    
    Returns:
        ctx_features: 字典，键为实例ID，值为上下文特征向量
    """
    ctx_features = {}
    
    # 计算每个实例的中心点
    centers = {}
    for ins_id, member_indices in ins_members.items():
        if len(member_indices) > 0:
            centers[ins_id] = np.mean(points[member_indices], axis=0)
        else:
            centers[ins_id] = np.zeros(3)
    
    for ins_id in range(ins_num):
        # 计算邻居数量
        neighbor_count = np.sum(ins_neighbors[ins_id])
        
        # 计算到邻居的距离统计
        neighbor_dists = []
        for neighbor_id in np.where(ins_neighbors[ins_id])[0]:
            if ins_id in centers and neighbor_id in centers:
                dist = np.linalg.norm(centers[ins_id] - centers[neighbor_id])
                neighbor_dists.append(dist)
        
        if neighbor_dists:
            avg_dist = np.mean(neighbor_dists)
            min_dist = np.min(neighbor_dists)
            max_dist = np.max(neighbor_dists)
        else:
            avg_dist = min_dist = max_dist = 0.0
        
        # 组合上下文特征
        ctx_feature = np.array([
            neighbor_count,
            avg_dist,
            min_dist,
            max_dist
        ])
        
        ctx_features[ins_id] = ctx_feature
    
    return ctx_features

def compute_geometric_similarity(geo_feat1, geo_feat2):
    """
    计算两个实例的几何相似度
    
    Args:
        geo_feat1: 第一个实例的几何特征
        geo_feat2: 第二个实例的几何特征
    
    Returns:
        similarity: 几何相似度得分
    """
    # 提取特征值部分（前3个元素）
    ev1 = geo_feat1[:3]
    ev2 = geo_feat2[:3]
    
    # 提取方向部分（后3个元素）
    dir1 = geo_feat1[3:6]
    dir2 = geo_feat2[3:6]
    
    # 计算特征值的相似度（使用余弦相似度）
    # 归一化特征值以便比较
    ev1_norm = ev1 / (np.linalg.norm(ev1) + 1e-8)
    ev2_norm = ev2 / (np.linalg.norm(ev2) + 1e-8)
    ev_similarity = 1 - cosine(ev1_norm, ev2_norm)
    
    # 计算方向的相似度（使用余弦相似度）
    dir_similarity = np.abs(np.dot(dir1, dir2))
    
    # 组合相似度得分
    # 特征值相似度权重0.6，方向相似度权重0.4
    similarity = 0.6 * ev_similarity + 0.4 * dir_similarity
    
    return similarity

def compute_semantic_similarity(sem_feat1, sem_feat2):
    """
    计算两个实例的语义相似度
    
    Args:
        sem_feat1: 第一个实例的语义特征
        sem_feat2: 第二个实例的语义特征
    
    Returns:
        similarity: 语义相似度得分
    """
    # 余弦相似度
    return 1 - cosine(sem_feat1, sem_feat2)

def compute_context_similarity(ctx_feat1, ctx_feat2):
    """
    计算两个实例的上下文相似度
    
    Args:
        ctx_feat1: 第一个实例的上下文特征
        ctx_feat2: 第二个实例的上下文特征
    
    Returns:
        similarity: 上下文相似度得分
    """
    # 提取各个特征组件
    # [邻居数量, 平均距离, 最小距离, 最大距离]
    neighbor_count1, avg_dist1, min_dist1, max_dist1 = ctx_feat1
    neighbor_count2, avg_dist2, min_dist2, max_dist2 = ctx_feat2
    
    # 邻居数量相似度（使用相对差异）
    if neighbor_count1 + neighbor_count2 > 0:
        count_diff = abs(neighbor_count1 - neighbor_count2) / max(neighbor_count1, neighbor_count2)
        count_similarity = 1 - count_diff
    else:
        count_similarity = 1.0
    
    # 距离统计相似度
    dist_diffs = [
        abs(avg_dist1 - avg_dist2) / (max(avg_dist1, avg_dist2) + 1e-8),
        abs(min_dist1 - min_dist2) / (max(min_dist1, min_dist2) + 1e-8),
        abs(max_dist1 - max_dist2) / (max(max_dist1, max_dist2) + 1e-8)
    ]
    dist_similarity = 1 - np.mean(dist_diffs)
    
    # 组合相似度得分
    similarity = 0.3 * count_similarity + 0.7 * dist_similarity
    
    return similarity

# @breif: cal graph edge: similarity and confidence
# @param ins_neighbors [n_ins, n_ins]: 1 if two instance are neighbors
# @param ins_label [n_points]: 3D instance id of every point
# @param ins_vis_ratio [n_ins, n_imagse]: ratio of seen part of primitives in every view
# @param points_mask_label [n_points, n_images]: labels of all points in all views
# @param geo_features: 几何特征字典
# @param sem_features: 语义特征字典
# @param ctx_features: 上下文特征字典
# @return similar [n_ins, n_ins]: wighted sum of how much the two primitives are similar in every view
# @return confidence [n_ins, n_ins]: sum of wight of how much we can trust the similar score in every view
@torch.inference_mode()
def get_similar_confidence_matrix(ins_neighbors, ins_label, ins_vis_ratio, points_mask_label, 
                                 geo_features=None, sem_features=None, ctx_features=None,
                                 feature_weights=[0.3, 0.3, 0.2, 0.2],
                                 mask_quality_scores=None, quality_weight=0.5):
    """
    计算基于多视图融合的相似度和置信度矩阵，支持多特征融合和掩码质量分数加权
    
    Args:
        ins_neighbors: 实例邻居关系矩阵
        ins_label: 实例标签
        ins_vis_ratio: 实例可见性比率
        points_mask_label: 点的掩码标签
        geo_features: 几何特征字典 {ins_id: geo_feature}
        sem_features: 语义特征字典 {ins_id: sem_feature}
        ctx_features: 上下文特征字典 {ins_id: ctx_feature}
        feature_weights: 特征权重 [几何权重, 语义权重, 上下文权重, 可见性权重]
        mask_quality_scores: 各视图的掩码质量分数字典列表 [view1_scores, view2_scores, ...]
        quality_weight: 掩码质量在置信度中的权重
        
    Returns:
        similar_sum: 相似度矩阵
        confidence_sum: 置信度矩阵
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    view_num = ins_vis_ratio.shape[1]
    ins_num = ins_vis_ratio.shape[0]

    gpu_ins_label = torch.tensor(ins_label, device=device, dtype=torch.int32)  # (n,)
    gpu_points_mask_label = torch.tensor(points_mask_label, device='cpu', dtype=torch.float32)  # (n,m)
    gpu_ins_neighbors = torch.tensor(ins_neighbors, device=device, dtype=torch.bool)  # (s, s)
    gpu_ins_vis_ratio = torch.tensor(ins_vis_ratio, device=device, dtype=torch.float32)  # (s, m)

    similar_sum = torch.zeros([ins_num, ins_num], device=device, dtype=torch.float32)  # (s, s)
    confidence_sum = torch.zeros([ins_num, ins_num], device=device, dtype=torch.float32)
    one_view_similar = torch.zeros([ins_num, ins_num], device=device, dtype=torch.float32)
    one_view_confidence = torch.zeros([ins_num, ins_num], device=device, dtype=torch.float32)

    for m in tqdm(range(view_num)):
        plabels = gpu_points_mask_label[:, m].to(device)  # (n_points,)
        one_view_similar.fill_(0.)
        one_view_confidence.fill_(0.)

        label_range = int(torch.max(plabels) + 1) # exist 0
        if (label_range < 2):
            continue  

        # cal 3d_id & 2d_mask_id distribution
        seglabels = torch.zeros([ins_num, label_range], device=device, dtype=torch.float32)  # (s, lr)
        # concat: 2d mask label || 3d instance label
        p_maskIds_insIds = torch.stack([plabels, gpu_ins_label], dim=1) #  [n_points, 2]
        # unique_maskIds_insIds: [n_uniques, 2], unique_counts: [n_uniques]
        unique_maskIds_insIds, unique_counts = torch.unique(p_maskIds_insIds, return_counts=True, dim=0)
        unique_maskIds_insIds = unique_maskIds_insIds.type(torch.long)
        unique_counts = unique_counts.type(torch.float32)
        
        # 如果有掩码质量分数，应用质量加权
        if mask_quality_scores and m < len(mask_quality_scores):
            view_scores = mask_quality_scores[m]
            # 对每个掩码ID应用质量权重
            for idx, (mask_id, ins_id) in enumerate(unique_maskIds_insIds):
                if mask_id > 0 and mask_id in view_scores:
                    # 将掩码质量分数应用到计数上
                    quality_factor = view_scores[mask_id]
                    unique_counts[idx] *= (1 + quality_weight * quality_factor)
        
        seglabels[unique_maskIds_insIds[:, 1], unique_maskIds_insIds[:, 0]] = unique_counts  # (n_ins, label_range)

        # 2D mask id 0 is invalid
        nonzero_seglabels = seglabels[:, 1:]  # (s,lr-1)
        nonzero_seglabels = torch.divide(nonzero_seglabels, torch.clamp(torch.norm(nonzero_seglabels, dim=-1), 1e-8)[:, None])

        del unique_counts, unique_maskIds_insIds
        del p_maskIds_insIds, plabels

        # in every iter, we process one batch primitives and its all neighbors
        batch_size = 200
        for start_id in range(0, ins_num, batch_size):
            if (ins_neighbors[start_id:start_id+batch_size].nonzero()[0].size == 0):
                continue
            
            # [batch_size, ins_num] --> sum --> [ins_num]
            all_neighbors_mask = torch.sum(gpu_ins_neighbors[start_id:start_id+batch_size], dim=0) > 0
            neighbors_labels = nonzero_seglabels[all_neighbors_mask] # [neibors, lr-1]
            
            one_view_similar[start_id:start_id+batch_size, all_neighbors_mask] = \
                calcu_all_jsd_similar(nonzero_seglabels[start_id:start_id+batch_size], neighbors_labels)
        
        # 基础可见性置信度
        visibility_confidence = gpu_ins_vis_ratio[:,m][:, None] @ gpu_ins_vis_ratio[:, m][None, :]
        
        # 如果有掩码质量分数，创建基于质量的置信度
        quality_confidence = torch.ones_like(visibility_confidence)
        if mask_quality_scores and m < len(mask_quality_scores):
            view_scores = mask_quality_scores[m]
            # 为每个实例计算平均质量分数
            ins_quality_scores = torch.ones(ins_num, device=device, dtype=torch.float32)
            
            # 统计每个实例关联的掩码质量分数
            for idx, (mask_id, ins_id) in enumerate(unique_maskIds_insIds):
                if mask_id > 0 and mask_id in view_scores:
                    if ins_quality_scores[ins_id] == 1.0:  # 初始值
                        ins_quality_scores[ins_id] = view_scores[mask_id]
                    else:
                        # 取最大值作为实例的质量分数
                        ins_quality_scores[ins_id] = max(ins_quality_scores[ins_id], view_scores[mask_id])
            
            # 创建质量置信度矩阵
            quality_confidence = ins_quality_scores[:, None] @ ins_quality_scores[None, :]
        
        # 组合可见性和质量置信度
        one_view_confidence = visibility_confidence * (1 - quality_weight) + quality_confidence * quality_weight
        
        # 如果提供了额外特征，计算多特征融合的相似度
        if geo_features is not None and sem_features is not None and ctx_features is not None:
            # 初始化特征相似度矩阵
            geo_similarity = torch.zeros([ins_num, ins_num], device=device, dtype=torch.float32)
            sem_similarity = torch.zeros([ins_num, ins_num], device=device, dtype=torch.float32)
            ctx_similarity = torch.zeros([ins_num, ins_num], device=device, dtype=torch.float32)
            
            # 计算特征相似度
            for i in range(ins_num):
                for j in range(ins_num):
                    if ins_neighbors[i, j]:
                        # 几何相似度
                        if i in geo_features and j in geo_features:
                            geo_sim = compute_geometric_similarity(geo_features[i], geo_features[j])
                            geo_similarity[i, j] = geo_sim
                        
                        # 语义相似度
                        if i in sem_features and j in sem_features:
                            sem_sim = compute_semantic_similarity(sem_features[i], sem_features[j])
                            sem_similarity[i, j] = sem_sim
                        
                        # 上下文相似度
                        if i in ctx_features and j in ctx_features:
                            ctx_sim = compute_context_similarity(ctx_features[i], ctx_features[j])
                            ctx_similarity[i, j] = ctx_sim
            
            # 多特征融合相似度
            # 从feature_weights获取各特征权重
            geo_weight, sem_weight, ctx_weight, vis_weight = feature_weights
            
            # 可见性特征（基于one_view_similar）
            # 归一化可见性相似度
            vis_similarity = one_view_similar.clone()
            vis_max = torch.max(vis_similarity)
            if vis_max > 0:
                vis_similarity = vis_similarity / vis_max
            
            # 组合所有特征相似度
            fused_similarity = (
                geo_weight * geo_similarity +
                sem_weight * sem_similarity +
                ctx_weight * ctx_similarity +
                vis_weight * vis_similarity
            )
            
            # 更新one_view_similar为融合后的相似度
            one_view_similar = fused_similarity
        del seglabels, nonzero_seglabels

        confidence_sum += one_view_confidence
        similar_sum += (one_view_similar * one_view_confidence)
    
    # [ins_num, ins_num], [ins_num, ins_num]
    return [similar_sum.cpu().numpy(), confidence_sum.cpu().numpy()]