import os
from typing import Any
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
from argparse import ArgumentParser
import random
import yaml
from munch import munchify
import cv2
import open3d as o3d
import imgviz
import numpy as np
from tqdm import tqdm
import scipy
from plyfile import PlyData, PlyElement

import torch
import matplotlib.pyplot as plt
from collections import deque

from gaussian_splatting.scene.gaussian_model import GaussianModel
from utils.config_utils import load_config
from utils.seg_utils import (
    num_to_natural, Universe, Edge, get_similar_confidence_matrix,
    extract_geometric_features, extract_semantic_features, extract_context_features
)
from utils.mask_edge_utils import process_mask_edges_for_pointcloud
from utils.eval_utils import calculate_iou_3d
from gaussian_splatting.utils.system_utils import mkdir_p
from datasets.load_func import load_dataset
from decoders.decoder import FeatureDecoder

class GSegmentation:
    def __init__(self, config):
        self.config = config
        self.save_dir = config["Results"]["save_dir"]

        self.seg_min_verts = config['segmentation']['seg_min_verts']
        self.k_thr = config['segmentation']['k_thresh']
        self.k_neigbor = config['segmentation']['k_neigbor']
        self.clustering_thres = config['segmentation']['thres_connect']
        self.thres_merge = config['segmentation']['thres_merge']

        self.discard_unseen = config["Training"]["discard_unseen"]
        self.thres_vis_dis = config['Training']['thres_vis_dis']
        self.kf_inter = self.config["Training"]["kf_inter"]

        self.n_workers = 20
        self.feat_decoder = None

    def load_decoder(self):
        ckpt_path = os.path.join(self.save_dir, 'decoder/ckpt.pth')
        print('Load feature decoder from: ', ckpt_path)
        # 确保在加载解码器时使用相同的配置，包括位置编码设置
        self.feat_decoder = FeatureDecoder(self.config).cuda()
        
        # 加载模型权重，并处理可能的键名不匹配
        checkpoint = torch.load(ckpt_path)
        
        # 过滤掉可能不存在的权重（如新添加的位置编码网络参数）
        model_dict = self.feat_decoder.state_dict()
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        
        # 更新模型权重
        model_dict.update(filtered_checkpoint)
        self.feat_decoder.load_state_dict(model_dict)
        
        # 显示加载的权重信息
        print(f"Loaded {len(filtered_checkpoint)} out of {len(checkpoint)} weights")
        if len(filtered_checkpoint) < len(checkpoint):
            print("Some weights were not loaded (possibly due to model architecture changes)")
        
        # 检查是否存在特征空间优化相关的新参数
        new_params_exist = any(
            k.startswith(('cross_res_interaction.', 'feature_attention.', 'feature_decomposition.')) 
            for k in checkpoint.keys()
        )
        
        if new_params_exist:
            print("Decoder contains feature space optimization parameters")
            # 确保特征空间优化选项与模型匹配
            if not hasattr(self.feat_decoder, 'optimize_feature_space') or not self.feat_decoder.optimize_feature_space:
                print("Warning: Decoder has optimization parameters but optimize_feature_space is not enabled")
        else:
            print("Decoder does not contain feature space optimization parameters")
    
    # @breif: graph cuts with geo. and lang.
    def segment_graph(self, num_vertices, edges, threshold, feat, mask_edges=None, mask_edge_weight=0.3):
        """
        图割分割算法，支持掩码边缘约束
        
        Args:
            num_vertices: 顶点数量
            edges: 边列表
            threshold: 阈值
            feat: 特征
            mask_edges: 掩码边缘点对集合 {(p1, p2), ...}
            mask_edge_weight: 掩码边缘约束的权重
            
        Returns:
            Universe: 分割结果
        """
        edges = sorted(edges, key=lambda e: e.w)
        u = Universe(num_vertices, feat) 
        normal_t = [threshold] * num_vertices  
        feat_thre = 0.99
        feat_t = [feat_thre] * num_vertices 
        
        # 初始化掩码边缘权重
        if mask_edges is None:
            mask_edges = set()
            mask_edge_weight = 0.0
        
        # 构建掩码边缘的快速查询结构
        mask_edge_set = set(mask_edges)
        
        for edge in edges:
            a = u.find(edge.a)
            b = u.find(edge.b)
            if a != b:  
                normal = edge.w <= normal_t[a] and edge.w <= normal_t[b]
                sim = u.get_feat_sim(a,b)
                feat = sim > 0.9
                # feat = ((sim >= feat_t[a]) and (sim >= feat_t[b]))
                
                # 检查是否为掩码边缘
                is_mask_edge = False
                if mask_edge_weight > 0:
                    is_mask_edge = ((edge.a, edge.b) in mask_edge_set or (edge.b, edge.a) in mask_edge_set)
                
                # 调整合并条件：掩码边缘降低合并概率
                should_merge = normal and feat
                if is_mask_edge:
                    # 对于掩码边缘，需要更高的特征相似度才能合并
                    should_merge = should_merge and (sim > 0.95)
                    # 或者直接禁止合并
                    # should_merge = False
                
                if should_merge:
                    u.union(a, b) 
                    new_root = u.find(a)
                    normal_t[new_root] = edge.w + threshold / u.component_size(new_root)
                    feat_t[new_root] = sim + feat_thre / u.component_size(new_root)

        return u

    # @breif: build super-gaussian or super-primitives
    def build_super_gaussians(self, mask_edges=None, mask_edge_weight=0.3):
        """
        构建超高斯体，支持掩码边缘约束
        
        Args:
            mask_edges: 掩码边缘点对集合 {(p1, p2), ...}
            mask_edge_weight: 掩码边缘约束的权重
            
        Returns:
            segments: 分割结果
        """
        # points normals neighbors
        ply = o3d.geometry.PointCloud()
        ply.points = o3d.utility.Vector3dVector(self.points_w)
        ply.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals = np.array(ply.normals)

        points_kdtree = scipy.spatial.KDTree(self.points_w)
        # (n_points, k_neighbors)  the first one is itself
        self.points_neighbors = points_kdtree.query(self.points_w, self.k_neigbor, workers=self.n_workers)[1]
        # distances, indices = points_kdtree.query(self.points_w, k=k)

        num_points = self.points_w.shape[0] 
        num_neighbors = self.points_neighbors.shape[1]
        edges = []

        if self.feat_decoder is None:
            self.load_decoder()

        # 如果没有提供mask_edges，使用基于几何特征的边缘检测
        if mask_edges is None or len(mask_edges) == 0:
            print("No mask edges provided, using geometric feature-based edge detection")
            mask_edges = self._compute_geometric_edges()
        
        # 检查是否支持掩码引导的特征注意力
        use_mask_guided_attention = hasattr(self.feat_decoder, 'set_mask_attention') and mask_edges is not None
        
        # 生成point_indices - 用于掩码引导的特征注意力
        point_indices = None
        if use_mask_guided_attention:
            # 设置掩码边缘信息用于特征注意力计算
            self.feat_decoder.set_mask_attention(mask_edges, self.points_w)
            print("Using mask-guided feature attention")
            # 生成点索引数组 (从0到num_points-1)
            point_indices = self._generate_geometric_point_indices()
        else:
            # 如果不使用掩码引导注意力，仍然生成点索引以提高性能
            point_indices = self._generate_geometric_point_indices()
        
        # 调用decoder时传递point_indices参数
        pc_feat = self.feat_decoder(torch.from_numpy(self.points_w), point_indices=point_indices).cpu().detach() # [N, n_dim]
        pc_feat = torch.nn.functional.normalize(pc_feat, p=2, dim=1).numpy()

        # construct edges
        for i in range(num_points):
            for j in range(1, num_neighbors): # ignore itself
                a = i
                b = self.points_neighbors[i, j]
                norm_dist = 1.0 - np.dot(normals[a], normals[b])

                # for convex surface
                d_xyz = self.points_w[b] - self.points_w[a]
                d_xyz /= np.linalg.norm(d_xyz)
                dot2 = np.dot(normals[b], d_xyz)
                
                feat_dist = 1 - np.dot(pc_feat[a], pc_feat[b])

                if dot2 > 0:
                    norm_dist = norm_dist * norm_dist 
                    
                # W[i, j] = np.exp(-color_dist**2 / (2 * sigma_c**2)) * np.exp(-normal_dist**2 / (2 * sigma_n**2))
                if feat_dist > 0.1:
                    norm_dist = norm_dist * 20

                edge_w = norm_dist
                edges.append(Edge(a, b, edge_w, feat_dist))  

        # 使用掩码边缘约束进行图割分割
        u = self.segment_graph(num_points, edges, self.k_thr, pc_feat, mask_edges, mask_edge_weight)

        # merge small segments
        for edge in edges:
            a = u.find(edge.a)
            b = u.find(edge.b)
            if (a != b) and (u.component_size(a) < self.seg_min_verts or u.component_size(b) < self.seg_min_verts):
                u.union(a, b)

        segments = [u.find(i) for i in range(num_points)]
        print('Graph Cuts.: ', len(np.unique(segments, return_counts=True)[1]), ' Super-Primitives.')
        return segments

    # @breif: update graph data before each iteration
    def build_graph(self, ins_label):
        ins_label = num_to_natural(ins_label)
        unique_gs_ids, ins_member_count = np.unique(ins_label, return_counts=True)  # from 0 to instance_num - 1
        ins_num = len(unique_gs_ids)

        # build dict: super-Gaussian ID -> 3D Gaussians index
        ins_members = {ins_id: np.where(ins_label == ins_id)[0] for ins_id in unique_gs_ids}

        ins_neighbors = np.zeros((ins_num, ins_num), dtype=bool)
        for ins_id, members in ins_members.items():
            # Gaussians with current 3d ins_label -> its neighbors -> its neighbors' 3d ins_label
            neighbor_ins_ids = ins_label[self.points_neighbors[members]].flatten()
            ins_neighbors[ins_id, neighbor_ins_ids] = True
        
        # neighboring matrix symmetric and exclude self
        ins_neighbors = np.maximum(ins_neighbors, ins_neighbors.T)
        np.fill_diagonal(ins_neighbors, 0)

        # indirect neighbor pool
        ins_neighbor_pool = np.zeros((ins_num, ins_num), dtype=bool)
        for ins_id in range(ins_num):
            neigbs = ins_neighbors[ins_id]
            neighbors_pool = ins_neighbors[neigbs].sum(0) > 0
            ins_neighbor_pool[ins_id] = neighbors_pool
        
        # exclude self
        np.fill_diagonal(ins_neighbor_pool, 0)
        ins_neighbor_pool[ins_neighbors] = 1
        ins_neighbor_pool[ins_neighbor_pool.T] = 1

        # update graph data
        # list of int
        self.ins_member_count = ins_member_count
        # binary matrix, (ins_num, ins_num)
        self.ins_neighbors = ins_neighbors
        # binary matrix, (ins_num, ins_num)
        self.ins_neighbor_pool = ins_neighbor_pool
        # (n_points)
        self.ins_label = ins_label
        # int
        self.ins_num = ins_num
        # dict ins_id : list of points id
        self.ins_members = ins_members

    # @breif: compute affinity between diff instance
    def compute_edge_affinity(self, points_mask_label, points_seen, points=None, feat=None, 
                             use_multifeature=True, feature_weights=[0.3, 0.3, 0.2, 0.2]):
        """
        计算实例间的亲和度矩阵，支持多特征融合
        
        Args:
            points_mask_label: 点的掩码标签
            points_seen: 点可见性信息
            points: 点云数据 [n_points, 3]
            feat: 点特征向量 [n_points, feat_dim]
            use_multifeature: 是否使用多特征融合
            feature_weights: 特征权重 [几何权重, 语义权重, 上下文权重, 可见性权重]
            
        Returns:
            adjacency_mat: 亲和度矩阵
        """
        # 计算每个实例在各视图的可见率
        ins_vis_ratio = np.zeros([self.ins_num, self.n_images], dtype=np.float32)  # (ins_num, n_images)
        for ins_id, members in self.ins_members.items(): # dict {ins_id: point_array}
            ins_vis_ratio[ins_id] = ((points_mask_label[members] > 0).sum(axis=0)) / members.shape[0]  # (n_members, n_images) sum -> (n_images)
        
        # 如果使用多特征融合
        geo_features = None
        sem_features = None
        ctx_features = None
        
        if use_multifeature and points is not None and feat is not None:
            # 提取几何特征（使用协方差矩阵表示）
            geo_features = extract_geometric_features(points, self.ins_members)
            
            # 提取语义特征（基于超高斯点特征平均）
            sem_features = extract_semantic_features(feat, self.ins_members)
            
            # 提取上下文特征
            ctx_features = extract_context_features(self.ins_members, points, self.ins_neighbor_pool, self.ins_num)
        
        # 计算相似度矩阵和置信度矩阵
        similar_mat, confidence_mat = get_similar_confidence_matrix(
            self.ins_neighbor_pool, 
            self.ins_label, 
            ins_vis_ratio, 
            points_mask_label,
            geo_features=geo_features,
            sem_features=sem_features,
            ctx_features=ctx_features,
            feature_weights=feature_weights
        )

        # 计算亲和度矩阵
        assert similar_mat.nonzero()[0].size > 0
        adjacency_mat = np.zeros([self.ins_num, self.ins_num])

        # 亲和度计算：similarity / confidence
        # https://blog.csdn.net/monchin/article/details/79750216 
        adjacency_mat[confidence_mat.nonzero()] = similar_mat[confidence_mat.nonzero()] / confidence_mat[confidence_mat.nonzero()]
        
        # 确保亲和度矩阵的对称性
        r, c = adjacency_mat.nonzero()
        adjacency_mat[r, c] = adjacency_mat[c, r] = np.maximum(adjacency_mat[r, c], adjacency_mat[c, r])

        return adjacency_mat
    
    # @breif: 原始版本的compute_edge_affinity，保持向后兼容
    def compute_edge_affinity_legacy(self, points_mask_label, points_seen):
        """原始版本的亲和度计算方法，仅使用可见率信息"""
        # cal visible ratio of each sup-primitive/vertex in each view
        ins_vis_ratio = np.zeros([self.ins_num, self.n_images], dtype=np.float32)  # (ins_num, n_images)
        for ins_id, members in self.ins_members.items(): # dict {ins_id: point_array}
            ins_vis_ratio[ins_id] = ((points_mask_label[members] > 0).sum(axis=0)) / members.shape[0]  # (n_members, n_images) sum -> (n_images)

        # similar_mat [ins_num, ins_num]: weight sum of similar score in every view 
        # confidence_mat [ins_num, ins_num]: sum of confidence of how much we can trust the similar score in every view
        similar_mat, confidence_mat = get_similar_confidence_matrix(
            self.ins_neighbor_pool, 
            self.ins_label, 
            ins_vis_ratio, 
            points_mask_label
        )

        # cal. adjacency_mat
        assert similar_mat.nonzero()[0].size > 0
        adjacency_mat = np.zeros([self.ins_num, self.ins_num])

        # https://blog.csdn.net/monchin/article/details/79750216 
        adjacency_mat[confidence_mat.nonzero()] = similar_mat[confidence_mat.nonzero()] / confidence_mat[confidence_mat.nonzero()]
        r, c = adjacency_mat.nonzero()
        adjacency_mat[r, c] = adjacency_mat[c, r] = np.maximum(adjacency_mat[r, c], adjacency_mat[c, r])

        return adjacency_mat

    # @breif: postprocess segmentation results by merging small regions into neighbor regions with high affinity 
    def merge_small_segs(self, ins_labels, merge_thres, adj):
        ins_member_count = self.ins_member_count
        unique_labels, ins_count = np.unique(ins_labels, return_counts=True)
        region_num = unique_labels.shape[0]

        merged_labels = ins_labels.copy()
        merge_count = 0
        # 0 means the superpoint is remain to merge
        merged_mask = np.ones_like(ins_labels)
        for i in range(region_num):
            if ins_count[i] > 2:
                continue
            label = unique_labels[i]
            seg_ids = (ins_labels == label).nonzero()[0]
            if ins_member_count[seg_ids].sum() < merge_thres:
                merged_mask[seg_ids] = 0

        finished = False
        while not finished:
            flag = False  # mark whether merging happened in this iteration
            for i in range(region_num):
                label = unique_labels[i]
                seg_ids = (ins_labels == label).nonzero()[0]
                if merged_mask[seg_ids[0]] > 0:
                    continue
                seg_sims = adj[seg_ids].sum(0)
                adj_sort = np.argsort(seg_sims)[::-1]

                for i in range(adj_sort.shape[0]):
                    target_seg_id = adj_sort[i]
                    if merged_mask[target_seg_id] == 0:
                        continue  # if the target region is also too samll and has not been merged, find next target
                    if seg_sims[target_seg_id] == 0:
                        break  # no more target region can be found
                    target_label = merged_labels[target_seg_id]
                    merged_labels[seg_ids] = target_label
                    merge_count += 1
                    merged_mask[seg_ids] = 1
                    flag = True
                    break
            if not flag:
                finished = True

        # for small regions that cannot be merged, set their labels to 0
        merged_labels[merged_mask == 0] = 0
        print('original region number:', ins_count.shape[0])
        print('mreging count:', merge_count)
        print("remove count:", (merged_mask == 0).sum())
        return merged_labels

    # @breif: graph clustering
    # @return affinity [n_ins, n_ins] :
    # @return ins_labels [n_ins] : 3D instance label of points
    def clustering(self, affinity, thres_connect):
        current_label = 1
        ins_labels = np.zeros(self.ins_num, dtype=np.float32)
        visited = np.zeros(self.ins_num, dtype=bool)

        for i in range(self.ins_num):
            if not visited[i]:
                queue = deque()
                queue.append(i)

                visited[i] = True
                ins_labels[i] = current_label

                while queue:
                    v = queue.popleft()
                    js = np.where(self.ins_neighbors[v])[0]

                    for j in js:
                        if visited[j]:
                            continue

                        # 
                        # direct neighbor
                        neighbor_ids = self.ins_neighbors[j]  # (n_ins.)
                        neighbor_ids = np.logical_and(neighbor_ids, ins_labels == current_label)  # (n_ins.)
                        neighbor_ids = neighbor_ids.nonzero()[0]  # (nei,)

                        # # (n_ins.)*(n_ins.) -> (n_ins.) -> (1.)
                        affinity_sum = (affinity[j, neighbor_ids] * self.ins_member_count[neighbor_ids]).sum(0)
                        weight_sum = (self.ins_member_count[neighbor_ids]).sum(0)  # (s.) -> (1.)

                        # indirect neighbor
                        neighbor_ids = self.ins_neighbor_pool[j]  # (n_ins.)
                        neighbor_ids = np.logical_and(neighbor_ids, np.logical_not(self.ins_neighbors[j]))
                        neighbor_ids = np.logical_and(neighbor_ids, ins_labels == current_label)  # (n_ins.)
                        neighbor_ids = neighbor_ids.nonzero()[0]  # (nei,)

                        # # (n_ins.)*(n_ins.) -> (n_ins.) -> (1.)
                        affinity_sum += (0.5 * affinity[j, neighbor_ids] * self.ins_member_count[neighbor_ids]).sum(0)
                        weight_sum += 0.5 * (self.ins_member_count[neighbor_ids]).sum(0)  # (s.) -> (1.)
                        
                        score = affinity_sum / weight_sum
                        connect = (score >= thres_connect)
                        # 
                        
                        if not connect:
                            continue

                        visited[j] = True
                        ins_labels[j] = current_label
                        queue.append(j)

                current_label += 1

        return ins_labels  # (n_ins, )

    # @breif: project N (1e6) points to M (1e3) images
    # @param: points_w: [n_points, 3], intrinsic: [n_images, 3, 3], poses: [n_images, 4, 4]
    # @return points_c: [n_points, n_images, 3], 3D coordinates of spatial points in different image-views (in camera coordinate sys.)
    # @return uv_pixels: [n_points, n_images, 2], pixel coordinates of spatial points in different image-views
    @torch.inference_mode()
    def get_camp_pixel(self):
        batch_size = 10000
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        intrinsics = torch.tensor(self.intrinsics, device=device, dtype=torch.float32)
        poses = torch.tensor(self.poses, device=device, dtype=torch.float32)
        poses_inv = torch.linalg.inv(poses)   # (n_images, 4, 4)
        del poses

        N = self.points_w.shape[0]
        M = poses_inv.shape[0]
        points_c = np.zeros((N, M, 3), dtype=np.float32)
        uv_pixels = np.zeros((N, M, 2), dtype=int)

        for batch_start in tqdm(range(0, N, batch_size)):
            points_world = torch.tensor(self.points_w[batch_start: batch_start+batch_size], device=device, dtype=torch.float32)
            points_world_homo = torch.cat((points_world, torch.ones((points_world.shape[0], 1), dtype=torch.float32, device=device)), 1)

            points_cam_homo = torch.matmul(poses_inv[None], points_world_homo[:, None, :, None])
            points_cam_homo = points_cam_homo[..., 0]        # (N, M, 4)
            points_cam = torch.div(points_cam_homo[..., :-1], points_cam_homo[..., [-1]])  # (N, M, 3)

            # (M, 3, 3) @ (N, M, 3, 1) = (N, M, 3, 1)
            points_pixel_homo = torch.matmul(intrinsics, points_cam[..., None])
            # (N, M, 3)
            points_pixel_homo = points_pixel_homo[..., 0]
            # (u, v) coordinate, (N, M, 2)
            points_pixel = torch.div(points_pixel_homo[..., :-1], torch.clip(points_pixel_homo[..., [-1]], min=1e-8)).round().to(torch.int32)
            points_c[batch_start: batch_start + batch_size] = points_cam.cpu().numpy()
            uv_pixels[batch_start: batch_start + batch_size] = points_pixel.cpu().numpy()

        torch.cuda.empty_cache()
        return points_c, uv_pixels

    # @breif: get 2D mask labels and vis flag of all spatial points in all views
    # @param points_c: [n_points, n_images, 3], 3D coordinates of spatial points in different image-views (in camera coordinate sys.)
    # @param uv_pixels: [n_points, n_images, 2], pixel coordinates of spatial points in different image-views
    # @return all_label: [n_points, n_images], 2D mask labels of all points in all views
    # @return all_seen_flag: [n_points, n_images], seen flag of all points in all views
    # @NOTE: 0 in all_label is invalid
    def get_points_label_seen(self, points_c, uv_pixels, discard_unseen, thres_vis_dis=0.15):
        # @TODO: hardcode
        batch_size = 50000
        all_label = np.zeros([self.n_points, self.n_images], dtype=np.float32)
        all_seen_flag = np.zeros([self.n_points, self.n_images], dtype=bool)

        # projected by batch
        for start_id in tqdm(range(0, self.n_points, batch_size)):
            p_cam0 = points_c[start_id: start_id + batch_size]
            pix0 = uv_pixels[start_id: start_id + batch_size]
            w0, h0 = np.split(pix0, 2, axis=-1)
            w0, h0 = w0[..., 0], h0[..., 0]  # (n_points_sub, n_images)
            bounded_flag = (0 <= w0)*(w0 <= self.width - 1)*(0 <= h0)*(h0 <= self.height - 1)  # (n_points_sub, n_images)

            # (n_points_sub, n_images), querying labels from masks (n_images, H, W) by h (n_points_sub, n_images) and w (n_points_sub, n_images)
            label_iter = self.masks[np.arange(self.n_images), h0.clip(0, self.height - 1), w0.clip(0, self.width - 1)]

            # visible check
            real_depth = p_cam0[..., -1]  # (n_points_sub, n_images)
            # (n_points_sub, n_images), querying depths
            capture_depth = self.depths[np.arange(self.n_images), h0.clip(0, self.height - 1), w0.clip(0, self.width - 1)]  
            visible_flag = np.isclose(real_depth, capture_depth, rtol=thres_vis_dis)
            seen_flag = bounded_flag * visible_flag

            if discard_unseen:
                label_iter = label_iter * seen_flag  # set label of invalid point to 0

            all_seen_flag[start_id: start_id + batch_size] = seen_flag
            all_label[start_id: start_id + batch_size] = label_iter

        return all_label, all_seen_flag

    # @breif: segmentation
    def do_segmentation(self):
        seg_save_path = os.path.join(self.save_dir, 'segmentation')
        os.makedirs(seg_save_path, exist_ok=True)
        
        # Graph Vertex Construction 
        print("====> Construct Vertex.")
        
        # 检查配置中是否启用掩码边缘约束
        use_mask_edge = self.config.get('segmentation', {}).get('mask_edge', {}).get('enabled', False)
        mask_edge_weight = self.config.get('segmentation', {}).get('mask_edge', {}).get('weight', 0.3)
        
        # 尝试多种方式获取mask_edges
        mask_edges = None
        # 1. 从dataset获取，使用2D到3D投影方法
        if use_mask_edge and hasattr(self.dataset, 'get_mask_edges'):
            try:
                # 获取全局点云数据
                global_pointcloud = self.gaussians.get_xyz.detach().cpu().numpy()
                # 调用更新后的get_mask_edges方法，传入全局点云
                mask_edges = self.dataset.get_mask_edges(global_pointcloud=global_pointcloud)
                print(f"Loaded {len(mask_edges)} mask edges from dataset using 2D-3D projection")
            except Exception as e:
                print(f"Warning: Failed to get mask edges from dataset: {e}")
        
        # 2. 如果没有获取到，将在build_super_gaussians中使用基于几何特征的边缘检测
        
        points_ins_label = self.build_super_gaussians(mask_edges=mask_edges, mask_edge_weight=mask_edge_weight)
        points_ins_label = np.array(points_ins_label)

        init_seg_path = os.path.join(seg_save_path, 'init_seg.npy')
        np.save(init_seg_path, points_ins_label)

        # Graph Clustering based Segmentation
        # return: [n_points, n_images, 3] [n_points, n_images, 2]
        points_c, uv_pixels = self.get_camp_pixel()
        # return: [n_points, n_images], [n_points, n_images]
        points_mask_label, points_seen = self.get_points_label_seen(points_c, uv_pixels, self.discard_unseen, self.thres_vis_dis)

        print("====> Perform region clustering.")
        steps = len(self.clustering_thres)
        for i in range(steps):
            # (a) Build Graph Edges
            self.build_graph(points_ins_label)
            
            # (b) Edge Affinity Computation
            edge_affinity = self.compute_edge_affinity(points_mask_label, points_seen)

            # (c) Clustering on instance-level
            ins_labels = self.clustering(edge_affinity, self.clustering_thres[i])

            # (last iter) Filter noise
            if i == (steps - 1) and self.thres_merge > 0:
                ins_labels = self.merge_small_segs(ins_labels, self.thres_merge, edge_affinity)

            # (d) Assign primitive labels to member points
            points_ins_label = np.zeros(self.n_points, dtype=int)
            for j in range(self.ins_num):
                label = ins_labels[j]
                points_ins_label[self.ins_members[j]] = label

        # save results
        final_seg_path = os.path.join(seg_save_path, 'final_seg.npy')
        np.save(final_seg_path, points_ins_label)

        # export results to .txt
        self.export_point_wise_segmentation(True)

    # @breif: load trained 3DGS model, xyz and feature decoder
    def load_data(self):
        self.dataset = load_dataset(config=self.config)

        # ply_path = os.path.join(self.save_dir, 'point_cloud/final/point_cloud.ply')
        ply_path = os.path.join(self.save_dir, 'point_cloud/point_cloud.ply')

        self.gaussians = GaussianModel(self.config["model_params"]['sh_degree'], config=self.config)
        self.gaussians.load_ply(ply_path)
        print('Load 3DGS model from: ', ply_path)

        self.points_w = (self.gaussians.get_xyz).cpu().detach().numpy()        
        self.n_points = self.points_w.shape[0]
        self.width = self.dataset.width
        self.height = self.dataset.height

        # [N_images, 4, 4] # [N_images, H, W] # [N_images, 3, 3] # [N_images, H, W]
        self.poses, self.depths, self.intrinsics, self.masks = self.dataset.load_data_for_seg(self.kf_inter)
        self.n_images = self.poses.shape[0]

    def export_point_wise_segmentation(self, smooth):
        if self.feat_decoder is None:
            self.load_decoder()
        
        # 使用修改后的解码器获取特征
        # 使用基于几何特征的点索引生成
        point_indices = self._generate_geometric_point_indices()
        
        # 调用decoder时传递point_indices参数
        pc_feat = self.feat_decoder(torch.from_numpy(self.points_w), point_indices=point_indices).cpu().detach() # [N, n_dim]
        class_feat = torch.from_numpy(self.dataset.class_text_feat).float()

        pc_feat = torch.nn.functional.normalize(pc_feat, p=2, dim=1)
        class_feat = torch.nn.functional.normalize(class_feat, p=2, dim=1)
        similarity = torch.matmul(pc_feat, class_feat.t())     # [N1, N2]
        _, category = similarity.max(dim=1)

        # prediction vote
        if smooth:
            ind_mask_path = os.path.join(self.save_dir, 'segmentation', 'init_seg.npy')
            ind_mask = np.load(ind_mask_path)

            uni_ids = np.unique(ind_mask)
            for ind in uni_ids:
                mode_result = scipy.stats.mode(category[ind_mask==ind])
                mode_value = mode_result.mode[0]
                category[ind_mask==ind] = mode_value

        if self.config['Dataset']['type'] == 'scannet':
            seg_results = category.numpy() + 1 # 0 is invaliad
        else:
            seg_results = category.numpy()
        
        gt_semantics = self.dataset.gt_semantics
        n_classes = len(self.dataset.class_names)

        ious, accs, masks = calculate_iou_3d(seg_results, gt_semantics, n_classes)
        np.save(os.path.join(self.save_dir, 'pc_feat.npy'), pc_feat.numpy())
        np.save(os.path.join(self.save_dir, 'pre_semantic.npy'), seg_results)

        iou_file_path = os.path.join(self.save_dir, 'eval_pointwise_semantic.txt')
        self.record_results(ious, accs, iou_file_path, masks)

    def record_results(self, ious, accs, file_path, masks):
        """
        记录评估结果到文件
        
        Args:
            ious: 各类别的IOU值
            accs: 各类别的准确率
            file_path: 结果文件路径
            masks: 掩码信息
        """
        with open(file_path, 'w') as f:
            # 记录类别平均IOU和准确率
            f.write(f'Average IoU: {np.mean(ious)}\n')
            f.write(f'Average Accuracy: {np.mean(accs)}\n')
            
            # 记录每个类别的指标
            for i, (iou, acc) in enumerate(zip(ious, accs)):
                f.write(f'Class {i}: IoU={iou:.4f}, Accuracy={acc:.4f}\n')
            
            # 记录掩码数量
            if masks is not None:
                f.write(f'Total masks: {len(masks)}\n')

        
        
    def _compute_geometric_edges(self, edge_threshold=0.8):
        """
        基于几何特征计算边缘点对
        
        Args:
            edge_threshold: 边缘检测阈值
            
        Returns:
            边缘点对集合
        """
        import scipy.spatial
        
        # 构建KD树
        tree = scipy.spatial.KDTree(self.points_w)
        
        # 计算边缘点对
        edge_pairs = set()
        num_points = self.points_w.shape[0]
        
        # 采样部分点以避免计算量过大
        sample_size = min(10000, num_points)
        sample_indices = np.random.choice(num_points, sample_size, replace=False)
        
        for idx in sample_indices:
            # 查找最近邻
            distances, neighbors = tree.query(self.points_w[idx], k=10)  # 10个最近邻
            
            # 计算相邻点之间的特征差异
            for i in range(1, min(6, len(neighbors))):  # 只检查前5个最近邻
                neighbor_idx = neighbors[i]
                
                # 计算局部特征差异
                # 1. 距离变化
                if i > 1:
                    prev_dist = distances[i-1]
                    curr_dist = distances[i]
                    dist_ratio = curr_dist / (prev_dist + 1e-10)
                    
                    # 如果距离突然增大，认为是边缘
                    if dist_ratio > edge_threshold:
                        edge_pairs.add((int(idx), int(neighbor_idx)))
        
        print(f"Computed {len(edge_pairs)} geometric edge pairs")
        return edge_pairs
        
    def _generate_geometric_point_indices(self):
        """
        基于几何特征生成点索引
        
        Returns:
            排序后的点索引tensor
        """
        import scipy.spatial
        
        num_points = self.points_w.shape[0]
        
        # 构建KD树
        tree = scipy.spatial.KDTree(self.points_w)
        
        # 计算每个点的局部密度和特征
        point_scores = np.zeros(num_points)
        
        # 采样部分点进行计算
        sample_size = min(20000, num_points)
        sample_indices = np.random.choice(num_points, sample_size, replace=False)
        
        for idx in sample_indices:
            # 查找最近邻
            distances, neighbors = tree.query(self.points_w[idx], k=20)  # 20个最近邻
            
            if len(distances) > 1:
                # 计算局部密度分数
                mean_dist = np.mean(distances[1:6])  # 前5个最近邻的平均距离
                std_dist = np.std(distances[1:6])    # 前5个最近邻的距离标准差
                
                # 密度分数 = 1/(平均距离 + 标准差)，距离越小分数越高
                density_score = 1.0 / (mean_dist + std_dist + 1e-10)
                
                # 几何特征分数 = 距离变化率
                if len(distances) > 5:
                    dist_change = np.std(distances[1:]) / (np.mean(distances[1:]) + 1e-10)
                else:
                    dist_change = 0
                
                # 综合分数 = 密度分数 + 几何变化分数
                point_scores[idx] = density_score + dist_change
        
        # 对所有点进行排序，分数高的点优先
        all_indices = np.arange(num_points)
        sorted_indices = all_indices[np.argsort(point_scores)[::-1]]
        
        # 转换为tensor并移至GPU
        return torch.tensor(sorted_indices, dtype=torch.long, device='cuda')
        
    def _fill_unlabeled_points(self, point_labels, k=5):
        """
        使用k近邻填充未标记的点
        
        Args:
            point_labels: 点标签，0表示未标记
            k: 近邻数量
        """
        import scipy.spatial
        
        # 找出已标记和未标记的点
        labeled_indices = np.where(point_labels != 0)[0]
        unlabeled_indices = np.where(point_labels == 0)[0]
        
        if len(labeled_indices) == 0 or len(unlabeled_indices) == 0:
            return
        
        # 构建已标记点的KD树
        labeled_points = self.points_w[labeled_indices]
        tree = scipy.spatial.KDTree(labeled_points)
        
        # 为每个未标记点找到最近的已标记点
        unlabeled_points = self.points_w[unlabeled_indices]
        distances, neighbors = tree.query(unlabeled_points, k=min(k, len(labeled_points)))
        
        # 使用投票机制确定标签
        filled_count = 0
        for i, unlabeled_idx in enumerate(unlabeled_indices):
            # 获取k个最近邻的标签
            if k == 1:
                neighbor_labels = [point_labels[labeled_indices[neighbors[i]]]]
            else:
                neighbor_labels = [point_labels[labeled_indices[n]] for n in neighbors[i]]
            
            # 投票选择最常见的标签
            from collections import Counter
            label_counts = Counter(neighbor_labels)
            if label_counts:
                most_common_label = label_counts.most_common(1)[0][0]
                if most_common_label != 0:
                    point_labels[unlabeled_idx] = most_common_label
                    filled_count += 1
        
        print(f"Filled {filled_count} unlabeled points using nearest neighbors")
        return point_labels


if __name__ == "__main__":
    parser = ArgumentParser(description="3D Gaussian Panoptic Segmentation.")
    parser.add_argument("--config", type=str)
    args = parser.parse_args(sys.argv[1:])
    config = load_config(args.config)

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        path = config["Dataset"]["dataset_path"].split("/")
        scene_id = config["Dataset"]["scene_id"]

        # set save_dir
        if config['Dataset']['type'] == 'replica':
            save_dir = os.path.join(config["Results"]["save_dir"], path[-1], scene_id)
        elif config['Dataset']['type'] == 'scannet':
            save_dir = os.path.join(config["Results"]["save_dir"], path[-2], scene_id)
        else:
            print('Dataset type should be replica or scannet')
            exit()

        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)

        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)

    seg = GSegmentation(config)

    seg.load_data()
    seg.do_segmentation()
