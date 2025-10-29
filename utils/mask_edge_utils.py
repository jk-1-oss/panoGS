import cv2
import numpy as np
import torch
import os
import scipy.spatial
from typing import Tuple, List, Set, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_mask_edges(mask: np.ndarray, edge_thickness: int = 2) -> np.ndarray:
    """
    从掩码图像中提取边缘
    
    Args:
        mask: 输入掩码图像，形状为 [H, W]
        edge_thickness: 边缘厚度
    
    Returns:
        edge_mask: 边缘掩码，形状为 [H, W]
    """
    # 确保掩码是二值图像
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    
    # 转换为二值图像
    _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)
    
    # 使用Canny边缘检测，调整阈值为合理的整数值
    edges = cv2.Canny(binary_mask * 255, 50, 150)
    
    # 使用膨胀操作增加边缘厚度
    if edge_thickness > 1:
        kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    
    return edges

def get_2d_edge_points(edge_mask: np.ndarray, max_points: int = None) -> List[Tuple[int, int]]:
    """
    从边缘掩码中获取2D边缘点坐标
    
    Args:
        edge_mask: 边缘掩码，形状为 [H, W]
        max_points: 最大返回点数量，如果为None则返回所有点
    
    Returns:
        edge_points: 2D边缘点坐标列表 [(y1, x1), (y2, x2), ...]
    """
    # 获取所有边缘点的坐标
    y_coords, x_coords = np.where(edge_mask > 0)
    edge_points = list(zip(y_coords, x_coords))
    
    # 如果需要限制点数量，随机采样
    if max_points is not None and len(edge_points) > max_points:
        indices = np.random.choice(len(edge_points), max_points, replace=False)
        edge_points = [edge_points[i] for i in indices]
    
    return edge_points

def generate_edge_pairs_2d(edge_points: List[Tuple[int, int]], max_pairs: int = 1000, sampling_step: int = 1) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    在2D空间中生成边缘点对，基于真实像素连通性追踪轮廓
    
    Args:
        edge_points: 2D边缘点坐标列表
        max_pairs: 最大生成点对数量
        sampling_step: 轮廓点采样步长，默认为1表示取所有相邻点对
    
    Returns:
        edge_pairs: 边缘点对列表 [((y1, x1), (y2, x2)), ...]
    """
    if len(edge_points) < 2:
        return []
    
    # 确定图像尺寸（基于边缘点坐标范围）
    y_coords, x_coords = zip(*edge_points)
    max_y, max_x = max(y_coords) + 1, max(x_coords) + 1
    
    # 创建边缘掩码图像
    edge_mask = np.zeros((max_y, max_x), dtype=np.uint8)
    for y, x in edge_points:
        edge_mask[y, x] = 255
    
    # 使用cv2.findContours提取轮廓
    # 使用RETR_EXTERNAL获取外轮廓，CHAIN_APPROX_NONE获取所有轮廓点
    contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    edge_pairs = []
    
    # 处理每个轮廓
    for contour in contours:
        # 轮廓点数量
        contour_len = len(contour)
        if contour_len < 2:
            continue
        
        # 遍历轮廓点，生成相邻点对
        for i in range(0, contour_len - sampling_step, sampling_step):
            # 获取当前点和下一个采样点
            x1, y1 = contour[i][0]
            x2, y2 = contour[i + sampling_step][0]
            
            # 转换为元组形式
            point1 = (int(y1), int(x1))
            point2 = (int(y2), int(x2))
            
            # 确保点对顺序一致（避免重复）
            if point1 < point2:
                edge_pair = (point1, point2)
            else:
                edge_pair = (point2, point1)
            
            # 添加点对
            if edge_pair not in edge_pairs:
                edge_pairs.append(edge_pair)
                
                # 限制点对数量
                if len(edge_pairs) >= max_pairs:
                    return edge_pairs[:max_pairs]
        
        # 处理可能的剩余点（当轮廓长度不是采样步长的整数倍时）
        if contour_len % sampling_step != 0:
            last_idx = contour_len - 1
            prev_idx = last_idx - (last_idx % sampling_step)
            
            if prev_idx < last_idx:  # 确保不是同一个点
                x1, y1 = contour[prev_idx][0]
                x2, y2 = contour[last_idx][0]
                
                point1 = (int(y1), int(x1))
                point2 = (int(y2), int(x2))
                
                if point1 < point2:
                    edge_pair = (point1, point2)
                else:
                    edge_pair = (point2, point1)
                
                if edge_pair not in edge_pairs:
                    edge_pairs.append(edge_pair)
                    
                    if len(edge_pairs) >= max_pairs:
                        return edge_pairs[:max_pairs]
        
        # 对于闭合轮廓，添加最后一个点和第一个点的连接
        if len(contour) > sampling_step:
            x1, y1 = contour[-1][0]
            x2, y2 = contour[0][0]
            point1 = (int(y1), int(x1))
            point2 = (int(y2), int(x2))
            
            if point1 < point2:
                edge_pair = (point1, point2)
            else:
                edge_pair = (point2, point1)
            
            if edge_pair not in edge_pairs:
                edge_pairs.append(edge_pair)
                
                if len(edge_pairs) >= max_pairs:
                    return edge_pairs[:max_pairs]
    
    # 如果使用采样步长后点对数量仍然不足，可以尝试减少步长或使用其他轮廓
    # 为了保证返回足够的点对，可以从原始边缘点中补充一些点对
    if len(edge_pairs) < max_pairs:
        # 构建剩余需要的点对数量
        remaining_pairs = max_pairs - len(edge_pairs)
        
        # 构建KDTree用于补充点对
        points_array = np.array(edge_points)
        if len(points_array) > 1:
            kdtree = scipy.spatial.KDTree(points_array)
            
            # 创建已处理点的集合，避免重复处理
            processed_points = set()
            
            # 计算平均点间距作为距离阈值参考
            total_dist = 0
            count = 0
            sample_size = min(100, len(points_array))
            for i in range(sample_size):
                dists, _ = kdtree.query(points_array[i], k=2)
                if isinstance(dists, (int, float)):
                    continue
                if len(dists) > 1:
                    total_dist += dists[1]
                    count += 1
            
            # 设置合理的距离阈值
            dist_threshold = (total_dist / count * 2) if count > 0 else 10.0
            
            # 尝试补充点对，优先选择尚未处理过的点
            available_points = list(range(len(points_array)))
            np.random.shuffle(available_points)
            
            for i in available_points:
                if len(edge_pairs) >= max_pairs:
                    break
                
                point = points_array[i]
                point_tuple = (int(point[0]), int(point[1]))
                
                # 如果点已经处理过，跳过
                if point_tuple in processed_points:
                    continue
                
                # 标记为已处理
                processed_points.add(point_tuple)
                
                # 查找最近的几个点，但限制查找范围
                k_neighbors = min(10, len(points_array) - 1)
                dists, idxs = kdtree.query(point, k=k_neighbors)
                
                # 处理k=1的情况
                if isinstance(dists, (int, float)):
                    dists = np.array([dists])
                    idxs = np.array([idxs])
                
                # 尝试连接在合理距离内的点
                valid_neighbors = [(d, idx) for d, idx in zip(dists, idxs) 
                                  if d > 0 and d <= dist_threshold]
                valid_neighbors.sort(key=lambda x: x[0])  # 按距离排序
                
                for dist, neighbor_idx in valid_neighbors:
                    neighbor_point = points_array[neighbor_idx]
                    neighbor_tuple = (int(neighbor_point[0]), int(neighbor_point[1]))
                    
                    # 确保点对顺序一致
                    if point_tuple < neighbor_tuple:
                        edge_pair = (point_tuple, neighbor_tuple)
                    else:
                        edge_pair = (neighbor_tuple, point_tuple)
                    
                    # 添加点对
                    if edge_pair not in edge_pairs:
                        edge_pairs.append(edge_pair)
                        # 标记邻居点也为已处理，避免重复
                        processed_points.add(neighbor_tuple)
                        break
    
    return edge_pairs[:max_pairs]

def save_edge_visualization(image: np.ndarray, mask: np.ndarray, edges: np.ndarray, save_path: str):
    """
    保存2D边缘可视化结果
    
    Args:
        image: 原始彩色图像
        mask: 掩码图像
        edges: 边缘掩码
        save_path: 保存路径
    """
    # 确保图像为uint8类型并转换为0-1浮点范围
    if image.dtype == np.float32 or image.dtype == np.float64:
        # 如果已经是浮点型，确保在0-1范围
        img_float = np.clip(image, 0, 1)
    else:
        # 如果是整型，转换为浮点型并归一化到0-1
        img_float = image.astype(np.float32) / 255.0
    
    # 创建彩色边缘图（0-1范围）
    edge_color = np.zeros_like(img_float)
    edge_color[edges > 0] = [1.0, 0.0, 0.0]  # 归一化红色
    
    # 在0-1范围内混合图像
    vis = img_float * 0.6 + edge_color * 0.4
    
    # 裁剪值范围并转换为uint8
    vis_uint8 = (np.clip(vis, 0, 1) * 255).astype(np.uint8)
    
    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(vis_uint8, cv2.COLOR_RGB2BGR))
    print(f"2D edge visualization saved to {save_path}")


def visualize_2d_edges(edges: np.ndarray, output_path: str = "output/2d_edges_only.png", 
                      edge_color: Tuple[int, int, int] = (255, 0, 0), 
                      background_color: Tuple[int, int, int] = (0, 0, 0)) -> None:
    """
    保存仅显示边缘的2D图像
    
    Args:
        edges: 边缘掩码
        output_path: 保存图像的路径，默认为"output/2d_edges_only.png"
        edge_color: 边缘颜色 (R, G, B)，默认为红色
        background_color: 背景颜色 (R, G, B)，默认为黑色
    """
    # 创建彩色图像
    h, w = edges.shape[:2]
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 设置背景颜色
    color_image[:, :] = background_color
    
    # 设置边缘颜色
    color_image[edges > 0] = edge_color
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
    print(f"2D edges only image saved to {output_path}")

def pixel_to_3d(pixel: Tuple[int, int], depth_map: np.ndarray, K: np.ndarray, depth_scale: float = 1.0) -> np.ndarray:
    """
    将2D像素坐标转换为3D空间坐标
    
    Args:
        pixel: 像素坐标 (y, x)
        depth_map: 深度图
        K: 相机内参矩阵
        depth_scale: 深度缩放因子（1.0 = 米单位，1000 = 毫米转米）
    
    Returns:
        point_3d: 3D坐标 [x, y, z] or None if invalid
    """
    y, x = pixel
    
    # 检查坐标是否在有效范围内
    if y < 0 or y >= depth_map.shape[0] or x < 0 or x >= depth_map.shape[1]:
        return None
    
    # 获取深度值
    depth = depth_map[y, x]
    
    # 检查深度值是否有效（非NaN、非inf且大于0）
    if not np.isfinite(depth) or depth <= 0:
        return None
    
    # 应用深度缩放因子
    depth = float(depth) * depth_scale
    
    # 计算3D点
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x_3d = (x - cx) * depth / fx
    y_3d = (y - cy) * depth / fy
    z_3d = depth
    
    return np.array([x_3d, y_3d, z_3d])

def project_edges_to_3d(edge_pairs_2d: List[Tuple[Tuple[int, int], Tuple[int, int]]], 
                        depth_map: np.ndarray, 
                        K: np.ndarray, 
                        pose: np.ndarray, 
                        depth_scale: float = 1.0) -> Tuple[Set[Tuple[int, int]], np.ndarray]:
    """
    将2D边缘点对投影到3D空间
    
    Args:
        edge_pairs_2d: 2D边缘点对列表
        depth_map: 深度图
        K: 相机内参矩阵
        pose: 相机外参矩阵 [4x4]
        depth_scale: 深度缩放因子
    
    Returns:
        Tuple[Set[Tuple[int, int]], np.ndarray]: 3D点索引对集合和3D点数组
    """
    # 存储3D点和对应的索引
    points_3d = []
    point_to_index = {}
    edge_pairs_3d = set()
    
    # 提取相机旋转和平移
    R = pose[:3, :3]
    T = pose[:3, 3]
    
    # 处理每个2D点对
    for pixel1, pixel2 in edge_pairs_2d:
        # 转换为3D点
        pt1_camera = pixel_to_3d(pixel1, depth_map, K, depth_scale)
        pt2_camera = pixel_to_3d(pixel2, depth_map, K, depth_scale)
        
        if pt1_camera is None or pt2_camera is None:
            continue
        
        # 转换到世界坐标系
        pt1_world = R @ pt1_camera + T
        pt2_world = R @ pt2_camera + T
        
        # 添加到点云并记录索引
        for pt in [pt1_world, pt2_world]:
            # 使用四舍五入来减少精度问题
            pt_key = tuple(np.round(pt, 3))
            if pt_key not in point_to_index:
                point_to_index[pt_key] = len(points_3d)
                points_3d.append(pt)
        
        # 创建3D点对
        pt1_key = tuple(np.round(pt1_world, 3))
        pt2_key = tuple(np.round(pt2_world, 3))
        idx1, idx2 = point_to_index[pt1_key], point_to_index[pt2_key]
        
        # 确保点对顺序一致
        if idx1 < idx2:
            edge_pairs_3d.add((idx1, idx2))
        else:
            edge_pairs_3d.add((idx2, idx1))
    
    return edge_pairs_3d, np.array(points_3d)

def match_3d_points_with_pointcloud(edge_points_3d: np.ndarray, 
                                   global_pointcloud: np.ndarray, 
                                   max_distance: float = 0.05) -> Dict[int, int]:
    """
    将投影得到的3D点与全局点云进行匹配
    
    Args:
        edge_points_3d: 从边缘投影得到的3D点 [N, 3]
        global_pointcloud: 全局点云 [M, 3]
        max_distance: 最大匹配距离
    
    Returns:
        edge_to_global: 边缘点索引到全局点云索引的映射 {edge_idx: global_idx}
    """
    # 构建KD树加速最近邻查找
    kdtree = scipy.spatial.KDTree(global_pointcloud)
    
    edge_to_global = {}
    
    # 查找每个边缘点的最近邻
    for edge_idx, point in enumerate(edge_points_3d):
        # 查询最近邻，处理k=1时返回标量的情况
        result = kdtree.query(point, k=1)
        
        # 确保返回的是数组形式
        if isinstance(result[0], (int, float)):
            # k=1的情况，返回的是标量
            distance = float(result[0])
            index = int(result[1])
        else:
            # 一般情况，返回的是数组
            distance = float(result[0])
            index = int(result[1])
        
        # 如果距离小于阈值，则认为匹配成功
        if distance < max_distance:
            edge_to_global[edge_idx] = index
    
    return edge_to_global

def convert_edge_pairs_to_global_indices(edge_pairs_3d: Set[Tuple[int, int]], 
                                        edge_to_global: Dict[int, int]) -> Set[Tuple[int, int]]:
    """
    将边缘点对转换为全局点云索引对
    
    Args:
        edge_pairs_3d: 边缘点索引对集合
        edge_to_global: 边缘点索引到全局点云索引的映射
    
    Returns:
        global_edge_pairs: 全局点云索引对集合
    """
    global_edge_pairs = set()
    
    for idx1, idx2 in edge_pairs_3d:
        # 检查两个点是否都匹配到了全局点云
        if idx1 in edge_to_global and idx2 in edge_to_global:
            g_idx1 = edge_to_global[idx1]
            g_idx2 = edge_to_global[idx2]
            
            # 确保顺序一致
            if g_idx1 < g_idx2:
                global_edge_pairs.add((g_idx1, g_idx2))
            else:
                global_edge_pairs.add((g_idx2, g_idx1))
    
    return global_edge_pairs

def process_mask_edges_for_pointcloud(mask: np.ndarray, 
                                     image: np.ndarray, 
                                     depth_map: np.ndarray, 
                                     K: np.ndarray, 
                                     pose: np.ndarray, 
                                     global_pointcloud: np.ndarray,
                                     max_points_per_frame: int = 500, 
                                     edge_thickness: int = 2, 
                                     depth_scale: float = 1.0,
                                     output_dir: str = None) -> Set[Tuple[int, int]]:
    """
    处理掩码边缘并投影到点云空间，返回点云索引对
    
    Args:
        mask: 掩码图像
        image: 原始彩色图像
        depth_map: 深度图
        K: 相机内参
        pose: 相机外参
        global_pointcloud: 全局点云
        max_points_per_frame: 每帧最大处理点数
        edge_thickness: 边缘厚度
        depth_scale: 深度缩放因子
        output_dir: 输出目录，用于保存2D边缘可视化图像
    
    Returns:
        global_edge_pairs: 全局点云索引对集合
    """
    # 提取边缘
    edge_mask = extract_mask_edges(mask, edge_thickness)
    
    # 保存2D边缘图像（如果提供了输出目录）
    if output_dir and image is not None:
        save_path = os.path.join(output_dir, "2d_edges.png")
        save_edge_visualization(image, mask, edge_mask, save_path)
    
    # 获取2D边缘点
    edge_points_2d = get_2d_edge_points(edge_mask, max_points_per_frame)
    
    # 生成2D边缘点对
    edge_pairs_2d = generate_edge_pairs_2d(edge_points_2d)
    
    # 投影到3D空间
    edge_pairs_3d, edge_points_3d = project_edges_to_3d(edge_pairs_2d, depth_map, K, pose, depth_scale)
    
    # 与全局点云匹配
    edge_to_global = match_3d_points_with_pointcloud(edge_points_3d, global_pointcloud)
    
    # 转换为全局点云索引对
    global_edge_pairs = convert_edge_pairs_to_global_indices(edge_pairs_3d, edge_to_global)
    
    print(f"Processed mask edges: {len(global_edge_pairs)} edge pairs found in pointcloud")
    
    return global_edge_pairs

def visualize_3d_edges(points_3d: np.ndarray, 
                      edge_pairs: Set[Tuple[int, int]], 
                      output_path: str = "output/3d_edges.png", 
                      point_size: float = 10.0,
                      line_width: float = 0.5,
                      point_color: str = 'b',
                      line_color: str = 'r') -> None:
    """
    保存3D边缘线段图像
    
    Args:
        points_3d: 3D点数组 [N, 3]
        edge_pairs: 边缘点对集合 {(idx1, idx2), ...}
        output_path: 保存图像的路径，默认为"output/3d_edges.png"
        point_size: 点的大小
        line_width: 线段宽度
        point_color: 点的颜色
        line_color: 线段的颜色
    """
    # 创建3D图形
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    ax.scatter(x, y, z, s=point_size, c=point_color, alpha=0.5)
    
    # 绘制边缘线段
    for idx1, idx2 in edge_pairs:
        x1, y1, z1 = points_3d[idx1]
        x2, y2, z2 = points_3d[idx2]
        ax.plot([x1, x2], [y1, y2], [z1, z2], linewidth=line_width, c=line_color)
    
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置坐标轴等比例
    max_range = max([max(x) - min(x), max(y) - min(y), max(z) - min(z)]) / 2
    mid_x = (max(x) + min(x)) / 2
    mid_y = (max(y) + min(y)) / 2
    mid_z = (max(z) + min(z)) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 设置标题
    ax.set_title(f'3D Edge Visualization: {len(edge_pairs)} edges, {len(points_3d)} points')
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"3D edge visualization saved to {output_path}")
    plt.close()


def visualize_3d_edges_with_open3d(points_3d: np.ndarray, 
                                  edge_pairs: Set[Tuple[int, int]],
                                  output_path: str = "output/3d_edges.ply",
                                  point_size: float = 3.0,
                                  line_width: float = 1.0,
                                  point_color: List[float] = [0, 0, 1],
                                  line_color: List[float] = [1, 0, 0]) -> None:
    """
    使用Open3D保存3D边缘线段数据（需要安装open3d库）
    
    Args:
        points_3d: 3D点数组 [N, 3]
        edge_pairs: 边缘点对集合 {(idx1, idx2), ...}
        output_path: 保存点云的路径（.ply格式），默认为"output/3d_edges.ply"
        point_size: 点的大小
        line_width: 线段宽度
        point_color: 点的颜色 [R, G, B]，范围0-1
        line_color: 线段的颜色 [R, G, B]，范围0-1
    """
    try:
        import open3d as o3d
        
        # 创建点云对象
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_3d)
        
        # 设置点云颜色
        point_cloud.colors = o3d.utility.Vector3dVector([point_color] * len(points_3d))
        
        # 创建线段集
        lines = []
        colors = []
        
        for idx1, idx2 in edge_pairs:
            lines.append([idx1, idx2])
            colors.append(line_color)
        
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_3d),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        # 保存数据
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        o3d.io.write_point_cloud(output_path.replace('.ply', '_points.ply'), point_cloud)
        o3d.io.write_line_set(output_path.replace('.ply', '_lines.ply'), line_set)
        print(f"3D edge data saved to {output_path.replace('.ply', '_points.ply')} and {output_path.replace('.ply', '_lines.ply')}")
    
    except ImportError:
        print("Open3D is not installed. Please install it with 'pip install open3d' to use this function.")
        # 如果没有Open3D，回退到matplotlib保存
        visualize_3d_edges(points_3d, edge_pairs, output_path.replace('.ply', '.png'), point_size, line_width)


if __name__ == "__main__":
    # 示例用法
    # 1. 创建一些示例3D点和边缘对
    sample_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0.5, 0.5, 1]
    ])
    
    sample_edges = {(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)}
    
    # 2. 保存3D边缘图像（默认保存到output目录）
    visualize_3d_edges(sample_points, sample_edges)
    
    # 3. 可以自定义保存路径
    visualize_3d_edges(sample_points, sample_edges, output_path="output/custom_3d_edges.png")
    
    # 4. 使用Open3D保存3D边缘数据（如果已安装）
    visualize_3d_edges_with_open3d(sample_points, sample_edges)
    
    # 5. 示例：创建简单的2D边缘掩码并保存2D边缘图像
    # 创建一个简单的边缘掩码示例
    h, w = 256, 256
    edges_2d = np.zeros((h, w), dtype=np.uint8)
    # 绘制一些线条作为边缘
    cv2.line(edges_2d, (50, 50), (200, 50), 255, 2)
    cv2.line(edges_2d, (200, 50), (200, 200), 255, 2)
    cv2.line(edges_2d, (200, 200), (50, 200), 255, 2)
    cv2.line(edges_2d, (50, 200), (50, 50), 255, 2)
    
    # 保存仅边缘的2D图像
    visualize_2d_edges(edges_2d)
    
    # 6. 创建简单的彩色图像和掩码用于测试save_edge_visualization
    image = np.ones((h, w, 3), dtype=np.uint8) * 255  # 白色背景
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (60, 60), (190, 190), 255, -1)  # 填充矩形
    
    # 保存带有图像背景的2D边缘可视化
    save_edge_visualization(image, mask, edges_2d, "output/2d_edges_with_background.png")
    
    # 7. 处理掩码边缘并保存2D边缘图像的完整流程示例
    # 创建简单的深度图和相机参数
    depth_map = np.ones((h, w), dtype=np.float32) * 1.0
    K = np.array([[300, 0, w//2], [0, 300, h//2], [0, 0, 1]], dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    global_pointcloud = np.random.rand(1000, 3)
    
    # 处理边缘并自动保存2D边缘图像
    global_edge_pairs = process_mask_edges_for_pointcloud(
        mask, image, depth_map, K, pose, global_pointcloud, 
        output_dir="output"
    )