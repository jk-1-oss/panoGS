import os
import numpy as np
import cv2
import torch
import argparse
from datasets.scannet import ScanNetDataset
from utils.mask_edge_utils import extract_mask_edges, save_edge_visualization
import open3d as o3d

def parse_args():
    parser = argparse.ArgumentParser(description="Test 2D to 3D edge projection")
    parser.add_argument("--config", type=str, default="configs/scannet/scene0000_00.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--frame-index", type=int, default=0, 
                        help="Frame index to test")
    parser.add_argument("--output-dir", type=str, default="outputs/edge_test", 
                        help="Output directory for visualization")
    return parser.parse_args()

def load_config(config_path):
    """加载配置文件"""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def visualize_3d_edges(points, edges, output_path):
    """可视化3D边缘"""
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色点云
    
    # 创建边缘线集
    lines = []
    colors = []
    for p1, p2 in edges:
        lines.append([p1, p2])
        colors.append([1, 0, 0])  # 红色边缘
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # 保存为PLY文件
    o3d.io.write_point_cloud(os.path.join(output_path, "pointcloud.ply"), pcd)
    o3d.io.write_line_set(os.path.join(output_path, "edges.ply"), line_set)
    
    # 可视化
    o3d.visualization.draw_geometries([pcd, line_set], window_name="3D Edges Visualization")

def main():
    # 解析参数
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置和数据集
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    print("Loading dataset...")
    dataset = ScanNetDataset(config)
    
    # 加载指定帧的数据
    print(f"Loading frame {args.frame_index}...")
    frame = dataset.get_frame(args.frame_index)
    image = frame["rgb"].numpy()
    depth = frame["depth"].numpy()
    K = frame["K"]
    pose = frame["c2w"].numpy()
    
    # 加载掩码
    mask = dataset.load_mask(args.frame_index)
    
    # 提取2D边缘
    print("Extracting 2D edges from mask...")
    edges_2d = extract_mask_edges(mask)
    
    # 可视化2D边缘
    edge_vis_path = os.path.join(args.output_dir, f"edge_visualization_{args.frame_index}.png")
    save_edge_visualization(image, mask, edges_2d, edge_vis_path)
    
    # 尝试使用3D投影功能
    print("Testing 3D projection functionality...")
    from utils.mask_edge_utils import process_mask_edges_for_pointcloud
    
    # 创建一个简单的点云（在实际应用中，这会是来自gaussians的数据）
    # 这里我们从深度图生成一些3D点
    H, W = depth.shape
    points = []
    
    # 采样一些点
    step = 10
    for y in range(0, H, step):
        for x in range(0, W, step):
            if depth[y, x] > 0:
                # 转换为3D点
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                x_3d = (x - cx) * depth[y, x] / fx
                y_3d = (y - cy) * depth[y, x] / fy
                z_3d = depth[y, x]
                
                # 转换到世界坐标系
                point_camera = np.array([x_3d, y_3d, z_3d, 1.0])
                point_world = pose @ point_camera
                points.append(point_world[:3])
    
    if len(points) > 0:
        points = np.array(points)
        print(f"Generated {len(points)} 3D points from depth map")
        
        # 处理掩码边缘
        edge_pairs = process_mask_edges_for_pointcloud(
            mask=mask,
            image=image,
            depth_map=depth,
            K=K,
            pose=pose,
            global_pointcloud=points,
            max_points_per_frame=200,
            edge_thickness=2,
            depth_scale=1.0  # 假设测试数据使用米单位深度
        )
        
        # 可视化3D边缘
        if len(edge_pairs) > 0:
            print(f"Projected {len(edge_pairs)} edge pairs to 3D space")
            visualize_3d_edges(points, edge_pairs, args.output_dir)
        else:
            print("No edge pairs were successfully projected to 3D space")
    else:
        print("Could not generate 3D points from depth map")
    
    print("Test completed!")

if __name__ == "__main__":
    main()