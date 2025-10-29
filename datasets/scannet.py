import csv
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from utils.mask_edge_utils import process_mask_edges_for_pointcloud

from gaussian_splatting.utils.graphics_utils import focal2fov
from datasets.scannet_class_utils import SCANNET_COLOR_MAP_20

class ScanNetDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = "cuda"
        self.dtype = torch.float32
        self.scene_id = config["Dataset"]["scene_id"]
        self.basedir = os.path.join(config["Dataset"]["dataset_path"], self.scene_id)
        self.generated_floder = config["Dataset"]["generated_floder"]

        self.color_paths = sorted(glob.glob(os.path.join(self.basedir, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(glob.glob(os.path.join(self.basedir, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.num_frames = len(self.color_paths)

        # 21:
        # unannotated due to invalid gt label
        self.class_names = ['unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', \
                 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refridgerator', 'picture', 'cabinet', \
                 'otherfurniture']

        self.num_classes = len(self.class_names)
        self.class_colors = SCANNET_COLOR_MAP_20
        self.class_text_feat = np.load('./datasets/scannet/SCANNET_LABELS_20.npy')

        # list of numpy array [4, 4]
        self.poses = self.load_poses(os.path.join(self.basedir, 'pose'))
        self.gt_semantics = np.load(os.path.join('./datasets/scannet/3d_sem_ins', self.scene_id, 'semantic_labels.npy'))
        self.gt_instances = np.load(os.path.join('./datasets/scannet/3d_sem_ins', self.scene_id, 'instance_labels.npy'))

        # gt point clound
        self.gt_ply = os.path.join(self.basedir, '{}_vh_clean_2.ply'.format(self.scene_id))
        
        # mask
        self.mask_name = config["Dataset"]["mask_name"]
        self.mask_path = os.path.join(self.generated_floder, self.scene_id, self.mask_name, 'raw')

        # Camera prameters
        calibration = config["Dataset"]["Calibration"]
        self.resize_wh = calibration["resize_wh"]
        w_scale = float(self.resize_wh[0] / 640.0)
        h_scale = float(self.resize_wh[1] / 480.0)
        self.depth_scale = calibration["depth_scale"]
        self.ignore_edge = calibration["ignore_edge"]
        self.crop = calibration["crop"]
        self.fx = calibration["fx"] * w_scale
        self.fy = calibration["fy"] * h_scale
        self.cx = calibration["cx"] * w_scale
        self.cy = calibration["cy"] * h_scale
        self.width = int(calibration["width"] * w_scale)
        self.height = int(calibration["height"] * h_scale)
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )

    def index_to_name(self, index):
        name = os.path.basename(self.color_paths[index])[:-4]    # 0.png
        return name       

    def load_depth(self, index):
        depth = cv2.imread(self.depth_paths[index], cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, self.resize_wh, interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(np.float32) / self.depth_scale
        return depth

    def load_image(self, index):
        rgb = cv2.imread(self.color_paths[index], -1)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, self.resize_wh)
        return rgb / 255.0

    def load_mask(self, index):
        name = self.index_to_name(index)
        mask_path = os.path.join(self.mask_path, '{}.png'.format(name))
        mask = cv2.imread(mask_path, -1).astype(np.float32)
        return mask

    def load_data_for_seg(self, kf_inter):
        poses = []
        depths = []
        intrinsics = []
        masks = []

        for i in range(0, self.num_frames, kf_inter):
            poses.append(self.poses[i])
            depths.append(self.load_depth(i))
            intrinsics.append(self.K)
            masks.append(self.load_mask(i))
        
        poses = np.stack(poses, 0) # [N_images, 4, 4]
        depths = np.stack(depths, 0) # [N_images, H, W]
        intrinsics = np.stack(intrinsics, 0) # [N_images, 3, 3]
        masks = np.stack(masks, 0) # [N_images, H, W]

        print('poses: ', poses.shape)
        print('depths: ', depths.shape)
        print('intrinsics: ', intrinsics.shape)
        print('masks: ', masks.shape)
        
        # 缓存这些数据，以便在get_mask_edges中使用
        self._seg_poses = poses
        self._seg_depths = depths
        self._seg_intrinsics = intrinsics
        self._seg_masks = masks
        self._seg_kf_inter = kf_inter

        return poses, depths, intrinsics, masks

    def get_frame(self, index):
        # rgb, depth, pose
        rgb = torch.from_numpy(self.load_image(index)).float()
        depth = torch.from_numpy(self.load_depth(index)).float()
        rgb = rgb
        depth = depth

        pose = self.poses[index]
        c2w = torch.from_numpy(pose).float()
        w2c = torch.from_numpy(np.linalg.inv(pose)).float()

        ret = {
            "K": self.K,    # [3, 3]
            "c2w": c2w,     # [4, 4]
            "w2c": w2c,     # [4, 4]
            "rgb": rgb,     # [H, W, 3]
            "depth": depth, # [H, W]
            "valid": True,  # bool: 
        }

        return ret
   
    def __len__(self):
        return self.num_frames
  
    def __getitem__(self, index):
        ret = self.get_frame(index)
        return ret

    def load_poses(self, path):
        poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')), key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            poses.append(c2w)

        return poses
    
    def get_mask_edges(self, kf_inter=1, max_points_per_frame=500, global_pointcloud=None):
        """
        从掩码图像中提取边缘信息，生成3D点对集合
        
        Args:
            kf_inter: 关键帧间隔
            max_points_per_frame: 每帧处理的最大边缘点数
            global_pointcloud: 全局点云数据 [N, 3]
            
        Returns:
            mask_edges: 掩码边缘点对集合 {(p1, p2), ...}
        """
        # 检查是否有预计算的边缘信息
        edge_file = os.path.join(self.mask_path, 'mask_edges.npy')
        if os.path.exists(edge_file):
            try:
                edges = np.load(edge_file)
                mask_edges = set(tuple(edge) for edge in edges)
                print(f"Loaded {len(mask_edges)} mask edges from {edge_file}")
                return mask_edges
            except:
                print(f"Failed to load mask edges from {edge_file}")
        
        # 检查是否有必要的数据进行2D到3D的投影
        if global_pointcloud is None:
            print("Warning: No global pointcloud provided. Cannot perform 2D to 3D edge projection.")
            return set()
        
        # 检查是否已经加载了分割所需的数据
        if not hasattr(self, '_seg_poses'):
            print("Loading segmentation data...")
            self.load_data_for_seg(kf_inter)
        
        print("Processing mask edges using 2D to 3D projection...")
        mask_edges = set()
        
        # 处理每个关键帧
        for i in range(0, len(self._seg_poses), kf_inter):
            print(f"Processing frame {i}/{len(self._seg_poses)}")
            
            # 获取当前帧的数据
            pose = self._seg_poses[i]
            depth = self._seg_depths[i]
            K = self._seg_intrinsics[i]
            mask = self._seg_masks[i]
            
            # 加载对应的彩色图像
            image_index = i * self._seg_kf_inter
            if image_index < len(self.color_paths):
                image = self.load_image(image_index)
            else:
                # 如果没有对应的图像，使用空白图像
                image = np.zeros((self.height, self.width, 3), dtype=np.float32)
            
            # 处理当前帧的掩码边缘
            frame_edge_pairs = process_mask_edges_for_pointcloud(
                mask=mask,
                image=image,
                depth_map=depth,
                K=K,
                pose=pose,
                global_pointcloud=global_pointcloud,
                max_points_per_frame=max_points_per_frame,
                edge_thickness=2,
                depth_scale=1.0  # ScanNet数据集通常使用米单位深度
            )
            
            # 添加到全局边缘点对集合
            mask_edges.update(frame_edge_pairs)
        
        # 保存处理结果以便下次使用
        try:
            os.makedirs(os.path.dirname(edge_file), exist_ok=True)
            np.save(edge_file, list(mask_edges))
            print(f"Saved {len(mask_edges)} mask edges to {edge_file}")
        except:
            print(f"Failed to save mask edges to {edge_file}")
        
        print(f"Total generated mask edges: {len(mask_edges)}")
        return mask_edges

