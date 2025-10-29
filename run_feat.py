import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from decoders.dataset import Decoder_dataset
import argparse
from utils.config_utils import load_config
import yaml
from datasets.load_func import load_dataset
from datetime import datetime
import numpy as np
from argparse import ArgumentParser

from decoders.decoder import FeatureDecoder

torch.autograd.set_detect_anomaly(True)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    res = {'Total': total_num/1024.0/1024.0, 'Trainable': trainable_num/1024.0/1024.0}
    print(res)
    return res

def coordinates(voxel_dim, device: torch.device, flatten=True):
    if type(voxel_dim) is int:
        nx = ny = nz = voxel_dim
    else:
        nx, ny, nz = voxel_dim[0], voxel_dim[1], voxel_dim[2]
    x = torch.arange(0, nx, dtype=torch.long, device=device)
    y = torch.arange(0, ny, dtype=torch.long, device=device)
    z = torch.arange(0, nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")

    if not flatten:
        return torch.stack([x, y, z], dim=-1)

    return torch.stack((x.flatten(), y.flatten(), z.flatten()))

def cos_loss(network_output, gt, weight):
    sim = torch.cosine_similarity(network_output, gt, dim=1)
    loss = (1 - sim) * weight
    return loss.mean()

def smoothness(decoder, bounding_box, sample_points=32, voxel_size=0.1, margin=0.05):
    '''
    Smoothness loss of feature grid
    '''
    volume = bounding_box[:, 1] - bounding_box[:, 0]

    grid_size = (sample_points-1) * voxel_size
    offset_max = bounding_box[:, 1] - bounding_box[:, 0] - grid_size - 2 * margin

    offset = torch.rand(3).to(offset_max) * offset_max + margin
    coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
    pts = (coords + torch.rand((1,1,1,3)).to(volume)) * voxel_size + bounding_box[:, 0] + offset

    # shape: [sample_points, sample_points, sample_points, 3]
    pts_tcnn = (pts - bounding_box[:, 0]) / (bounding_box[:, 1] - bounding_box[:, 0])
    pts_tcnn_ = pts_tcnn.reshape(-1, 3)
    
    # shape: [sample_points, sample_points, sample_points, n_dim or n_embed_dim]
    feat = decoder(pts_tcnn_, return_embed=True)
    # print('feat: ', feat.shape)
    feat = feat.reshape(*pts_tcnn.shape[:-1], -1)

    tv_x = torch.pow(feat[1:,...] - feat[:-1,...], 2).sum()
    tv_y = torch.pow(feat[:,1:,...] - feat[:,:-1,...], 2).sum()
    tv_z = torch.pow(feat[:,:,1:,...] - feat[:,:,:-1,...], 2).sum()
    loss = (tv_x + tv_y + tv_z)/ (sample_points**3)

    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)
    config = load_config(args.config)

    # params
    scene_id = config["Dataset"]["scene_id"]
    src_path = config["Dataset"]["dataset_path"].split("/")
    generated_path = config["Dataset"]["generated_floder"]
    num_epochs = args.num_epochs

    if config['Dataset']['type'] == 'replica':
        save_dir = os.path.join(config["Results"]["save_dir"], src_path[-1], scene_id)
    elif config['Dataset']['type'] == 'scannet':
        save_dir = os.path.join(config["Results"]["save_dir"], src_path[-2], scene_id)
    else:
        print('Dataset type should be replica or scannet')
        exit()

    # data path
    ply_path = os.path.join(save_dir, 'point_cloud', 'point_cloud.ply')
    feat_path = os.path.join(generated_path, scene_id, 'feats_weights', 'features.npy')
    weight_path = os.path.join(generated_path, scene_id, 'feats_weights', 'weights.npy')
    
    config["Results"]["save_dir"] = save_dir
    bounding_box = torch.from_numpy(np.array(config['scene']['bound']))

    # load decoder，dataset
    dataset = load_dataset(config=config)
    train_dataset = Decoder_dataset(config, ply_path, feat_path, weight_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, num_workers=16, drop_last=False)
    num_batch = len(train_loader)
    lr = args.lr #0.001 # 0.001 0.05
    
    # decoder
    decoder = FeatureDecoder(config).cuda()
    
    # 收集所有可训练参数，包括新添加的位置编码网络参数和特征空间优化参数
    trainable_parameters = []
    
    # 添加feature_net参数
    trainable_parameters.append({
        'params': decoder.feature_net.parameters(), 
        'weight_decay': 1e-6, 
        'lr': lr
    })
    
    # 添加encodings参数
    trainable_parameters.append({
        'params': decoder.encodings.parameters(), 
        'eps': 1e-15, 
        'lr': lr
    })
    
    # 添加位置编码相关参数（如果存在）
    if hasattr(decoder, 'position_mlp'):
        trainable_parameters.append({
            'params': decoder.position_mlp.parameters(),
            'weight_decay': 1e-6,
            'lr': lr * 0.1  # 为位置编码网络使用较小的学习率
        })
    
    if hasattr(decoder, 'hierarchical_weights'):
        trainable_parameters.append({
            'params': [decoder.hierarchical_weights],
            'lr': lr
        })
    
    if hasattr(decoder, 'spatial_importance') and hasattr(decoder, 'importance_scale'):
        trainable_parameters.append({
            'params': list(decoder.spatial_importance.parameters()) + [decoder.importance_scale],
            'weight_decay': 1e-6,
            'lr': lr * 0.1  # 为空间注意力网络使用较小的学习率
        })
    
    # 添加特征空间优化相关参数（如果启用）
    if hasattr(decoder, 'optimize_feature_space') and decoder.optimize_feature_space:
        # 跨分辨率特征交互参数
        if hasattr(decoder, 'cross_res_interaction'):
            trainable_parameters.append({
                'params': list(decoder.cross_res_interaction.interaction_weights) + 
                          list(decoder.cross_res_interaction.gate_weights),
                'weight_decay': 1e-6,
                'lr': lr * 0.5  # 为交互权重使用中等学习率
            })
        
        # 特征注意力机制参数
        if hasattr(decoder, 'feature_attention'):
            trainable_parameters.append({
                'params': decoder.feature_attention.parameters(),
                'weight_decay': 1e-6,
                'lr': lr
            })
        
        # 特征分解与重组参数
        if hasattr(decoder, 'feature_decomposition'):
            trainable_parameters.append({
                'params': list(decoder.feature_decomposition.projections.parameters()) + 
                          list(decoder.feature_decomposition.recombinations.parameters()) + 
                          [decoder.feature_decomposition.subspace_weights],
                'weight_decay': 1e-6,
                'lr': lr * 0.5  # 为分解重组参数使用中等学习率
            })
    optimizer = torch.optim.Adam(trainable_parameters, betas=(0.9, 0.99))
    
    decoder_save_dir = os.path.join(save_dir, 'decoder')
    os.makedirs(decoder_save_dir, exist_ok=True)

    # training
    for epoch in tqdm(range(num_epochs)):
        decoder.train()
        batch_i = 1
        for pts, features, weight in train_loader:
            weight = weight.detach().cuda()
            xyz, feat = pts, features.cuda()
            outputs = decoder(xyz)
            
            cosloss = cos_loss(outputs, feat, weight)
            regularization = 0.0
            # regularization = smoothness(decoder, bounding_box)
            loss = cosloss 
            # loss += 100 * regularization
            
            print('epoch: ', epoch, ' batch: {}/{}'.format(batch_i, num_batch), ' cosloss: ', cosloss, 'regularization: ', 0.1 * regularization)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_i += 1

    print('save ckpt to: ', f'{decoder_save_dir}/ckpt.pth')
    torch.save(decoder.state_dict(), f'{decoder_save_dir}/ckpt.pth')