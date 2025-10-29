import torch
import torch.nn as nn
import numpy as np
from nerfstudio.field_components import encodings
import scipy.spatial as spatial

def get_encoder(desired_resolution=512, num_components=16):
    '''
    https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/field_components/encodings.py
    
    https://docs.nerf.studio/reference/api/field_components/encodings.html#nerfstudio.field_components.encodings.TriplaneEncoding
    Learned triplane encoding
    The encoding at [i,j,k] is an n dimensional vector corresponding to the element-wise product of the three n dimensional vectors at plane_coeff[i,j], plane_coeff[i,k], and plane_coeff[j,k].
    This allows for marginally more expressivity than the TensorVMEncoding, and each component is self standing and symmetrical, 
    unlike with VM decomposition where we needed one component with a vector along all the x, y, z directions for symmetry.
    This can be thought of as 3 planes of features perpendicular to the x, y, and z axes, respectively and intersecting at the origin, 
    and the encoding being the element-wise product of the element at the projection of [i, j, k] on these planes.
    The use for this is in representing a tensor decomp of a 4D embedding tensor: (x, y, z, feature_size)
    This will return a tensor of shape (bs:…, num_components)
    TriplaneEncoding(resolution: int = 32, num_components: int = 64, init_scale: float = 0.1, reduce: Literal['sum', 'product'] = 'sum')
    
    plane_coef: Float[Tensor, "3 num_components resolution resolution"]
    Args:
    resolution: Resolution of grid.
    num_components: The number of scalar triplanes to use (ie: output feature size)
    init_scale: The scale of the initial values of the planes
    product: Whether to use the element-wise product of the planes or the sum
    '''

    embed = encodings.TriplaneEncoding(resolution=desired_resolution, num_components=num_components, init_scale=0.1, reduce='sum',)
    out_dim = embed.get_out_dim()

    return embed, out_dim


class FeatureAttention(nn.Module):
    def __init__(self, in_dim):
        super(FeatureAttention, self).__init__()
        # 重新设计为更高效的通道注意力机制
        # 在这里，Q和K是用于计算特征通道间的重要性权重
        self.attention_dim = 4  # 中间维度，用于计算通道重要性
        
        # Q和K用于计算通道间的相关性
        self.query_key = nn.Linear(in_dim, self.attention_dim)
        
        # 直接使用MLP作为注意力门控机制
        self.attention_gate = nn.Sequential(
            nn.Linear(in_dim, self.attention_dim),
            nn.ReLU(),
            nn.Linear(self.attention_dim, in_dim),
            nn.Sigmoid()
        )
        
        # 残差连接的投影层
        self.out = nn.Linear(in_dim, in_dim)
        
    def forward(self, x):
        # x: [B, D] - 点云特征，B是点的数量，D是特征维度
        
        # 计算通道注意力权重
        # 这里的注意力用于自适应调整每个通道的重要性
        attention_weights = self.attention_gate(x)  # [B, D]
        
        # 应用注意力权重到特征上
        # 这种实现更高效，且能保持维度一致性
        attended = x * attention_weights  # [B, D]
        
        # 通过投影层并应用残差连接
        out = self.out(attended)  # [B, D]
        
        return x + out  # 残差连接


class MaskGuidedAttention(nn.Module):
    def __init__(self, in_dim, spatial_weight=0.3):
        super(MaskGuidedAttention, self).__init__()
        self.in_dim = in_dim
        self.spatial_weight = spatial_weight
        
        # 基础注意力层
        self.attention_layer = FeatureAttention(in_dim=in_dim)
        
        # 空间权重投影层
        self.spatial_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 掩码边缘权重，在运行时设置
        self.mask_edge_weights = None
        
    def set_mask_edges(self, mask_edges, points):
        """
        设置掩码边缘信息
        
        Args:
            mask_edges: 掩码边缘点对集合
            points: 点云坐标
        """
        import scipy.spatial
        
        # 初始化掩码边缘权重数组
        num_points = points.shape[0] if torch.is_tensor(points) else len(points)
        self.mask_edge_weights = torch.zeros(num_points, device='cuda')
        
        # 方法1: 如果有提供的mask_edges
        if mask_edges and len(mask_edges) > 0:
            # 统计每个点出现在多少个边缘对中
            edge_count = {}
            for p1, p2 in mask_edges:
                if 0 <= p1 < num_points:
                    edge_count[p1] = edge_count.get(p1, 0) + 1
                if 0 <= p2 < num_points:
                    edge_count[p2] = edge_count.get(p2, 0) + 1
            
            # 计算边缘权重
            if edge_count:
                max_count = max(edge_count.values())
                for p_idx, count in edge_count.items():
                    # 权重与边缘出现次数成正比
                    self.mask_edge_weights[p_idx] = count / max_count
                print(f"Set mask edge weights for {len(edge_count)} points from provided edges")
        else:
            # 方法2: 如果没有提供mask_edges，使用几何特征计算边缘
            print("No mask edges provided, computing geometric edges...")
            
            # 转换为numpy数组以使用KDTree
            points_np = points.cpu().numpy() if torch.is_tensor(points) else points
            
            # 构建KD树用于查找近邻
            tree = scipy.spatial.KDTree(points_np)
            
            # 为每个点计算法向量变化率作为边缘度量
            # 随机采样部分点以避免计算量过大
            sample_size = min(10000, num_points)
            sample_indices = np.random.choice(num_points, sample_size, replace=False)
            
            # 计算每个采样点的法向量变化
            for i, idx in enumerate(sample_indices):
                # 查找k个最近邻
                distances, neighbors = tree.query(points_np[idx], k=10)
                
                # 计算点与其邻居之间的距离差异
                if len(distances) > 2:
                    # 计算距离变化率
                    dist_diff = np.std(distances[1:]) / np.mean(distances[1:]) if np.mean(distances[1:]) > 0 else 0
                    
                    # 如果距离变化率大，则认为是边缘点
                    if dist_diff > 0.5:
                        self.mask_edge_weights[idx] = min(1.0, dist_diff)
            
            print(f"Computed geometric edge weights for {sample_size} sample points")
    
    def forward(self, x, point_indices=None):
        """
        应用掩码引导的特征注意力
        
        Args:
            x: 输入特征 [B, C]
            point_indices: 点索引数组 [B]
            
        Returns:
            增强后的特征
        """
        # 基础特征注意力
        base_attention = self.attention_layer(x)
        
        # 如果没有掩码信息或点索引，直接返回基础注意力结果
        if self.mask_edge_weights is None or point_indices is None:
            return base_attention
        
        try:
            # 获取点的掩码权重
            batch_size = x.shape[0]
            spatial_weights = torch.zeros(batch_size, 1, device=x.device)
            
            # 向量化操作以提高效率
            valid_indices = (point_indices >= 0) & (point_indices < len(self.mask_edge_weights))
            spatial_weights[valid_indices] = self.mask_edge_weights[point_indices[valid_indices]].unsqueeze(1)
            
            # 通过MLP处理空间权重
            spatial_att = self.spatial_proj(spatial_weights)
            
            # 应用空间注意力到特征上
            # 对于边缘区域，增强特征的区分度
            spatial_enhanced = x * (1 + self.spatial_weight * spatial_att)
            
            # 结合基础注意力和空间增强
            combined = base_attention + self.spatial_weight * (spatial_enhanced - x)
            
            return combined
        except Exception as e:
            print(f"Error in MaskGuidedAttention forward: {e}")
            # 出错时返回基础注意力结果
            return base_attention


class CrossResolutionInteraction(nn.Module):
    """
    跨分辨率特征交互模块，增强不同分辨率特征间的信息流动
    """
    def __init__(self, num_resolutions, feat_dim):
        super(CrossResolutionInteraction, self).__init__()
        self.num_resolutions = num_resolutions
        self.interaction_weights = nn.ParameterList([
            nn.Parameter(torch.eye(feat_dim)) for _ in range(num_resolutions)
        ])
        self.gate_weights = nn.ParameterList([
            nn.Parameter(torch.ones(feat_dim)) for _ in range(num_resolutions)
        ])
        self.act = nn.Sigmoid()
    
    def forward(self, features):
        # features: list of features from different resolutions
        if len(features) != self.num_resolutions:
            raise ValueError(f"Expected {self.num_resolutions} features, got {len(features)}")
        
        # 计算交互后的特征
        interacted = []
        for i, feat in enumerate(features):
            # 基础特征是当前分辨率的特征
            combined = feat
            
            # 与其他分辨率特征进行交互
            for j, other_feat in enumerate(features):
                if i != j:
                    # 使用可学习权重矩阵进行特征转换
                    transformed = torch.matmul(other_feat, self.interaction_weights[j])
                    # 使用门控机制控制信息流动
                    gate = self.act(self.gate_weights[j])
                    combined = combined + gate * transformed
            
            interacted.append(combined)
        
        return interacted


class FeatureDecomposition(nn.Module):
    """
    特征分解模块，将特征分解为多个子空间，增强表示能力
    """
    def __init__(self, in_dim, num_subspaces=4):
        super(FeatureDecomposition, self).__init__()
        self.num_subspaces = num_subspaces
        self.subspace_dim = in_dim // num_subspaces
        
        # 子空间投影矩阵
        self.projections = nn.ModuleList([
            nn.Linear(in_dim, self.subspace_dim) for _ in range(num_subspaces)
        ])
        
        # 子空间重组矩阵
        self.recombinations = nn.ModuleList([
            nn.Linear(self.subspace_dim, in_dim) for _ in range(num_subspaces)
        ])
        
        # 子空间重要性权重
        self.subspace_weights = nn.Parameter(torch.ones(num_subspaces))
        
    def forward(self, x):
        # 将特征分解到多个子空间
        subspace_features = []
        for proj in self.projections:
            subspace_feat = proj(x)
            # 子空间内归一化
            subspace_feat = nn.functional.normalize(subspace_feat, dim=-1)
            subspace_features.append(subspace_feat)
        
        # 重组特征
        soft_weights = torch.softmax(self.subspace_weights, dim=0)
        recombined = 0
        for i, (sub_feat, recomb) in enumerate(zip(subspace_features, self.recombinations)):
            recombined = recombined + soft_weights[i] * recomb(sub_feat)
        
        return x + recombined  # 残差连接


class FeatureNet(nn.Module):
    def __init__(self, input_ch, dims):
        super(FeatureNet, self).__init__()
        self.input_ch = input_ch
        self.latent_dims = dims
        self.model = self.get_model()
        # 添加特征归一化层
        self.norm_layers = nn.ModuleList()
        current_dim = input_ch
        for dim in dims:
            self.norm_layers.append(nn.LayerNorm(current_dim))
            current_dim = dim
    
    def forward(self, input_feat):
        x = input_feat
        for i, layer in enumerate(self.model):
            # 在每一层线性变换前应用层归一化
            if i % 2 == 0 and i < len(self.norm_layers):
                x = self.norm_layers[i//2](x)
            x = layer(x)
        return x
    
    def get_model(self):
        net =  []
        for i in range(len(self.latent_dims)):
            if i == 0:
                in_dim = self.input_ch
            else:
                in_dim = self.latent_dims[i-1]
            
            out_dim = self.latent_dims[i]
            
            net.append(nn.Linear(in_dim, out_dim, bias=False))

            if i != (len(self.latent_dims) - 1):
                net.append(nn.ReLU(inplace=True))

        return nn.Sequential(*nn.ModuleList(net))

class FeatureDecoder(nn.Module):
    def __init__(self, config):
        super(FeatureDecoder, self).__init__()
        self.config = config
        self.latent_dims = config['decoder']['latent_dims']
        self.bounding_box = torch.from_numpy(np.array(self.config['scene']['bound']))
        dim_max = (self.bounding_box[:,1] - self.bounding_box[:,0]).max()

        self.resolutions = []
        for res in self.config['decoder']['resolutions']:
            res_int = int(dim_max / res)
            self.resolutions.append(res_int)
    
        self.encodings = torch.nn.ModuleList()
        # multi-resolution
        # input of triplane should be in range [0, resolution]
        self.num_components = config['decoder']['num_components']
        for res in self.resolutions:
            encoding, embed_dim = get_encoder(desired_resolution=res, num_components=self.num_components)
            self.encodings.append(encoding)

            # we performan add for different resolution, so the embed_dim should be same 
            self.embed_dim = embed_dim 
        
        # 添加位置编码改进
        self.position_encoding_type = config['decoder'].get('position_encoding', 'linear')
        
        # 1. 学习的非线性映射
        if self.position_encoding_type == 'learned_nonlinear':
            self.position_mlp = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            )
        
        # 2. 分层位置编码
        elif self.position_encoding_type == 'hierarchical':
            self.hierarchical_levels = config['decoder'].get('hierarchical_levels', 4)
            self.hierarchical_scales = torch.exp2(torch.arange(self.hierarchical_levels))
            self.hierarchical_weights = nn.Parameter(torch.ones(self.hierarchical_levels))
        
        # 3. 空间感知归一化 - 高斯注意力权重
        elif self.position_encoding_type == 'spatial_aware':
            self.spatial_importance = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            self.importance_scale = nn.Parameter(torch.tensor(1.0))

        # 添加特征表示空间优化
        self.optimize_feature_space = config['decoder'].get('optimize_feature_space', True)
        
        if self.optimize_feature_space:
            # 1. 跨分辨率特征交互
            self.cross_res_interaction = CrossResolutionInteraction(
                num_resolutions=len(self.resolutions), 
                feat_dim=self.embed_dim
            )
            
            # 2. 特征注意力机制
            self.feature_attention = FeatureAttention(in_dim=self.embed_dim)
            
            # 3. 特征分解与重组
            self.feature_decomposition = FeatureDecomposition(
                in_dim=self.embed_dim, 
                num_subspaces=config['decoder'].get('num_subspaces', 4)
            )
            
            # 4. 掩码引导的特征注意力
            self.use_mask_attention = config['decoder'].get('use_mask_attention', True)
            if self.use_mask_attention:
                spatial_weight = config['decoder'].get('mask_spatial_weight', 0.3)
                self.mask_attention = MaskGuidedAttention(
                    in_dim=self.embed_dim, 
                    spatial_weight=spatial_weight
                )
                print('Mask-guided attention enabled with spatial weight:', spatial_weight)

        print('Parametric encoding resolutions: ', self.resolutions)
        print('Parametric encoding dimensions: ', self.embed_dim)
        print('Position encoding type: ', self.position_encoding_type)
        print('Feature space optimization: ', self.optimize_feature_space)

        self.feature_net = FeatureNet(input_ch=self.embed_dim, dims=self.latent_dims)
    
    def normalize_position(self, pos):
        """归一化位置坐标，支持多种编码策略"""
        # 基础线性归一化
        normalized_pos = (pos - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        
        if self.position_encoding_type == 'linear':
            # 原始的线性归一化
            return normalized_pos
        
        elif self.position_encoding_type == 'learned_nonlinear':
            # 1. 学习的非线性映射
            # 先进行线性归一化，然后通过MLP进行非线性变换
            transformed_pos = self.position_mlp(normalized_pos)
            # 确保输出仍在[0,1]范围内
            transformed_pos = torch.sigmoid(transformed_pos)
            return transformed_pos
        
        elif self.position_encoding_type == 'hierarchical':
            # 2. 分层位置编码
            # 在多个分辨率级别上处理位置信息
            device = normalized_pos.device
            scales = self.hierarchical_scales.to(device)
            weights = torch.softmax(self.hierarchical_weights, dim=0).to(device)
            
            hierarchical_pos = 0
            for i in range(self.hierarchical_levels):
                # 对每个级别使用不同的缩放和权重
                level_pos = (torch.sin(normalized_pos * scales[i] * torch.pi) + 1) / 2
                hierarchical_pos = hierarchical_pos + weights[i] * level_pos
            
            return hierarchical_pos
        
        elif self.position_encoding_type == 'spatial_aware':
            # 3. 空间感知归一化
            # 计算位置的重要性权重
            importance = self.spatial_importance(pos) * self.importance_scale
            # 根据重要性调整归一化
            # 重要区域的位置将被扩展，非重要区域将被压缩
            scale_factor = 0.5 + importance * 0.5  # 缩放因子范围 [0.5, 1.0]
            
            # 调整归一化方式，重要区域获得更高分辨率
            center = (self.bounding_box[:, 1] + self.bounding_box[:, 0]) / 2
            range_half = (self.bounding_box[:, 1] - self.bounding_box[:, 0]) / 2
            
            # 计算相对于中心的偏移并应用重要性缩放
            offset = pos - center
            scaled_offset = offset * (1 - scale_factor)  # 重要区域的偏移被缩小，相当于分辨率提高
            adjusted_pos = (scaled_offset + center - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
            
            return adjusted_pos
        
        else:
            # 默认返回线性归一化
            return normalized_pos

    def get_embed(self, pos):
        pos = self.normalize_position(pos)
        pos = pos.float().cuda()

        multi_feat = []

        for i in range(len(self.encodings)):
            embed = self.encodings[i](pos).detach()
            multi_feat.append(embed)
        
        # 应用特征空间优化（如果启用）
        if self.optimize_feature_space and hasattr(self, 'cross_res_interaction'):
            multi_feat = self.cross_res_interaction(multi_feat)

        return multi_feat
            
    def set_mask_attention(self, mask_edges, points):
        """
        设置掩码边缘信息用于特征注意力计算
        
        Args:
            mask_edges: 掩码边缘点对集合
            points: 点云坐标
        """
        if hasattr(self, 'mask_attention') and self.use_mask_attention:
            self.mask_attention.set_mask_edges(mask_edges, points)
    
    def forward(self, pos, return_embed=False, point_indices=None):
        pos = self.normalize_position(pos)
        pos = pos.float().cuda()

        # 获取各个分辨率的特征
        multi_feat = []
        for i in range(len(self.encodings)):
            feat = self.encodings[i](pos).cuda()
            multi_feat.append(feat)
        
        # 应用特征空间优化
        if self.optimize_feature_space:
            # 1. 跨分辨率特征交互
            if hasattr(self, 'cross_res_interaction'):
                multi_feat = self.cross_res_interaction(multi_feat)
            
            # 2. 组合特征
            embed = sum(multi_feat)
            
            # 3. 应用特征分解与重组
            if hasattr(self, 'feature_decomposition'):
                embed = self.feature_decomposition(embed)
            
            # 4. 应用特征注意力机制
            if hasattr(self, 'feature_attention'):
                embed = self.feature_attention(embed)
            
            # 5. 应用掩码引导的特征注意力
            if hasattr(self, 'mask_attention') and self.use_mask_attention and point_indices is not None:
                embed = self.mask_attention(embed, point_indices)
        else:
            # 原始的特征相加方式
            embed = sum(multi_feat)
        
        feature = self.feature_net(embed)
        
        if return_embed:
            return feature, embed
        else:
            return feature