import numpy as np
import copy

def calculate_mask_quality_score(annotation, image_area, iou_weight=0.5, stability_weight=0.3, area_weight=0.2, max_area_ratio=0.1):
    """计算掩码质量分数
    
    Args:
        annotation: SAM生成的单个掩码注释
        image_area: 图像总面积
        iou_weight: predicted_iou的权重
        stability_weight: stability_score的权重
        area_weight: 面积归一化的权重
        max_area_ratio: 最大面积比例（图像的10%）
        
    Returns:
        float: 0-1之间的质量分数
    """
    # 获取SAM提供的置信度指标
    predicted_iou = annotation.get('predicted_iou', 0.0)
    stability_score = annotation.get('stability_score', 0.0)
    
    # 计算面积归一化分数（避免过大或过小的掩码）
    mask_area = np.sum(annotation['segmentation'])
    area_ratio = mask_area / image_area
    # 使用高斯函数对面积进行归一化，最优面积在max_area_ratio附近
    area_norm_score = np.exp(-(area_ratio - max_area_ratio/2)**2 / (2 * (max_area_ratio/2)**2))
    
    # 计算综合质量分数
    quality_score = (iou_weight * predicted_iou + 
                    stability_weight * stability_score + 
                    area_weight * area_norm_score)
    
    return quality_score

def get_single_mask(annotations, metric='predicted_iou', descending=False, 
                   quality_score=False, iou_weight=0.5, stability_weight=0.3, area_weight=0.2):
    # metric: predicted_iou or area or quality_score
    image_area = annotations[0]['segmentation'].shape[0] * annotations[0]['segmentation'].shape[1]
    
    # 如果使用质量分数排序
    if quality_score:
        for ann in annotations:
            ann['quality_score'] = calculate_mask_quality_score(
                ann, image_area, iou_weight, stability_weight, area_weight)
        sorted_masks = sorted(annotations, key=(lambda x: x['quality_score']), reverse=descending)
    else:
        sorted_masks = sorted(annotations, key=(lambda x: x['predicted_iou']), reverse=descending) # True: descending, False: ascending
    
    mask = np.full((sorted_masks[0]['segmentation'].shape[0], sorted_masks[0]['segmentation'].shape[1]), -1, dtype=int)
    
    # 保存质量分数的字典
    mask_quality_scores = {}
    mask_id = 0
    for ann in sorted_masks:
        m = ann['segmentation']
        mask_id += 1
        mask[m] = mask_id
        # 存储该掩码的质量分数
        if quality_score:
            mask_quality_scores[mask_id] = ann.get('quality_score', ann.get('predicted_iou', 0.0))
        else:
            mask_quality_scores[mask_id] = ann.get('predicted_iou', 0.0)
    
    # start from 1, 0 is invalid
    mask = num_to_natural(mask) + 1
    
    # 返回掩码和质量分数字典
    return mask, mask_quality_scores

def num_to_natural(mask):
    '''
    Change the group number to natural number arrangement (code credit: SAM3D)
    '''
    if np.all(mask == -1):
        return mask
    array = copy.deepcopy(mask).astype(int)

    unique_values = np.unique(array[array != -1])
    mapping = np.full(np.max(unique_values) + 2, -1)
    mapping[unique_values + 1] = np.arange(len(unique_values))  # map ith(start from 0) group_id to i
    array = mapping[array + 1]
    return array

def viz_mask(mask):
    array = np.zeros(tuple(mask.shape) + (3,))
    if np.all(mask == -1):
        return array
    unique_values = np.unique(mask[mask != -1])
    for i in unique_values:
        array[mask == i] = np.random.random((3))

    return array * 255
