"""
边界损失 (Boundary Loss) - 增强版
用于红外小目标分割，专门优化边缘质量

基于论文: "Boundary Loss for Remote Sensing Image Segmentation"
和 "Boundary Loss for Highly Unbalanced Segmentation"

使用方法:
    在 train.yaml 中添加:

    loss:
      loss_4:
        type: BD_loss
        weight: 0.5

    dataset:
      train:
        bd_loss: True
"""

import numpy as np
from scipy.ndimage import distance_transform_edt as eucl_distance
from basicseg.utils.registry import LOSS_REGISTRY
import torch.nn as nn
import torch


def get_dist_map(mask):
    """
    计算距离变换图

    Args:
        mask: numpy array, binary mask [h, w]

    Returns:
        距离变换图，背景区域为正值，目标区域为负值
    """
    # 确保是 numpy 数组
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    mask = mask.astype(np.bool_)
    resolution = [1, 1]

    # 背景距离: 从背景到最近目标的距离
    negmask = ~mask
    dist_from_bg = eucl_distance(negmask, sampling=resolution)

    # 目标距离: 从目标到最近背景的距离
    dist_from_target = eucl_distance(mask, sampling=resolution)

    # 边界感知距离图: 背景区域为正，目标区域为负
    # 越靠近边界，值越小
    dist_map = dist_from_bg - dist_from_target

    # 裁剪到非负
    dist_map = np.clip(dist_map, a_min=0, a_max=None)

    return dist_map


def compute_boundary_aware_distance(mask):
    """
    计算边界感知距离图

    使用内部距离和外部距离的组合
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    mask = mask.astype(np.bool_)
    resolution = [1, 1]

    # 外部距离: 从背景到目标的距离
    negmask = ~mask
    external_dist = eucl_distance(negmask, sampling=resolution)

    # 内部距离: 从目标内部的点到边界的距离
    # 使用负的距离变换，在目标内部为负值
    internal_dist = -(eucl_distance(mask, sampling=resolution) - 1)

    # 边界距离 = 外部距离 + 内部距离
    # 在边界附近值最小
    boundary_dist = external_dist + np.clip(internal_dist, 0, None)

    return boundary_dist


@LOSS_REGISTRY.register()
class BD_loss(nn.Module):
    """
    边界感知距离损失 (Boundary Distance Loss)

    基于距离变换的边界损失，专门针对小目标分割优化
    """
    def __init__(self, reduction='mean', boundary_width=5):
        super(BD_loss, self).__init__()
        self.reduction = reduction
        self.boundary_width = boundary_width

    def forward(self, pred, dist_map):
        """
        Args:
            pred: 预测的分割 logits [b, 1, h, w]
            dist_map: 距离变换图 [b, h, w] - 需要转换到设备上

        Returns:
            边界损失
        """
        pred = torch.sigmoid(pred)

        # 距离图需要扩展到与预测相同的维度
        if dist_map.dim() == 3:
            dist_map = dist_map.unsqueeze(1)

        # 确保 dist_map 在正确的设备上
        if dist_map.device != pred.device:
            dist_map = dist_map.to(pred.device)

        # 边界损失: 预测值 * 距离图
        # 距离图越小（越接近边界），权重越大
        bd_loss = pred * dist_map

        if self.reduction == 'mean':
            return bd_loss.mean()
        elif self.reduction == 'sum':
            return bd_loss.sum()
        else:
            return bd_loss


@LOSS_REGISTRY.register()
class EnhancedBoundaryLoss(nn.Module):
    """
    增强版边界损失

    结合多个边界感知策略:
    1. 距离变换加权的 BCE
    2. 梯度差异损失
    3. 边缘区域特定损失
    """
    def __init__(self, reduction='mean', edge_weight=3.0, max_dist=10.0):
        super(EnhancedBoundaryLoss, self).__init__()
        self.reduction = reduction
        self.edge_weight = edge_weight
        self.max_dist = max_dist

    def forward(self, pred, target):
        """
        Args:
            pred: 预测的分割 logits [b, 1, h, w]
            target: 目标 mask [b, 1, h, w]

        Returns:
            增强边界损失
        """
        pred = torch.sigmoid(pred)

        # 计算边缘感知权重
        # 使用梯度的最大值作为边缘指示
        grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        grad_magnitude = torch.zeros_like(pred)
        grad_magnitude[:, :, :, :-1] = torch.maximum(grad_magnitude[:, :, :, :-1], grad_x)
        grad_magnitude[:, :, :-1, :] = torch.maximum(grad_magnitude[:, :, :-1, :], grad_y)

        # 边缘区域的 BCE 损失
        bce = F.binary_cross_entropy(pred, target, reduction='none')

        # 对边缘区域加权
        edge_bce = bce * (1 + self.edge_weight * grad_magnitude)

        if self.reduction == 'mean':
            return edge_bce.mean()
        elif self.reduction == 'sum':
            return edge_bce.sum()
        else:
            return edge_bce


@LOSS_REGISTRY.register()
class IoUAwareLoss(nn.Module):
    """
    IoU 感知的损失函数

    直接优化 IoU 指标，比 BCE/Dice 更直接地优化目标
    """
    def __init__(self, reduction='mean', smooth=1e-6):
        super(IoUAwareLoss, self).__init__()
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        # 计算 IoU
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        # IoU 损失
        loss = 1 - iou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# 导入必要的模块
import torch.nn.functional as F


# 辅助函数: 用于在数据加载时计算距离图
def prepare_boundary_loss_data(mask):
    """
    准备边界损失所需的数据

    Args:
        mask: mask tensor [1, h, w] or [h, w]

    Returns:
        dist_map: 距离变换图
    """
    if mask.dim() == 3:
        mask = mask.squeeze(0)

    # 转换为 numpy
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask

    # 计算距离图
    dist_map = get_dist_map(mask_np)

    # 转换回 tensor
    dist_map = torch.from_numpy(dist_map).float()

    return dist_map