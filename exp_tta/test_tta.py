"""
TTA (Test Time Augmentation) 测试脚本
用于在推理时对图像进行多种变换并融合结果，提升分割精度

使用方法:
    python test_tta.py --config options/test.yaml

增强策略:
1. Original: 原图
2. Horizontal Flip: 水平翻转
3. Vertical Flip: 垂直翻转
4. 组合: 水平和垂直翻转
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import cv2
import sys
import numpy as np
import os
from tqdm import tqdm

#sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from basicseg.test_model import Test_model
from basicseg.utils.yaml_options import parse_options, dict2str
from basicseg.utils.path_utils import *
from basicseg.data import build_dataset


def tensor2img(inp):
    """将 tensor 转换为图像 [b,1,h,w] -> [b,h,w] -> numpy -> uint8"""
    inp = torch.sigmoid(inp) * 255.
    inp = inp.squeeze(1).cpu().numpy().astype(np.uint8)
    return inp


def save_batch_img(imgs, img_names, dire):
    """保存图像批次"""
    for i in range(len(imgs)):
        img = imgs[i]
        img_name = img_names[i]
        img_path = os.path.join(dire, img_name)
        cv2.imwrite(img_path, img)


def apply_tta_transform(img, transform_type):
    """
    应用 TTA 变换

    Args:
        img: 输入图像 tensor [b,c,h,w]
        transform_type: 变换类型

    Returns:
        变换后的图像 tensor
    """
    if transform_type == 'original':
        return img
    elif transform_type == 'hflip':
        return torch.flip(img, dims=[3])  # 水平翻转
    elif transform_type == 'vflip':
        return torch.flip(img, dims=[2])  # 垂直翻转
    elif transform_type == 'hvflip':
        return torch.flip(img, dims=[2, 3])  # 水平+垂直翻转
    elif transform_type == 'rotate90':
        # 逆时针旋转90度
        return torch.rot90(img, k=1, dims=[2, 3])
    elif transform_type == 'rotate180':
        return torch.rot90(img, k=2, dims=[2, 3])
    elif transform_type == 'rotate270':
        return torch.rot90(img, k=3, dims=[2, 3])
    else:
        return img


def inverse_tta_transform(pred, transform_type):
    """
    逆变换，将预测结果恢复到原始坐标系

    Args:
        pred: 预测结果 tensor [b,1,h,w]
        transform_type: 应用的变换类型

    Returns:
        逆变换后的预测结果
    """
    if transform_type == 'original':
        return pred
    elif transform_type == 'hflip':
        return torch.flip(pred, dims=[3])
    elif transform_type == 'vflip':
        return torch.flip(pred, dims=[2])
    elif transform_type == 'hvflip':
        return torch.flip(pred, dims=[2, 3])
    elif transform_type == 'rotate90':
        return torch.rot90(pred, k=-1, dims=[2, 3])
    elif transform_type == 'rotate180':
        return torch.rot90(pred, k=-2, dims=[2, 3])
    elif transform_type == 'rotate270':
        return torch.rot90(pred, k=-3, dims=[2, 3])
    else:
        return pred


class TTATestModel(Test_model):
    """支持 TTA 的测试模型"""

    def __init__(self, opt):
        super().__init__(opt)
        self.tta_enabled = opt['exp'].get('tta', {}).get('enabled', False)
        self.tta_transforms = opt['exp'].get('tta', {}).get('transforms',
            ['original', 'hflip', 'vflip'])
        self.tta_merge_mode = opt['exp'].get('tta', {}).get('merge_mode', 'mean')

        if self.tta_enabled:
            print(f"TTA enabled with transforms: {self.tta_transforms}")
            print(f"TTA merge mode: {self.tta_merge_mode}")

    def test_one_iter_tta(self, data):
        """
        使用 TTA 进行测试

        Args:
            data: (img, label, img_name)

        Returns:
            融合后的预测结果
        """
        img, label, img_name = data
        img = img.to(self.device)
        label = label.to(self.device)

        batch_size = img.shape[0]
        original_h, original_w = label.shape[2], label.shape[3]

        # 存储每种变换的结果
        pred_list = []

        for transform_type in self.tta_transforms:
            # 应用变换
            augmented_img = apply_tta_transform(img, transform_type)

            # 前向传播
            with torch.no_grad():
                pred = self.net(augmented_img)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                # 恢复到原始尺寸
                pred = F.interpolate(pred, (original_h, original_w),
                                     mode='bilinear', align_corners=False)

            # 逆变换
            pred = inverse_tta_transform(pred, transform_type)
            pred_list.append(pred)

        # 融合结果
        if self.tta_merge_mode == 'mean':
            # 取平均
            merged_pred = torch.stack(pred_list, dim=0).mean(dim=0)
        elif self.tta_merge_mode == 'max':
            # 取最大 (适用于多尺度)
            merged_pred = torch.stack(pred_list, dim=0).max(dim=0)[0]
        else:
            merged_pred = pred_list[0]

        # 更新指标
        self.metric.update(pred=merged_pred, target=label)

        return merged_pred


def init_dataset(opt):
    """初始化数据集"""
    test_opt = opt['dataset']['test']
    testset = build_dataset(test_opt)
    return testset


def init_dataloader(opt, testset):
    """初始化数据加载器"""
    test_loader = Data.DataLoader(
        dataset=testset,
        batch_size=opt['exp']['bs'],
        sampler=None,
        num_workers=opt['exp'].get('nw', 8)
    )
    return test_loader


def main():
    opt, args = parse_options()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt['exp']['device'])

    # 确认 TTA 配置
    tta_enabled = opt['exp'].get('tta', {}).get('enabled', False)

    # 初始化数据集
    testset = init_dataset(opt)
    test_loader = init_dataloader(opt, testset)

    # 初始化模型
    model = TTATestModel(opt)

    save_dir = opt['exp'].get('save_dir', False)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 加载模型权重
    if opt.get('resume'):
        if opt.get('resume').get('net_path'):
            model.load_network(model.net, opt['resume']['net_path'])
            print(f'load pretrained network from: {opt["resume"]["net_path"]}')

    model.net.eval()

    # 根据是否启用 TTA 选择测试方法
    test_method = 'test_one_iter_tta' if tta_enabled else 'test_one_iter'

    for idx, data in enumerate(tqdm(test_loader)):
        if tta_enabled:
            # TTA 测试
            pred = model.test_one_iter_tta(data)
        else:
            # 普通测试
            img, label, img_name = data
            pred = model.test_one_iter((img, label))

        if save_dir:
            img_np = tensor2img(pred)
            save_batch_img(img_np, data[2], save_dir)

    # 获取评估指标
    test_mean_metric = model.get_mean_metric()
    test_norm_metric = model.get_norm_metric()

    print(f"\n{'='*50}")
    print(f"测试完成")
    print(f"TTA 启用: {tta_enabled}")
    if tta_enabled:
        print(f"TTA 变换: {opt['exp'].get('tta', {}).get('transforms', [])}")
        print(f"TTA 融合模式: {opt['exp'].get('tta', {}).get('merge_mode', 'mean')}")
    print(f"{'='*50}")
    print(f"Mean IoU: {test_mean_metric['iou']:.4f}")
    print(f"Mean F1:  {test_mean_metric['fscore']:.4f}")
    print(f"Norm IoU: {test_norm_metric['iou']:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True,
                        help='配置文件路径 (YAML)')
    parser.add_argument('--tta', action='store_true', help='启用 TTA')
    parser.add_argument('--transforms', nargs='+',
                        default=['original', 'hflip', 'vflip', 'hvflip'],
                        help='TTA 变换类型')
    parser.add_argument('--merge', type=str, default='mean',
                        choices=['mean', 'max'], help='TTA 融合模式')
    args_parsed, unknown = parser.parse_known_args()

    # 将 --opt 转换为兼容格式
    import sys
    if args_parsed.opt:
        sys.argv = [sys.argv[0], '--opt', args_parsed.opt] + unknown
    else:
        sys.argv = [sys.argv[0]] + unknown

    # 解析配置
    opt, args = parse_options()

    # 命令行参数优先级更高
    if args_parsed.tta:
        if 'tta' not in opt['exp']:
            opt['exp']['tta'] = {}
        opt['exp']['tta']['enabled'] = True
        opt['exp']['tta']['transforms'] = args_parsed.transforms
        opt['exp']['tta']['merge_mode'] = args_parsed.merge

    # 如果配置文件中有 tta 设置，也使用它
    if 'tta' in opt['exp']:
        tta_enabled = opt['exp'].get('tta', {}).get('enabled', False)

    main()
