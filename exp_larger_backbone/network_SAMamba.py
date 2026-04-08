"""
SAMamba 模型 - 支持多种 SAM2 骨干网络

支持:
- sam2_hiera_s.yaml: SAM2 Hiera Small (默认)
- sam2_hiera_b+.yaml: SAM2 Hiera Base+
- sam2_hiera_l.yaml: SAM2 Hiera Large

使用方法:
    在 train.yaml 中设置:
    model:
      net:
        type: SAMamba
        config: sam2_hiera_b+  # 或 sam2_hiera_l
        checkpoint_path: /path/to/sam2_hiera_b+.pt
"""

import torch
import torch.nn as nn
from thop import profile
from basicseg.main_blocks import CSI, DPCF
from basicseg.mona_with_select import MonaOp
from basicseg.networks.sam2.build_sam import build_sam2
from thop import clever_format
from basicseg.utils.registry import NET_REGISTRY
import time


class Adapter(nn.Module):
    """适配器模块，用于包装 SAM2 编码器块"""
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        # 获取输入特征维度
        if hasattr(blk, 'attn') and hasattr(blk.attn, 'qkv'):
            dim = blk.attn.qkv.in_features
        elif hasattr(blk, 'mlp'):
            if hasattr(blk.mlp, 'fc1'):
                dim = blk.mlp.fc1.in_features
            else:
                dim = 96  # 默认值
        else:
            dim = 96  # 默认值

        self.monaOp = MonaOp(dim)

    def forward(self, x):
        x = self.monaOp(x)
        net = self.block(x)
        return net


@NET_REGISTRY.register()
class SAMamba(nn.Module):
    """
    SAMamba 模型 - 基于 SAM2 的红外小目标分割网络

    Args:
        checkpoint_path: SAM2 预训练权重路径
        config: 配置文件名 (sam2_hiera_s.yaml, sam2_hiera_b+.yaml, sam2_hiera_l.yaml)
    """
    def __init__(self, checkpoint_path='/media/data2/zhengshuchen/code/SAMamba/sam2_configs/sam2_hiera_small.pt',
                 config='sam2_hiera_s.yaml') -> None:
        super(SAMamba, self).__init__()

        # 根据配置选择不同的 SAM2 配置文件
        self.config = config

        # 通道数映射 (不同配置输出通道不同)
        channel_map = {
            'sam2_hiera_s.yaml': (96, 192, 384, 768),    # Small
            'sam2_hiera_b+.yaml': (112, 224, 448, 896),   # Base+
            'sam2_hiera_l.yaml': (144, 288, 576, 1152),    # Large
        }

        self.channel_map = channel_map.get(config, (96, 192, 384, 768))

        # 构建 SAM2 模型
        if checkpoint_path:
            model = build_sam2(config, checkpoint_path)
        else:
            model = build_sam2(config)

        # 删除不需要的模块
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck

        self.encoder = model.image_encoder.trunk

        # 冻结骨干网络
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 添加适配器
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(Adapter(block))
        self.encoder.blocks = nn.Sequential(*blocks)

        # CSI 模块 - 根据配置调整输入通道
        c1, c2, c3, c4 = self.channel_map
        self.mbhf1 = CSI(c1, 128)
        self.mbhf2 = CSI(c2, 128)
        self.mbhf3 = CSI(c3, 128)
        self.mbhf4 = CSI(c4, 128)

        # 解码器
        self.up1 = DPCF(128, 128)
        self.up2 = DPCF(128, 128)
        self.up3 = DPCF(128, 128)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.head = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            分割结果 [B, 1, H, W]
        """
        # 编码器
        x1, x2, x3, x4 = self.encoder(x)

        # CSI 特征增强
        x1, x2, x3, x4 = self.mbhf1(x1), self.mbhf2(x2), self.mbhf3(x3), self.mbhf4(x4)

        # 解码器 - 上采样融合
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # 输出
        out = self.head(self.deconv2(self.deconv1(x)))
        return out

    def get_config_info(self):
        """获取配置信息"""
        return {
            'config': self.config,
            'channels': self.channel_map,
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'total_params': sum(p.numel() for p in self.parameters())
        }


# 为了兼容性，保留原名
SAMamba_base = SAMamba
SAMamba_large = SAMamba


if __name__ == '__main__':
    # 测试不同配置
    configs = ['sam2_hiera_s.yaml', 'sam2_hiera_b+.yaml', 'sam2_hiera_l.yaml']

    for cfg in configs:
        print(f"\n{'='*50}")
        print(f"Testing config: {cfg}")
        print(f"{'='*50}")

        try:
            # 注意: 需要实际的 checkpoint 路径
            # net = SAMamba(checkpoint_path=None, config=cfg)
            # info = net.get_config_info()
            # print(f"Channels: {info['channels']}")
            # print(f"Trainable params: {info['trainable_params']:,}")
            # print(f"Total params: {info['total_params']:,}")
            print(f"Config {cfg} defined")
        except Exception as e:
            print(f"Error: {e}")

    print("\n可用配置:")
    print("- sam2_hiera_s.yaml: SAM2 Hiera Small (默认)")
    print("- sam2_hiera_b+.yaml: SAM2 Hiera Base+")
    print("- sam2_hiera_l.yaml: SAM2 Hiera Large")