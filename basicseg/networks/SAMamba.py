import torch
import torch.nn as nn
from basicseg.main_blocks import CSI, DPCF
from basicseg.mona_with_select import MonaOp
from basicseg.utils.registry import NET_REGISTRY
import logging


class SimpleEncoder(nn.Module):
    """简单 encoder,产生 4 级不同通道的特征"""

    def __init__(self, channels=(96, 192, 384, 768)):
        super().__init__()
        c1, c2, c3, c4 = channels

        # 初始卷积
        self.conv1 = nn.Conv2d(3, c1, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(c1)
        self.relu1 = nn.ReLU(inplace=True)

        # 后续阶段简化为不同步长的卷积
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(c4)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu1(self.bn1(self.conv1(x)))   # c1 channels
        x2 = self.relu2(self.bn2(self.conv2(x1)))    # c2 channels
        x3 = self.relu3(self.bn3(self.conv3(x2)))  # c3 channels
        x4 = self.relu4(self.bn4(self.conv4(x3)))  # c4 channels
        return x1, x2, x3, x4


@NET_REGISTRY.register()
class SAMamba(nn.Module):
    def __init__(self, checkpoint_path=None, config='sam2_hiera_s.yaml') -> None:
        super(SAMamba, self).__init__()

        logging.info(f"SAMamba initializing with config: {config}")

        channel_map = {
            'sam2_hiera_s.yaml': (96, 192, 384, 768),
            'sam2_hiera_b+.yaml': (112, 224, 448, 896),
            'sam2_hiera_l.yaml': (144, 288, 576, 1152),
        }

        self.channels = channel_map.get(config, (96, 192, 384, 768))

        # 创建 encoder
        self.encoder = SimpleEncoder(self.channels)

        # 冻结 encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # CSI 模块 (每个使用对应的通道数)
        c1, c2, c3, c4 = self.channels
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
        x1, x2, x3, x4 = self.encoder(x)
        x1, x2, x3, x4 = self.mbhf1(x1), self.mbhf2(x2), self.mbhf3(x3), self.mbhf4(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        out = self.head(self.deconv2(self.deconv1(x)))
        return out


if __name__ == '__main__':
    input_tensor = torch.randn(1, 3, 1024, 1024)
    net = SAMamba()
    output = net(input_tensor)
    print(f"Input: {input_tensor.shape}")
    print(f"Output: {output.shape}")
    print(f"Channels: {net.channels}")