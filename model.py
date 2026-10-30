"""模型：针对 48x48 灰度图改造的 ResNet18，从头训练（不使用预训练权重）。"""

import torch.nn as nn
from torchvision.models import resnet18


def build_model(num_classes: int = 7, dropout: float = 0.4) -> nn.Module:
    """标准 ResNet18 的三处改造：

    1. conv1：输入通道 3 -> 1（灰度），卷积核 7x7/stride2 -> 3x3/stride1，避免小图一开始就缩小。
    2. maxpool：去掉，保留空间分辨率。特征图依次为 48 -> 24 -> 12 -> 6，再全局平均池化。
    3. fc：加 Dropout 抑制过拟合，输出 7 类。
    """
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))
    return model
