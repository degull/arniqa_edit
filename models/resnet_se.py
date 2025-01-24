import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision.models import resnet50
from models.attention_se import DistortionAttention, HardNegativeCrossAttention


# SEBlock (Squeeze-and-Excitation Block)
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze: 채널 별 전역 평균 계산
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),   # 채널 축소
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),   # 채널 복원
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()   # (배치 크기, 채널, 높이, 너비)
        y = self.global_avg_pool(x).view(b, c)  # Squeeze 평균값 계산
        y = self.fc(y).view(b, c, 1, 1) # Excitation : 채널 중요도 계산
        return x * y    # 입력 텐서와 중요도를 곱하여 출력


# ResNetSE 클래스 (ResNet + SEBlock + Distortion Attention + Hard Negative Cross Attention)
# ResNet50기반 -> SEBlock & Attention 메커니즘 추가 -> 왜곡 패턴 강조
class ResNetSE(nn.Module):
    def __init__(self):
        super(ResNetSE, self).__init__()
        base_model = resnet50(pretrained=True)
        self.layer0 = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool
        )
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # Distortion Attention (각 레이어에 왜곡 주의 메커니즘 추가)
        self.distortion_attention1 = DistortionAttention(256)
        self.distortion_attention2 = DistortionAttention(512)
        self.distortion_attention3 = DistortionAttention(1024)
        self.distortion_attention4 = DistortionAttention(2048)

        # SEBlock (각 레이어에 채널 중요도 강조)
        self.se1 = SEBlock(256)
        self.se2 = SEBlock(512)
        self.se3 = SEBlock(1024)
        self.se4 = SEBlock(2048)

        # Hard Negative Cross Attention (Layer 4 출력에서 교차 주의 계산)
        self.hard_negative_attention = HardNegativeCrossAttention(2048)

        # Global Average Pooling (최종 출력 압축)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        # Layer 0
        x = self.layer0(x)
        print(f"Layer0 output: {x.size()}")

        # Layer 1
        x = self.layer1(x)  # ResNet Layer1 적용
        x = self.distortion_attention1(x)   # Distortion Attention 적용
        x = self.se1(x) # SEBlock 적용
        print(f"Layer1 output: {x.size()}")

        # Layer 2
        x = self.layer2(x)
        x = self.distortion_attention2(x)
        x = self.se2(x)
        print(f"Layer2 output: {x.size()}")

        # Layer 3
        x = self.layer3(x)
        x = self.distortion_attention3(x)
        x = self.se3(x)
        print(f"Layer3 output: {x.size()}")

        # Layer 4
        x = self.layer4(x)  # ResNet Layer4 적용
        x_attr = self.distortion_attention4(x)  # Distortion Attention 적용
        x_texture = self.se4(x) # SEBlock 적용
        print(f"Layer4 output (before attention): {x.size()}")

        # Hard Negative Cross Attention 적용
        x = self.hard_negative_attention(x_attr, x_texture)
        print(f"Layer4 output (after attention): {x.size()}")

        # Global Average Pooling
        x = self.global_avg_pool(x)
        print(f"Global Avg Pool output: {x.size()}")
        return x.view(x.size(0), -1)  # (batch_size, 2048)


"""
ResNetSE는 ResNet50의 사전 학습된 구조 기반
Distortion Attention, SEBlock, Hard Negative Cross Attention 추가
-> 왜곡 패턴에 민감한 특징 맵 생성
"""