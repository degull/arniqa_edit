import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 왜곡 타입 & 인덱스 매핑
distortion_map = {
    "gaussian_blur": 0,  # Sobel 필터
    "lens_blur": 1,  # Sobel 필터
    "motion_blur": 2,  # Sobel 필터
    "color_diffusion": 3,  # Sobel 필터
    "color_shift": 4,  # Sobel 필터
    "color_quantization": 5,  # Sobel 필터
    "color_saturation_1": 6,  # HSV 색공간 분석
    "color_saturation_2": 7,  # HSV 색공간 분석
    "jpeg2000": 8,  # Sobel 필터
    "jpeg": 9,  # Sobel 필터
    "white_noise": 10,  # Sobel 필터
    "white_noise_color_component": 11,  # Sobel 필터
    "impulse_noise": 12,  # Sobel 필터
    "multiplicative_noise": 13,  # Sobel 필터
    "denoise": 14,  # Fourier Transform
    "brighten": 15,  # HSV 색공간 분석
    "darken": 16,  # HSV 색공간 분석
    "mean_shift": 17,  # 히스토그램 분석
    "jitter": 18,  # Fourier Transform
    "non_eccentricity_patch": 19,  # Sobel 필터
    "pixelate": 20,  # Sobel 필터
    "quantization": 21,  # Fourier Transform
    "color_block": 22,  # Fourier Transform
    "high_sharpen": 23,  # Fourier Transform
    "contrast_change": 24  # Fourier Transform
}

# 왜곡 분류 -> CNN 기반 분류기
class DistortionClassifier(nn.Module):
    def __init__(self, in_channels, num_distortions=25):
        super(DistortionClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))    # 특징 맵 : 1x1로 축소
        )
        self.fc = nn.Linear(128, num_distortions)   # 최종 분류 레이어

    def forward(self, x):
        x = self.conv(x)    # Convolution 적용
        x = x.view(x.size(0), -1)  # 1D로 변환
        x = self.fc(x)  # Fully Connected Layer로 분류 수행
        return x


# HNCA 속성(feature) 처리 -> CNN 모듈
class AttributeFeatureProcessor(nn.Module):
    def __init__(self, in_channels):
        super(AttributeFeatureProcessor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),  # 첫 번째 Conv
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),  # 두 번째 Conv
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),  # 세 번째 Conv
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x) # 입력 특징 맵 처리


# 텍스처 데이터를 처리 모듈
class TextureBlockProcessor(nn.Module):
    def __init__(self, in_channels):
        super(TextureBlockProcessor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2),   # 다운샘플링
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)   # 업샘플링
        )

    def forward(self, x):
        return self.conv(x) # 텍스처 정보 처리


# distortion attention 모듈 정의
class DistortionAttention(nn.Module):
    def __init__(self, in_channels):
        super(DistortionAttention, self).__init__()
        query_key_channels = max(1, in_channels // 8)   # Query / Key 채널 수 설정
        self.query_conv = nn.Conv2d(in_channels, query_key_channels, kernel_size=1)  # Query 생성
        self.key_conv = nn.Conv2d(in_channels, query_key_channels, kernel_size=1)   # Key 생성
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)    # Value 생성
        self.softmax = nn.Softmax(dim=-1)   # Softmax를 통한 정규화
        self.distortion_classifier = DistortionClassifier(in_channels)  # 왜곡 분류기

    def forward(self, x):
        b, c, h, w = x.size()   
        distortion_logits = self.distortion_classifier(x)   # 왜곡 유형 분류
        distortion_types = torch.argmax(distortion_logits, dim=1)   # 최종 왜곡 유형 선택
        query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)  # Query 생성 및 변환
        key = self.key_conv(x).view(b, -1, h * w)   # Key 생성
        value = self.value_conv(x).view(b, -1, h * w)   # Value 생성
        scale = query.size(-1) ** -0.5  # 스케일 팩터 계산
        attention = self.softmax(torch.bmm(query, key) * scale) # Attention 계산
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(b, c, h, w) # Attention 적용
        return out + x  # # 잔차 연결(residual connection) 추가


    # 왜곡 유형에 따른 필터 적용
    def _apply_filter(self, x, distortion_type):
        if isinstance(distortion_type, str):
            distortion_type = distortion_map.get(distortion_type, -1)

        if distortion_type in [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 19, 20]:
            return self._sobel_filter(x)    # Sobel 필터 적용
        elif distortion_type in [6, 7, 15, 16]:
            return self._hsv_analysis(x)    # HSV 분석
        elif distortion_type == 17:
            return self._histogram_analysis(x)  # 히스토그램 분석
        elif distortion_type in [14, 18, 21, 22, 23, 24]:
            return self._fourier_analysis(x)    # Fourier 분석
        else:
            return torch.ones_like(x[:, :1, :, :])  # 기본 필터 반환


    # Sobel 필터 구현
    def _sobel_filter(self, x):
        sobel_x = self.sobel_x.repeat(x.size(1), 1, 1, 1).to(x.device)
        sobel_y = self.sobel_y.repeat(x.size(1), 1, 1, 1).to(x.device)

        grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.size(1))
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Normalize gradient magnitude
        return torch.sigmoid(gradient_magnitude.mean(dim=1, keepdim=True))

    # HSV 분석
    def _hsv_analysis(self, x):
        hsv = self._rgb_to_hsv(x)
        return hsv[:, 1:2, :, :]

    # 히스토그램 분석
    def _histogram_analysis(self, x):
        hist_map = torch.mean(x, dim=1, keepdim=True)
        return torch.sigmoid(hist_map)

    # Fourier 분석
    def _fourier_analysis(self, x):
        h, w = x.shape[-2:]
        new_h = 2 ** int(np.ceil(np.log2(h)))
        new_w = 2 ** int(np.ceil(np.log2(w)))
        
        padded_x = F.pad(x, (0, new_w - w, 0, new_h - h))
        
        fft = torch.fft.fft2(padded_x, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
        magnitude = torch.sqrt(fft_shift.real ** 2 + fft_shift.imag ** 2)
        
        magnitude = magnitude[:, :, :h, :w]
        
        return torch.sigmoid(magnitude.mean(dim=1, keepdim=True))

    # RGB 이미지를 HSV로 변환
    def _rgb_to_hsv(self, x):
        max_rgb, _ = x.max(dim=1, keepdim=True)
        min_rgb, _ = x.min(dim=1, keepdim=True)
        delta = max_rgb - min_rgb + 1e-6
        saturation = delta / (max_rgb + 1e-6)
        value = max_rgb
        return torch.cat((delta, saturation, value), dim=1)


# Hard Negative Cross Attention 모듈 정의
class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(HardNegativeCrossAttention, self).__init__()
        self.num_heads = num_heads  # Multi-head Attention
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)    # Query 생성
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # Key 생성
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)    # Value 생성
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)   # Output Projection
        self.softmax = nn.Softmax(dim=-1)   # Attention 정규화
        self.layer_norm = None  # Layer Normalization

        # 속성 및 텍스처 프로세서
        self.attribute_processor = AttributeFeatureProcessor(in_channels)
        self.texture_processor = TextureBlockProcessor(in_channels)

    def forward(self, x_attr, x_texture):
        x_attr = self.attribute_processor(x_attr)  # 속성 정보 처리
        x_texture = self.texture_processor(x_texture)  # 텍스처 정보 처리

        # 입력 크기 조정
        if x_attr.size(2) != x_texture.size(2) or x_attr.size(3) != x_texture.size(3):
            min_h = min(x_attr.size(2), x_texture.size(2))
            min_w = min(x_attr.size(3), x_texture.size(3))
            x_attr = F.adaptive_avg_pool2d(x_attr, (min_h, min_w))
            x_texture = F.adaptive_avg_pool2d(x_texture, (min_h, min_w))

        b, c, h, w = x_attr.size()
        head_dim = c // self.num_heads  # Head 당 차원 계산
        assert c % self.num_heads == 0

        # Multi-head Query, Key, Value 생성 및 변환
        multi_head_query = self.query_conv(x_attr).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)
        multi_head_key = self.key_conv(x_texture).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 2, 3)
        multi_head_value = self.value_conv(x_texture).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)

        # Attention 계산
        scale = torch.sqrt(torch.tensor(head_dim, dtype=torch.float32).clamp(min=1e-6)).to(multi_head_query.device)
        attention = self.softmax(torch.matmul(multi_head_query, multi_head_key) / scale)
        out = torch.matmul(attention, multi_head_value).permute(0, 1, 3, 2).contiguous()

        # 출력 재구성 및 Output Projection 적용
        out = out.view(b, c, h, w)
        out = self.output_proj(out)
        out = nn.Dropout(p=0.1)(out) + x_attr  # 잔차 연결 추가

        # Layer Normalization 적용
        if self.layer_norm is None or self.layer_norm.normalized_shape != (c, h, w):
            self.layer_norm = nn.LayerNorm([c, h, w]).to(out.device)

        out = self.layer_norm(out)
        return out





