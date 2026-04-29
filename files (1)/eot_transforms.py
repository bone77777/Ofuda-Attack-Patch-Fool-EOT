"""
eot_transforms.py

Expectation Over Transformation (Athalye et al., ICML 2018) のための
物理世界変換分布。

各最適化ステップで以下をランダムサンプリングして合成画像に適用する:
  - スケーリング (撮影距離の変動)
  - 平面回転 (カメラ・パッチの相対傾き)
  - 透視変換 (浅い角度からの撮影)
  - 明度・コントラスト・色相シフト (照明条件)
  - ガウシアンぼかし (ピント外れ)
  - センサノイズ (CMOS の読み出しノイズ)
  - 色再現誤差 (印刷-スキャンサイクルでの色ずれをシミュレート)

Eykholt et al. (CVPR 2018) "Robust Physical-World Attacks on Deep Learning
Visual Classification" が道路標識への物理攻撃で示した通り、印刷物攻撃の
物理ロバスト性は最適化中に十分広い変換分布を取ることに大きく依存する。

注: 本実装は印刷キャリブレーション (具体的なプリンタの色再現プロファイル
やインクの分光反射特性) は含まない。これは本概念実証を実機攻撃の
レシピにしないための意図的な省略である。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class EOTConfig:
    """物理変換分布のパラメータ。

    全てのレンジは (low, high) の一様分布として解釈する。
    """
    scale_range: tuple[float, float] = (0.85, 1.15)
    rotation_deg_range: tuple[float, float] = (-12.0, 12.0)
    perspective_strength: float = 0.06  # 0.0 で無効、上限 ~0.1 程度を推奨
    brightness_range: tuple[float, float] = (0.75, 1.25)
    contrast_range: tuple[float, float] = (0.85, 1.15)
    hue_shift_range: tuple[float, float] = (-0.05, 0.05)
    blur_sigma_range: tuple[float, float] = (0.0, 1.5)
    sensor_noise_std_range: tuple[float, float] = (0.0, 0.02)
    color_reproduction_jitter: float = 0.04  # チャネルごとの線形ゲイン揺らぎ


def _affine_grid(
    batch: int,
    height: int,
    width: int,
    scale: torch.Tensor,
    rotation_rad: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """affine_grid 用の (B, 2, 3) theta を作る。"""
    cos = torch.cos(rotation_rad)
    sin = torch.sin(rotation_rad)
    inv_scale = 1.0 / scale  # affine_grid は inverse 行列扱いのため

    theta = torch.zeros(batch, 2, 3, device=device, dtype=torch.float32)
    theta[:, 0, 0] = cos * inv_scale
    theta[:, 0, 1] = -sin * inv_scale
    theta[:, 1, 0] = sin * inv_scale
    theta[:, 1, 1] = cos * inv_scale
    return theta


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """軽量なガウシアンぼかし (sigma が小さい場合は素通し)。"""
    if sigma < 1e-3:
        return image

    # カーネルサイズは sigma に比例
    radius = max(1, int(math.ceil(3.0 * sigma)))
    kernel_size = 2 * radius + 1

    coords = torch.arange(kernel_size, device=image.device, dtype=torch.float32) - radius
    g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    g1d = g1d / g1d.sum()
    g2d = g1d[:, None] * g1d[None, :]
    g2d = g2d.expand(image.shape[1], 1, kernel_size, kernel_size)

    return F.conv2d(image, g2d, padding=radius, groups=image.shape[1])


def _rgb_to_hsv_shift(image: torch.Tensor, hue_shift: float) -> torch.Tensor:
    """簡易な hue shift。chrominance 系の摂動として近似的に動作。

    厳密な HSV 変換ではなく、RGB チャネル間の循環シフトを連続化したもの。
    画像が [0, 1] 範囲の RGB であることを仮定する。
    """
    if abs(hue_shift) < 1e-4:
        return image

    # YIQ 風の chrominance 回転で近似
    angle = hue_shift * 2.0 * math.pi
    cos = math.cos(angle)
    sin = math.sin(angle)

    # luma + 2 つの chroma の単純なローテーション
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    i_chroma = 0.596 * r - 0.274 * g - 0.322 * b
    q_chroma = 0.211 * r - 0.523 * g + 0.312 * b

    i_rot = cos * i_chroma - sin * q_chroma
    q_rot = sin * i_chroma + cos * q_chroma

    r_out = y + 0.956 * i_rot + 0.621 * q_rot
    g_out = y - 0.272 * i_rot - 0.647 * q_rot
    b_out = y - 1.106 * i_rot + 1.703 * q_rot

    return torch.cat([r_out, g_out, b_out], dim=1).clamp(0.0, 1.0)


def sample_eot_batch(
    image: torch.Tensor,
    config: EOTConfig,
    num_samples: int,
) -> torch.Tensor:
    """
    画像を 1 枚受け取り、EOT 変換分布から num_samples 個の変換版をバッチで返す。

    Args:
        image: (1, C, H, W) の画像 ([0, 1] 正規化済み)
        config: 変換パラメータ
        num_samples: バッチサイズ (= EOT サンプル数)

    Returns:
        (num_samples, C, H, W) のテンソル
    """
    if image.dim() != 4 or image.shape[0] != 1:
        raise ValueError(f"expected (1, C, H, W), got {tuple(image.shape)}")

    device = image.device
    _, channels, height, width = image.shape

    # 同一画像を num_samples 回複製
    batch = image.expand(num_samples, channels, height, width)

    # スケール + 回転 (アフィン変換)
    scale = torch.empty(num_samples, device=device).uniform_(*config.scale_range)
    rot_deg = torch.empty(num_samples, device=device).uniform_(*config.rotation_deg_range)
    rot_rad = rot_deg * (math.pi / 180.0)
    theta = _affine_grid(num_samples, height, width, scale, rot_rad, device)
    grid = F.affine_grid(theta, batch.shape, align_corners=False)
    transformed = F.grid_sample(batch, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # 明度・コントラスト
    brightness = torch.empty(num_samples, 1, 1, 1, device=device).uniform_(*config.brightness_range)
    contrast = torch.empty(num_samples, 1, 1, 1, device=device).uniform_(*config.contrast_range)
    mean = transformed.mean(dim=(2, 3), keepdim=True)
    transformed = ((transformed - mean) * contrast + mean) * brightness
    transformed = transformed.clamp(0.0, 1.0)

    # 色再現誤差 (チャネルごとの線形ゲイン)
    if config.color_reproduction_jitter > 0:
        jitter = (
            torch.empty(num_samples, channels, 1, 1, device=device)
            .uniform_(-config.color_reproduction_jitter, config.color_reproduction_jitter)
        )
        transformed = (transformed * (1.0 + jitter)).clamp(0.0, 1.0)

    # 色相シフト (バッチ要素ごとに個別に適用)
    if config.hue_shift_range[1] > config.hue_shift_range[0]:
        out = []
        for i in range(num_samples):
            hue = torch.empty(1, device=device).uniform_(*config.hue_shift_range).item()
            out.append(_rgb_to_hsv_shift(transformed[i:i+1], hue))
        transformed = torch.cat(out, dim=0)

    # ガウシアンぼかし (バッチ要素ごとに個別に sigma)
    blurred_list = []
    for i in range(num_samples):
        sigma = torch.empty(1, device=device).uniform_(*config.blur_sigma_range).item()
        blurred_list.append(_gaussian_blur(transformed[i:i+1], sigma))
    transformed = torch.cat(blurred_list, dim=0)

    # センサノイズ
    noise_std = torch.empty(num_samples, 1, 1, 1, device=device).uniform_(*config.sensor_noise_std_range)
    noise = torch.randn_like(transformed) * noise_std
    transformed = (transformed + noise).clamp(0.0, 1.0)

    return transformed
