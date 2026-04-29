"""
visualize.py

Attention rollout (Abnar & Zuidema, ACL 2020) による可視化。

attention rollout は ViT の各層の attention 行列を、residual connection を
考慮した上で順次乗算し、入力 patch から最終層までの「情報フロー」を
近似する手法。adversarial patch がどの程度 attention を奪えているかの
直感的な可視化に有用。
"""

from __future__ import annotations

import torch
import numpy as np
import matplotlib.pyplot as plt


def attention_rollout(
    attentions: tuple[torch.Tensor, ...],
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
) -> torch.Tensor:
    """
    Args:
        attentions: 各層の attention weights のタプル。各要素 (B, H, N, N)。
        head_fusion: ヘッドの集約方法 ("mean", "max", "min")。
        discard_ratio: 各層で attention の小さい上位何割を 0 にするか
            (ノイズ除去用、0 で素通し)。

    Returns:
        rollout 行列 (B, N, N)。row i, col j が「i から j への累積寄与」。
    """
    if len(attentions) == 0:
        raise ValueError("empty attentions")

    device = attentions[0].device
    batch, _, num_tokens, _ = attentions[0].shape

    rollout = torch.eye(num_tokens, device=device).unsqueeze(0).expand(batch, -1, -1).clone()

    for layer_attn in attentions:
        if head_fusion == "mean":
            fused = layer_attn.mean(dim=1)
        elif head_fusion == "max":
            fused = layer_attn.max(dim=1).values
        elif head_fusion == "min":
            fused = layer_attn.min(dim=1).values
        else:
            raise ValueError(f"unknown head_fusion: {head_fusion}")

        if discard_ratio > 0.0:
            flat = fused.view(batch, -1)
            k = int(flat.shape[1] * discard_ratio)
            if k > 0:
                threshold = flat.kthvalue(k, dim=-1, keepdim=True).values
                fused = torch.where(
                    fused < threshold.unsqueeze(-1),
                    torch.zeros_like(fused),
                    fused,
                )

        # residual connection を考慮 (Abnar & Zuidema 2020)
        identity = torch.eye(num_tokens, device=device).unsqueeze(0)
        fused = fused + identity
        fused = fused / fused.sum(dim=-1, keepdim=True)

        rollout = torch.bmm(fused, rollout)

    return rollout


def cls_attention_map(
    rollout: torch.Tensor,
    patch_grid_size: int,
) -> torch.Tensor:
    """
    rollout 行列から CLS トークンに集まる attention を 2D ヒートマップに変形。

    Args:
        rollout: (B, N, N)
        patch_grid_size: パッチの縦横数 (N = 1 + grid^2 を仮定)

    Returns:
        (B, grid, grid) のヒートマップ
    """
    cls_to_patches = rollout[:, 0, 1:]  # CLS から各パッチへの寄与
    return cls_to_patches.view(-1, patch_grid_size, patch_grid_size)


def save_comparison(
    clean_image: torch.Tensor,
    perturbed_image: torch.Tensor,
    clean_heatmap: torch.Tensor,
    perturbed_heatmap: torch.Tensor,
    output_path: str,
) -> None:
    """
    clean / 攻撃後の画像と attention map を 2x2 で並べて保存する。

    画像は (1, C, H, W)、heatmap は (1, grid, grid) を想定。
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    img_clean = clean_image[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    img_perturbed = perturbed_image[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    hm_clean = clean_heatmap[0].cpu().numpy()
    hm_perturbed = perturbed_heatmap[0].cpu().numpy()

    axes[0, 0].imshow(img_clean)
    axes[0, 0].set_title("clean image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_perturbed)
    axes[0, 1].set_title("with adversarial patch")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(hm_clean, cmap="hot")
    axes[1, 0].set_title("clean attention rollout")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(hm_perturbed, cmap="hot")
    axes[1, 1].set_title("hijacked attention rollout")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
