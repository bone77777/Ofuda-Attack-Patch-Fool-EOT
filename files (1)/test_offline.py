"""
test_offline.py

ネットワーク不要のオフライン動作確認用テスト。

HuggingFace の CLIP の代わりに、torchvision の ViT-B/16 をランダム初期化で
使って、attention_loss / eot_transforms / 最適化ループ全体が
意図通りに動くかを検証する。

注: ランダム初期化ViTでは「意味的な hijack」は当然起きないが、
    "attention 重みが指定領域に集中するように画像を最適化できるか"
    というメカニズムレベルの正しさだけは確認できる。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from attention_loss import AttentionHijackLoss, patch_indices_for_region
from eot_transforms import EOTConfig, sample_eot_batch


# ---------------------------------------------------------------------------
# 最小 ViT (検証専用、ランダム初期化)
# ---------------------------------------------------------------------------

class TinyViT(nn.Module):
    """検証用の小さな ViT。本物の CLIP の代替。

    - 入力: (B, 3, 64, 64), patch=16 → 4x4=16 patch + CLS = 17 tokens
    - 各層で attention weights を返せる
    """

    def __init__(self, num_layers: int = 4, num_heads: int = 4, dim: int = 64) -> None:
        super().__init__()
        self.image_size = 64
        self.patch_size = 16
        self.patch_grid = self.image_size // self.patch_size
        self.num_patches = self.patch_grid ** 2
        self.num_tokens = self.num_patches + 1
        self.dim = dim
        self.num_heads = num_heads

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, dim))
        self.layers = nn.ModuleList([
            _AttentionBlock(dim, num_heads) for _ in range(num_layers)
        ])

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.patch_embed(image)
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, dim)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        attentions = []
        for layer in self.layers:
            x, attn = layer(x)
            attentions.append(attn)
        return tuple(attentions)


class _AttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm1(x)
        # need_weights=True で attention weights を取得 (averaged over heads)
        # ヘッドごとに必要なので average_attn_weights=False を使う
        attn_out, attn_weights = self.attn(
            normed, normed, normed,
            need_weights=True,
            average_attn_weights=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


# ---------------------------------------------------------------------------
# 検証テスト
# ---------------------------------------------------------------------------

def test_attention_loss_decreases() -> None:
    """ターゲットへの attention 集中度が最適化で実際に上がるかを確認する。"""
    torch.manual_seed(0)
    device = torch.device("cpu")

    model = TinyViT().to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # ターゲット: 中央の 2x2 patch 領域
    target_indices = patch_indices_for_region(
        patch_grid_size=model.patch_grid,
        region_xy=(1, 1, 2, 2),
        include_cls=True,
    )
    print(f"target patch indices: {target_indices}")

    loss_fn = AttentionHijackLoss(target_indices)

    # 画像領域: 中央 32x32 ピクセル
    base = torch.rand(1, 3, model.image_size, model.image_size, device=device)
    mask = torch.zeros(1, 1, model.image_size, model.image_size, device=device)
    mask[:, :, 16:48, 16:48] = 1.0

    patch = (torch.rand_like(base) * mask).requires_grad_(True)
    optimizer = torch.optim.Adam([patch], lr=0.05)

    config = EOTConfig(
        scale_range=(0.9, 1.1),
        rotation_deg_range=(-5.0, 5.0),
        blur_sigma_range=(0.0, 0.5),
        sensor_noise_std_range=(0.0, 0.01),
    )

    initial_score = None
    final_score = None

    for step in range(60):
        optimizer.zero_grad()
        composed = (base * (1 - mask) + patch * mask).clamp(0, 1)
        eot_batch = sample_eot_batch(composed, config, num_samples=4)
        attentions = model(eot_batch)
        loss = loss_fn(attentions)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            patch.data = (patch.data * mask).clamp(0, 1)

        if step == 0:
            initial_score = -loss.item()
        if step == 59:
            final_score = -loss.item()

        if step % 10 == 0:
            print(f"  step {step:3d}: hijack_score = {-loss.item():.4f}")

    print(f"\ninitial hijack score: {initial_score:.4f}")
    print(f"final   hijack score: {final_score:.4f}")
    assert final_score > initial_score, (
        "attention hijack score did not improve — optimization broken?"
    )
    print("[OK] hijack score improved — Patch-Fool style optimization is working")


def test_eot_invariance() -> None:
    """EOT サンプルが意味のある分布を作れているか確認する (簡易チェック)。"""
    image = torch.rand(1, 3, 64, 64)
    config = EOTConfig()
    batch = sample_eot_batch(image, config, num_samples=8)

    assert batch.shape == (8, 3, 64, 64), f"unexpected shape: {batch.shape}"
    assert batch.min() >= 0.0 and batch.max() <= 1.0, "out of [0, 1]"

    # サンプル間の差が無視できないほどあること = 多様な変換が効いている
    pairwise_diff = (batch[1:] - batch[:-1]).abs().mean().item()
    assert pairwise_diff > 0.01, f"EOT samples too similar: diff={pairwise_diff}"
    print(f"[OK] EOT samples are diverse (mean pairwise diff = {pairwise_diff:.4f})")


def test_patch_indices_helper() -> None:
    """patch_indices_for_region のインデックス計算が正しいか確認する。"""
    indices = patch_indices_for_region(
        patch_grid_size=7,
        region_xy=(0, 0, 1, 1),
        include_cls=True,
    )
    expected = [1, 2, 8, 9]
    assert indices == expected, f"expected {expected}, got {indices}"
    print(f"[OK] patch_indices_for_region: 2x2 corner region → {indices}")


if __name__ == "__main__":
    print("=" * 60)
    print("test_patch_indices_helper")
    print("=" * 60)
    test_patch_indices_helper()

    print("\n" + "=" * 60)
    print("test_eot_invariance")
    print("=" * 60)
    test_eot_invariance()

    print("\n" + "=" * 60)
    print("test_attention_loss_decreases")
    print("=" * 60)
    test_attention_loss_decreases()

    print("\nall tests passed.")
