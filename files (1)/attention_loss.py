"""
attention_loss.py

Patch-Fool スタイルの attention hijack 損失。

Patch-Fool (Fu et al., ICLR 2022) の中心アイデア:
    ViT の各 self-attention 層において、ターゲットパッチ位置 p* に対して
    他の全パッチから集まる attention weight の総和を最大化する勾配を
    入力 patch ピクセルに伝搬させる。

すなわち、l 層目、h ヘッド目の attention weight 行列を A^{(l,h)} ∈ R^{N×N}
(N はパッチトークン数) としたとき、

    L_attn(p*) = - sum_{l, h} sum_{i ≠ p*} A^{(l,h)}_{i, p*}

を最小化する (= ターゲットパッチへの集中度を最大化する)。

本実装では HuggingFace transformers の CLIPVisionModel から各層の
attention weights を出力させ、ターゲットパッチ列への流入総和を集計する。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHijackLoss(nn.Module):
    """
    Patch-Fool 風 attention hijack 損失。

    Args:
        target_patch_indices: ターゲットにする patch token インデックスのリスト。
            CLIP ViT-B/32 (224x224 入力, patch=32) なら、CLS トークンを含めて
            1 + 7*7 = 50 トークン。インデックス 0 が CLS、1..49 が patch。
        layer_weights: 各層の損失への寄与重み。None なら全層等価。
        head_reduction: マルチヘッドの集約方法 ("mean" or "sum")。
    """

    def __init__(
        self,
        target_patch_indices: list[int],
        layer_weights: list[float] | None = None,
        head_reduction: str = "mean",
    ) -> None:
        super().__init__()
        if len(target_patch_indices) == 0:
            raise ValueError("target_patch_indices must be non-empty")
        self.register_buffer(
            "target_indices",
            torch.tensor(target_patch_indices, dtype=torch.long),
            persistent=False,
        )
        self.layer_weights = layer_weights
        self.head_reduction = head_reduction

    def forward(self, attentions: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Args:
            attentions: HuggingFace の output_attentions=True で得られる
                各層の attention weights のタプル。各要素は
                (batch, heads, num_tokens, num_tokens)。

        Returns:
            損失スカラー (= -hijack score)。最小化すると hijack が強まる。
        """
        if self.layer_weights is None:
            weights = [1.0 / len(attentions)] * len(attentions)
        else:
            if len(self.layer_weights) != len(attentions):
                raise ValueError(
                    f"layer_weights length {len(self.layer_weights)} "
                    f"does not match number of attention layers {len(attentions)}"
                )
            weights = self.layer_weights

        target = self.target_indices.to(attentions[0].device)

        total = 0.0
        for layer_attn, w in zip(attentions, weights):
            # layer_attn: (B, H, N, N) — A[i, j] = i から j への attention
            # ターゲット列 j = target_indices への流入総和を取る
            # selected: (B, H, N, |target|)
            selected = layer_attn.index_select(dim=-1, index=target)

            # 他パッチからターゲットへの流入を全インデックスで合計
            # (CLS と target 自身を含むことになるが、Patch-Fool 原典に従い
            #  target 自身は除外しない — self-attention の対角成分は通常小さく
            #  最適化の挙動には大きく影響しない)
            # influx: (B, H, |target|)
            influx = selected.sum(dim=-2)

            # |target| 軸を合計してターゲット集合への総流入にする
            # (B, H)
            head_total = influx.sum(dim=-1)

            if self.head_reduction == "mean":
                layer_score = head_total.mean(dim=-1)  # (B,)
            elif self.head_reduction == "sum":
                layer_score = head_total.sum(dim=-1)
            else:
                raise ValueError(f"unknown head_reduction: {self.head_reduction}")

            total = total + w * layer_score  # (B,)

        # 最小化問題に変換 (hijack を強めるほど total は大きくなる)
        return -total.mean()


def patch_indices_for_region(
    patch_grid_size: int,
    region_xy: tuple[int, int, int, int],
    include_cls: bool = True,
) -> list[int]:
    """
    画像上の矩形領域 (x0, y0, x1, y1) を覆う patch の token インデックスを返す。

    Args:
        patch_grid_size: 一辺あたりのパッチ数 (例: 224x224 入力 + patch=32 → 7)
        region_xy: パッチグリッド座標系での (x0, y0, x1, y1)。両端含む。
        include_cls: CLS トークン分のオフセット +1 を入れるか。

    Returns:
        token インデックスのリスト。
    """
    x0, y0, x1, y1 = region_xy
    if not (0 <= x0 <= x1 < patch_grid_size and 0 <= y0 <= y1 < patch_grid_size):
        raise ValueError(f"region {region_xy} out of grid {patch_grid_size}")

    indices = []
    offset = 1 if include_cls else 0
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            indices.append(offset + y * patch_grid_size + x)
    return indices
