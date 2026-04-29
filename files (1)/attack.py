"""
attack.py

Patch-Fool 風 attention hijack と EOT 物理ロバスト化を組み合わせた
adversarial patch 最適化のメインループ。

最適化対象は「画像内の指定矩形領域に貼る patch ピクセル」。
画像本体は変更しない。これは物理世界での「お札」を貼る挙動を模している。

最小化する目的関数:
    L_total = L_attn(patched_image_under_eot)
            + lambda_tv * TV(patch)
            + lambda_nps * NPS(patch)

ただし NPS (Non-Printability Score) は実機印刷キャリブレーションを伴う
ため、本概念実証では参考実装にとどめ、デフォルトでは無効化している。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModel, CLIPImageProcessor

from attention_loss import AttentionHijackLoss, patch_indices_for_region
from eot_transforms import EOTConfig, sample_eot_batch
from visualize import attention_rollout, cls_attention_map, save_comparison


# ---------------------------------------------------------------------------
# モデルラッパ
# ---------------------------------------------------------------------------

class CLIPViTWrapper(torch.nn.Module):
    """HuggingFace CLIP ViT を attention 出力モードで包む薄いラッパ。"""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32") -> None:
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # CLIP の前処理パラメータを取得
        processor = CLIPImageProcessor.from_pretrained(model_id)
        self.register_buffer(
            "image_mean",
            torch.tensor(processor.image_mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(processor.image_std).view(1, 3, 1, 1),
            persistent=False,
        )
        self.input_size = processor.size["shortest_edge"]
        self.patch_size = self.model.config.patch_size
        self.patch_grid = self.input_size // self.patch_size

    def forward(self, image_01: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """[0, 1] 範囲の画像を受け取り、各層の attention weights を返す。"""
        normalized = (image_01 - self.image_mean) / self.image_std
        outputs = self.model(pixel_values=normalized, output_attentions=True)
        return outputs.attentions


# ---------------------------------------------------------------------------
# パッチマスクの管理
# ---------------------------------------------------------------------------

def make_patch_mask(
    image_size: int,
    region_pixels: tuple[int, int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """画像サイズに合わせて、パッチ領域のみ 1 のマスクを作る。

    Args:
        image_size: 画像の一辺 (正方形を仮定)
        region_pixels: (x0, y0, x1, y1) ピクセル座標で両端含む

    Returns:
        (1, 1, H, W) のマスク
    """
    x0, y0, x1, y1 = region_pixels
    mask = torch.zeros(1, 1, image_size, image_size, device=device)
    mask[:, :, y0:y1+1, x0:x1+1] = 1.0
    return mask


def total_variation(patch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """パッチ領域内の TV ノルム (印刷可能性向上のため)。"""
    masked = patch * mask
    diff_h = torch.abs(masked[:, :, 1:, :] - masked[:, :, :-1, :])
    diff_w = torch.abs(masked[:, :, :, 1:] - masked[:, :, :, :-1])
    return diff_h.mean() + diff_w.mean()


# ---------------------------------------------------------------------------
# 最適化ループ
# ---------------------------------------------------------------------------

def optimize_patch(
    base_image_path: Path,
    output_dir: Path,
    target_patch_xy: tuple[int, int, int, int],
    region_pixels: tuple[int, int, int, int],
    steps: int = 500,
    eot_samples: int = 8,
    learning_rate: float = 0.02,
    tv_weight: float = 0.001,
    log_every: int = 25,
    model_id: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
) -> None:
    """
    Args:
        base_image_path: ベース画像 (パッチを貼る背景)。何でもよい。
        output_dir: 出力先
        target_patch_xy: hijack ターゲットにする patch グリッド領域 (x0, y0, x1, y1)。
            ここに集中させたい。普通は region_pixels と同じ場所を指定する
            (= パッチ領域自身に attention を集中させる) が、別領域も可能。
        region_pixels: 摂動を加える画像ピクセル領域 (x0, y0, x1, y1)。
            これがそのまま「お札」の物理サイズ・位置の縮尺。
        steps: 最適化ステップ数
        eot_samples: 各ステップでサンプリングする EOT 変換数
        learning_rate: 学習率
        tv_weight: total variation 損失の重み
        log_every: 何ステップごとにログを出すか
        model_id: HuggingFace モデル ID
        device: "cpu" or "cuda"
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    # モデル
    model = CLIPViTWrapper(model_id).to(dev)
    print(f"loaded {model_id}: input_size={model.input_size}, "
          f"patch_size={model.patch_size}, grid={model.patch_grid}")

    # ターゲットインデックス
    target_indices = patch_indices_for_region(
        patch_grid_size=model.patch_grid,
        region_xy=target_patch_xy,
        include_cls=True,
    )
    print(f"hijack target patch indices ({len(target_indices)} tokens): {target_indices}")

    loss_fn = AttentionHijackLoss(target_indices).to(dev)

    # ベース画像のロード
    base_pil = Image.open(base_image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(model.input_size),
        transforms.CenterCrop(model.input_size),
        transforms.ToTensor(),  # [0, 1]
    ])
    base_image = transform(base_pil).unsqueeze(0).to(dev)

    # パッチマスクと初期パッチ
    mask = make_patch_mask(model.input_size, region_pixels, dev)
    patch = torch.rand_like(base_image) * 0.5 + 0.25
    patch = patch * mask  # マスク外は 0
    patch.requires_grad = True

    optimizer = torch.optim.Adam([patch], lr=learning_rate)
    eot_config = EOTConfig()

    # 最適化ループ
    for step in range(steps):
        optimizer.zero_grad()

        # パッチを画像に貼る (マスク領域だけ patch、他は base_image)
        patched = base_image * (1.0 - mask) + patch * mask
        patched = patched.clamp(0.0, 1.0)

        # EOT サンプリング
        eot_batch = sample_eot_batch(patched, eot_config, eot_samples)

        # attention 取得
        attentions = model(eot_batch)

        # 損失
        attn_loss = loss_fn(attentions)
        tv_loss = total_variation(patch, mask)
        total_loss = attn_loss + tv_weight * tv_loss

        total_loss.backward()
        optimizer.step()

        # マスク外を強制ゼロ + [0, 1] へクランプ
        with torch.no_grad():
            patch.data = patch.data * mask
            patch.data = patch.data.clamp(0.0, 1.0)

        if step % log_every == 0 or step == steps - 1:
            print(
                f"step {step:4d}/{steps} | "
                f"attn_loss={attn_loss.item():+.4f} "
                f"tv={tv_loss.item():.4f} "
                f"total={total_loss.item():+.4f}"
            )

    # 最終評価
    with torch.no_grad():
        patched_final = base_image * (1.0 - mask) + patch * mask
        patched_final = patched_final.clamp(0.0, 1.0)

        clean_attn = model(base_image)
        adv_attn = model(patched_final)

        clean_rollout = attention_rollout(clean_attn)
        adv_rollout = attention_rollout(adv_attn)

        clean_map = cls_attention_map(clean_rollout, model.patch_grid)
        adv_map = cls_attention_map(adv_rollout, model.patch_grid)

        save_comparison(
            base_image, patched_final, clean_map, adv_map,
            str(output_dir / "comparison.png"),
        )

        # ターゲット領域への CLS 集中度を測る
        ty0, tx0, ty1, tx1 = (
            target_patch_xy[1], target_patch_xy[0],
            target_patch_xy[3], target_patch_xy[2],
        )
        clean_target_mass = clean_map[:, ty0:ty1+1, tx0:tx1+1].sum().item()
        adv_target_mass = adv_map[:, ty0:ty1+1, tx0:tx1+1].sum().item()
        print(
            f"\nCLS attention mass on target region:\n"
            f"  clean : {clean_target_mass:.4f}\n"
            f"  adv   : {adv_target_mass:.4f}\n"
            f"  ratio : {adv_target_mass / max(clean_target_mass, 1e-8):.2f}x"
        )

        # パッチ画像も保存
        patch_only = (patch * mask).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy()
        from matplotlib import pyplot as plt
        plt.imsave(str(output_dir / "patch.png"), patch_only)
        print(f"saved outputs to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Patch-Fool x EOT adversarial patch PoC")
    parser.add_argument("--base-image", type=Path, required=True,
                        help="ベース背景画像のパス")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eot-samples", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--target-patch", type=int, nargs=4, default=[1, 1, 2, 2],
                        metavar=("X0", "Y0", "X1", "Y1"),
                        help="hijack するパッチグリッド領域")
    parser.add_argument("--region-pixels", type=int, nargs=4, default=[32, 32, 95, 95],
                        metavar=("X0", "Y0", "X1", "Y1"),
                        help="お札を貼る画像ピクセル領域")
    args = parser.parse_args()

    optimize_patch(
        base_image_path=args.base_image,
        output_dir=args.output_dir,
        target_patch_xy=tuple(args.target_patch),
        region_pixels=tuple(args.region_pixels),
        steps=args.steps,
        eot_samples=args.eot_samples,
        learning_rate=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
