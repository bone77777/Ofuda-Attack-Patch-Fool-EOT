"""
demo.py

CPU で数分で完走する縮小デモ。

実行例:
    python demo.py

出力:
    ./demo_outputs/comparison.png   (clean / 攻撃後の attention map 比較)
    ./demo_outputs/patch.png        (生成されたお札パッチ)
"""

from __future__ import annotations

from pathlib import Path

import torch
import numpy as np
from PIL import Image

from attack import optimize_patch


def make_synthetic_warehouse_image(size: int = 224) -> Image.Image:
    """
    テスト用の合成「倉庫風」画像を生成する。

    実物の写真を使う代わりに、棚と床と通路風のグラデーションを描いた
    合成画像で動作確認する。これにより外部画像ファイルへの依存をなくし、
    PoCの再現性を確保する。
    """
    rng = np.random.default_rng(seed=42)

    # 床 (下半分): 灰色のグラデーション
    floor = np.linspace(0.45, 0.65, size // 2)[:, None] * np.ones(size)
    floor = np.stack([floor, floor, floor * 0.95], axis=-1)

    # 壁 (上半分): クリーム色
    wall = np.full((size - size // 2, size, 3), [0.82, 0.78, 0.70])

    image = np.concatenate([wall, floor], axis=0)

    # 棚 (縦の帯)
    for shelf_x in [40, 110, 180]:
        image[20:140, shelf_x:shelf_x+15] = [0.55, 0.40, 0.25]

    # 棚の上に箱 (青いコンテナを模す)
    image[60:90, 45:75] = [0.20, 0.35, 0.65]
    image[60:90, 115:145] = [0.65, 0.20, 0.20]  # 赤タグ風
    image[60:90, 185:215] = [0.20, 0.55, 0.30]

    # ノイズ少々
    image = image + rng.normal(0, 0.015, image.shape)
    image = np.clip(image, 0, 1)

    pil = Image.fromarray((image * 255).astype(np.uint8))
    return pil


def main() -> None:
    output_dir = Path("./demo_outputs")
    output_dir.mkdir(exist_ok=True)

    base_image_path = output_dir / "base.png"
    if not base_image_path.exists():
        synth = make_synthetic_warehouse_image(size=224)
        synth.save(base_image_path)
        print(f"created synthetic base image at {base_image_path}")

    # 縮小設定 (CPU 想定)
    optimize_patch(
        base_image_path=base_image_path,
        output_dir=output_dir,
        target_patch_xy=(2, 3, 4, 5),       # 中央付近のパッチ領域 (3x3)
        region_pixels=(64, 96, 159, 191),   # 96x96 ピクセル領域
        steps=80,
        eot_samples=4,
        learning_rate=0.03,
        log_every=10,
        device="cpu",
    )


if __name__ == "__main__":
    main()
