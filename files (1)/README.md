# Ofuda Attack — Patch-Fool × EOT 概念実証

セキュリティキャンプ 問５ 応募エッセイで言及した「お札貼り攻撃」の **ローカル概念実証** リポジトリです。Vision Transformer (ViT) ベースの **CLIP** モデルに対し、Patch-Fool 方式の attention hijack と Expectation Over Transformation (EOT) による物理ロバスト化を組み合わせた敵対的パッチを最適化します。

## このリポジトリでやっていること / やっていないこと

### やっていること
- **CLIP ViT-B/32 (HuggingFace) のローカル推論** に対して、attention 重みが指定パッチに集中するような adversarial patch を勾配降下で最適化
- 視点・照明・印刷色再現誤差・カメラノイズをサンプリングする EOT 変換ループの実装
- attention rollout による可視化と、変換に対するロバスト性の評価

### やっていないこと（意図的に含めていない）
- 実在のロボット (OpenVLA, RT-2, VLAS, π0 等) への攻撃適用
- 物理印刷キャリブレーション (色再現プロファイル特定、用紙選定、貼付位置最適化など)
- 特定企業・特定施設を想定したペイロード設計
- 言語側 typographic injection と組み合わせた end-to-end のVLA goal hijack

これは **「ViT の attention は物理ロバストな patch で hijack 可能か」** という学術的問いに対する最小限の概念実証であって、実機攻撃のレシピではありません。物理世界への適用には、ターゲットVLAの正確な仕様取得、ファインチューニング後の attention 構造解析、印刷再現性キャリブレーション、対象施設への合法的アクセス、そしていずれも欠けてはならない倫理審査と当事者の同意が、本コードの外側で別途必要です。

## 想定する読み手

防御研究者、レッドチーマー、VLA セキュリティ研究を始めたい学部生 (自分含む)。

## 学術的背景

| 手法 | 典拠 | 本実装での役割 |
|------|------|---------------|
| Adversarial Patch | Brown et al., arXiv:1712.09665, 2017 | パッチ攻撃の出発点 |
| Patch-Fool | Fu et al., ICLR 2022 | ViT self-attention をターゲットにする損失関数の定式化 |
| EOT | Athalye et al., ICML 2018 | 物理変換に対するロバスト最適化 |
| Robust Physical-World Attacks | Eykholt et al., CVPR 2018 | 印刷物理攻撃でのEOT実証 (本実装は道路標識ではなくViTを対象) |
| Multimodal Neurons / Typographic Attack | Goh et al., Distill, 2021 | 攻撃の意味論的メカニズム説明 |

## 構成

```
ofuda_attack/
├── README.md             # これ
├── requirements.txt      # 依存ライブラリ
├── attack.py             # メイン: 最適化ループ
├── attention_loss.py     # Patch-Fool 風 attention hijack 損失
├── eot_transforms.py     # EOT 変換セット
├── visualize.py          # attention rollout 可視化
└── demo.py               # 最小実行例 (CPUでも数分で完走)
```

## 実行方法

```bash
pip install -r requirements.txt
python demo.py  # CPU で数分の縮小版
python attack.py --steps 500 --eot-samples 8  # GPU 推奨
```

## 倫理的使用について

本コードの利用にあたっては以下を遵守してください:
1. 自分が所有または明示的許諾を得たモデルに対してのみ実行する
2. 実機ロボットを稼働させる現場での試験は行わない
3. 公開・配布する場合は本リポジトリと同等の注意書きを必ず付ける
4. 研究成果は防御メカニズムの設計提案とセットで公表する
