# AIxSuture × SAM 3 統合 — Phase 0 + Phase 1 記録

- 日付: 2026-04-20
- 目的: 縫合シーンで SAM 3 がどの物体をどの程度取得できるかを単一フレーム画像で探索

## 確定事項

| 項目 | 値 | 備考 |
|---|---|---|
| Model version | **SAM 3(無印)** | SAM 3.1 image checkpoint は HF に存在しない(`download_ckpt_from_hf` が `sam3.1_multiplex.pt` のみ提供)。Phase 2 以降の video 伝播は multiplex 3.1 を使用予定 |
| Image API | `build_sam3_image_model()` + `Sam3Processor` | `sam3/README.md:119-137` に準拠 |
| Checkpoint path | HF default: `facebook/sam3` (`sam3.pt`) | CSV 列 `checkpoint_path` に記録 |
| Confidence threshold | **0.3** | `Sam3Processor(confidence_threshold=0.3)` 明示指定 |
| Device | `cuda` (RTX 3060 12GB) | |
| Seed | 0 | overlay 色パレット用 |
| Autocast | `torch.autocast("cuda", dtype=torch.bfloat16)` 必須 | `sam3_image_predictor_example.ipynb:98-99` に準拠 |

## 対象動画(Phase 1 sweep)

全 3 本、いずれも 1920×1080 / 29.97 fps / ≈303 秒 / cv2 読取可:

| Package | File | Duration | FPS | Resolution |
|---|---|---|---|---|
| Package 01 | `A31H.mp4` | 302.8s | 29.97 | 1920×1080 |
| Package 05 | `K78J.mp4` | 303.7s | 29.97 | 1920×1080 |
| Package 10 | `X14A.mp4` | 301.9s | 29.97 | 1920×1080 |

各動画から 3 フレーム(`t=5s`, `t=duration/2`, `t=duration-5s`)抽出 → 合計 9 フレーム。

## プロンプト語彙 v1

単数形 + 複合語を併用し、open vocabulary での表現差を比較:

```
hand, glove, finger,
needle, surgical needle, needle holder,
forceps, surgical forceps, scissors,
thread, suture thread,
skin, tissue, practice pad
```

合計 **14 語**。

## Phase 0 完了条件(達成状況)

| 条件 | 状態 |
|---|---|
| 9 フレーム JPG 抽出 | ✅ `tmp/aixsuture/probe_frames/` に 9 枚 |
| smoke test 完走 | ✅ `tmp/aixsuture/probe_results/smoke.log` |
| VRAM 実測 | ✅ build 後 allocated=3.33 GiB / reserved=3.43 GiB、end 時 allocated=3.46 GiB / reserved=4.13 GiB(12 GB 上限に十分な余裕) |
| CSV 1 行 + overlay 1 枚 | ✅ `smoke.csv` / `overlays/A31H_t5_hand.png` |
| checkpoint/threshold 記録 | ✅ 本 README |

### Smoke test メトリクス

| 項目 | 値 |
|---|---|
| model build time | 6.1s(HF cache hit 後) |
| set_image time | 0.40s / frame |
| set_text_prompt time(hand) | 0.16s |
| 検出数 | 2 |
| max score | 0.973 |
| mask area ratio | 0.177(フレーム面積の 17.7%) |

## ファイル構成

```
/home/takenouchi/AIxSuture/
├── scripts/
│   ├── extract_probe_frames.py   # Step 0.2
│   └── probe_image.py            # Step 0.3 smoke + Phase 1 sweep
├── tmp/aixsuture/
│   ├── probe_frames/             # 9 JPG
│   └── probe_results/
│       ├── overlays/             # overlay PNG
│       ├── smoke.csv / smoke.log
│       └── summary.csv / run.log (Phase 1 で生成)
└── docs/2026-04-20_aixsuture_integration/
    ├── README.md                    # 本ファイル
    ├── phase1_results.md            # Phase 1 raw 結果
    └── data_capability_matrix.md    # 取得可否マトリクス(Phase 2 設計用)
```

## ドキュメント構成

| ファイル | 用途 |
|---|---|
| `README.md` | 全体インデックス、確定事項、実行コマンド |
| `phase1_results.md` | Phase 1 の raw 実験結果(プロンプト別集計、動画横断性、定性評価) |
| `data_capability_matrix.md` | CSV 多角分析による「取れる / 取れない」の判断材料と Phase 2 設計への示唆 |

## 実行コマンド

```bash
cd /home/takenouchi/AIxSuture/sam3

# Phase 0.2 フレーム抽出
uv run python ../scripts/extract_probe_frames.py \
    --videos \
        "../datasets/Package 01/A31H.mp4" \
        "../datasets/Package 05/K78J.mp4" \
        "../datasets/Package 10/X14A.mp4" \
    --out-dir ../tmp/aixsuture/probe_frames

# Phase 0.3 smoke test
uv run python ../scripts/probe_image.py \
    --frames-dir ../tmp/aixsuture/probe_frames \
    --output-dir ../tmp/aixsuture/probe_results \
    --smoke

# Phase 1 full sweep(14 prompts × 9 frames = 126 runs)
uv run python ../scripts/probe_image.py \
    --frames-dir ../tmp/aixsuture/probe_frames \
    --output-dir ../tmp/aixsuture/probe_results
```
