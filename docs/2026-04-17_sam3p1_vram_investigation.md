# SAM 3.1 multiplex video predictor の RTX 3060 (12 GB) 適合化 — 調査記録

- 日付: 2026-04-17
- 対象: `sam3/scripts/verify_sam3p1_phase3.py` (propagate_in_video フルビデオ検証)
- 結論: `sam3/model_builder.py:1190` の `batched_grounding_batch_size` を `16 → 1` に変更することで 12 GB GPU 上で動作可能化

---

## 1. 環境

- GPU: NVIDIA GeForce RTX 3060, 12 GB (Ampere, sm_86)
- CPU RAM: 15 GiB, Swap: 2 GiB
- OS: Linux 6.8.0-107-generic
- Python: 3.12, PyTorch: 2.10+cu128
- Repo: `facebookresearch/sam3` (SAM 3.1 release 2026-03-27)

## 2. 発生した問題

### 初回実行時
- `verify_sam3p1_phase3.py` を `assets/videos/0001` (270 フレーム, 1280×720) で実行したところ、**PC 全体がフリーズ**した。
- 原因は GPU OOM とホスト側 swap 圧迫の複合と推定。

### 最小条件での再現
- 30 フレーム・`max_num_objects=4`・overlay 無し条件でも `propagate_in_video` の**1 フレーム目で CUDA OOM**。
- エラー箇所: `sam3/model/maskformer_segmentation.py:222`
  `prev_fpn = curr_fpn + F.interpolate(prev_fpn, size=curr_fpn.shape[-2:], ...)`
- メモリ推移:
  | フェーズ | Allocated | Reserved |
  |---|---|---|
  | predictor build 後 | 3.65 GiB | 3.75 GiB |
  | add_prompt 後 | 4.34 GiB | 5.27 GiB |
  | 1 フレーム目 OOM 時 | - | **11.14 GiB** |

→ 単一フレーム推論で **+7 GiB 近くのスパイク**。

## 3. 原因の切り分け

### 真因(Codex 検証 + コード読解で確定)
1. **Detector 段のフレームチャンク**: `sam3/model_builder.py:1190` で `batched_grounding_batch_size=16` がハードコードされており、`forward_video_grounding_batched_multigpu` が **16 フレーム分の tri-neck 特徴 (288², 144², 72²) を同時にバッファリング** → 約 5.1 GiB
2. **per-query FPN 複製**: `sam3/model/maskformer_segmentation.py:119-127` の bs>1 パスで backbone 特徴を query 数分コピー → 約 0.85 GiB
3. **PixelDecoder の中間テンソル**: 約 0.6–1.0 GiB

合計で +7 GiB 前後のスパイクとなり、モデル常駐 (4.34 GiB) と合わせて 12 GB を超過。

### 誤っていた初期仮説
- `outputs_per_frame` 辞書へのマスク累積 — **無関係**。1 フレーム目で OOM するため到達しない。
- GitHub issue #511 の "split video / downsample resolution" — **別の障害モード** (accumulation 型 OOM) への助言で、per-frame ピーク超過には無効。

## 4. 試して効かなかった手段

| 手段 | 結論 | 理由 |
|---|---|---|
| `--use-fa3` (FlashAttention 3) | 使用不可 | **Hopper 専用**。`scripts/verify_sam3p1_smoke.py:63` のヘルプに "Hopper only; leave off for Ampere" と明記 |
| bf16/fp16 量子化の追加 | 既に適用済み | `sam3/model/sam3_multiplex_base.py:170-172` で `torch.autocast(..., dtype=bfloat16)` が常時 ON |
| `image_size` の削減 | 非現実的 | `sam3/model_builder.py` の 7+ 箇所にハードコード (`1008`)。pretrain 位置エンコーディング (pretrain_img_size=336) と RoPE との整合性崩壊、精度劣化必至 |
| フレーム分割 / 動画短縮 | 無効 | 1 フレーム目で OOM するため |
| 入力 JPEG 解像度ダウンサンプル | 無効 | モデル内部で 1008×1008 に強制リサイズされる |
| `torch.cuda.empty_cache()` | 実質無効 | 初回チャンクで evict できるものが無い |
| CPU offload | 実質無効 | text tensor は <0.1 GiB、backbone feature は forward で CUDA に戻される |
| `postprocess_batch_size=1` | 今回の OOM には無効 | スパイク地点 (detector 段) と異なる (出力後処理段) |

## 5. 適用した修正

### 変更箇所: `sam3/model_builder.py:1190`

```python
# 修正前
batched_grounding_batch_size=16,
# 修正後
batched_grounding_batch_size=1,
```

### なぜこれが安全か(Codex コード検証済み)

SAM 3.1 multiplex の**核心機能は `multiplex_count` が司る共有メモリ機構**であり、`batched_grounding_batch_size` とは完全独立。

| パラメータ | 役割 | 品質への影響 |
|---|---|---|
| `multiplex_count` | Mask decoder / Memory encoder の**オブジェクト間共有メモリ** (`mux()` / `demux()`) | SAM 3.1 品質の本体 |
| `batched_grounding_batch_size` | detector 段の**フレーム単位バッチ化** (throughput 最適化のみ) | **品質不変** |
| `postprocess_batch_size` | 出力後処理の**フレーム単位バッチ化** (throughput 最適化のみ) | 品質不変 |

`batched_grounding_batch_size` は `sam3/model/sam3_multiplex_base.py:696` で `detector.forward_video_grounding_batched_multigpu(..., batch_size=...)` に渡されるだけで、tracker が受け取る per-frame `det_out` は batch size に依存せず同一。downstream の propagation、mux/demux、object association、bucket management はすべて `batched_grounding_batch_size` に触れない。

副作用: **detector 段のスループット低下のみ**。

## 6. 動作確認結果

### 条件
- 動画: `tmp/sam3_small/` (30 フレーム, 1280×720, `assets/videos/0001/0.jpg`〜`29.jpg` をコピー)
- `max_num_objects=4`, overlay 無し, `batched_grounding_batch_size=1`

### 出力(`tmp/log/run_bg1.log`)
```
predictor built in 16.90s
after build          | allocated=3.65 GiB reserved=3.75 GiB
add_prompt text='person' in 1.19s, initial obj_ids=[0, 1, 2, 3]
before propagate     | allocated=4.34 GiB reserved=5.18 GiB
propagate: 30 frames in 13.99s (2.15 fps)
after propagate      | allocated=5.06 GiB reserved=6.10 GiB
seen_ids=[0, 1, 2, 3], nonempty_masks=120
Bucket utilization rate: 25.00%, subscription rate: 400.00%
Phase 3 propagation test PASSED
```

### メモリピーク比較

| 指標 | 修正前 | 修正後 |
|---|---|---|
| propagate peak (allocated) | 11+ GiB → **OOM** | **5.06 GiB** |
| propagate peak (reserved) | 11.14 GiB → OOM | 6.10 GiB |
| モニタ実測 GPU used peak | - | 6.56 GiB (6721 MiB) |
| モニタ実測 RAM used peak | - | 11.1 GiB (11331 MiB) |

→ Codex の推定 (5–7 GiB) とほぼ一致。12 GB GPU に対して約 5 GiB の余裕。

### スループット
- **2.15 fps** (30 frames / 14.0s)
- H100 の multiplex スループット目標 (128 オブジェクト 7x 高速化) とは別次元だが、動作検証用途としては実用的。

## 7. ディレクトリ構成(今回追加)

```
/home/takenouchi/AIxSuture/
├── tmp/
│   ├── sam3_small/              # 30 フレームサブセット (0.jpg〜29.jpg)
│   ├── monitor.sh               # free -h + nvidia-smi を 1s 間隔で記録
│   └── log/
│       ├── latest.env           # 最新モニタログのパス
│       ├── monitor_*.log        # タイムスタンプ付きリソース記録
│       └── run_bg1.log          # phase3 成功時の stdout
└── docs/
    └── 2026-04-17_sam3p1_vram_investigation.md   # 本ドキュメント
```

## 8. 今後の調査項目

### A. スケール耐性(優先度: 高)
1. **270 フレームフル動画 `assets/videos/0001`** で phase3 再試行
   - per-frame peak は不変のはずだが、`outputs_per_frame` 辞書累積でホスト RAM が膨らむ懸念
   - モニタログで RAM 推移を観測
2. **`max_num_objects=16`**(デフォルト)での動作確認
   - `subscription rate: 400%` の警告あり → 実運用では重要
3. **`bedroom.mp4`** など異なる動画での動作確認

### B. overlay 出力パイプライン(優先度: 中)
4. `--overlay-dir` + `--overlay-mp4` 有効化時のホスト RAM / ディスク挙動
5. 視覚的妥当性の目視確認

### C. スループット最適化(優先度: 中)
6. `batched_grounding_batch_size=2` / `4` が 12 GB に収まるか検証(速度向上の余地)
7. 2.15 fps からどこまで伸ばせるかの上限調査

### D. 品質回帰の確認(優先度: 中)
8. `batched_grounding_batch_size=1` vs `=16` での**マスク出力同一性**の数値検証
   - Codex はコード上同一性を保証しているが実測確認が理想(大容量 GPU 環境が必要)

### E. 他フェーズ(優先度: 低)
9. `verify_sam3p1_phase2.py` が同修正で動くか
10. `verify_sam3p1_smoke.py` (build のみ、影響なし見込み)

### F. AIxSuture 本体への統合(優先度: プロジェクト依存)
11. 画像単位 (`build_sam3_image_model`) / 動画単位 (修正済 multiplex) / 非-multiplex (`build_sam3_predictor(version="sam3")`) の使い分け設計
12. 上位 API 設計

## 9. 参考情報

### 検討したが採用しなかった代替案

- **(A) 画像単位処理 `build_sam3_image_model`**: 確実に 12 GB に収まるが、時間的追跡 (obj_id 永続) を失う。SAM 3 (非 3.1) 重みを使用
- **(B) 非-multiplex 動画 `build_sam3_predictor(version="sam3")`**: detector チャンクが単一 GPU 時 1 フレーム固定、追跡保持、SAM 3 重み。推定 peak 6–7 GiB。今回の修正が通ったため未検証

### 関連 Issue
- [facebookresearch/sam3 #511](https://github.com/facebookresearch/sam3/issues/511): RTX 4090 (24 GB) での OOM 事例。ただし accumulation 型で今回の per-frame peak 型とは別障害モード。

### 参考コード
- 修正箇所: `sam3/model_builder.py:1190`
- 元のデフォルト値(モジュール側): `sam3/model/sam3_multiplex_base.py:258` (`=1`) — builder が 16 で上書きしていた
- スパイクの原因箇所: `sam3/model/sam3_multiplex_base.py:671-697`, `sam3/model/maskformer_segmentation.py:119-127`
