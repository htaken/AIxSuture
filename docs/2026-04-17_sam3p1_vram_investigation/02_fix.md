# §4–6, §9 試行・修正・初回動作確認・参考情報

→ [目次に戻る](README.md)

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
