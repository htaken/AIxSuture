# §1–3 環境・発生した問題・原因の切り分け

→ [目次に戻る](README.md)

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
