# §13 スループット最適化検証

→ [目次に戻る](README.md)

## 13.1 検証目的

[§8.C](remaining_tasks.md#c-スループット最適化優先度-中--§13-で完了) の積み残しに対応:`batched_grounding_batch_size`(以下 `bs`)を
1 → 2 / 3 / 4 にスイープし、**(E1-1) 12 GB GPU での適合上限**、
**(E1-2) スループット改善余地**(2.15 fps からの伸び)を測定する。

## 13.2 試験条件

- 入力: `tmp/bedroom_jpg/` (200 frames, 960×540) — [§11](04_video.md) C1-JPG / [§12](05_overlay.md) D1 と同一
- `obj=4`, `prompt="person"`, 検出 obj 数 = 2(全試験で一定)
- 編集箇所: `sam3/model_builder.py:1190` の `batched_grounding_batch_size`(各試行ごとに書き換え)
- 試験後 **bs=1 に復元済**(プロジェクト規約準拠)
- ログ: `tmp/log/run_E{1,2,3}_*.log`、`tmp/log/scale_E{1,2,3}_*.log`

## 13.3 結果一覧

| 試行 | bs | 検出 obj | 推論 fps | 推論時間 | before-prop alloc | after-prop alloc / res (GiB) | **GPU peak (MiB)** | RAM peak (MiB) | 結果 |
|---|---|---|---|---|---|---|---|---|---|
| C1-JPG (基準) | 1 | 2 | **2.15** | 93.0 s | 4.40 GiB | 7.48 / 8.83 | **9,519** | — | ✓ |
| E1 | 2 | 2 | 2.08 | 96.15 s | — | 7.82 / 9.62 | **10,301** | 10,330 | ✓ |
| E3 | **3** | 2 | **2.15** | 93.13 s | 4.40 GiB | 7.82 / **11.00** | **11,739** | 9,656 | ✓ (極限) |
| E2 | 4 | — | — | — | 4.41 GiB | — / — | **11,901+** | — | ✗ **OOM** |

**OOM 詳細(E2, bs=4):** 1 フレーム目の detector 段で `Tried to allocate 324.00 MiB. GPU 0 has a total capacity of 11.63 GiB of which 170.25 MiB is free`(PyTorch 10.11 GiB + reserved-unallocated 916 MiB)。

## 13.4 主要な定量的知見

### E1-1: 12 GB GPU での適合上限は **bs=3**

| bs | GPU peak | 12 GB 残量 | 増分 (vs 前段) |
|---|---|---|---|
| 1 | 9,519 MiB | 2,769 MiB | — |
| 2 | 10,301 MiB | 1,987 MiB | +782 |
| 3 | 11,739 MiB | **549 MiB** | +1,438 |
| 4 | OOM (11,901+) | — | OOM |

- bs=1→2 の増分(+782 MiB)は [§3](01_problem.md#3-原因の切り分け) の「5.1 GiB / 16 frames ≒ 327 MiB/frame」予測の約 2.4 倍 — 検出 obj が少ないため per-bs オーバーヘッドが相対的に大きい
- bs=2→3 の増分(+1,438 MiB)は非線形に跳ね上がり、PyTorch reserved も 9.62 → 11.00 GiB へ急増
- **bs=4 は detector 段の 4 フレーム同時バッファリングで即 OOM** — 12 GB GPU では **bs=3 が硬い上限**

### E1-2: スループット改善は **得られない**(本シーンでは)

| bs | fps | vs bs=1 |
|---|---|---|
| 1 | 2.15 | — |
| 2 | 2.08 | **-3.3%** |
| 3 | 2.15 | ±0% |

- **bs を上げても fps は伸びない**。bs=2 はむしろ僅かに遅くなり、bs=3 で bs=1 と同値に戻るのみ
- 原因の推測: 検出 obj 数が **2 件しかない**ため、detector 段の per-frame コストが既に最小化されており、フレームバッチ化のオーバーヘッド(同期・メモリアロケーション)が利得を相殺
- **結論: 本シーンでは bs=1 が時間効率・メモリ効率ともに最適**

### E1-3: ボトルネックは detector ではない

[§3](01_problem.md#3-原因の切り分け) では bs=16 由来の +5.1 GiB が detector 段のスパイクと同定したが、本検証は
**スループット律速は detector 段ではない**ことを示唆する:

- bs=1〜3 で fps がほぼ一定(2.08〜2.15) → detector 段の処理時間は per-frame 全体の小さな割合
- 律速は **propagation 段(memory encoder / mux/demux / mask decoder)** にあると推定
- 本物の高速化には bs スイープではなく以下が必要:
  1. **`compile=True`**(現在 `False`)→ torch.compile による fused kernel(要 warm-up)
  2. **`use_fa3=True`**(Hopper 専用、Ampere 不可)
  3. **propagation 段の autocast / kv-cache 最適化**

## 13.5 多 obj シーンでの再評価が必要

本検証は検出 obj=2 の小ケース。obj 数が増えると detector の per-query コストが上がるため、
batching の利得が顕在化する可能性がある:

| 仮説 | 検証提案 |
|---|---|
| obj=8〜16 で bs=2/3 が +5〜10% 速くなる | `--prompt-text window` (16 obj) で bs=1/2/3 を再測 |
| 高解像度(1280×720)でも同様の OOM 上限 | `assets/videos/0001` で bs スイープ |

## 13.6 運用ガイドライン(更新版)

- **デフォルトは `bs=1` を維持**(本リポジトリの修正値)— 12 GB GPU での安全マージン最大、本シーンではスループットも最適
- **`bs=2` を試す価値があるケース** — 検出 obj が 8 以上(detector per-query コスト大)、かつ GPU に 1.5 GiB 以上の余裕があるとき
- **`bs=3` は非推奨** — 12 GB の残量がわずか 549 MiB、わずかな入力変動で OOM 危険
- **`bs=4+` は 12 GB GPU では不可** — detector 段で確実に OOM
- 真のスループット改善は `compile=True` と warm-up の組み合わせを次に検証すべき([§8.C](remaining_tasks.md#c-スループット最適化優先度-中--§13-で完了) の余地拡大)

## 13.7 再現方法

```bash
# 各 bs ごとに sam3/model_builder.py:1190 を編集
# (bs=1 ↔ bs=2 ↔ bs=3 ↔ bs=4)

# モニタ起動
TS=$(date +%Y%m%d_%H%M%S)
LOG=/home/takenouchi/AIxSuture/tmp/log/scale_E_bs${BS}_${TS}.log
bash /home/takenouchi/AIxSuture/tmp/monitor.sh "$LOG" &

# 推論
cd /home/takenouchi/AIxSuture/sam3
uv run python scripts/verify_sam3p1_phase3.py \
    --video-path /home/takenouchi/AIxSuture/tmp/bedroom_jpg \
    --max-num-objects 4 \
    --prompt-text person
```

**重要:** 試験後は必ず `batched_grounding_batch_size=1` に復元すること(本ドキュメントの主修正)。
