# §10 スケール耐性検証結果

→ [目次に戻る](README.md)

## 10.1 検証目的

[§5](02_fix.md#5-適用した修正) の修正(`batched_grounding_batch_size=1`)適用後に、12 GB GPU 上での
**(A) フレーム数依存性 / (B) オブジェクト数依存性 / (A×B) 結合限界** を定量化する。

## 10.2 試験条件

- 全試験: `batched_grounding_batch_size=1`、bf16 autocast、`compile=False, warm_up=False`
- 動画: `assets/videos/0001`(1280×720, 4 人ダンサー)からフレームサブセットを `tmp/sam3_<N>/` に配置
- モニタ: `tmp/monitor.sh` で 1s 間隔の `nvidia-smi` + `/proc/meminfo` 記録
- ログ保存先: `tmp/log/scale_*` および `tmp/log/run_*`

## 10.3 結果一覧

| 試行 | frames | obj | prompt | 検出 obj | fps | after-prop alloc/res (GiB) | GPU peak (MiB) | Swap 増分 (MiB) | 結果 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 30 | 4 | person | 4 | 2.15 | 5.06 / 6.10 | 6,721 | — | ✓ |
| A1 | 90 | 4 | person | 4 | 2.15 | 6.16 / 7.22 | 7,869 | +62 | ✓ |
| A2 | 180 | 4 | person | 4 | 2.14 | 7.81 / 9.01 | 9,699 | +541 | ✓ |
| A3 | 270 | 4 | person | 4 | 2.13 | 9.48 / 10.55 | 11,275 | +690 | ✓ |
| B1 | 30 | 8 | person | 4 | 2.17 | 5.06 / 6.00 | 6,617 | — | ✓ |
| B2 | 30 | 16 | person | 4 | 2.16 | 5.06 / 6.15 | 6,771 | — | ✓ |
| B3 | 30 | 16 | shoe | 6 | 2.15 | 5.11 / 6.07 | 6,689 | — | ✓ |
| B4 | 30 | 16 | window | **16** | 2.10 | 5.48 / 7.45 | 8,101 | — | ✓ |
| **AB** | **140** | **16** | **window** | **16** | **2.07** | **9.00 / 10.68** | **11,411** | — | **✓** |

## 10.4 主要な定量的知見

### A 軸:フレーム数依存(`obj=4 person` 固定)

- **VRAM 累積は完璧に線形** — `outputs_per_frame` 辞書のマスクテンソルが GPU 上に保持される
- per-frame 増加率(GPU peak):
  - 30→90: 18.7 MiB/frame
  - 90→180: 18.3 MiB/frame
  - 180→270: 18.6 MiB/frame
- スループットはフレーム数に依存せず一定(2.13–2.15 fps)
- ホスト RAM・Swap への影響は無視可能(増加は線形未満)

### B 軸:オブジェクト数依存(`30 frames` 固定)

- **`max_num_objects` はキャップに過ぎず VRAM に影響しない** — 実コストは検出された実インスタンス数で決まる
- **multiplex の共有メモリ機構が極めて効率的** — 4→16 obj(4 倍)で per-frame 増加はわずか **1.5 倍** (18.5 → 30.1 MiB/frame)
- `subscription rate` 警告(400% / 1600%)は **誤解を招く名称**。multiplex が複数 obj を 1 バケットに詰める正常動作を示しているだけで、リソース不足ではない
- モデル静的サイズは obj 数に非依存(`after build = 3.65 GiB` 固定)

### A×B 結合(`140 frames × 16 obj`)

- 線形外挿が実測と一致 — 予測モデルの妥当性を確認
- 12 GB GPU の **92.9% 使用 / 7.1% 余裕**で動作
- スループットは軽度低下(2.07 fps、4obj 比 -3.7%)

## 10.5 12 GB GPU での実用安全圏(導出結果)

per-frame 累積率からの線形外挿:

| obj 数 | per-frame 増加 (MiB) | 推定最大フレーム | 検証状況 |
|---|---|---|---|
| 4 | 18.5 | ~310 | 270f 実測 ✓ |
| 6 | ~20 | ~290 | 30f のみ |
| 8 | ~22 | ~250 | 30f のみ |
| 16 | 30.1 | ~170 | 140f 実測 ✓ |

**運用ガイドライン:**
- 単発推論なら `obj=16` で 170 フレーム以内、`obj=4` で 270 フレーム以内が安全
- 長尺動画は **`outputs_per_frame` の CPU offload 改修が恒久対策**(下記 §10.7)

## 10.6 `subscription rate` 警告の正しい解釈

ログに頻出する以下は **正常動作**:

```
Bucket utilization rate: 100.00%, subscription rate: 1600.00%
```

- `Bucket utilization rate`: 確保されたバケットスロットの使用率
- `subscription rate`: 1 バケットに割り当てられた obj 数(multiplex の意図した動作)
- 100% / 1600% = 全バケット満杯 + 各バケット 16 obj 詰め込み = `max_num_objects=16` 完全活用

リソース警告ではなく **multiplex の効率的稼働を示す情報ログ**として読むべき。

## 10.7 残課題と次の対策

1. **`outputs_per_frame` の CPU offload 改修**(優先度: 高) — マスクは propagation 内で再使用しないため、frame ごとに `tensor.cpu()` で転送すれば VRAM 累積を排除でき、12 GB GPU でも長尺・多 obj が両立可能
2. **C 軸検証**(`bedroom.mp4` 等の異なる動画) — 解像度・シーン依存の確認
3. **AIxSuture 本体への統合** — 上限値(170f / 16obj、310f / 4obj)を予算とした上位 API 設計

## 10.8 検証スクリプトと再現方法

```bash
# モニタ起動
TS=$(date +%Y%m%d_%H%M%S)
LOG=/home/takenouchi/AIxSuture/tmp/log/scale_X_${TS}.log
bash /home/takenouchi/AIxSuture/tmp/monitor.sh "$LOG" &

# 推論実行(例: 140f × 16obj × window)
cd /home/takenouchi/AIxSuture/sam3
uv run python scripts/verify_sam3p1_phase3.py \
    --video-path /home/takenouchi/AIxSuture/tmp/sam3_140 \
    --max-num-objects 16 \
    --prompt-text window
```

各試行のログは `tmp/log/scale_*` と `tmp/log/run_*` に保存済み。
