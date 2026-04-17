# §12 overlay 出力パイプライン検証

→ [目次に戻る](README.md)

## 12.1 検証目的

[§8.B](remaining_tasks.md#b-overlay-出力パイプライン優先度-中--§12-で完了) の積み残しに対応:`--overlay-dir` (PNG) と `--overlay-mp4` (動画) を
同時有効化した際の **(D1-1) ホスト RAM / Swap / ディスクへの追加コスト**、
**(D1-2) GPU リソースへの影響**、**(D1-3) 出力の視覚的妥当性** を測定する。

## 12.2 試験条件

- 入力: `tmp/bedroom_jpg/` (200 frames, 960×540) — [§11](04_video.md) C1-JPG と完全同一
- `obj=4`, `prompt="person"`, `batched_grounding_batch_size=1`
- `--overlay-dir tmp/overlay_test/png --overlay-stride 30`
- `--overlay-mp4 tmp/overlay_test/output.mp4 --mp4-fps 30`
- ログ: `tmp/log/run_D1_overlay_20260417_171113.log`、`tmp/log/scale_D1_overlay_*.log`

## 12.3 結果一覧(C1-JPG ベースラインとの比較)

| 指標 | C1-JPG (overlay 無) | D1 (overlay 有) | 差分 |
|---|---|---|---|
| 検出 obj | 2 | 2 | — |
| nonempty_masks | — | 400 (200×2) | 完全追跡 |
| 推論 fps | 2.15 | 2.15 | ±0 |
| 推論時間 | — | 92.91 s | — |
| overlay 後処理時間 | — | +2.1 s | — |
| after-prop alloc | 7.48 GiB | 7.47 GiB | -0.01 |
| after-prop reserved | 8.83 GiB | 8.98 GiB | +0.15 |
| **GPU peak** | **9,519 MiB** | **9,665 MiB** | **+146 (+1.5%)** |
| **RAM peak** | — | **10,505 MiB** | — |
| **Swap 増分** | — | **+30 MiB** (1,792→1,822) | 軽微 |
| ディスク使用 | — | **PNG 4.4 MiB + MP4 3.7 MiB = 8.1 MiB** | — |
| 出力 PNG 数 | — | 7 (frame 0,30,60,…,180) | — |
| 出力 MP4 | — | 200 frames @ 30fps, 960×540, mp4v | — |

## 12.4 主要な定量的知見

### D1-1: ホストリソースへの影響は軽微

- RAM peak +100 MiB 程度 ≒ overlay 描画用バッファ(`cv2.imread` + numpy + `VideoWriter` キュー)
- Swap +30 MiB は無視できる(プロセス開始前の常駐分が大半)
- ディスクは PNG 4.4 MiB + MP4 3.7 MiB = **8.1 MiB のみ**(960×540 × 200 frames)

### D1-2: GPU リソースへの影響は実質ゼロ

- GPU peak +146 MiB (+1.5%) は計測誤差範囲
- alloc/reserved も C1-JPG とほぼ同一 → overlay は **完全に CPU 処理**
- 推論 fps が 2.15 で完全一致 → overlay は inference path に副作用なし

### D1-3: 視覚的妥当性

- `seen_ids=[0, 1]` × 200 frames、`nonempty_masks=400 / 400` → multiplex の obj_id 永続性が完全機能
- `tmp/overlay_test/png/00000.png` / `00090.png` / `00180.png` を目視確認:
  - 子供 2 名のシルエットに半透明 (alpha=0.5) オーバーレイが乗り、ベッド/壁との分離が明瞭
  - frame 90 では右側の子が一部画面外に出ても obj_id 維持
- mp4v コーデックの 200-frame 動画は再生可能(`cv2.VideoCapture` で 200 frames / 30 fps / 960×540 confirmed)

### D1-4: overlay 後処理コスト

- PNG 7 枚 + MP4 200 frames 書き出しで **2.1 秒**(全体時間の 2.2%)
- 内訳推定: `cv2.imread` × 200 + `_render_overlay` × 200 + `writer.write` × 200 + `imwrite` × 7
- 約 **10 ms/frame** の CPU バウンド処理(GPU 不使用)→ スループット律速にならない

## 12.5 出力形式選択ガイドライン

| 用途 | 推奨設定 | 理由 |
|---|---|---|
| デバッグ確認 | `--overlay-dir` のみ + stride=30 | PNG だけで足り、書き出しコスト最小 |
| 全フレーム動画化 | `--overlay-mp4` のみ | ディスク効率良好(200f → 3.7 MiB) |
| 詳細レビュー | 両方 | 後処理 +2 s、ディスク +8 MiB(本実験の構成) |
| 長尺 (1000+ frames) | `--overlay-mp4` のみ + bitrate 調整 | PNG 累積を回避 |

## 12.6 副次知見

- `VideoWriter` の `mp4v` コーデックは `opencv-python-headless` で動作(追加 codec install 不要)
- PNG ファイル名は `{frame_idx:05d}.png` 形式で自然順整列
- `_render_overlay`(`sam3/scripts/verify_sam3p1_phase3.py:129`)は numpy のみで GPU 不使用
- `before propagate` 時点(4.38 GiB)→ `after propagate`(7.47 GiB) の +3.09 GiB 増は [§10](03_scale.md) で示した `outputs_per_frame` の累積であり、overlay とは独立

## 12.7 再現方法

```bash
# モニタ起動
TS=$(date +%Y%m%d_%H%M%S)
LOG=/home/takenouchi/AIxSuture/tmp/log/scale_D1_overlay_${TS}.log
bash /home/takenouchi/AIxSuture/tmp/monitor.sh "$LOG" &

# overlay 付き推論
cd /home/takenouchi/AIxSuture/sam3
uv run python scripts/verify_sam3p1_phase3.py \
    --video-path /home/takenouchi/AIxSuture/tmp/bedroom_jpg \
    --max-num-objects 4 \
    --prompt-text person \
    --overlay-dir /home/takenouchi/AIxSuture/tmp/overlay_test/png \
    --overlay-stride 30 \
    --overlay-mp4 /home/takenouchi/AIxSuture/tmp/overlay_test/output.mp4 \
    --mp4-fps 30
```

出力先: `tmp/overlay_test/png/*.png`, `tmp/overlay_test/output.mp4`
