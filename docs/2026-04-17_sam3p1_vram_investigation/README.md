# SAM 3.1 multiplex video predictor の RTX 3060 (12 GB) 適合化 — 調査記録

- 日付: 2026-04-17
- 対象: `sam3/scripts/verify_sam3p1_phase3.py` (propagate_in_video フルビデオ検証)
- 結論: `sam3/model_builder.py:1190` の `batched_grounding_batch_size` を `16 → 1` に変更することで 12 GB GPU 上で動作可能化

## 目次

| # | ファイル | 対応章 | 内容 |
|---|---|---|---|
| 1 | [01_problem.md](01_problem.md) | §1–3 | 環境・発生した問題・原因の切り分け |
| 2 | [02_fix.md](02_fix.md) | §4–6, §9 | 効かなかった手段・適用した修正・初回動作確認・参考情報 |
| 3 | [03_scale.md](03_scale.md) | §10 | スケール耐性検証 (A/B 軸) |
| 4 | [04_video.md](04_video.md) | §11 | 動画依存性検証 (C 軸) |
| 5 | [05_overlay.md](05_overlay.md) | §12 | overlay 出力パイプライン検証 (D 軸) |
| 6 | [06_throughput.md](06_throughput.md) | §13 | スループット最適化検証 (E 軸) |
| 7 | [07_phases.md](07_phases.md) | §14 | 他フェーズ動作検証 (F 軸) |
| — | [remaining_tasks.md](remaining_tasks.md) | §8 | 残タスク(§8.D 品質回帰、§8.F AIxSuture 統合) |

## 成果物ディレクトリ構成

```
/home/takenouchi/AIxSuture/
├── tmp/
│   ├── sam3_small/              # 30 フレームサブセット (0.jpg〜29.jpg)
│   ├── sam3_<N>/                # A 軸用サブセット (N=90/140/180/270)
│   ├── bedroom_jpg/             # C 軸用 200 frame JPG 列
│   ├── overlay_test/            # D 軸出力 (PNG + MP4)
│   ├── phase2_overlay/          # F 軸出力 (phase2 確認用 PNG)
│   ├── monitor.sh               # free -h + nvidia-smi を 1s 間隔で記録
│   └── log/
│       ├── latest.env           # 最新モニタログのパス
│       ├── monitor_*.log        # タイムスタンプ付きリソース記録
│       ├── scale_*.log          # 各検証のリソースログ
│       └── run_*.log            # 各検証の stdout
└── docs/
    └── 2026-04-17_sam3p1_vram_investigation/   # 本ドキュメント群
```
