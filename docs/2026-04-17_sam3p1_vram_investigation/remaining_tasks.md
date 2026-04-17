# §8 残タスク(今後の調査項目)

→ [目次に戻る](README.md)

## A. スケール耐性(優先度: 高) — [§10](03_scale.md)/[§11](04_video.md) で完了
1. ✅ **270 フレームフル動画 `assets/videos/0001`** で phase3 再試行 — PASSED
2. ✅ **`max_num_objects=16`** での動作確認 — `subscription rate` は誤解を招く名称で、実態は multiplex の正常動作
3. ✅ **`bedroom.mp4`** での動作確認 — [§11](04_video.md) で MP4/JPG 両形式で検証済

## B. overlay 出力パイプライン(優先度: 中) — [§12](05_overlay.md) で完了
4. ✅ `--overlay-dir` + `--overlay-mp4` 有効化時のホスト RAM / ディスク挙動
5. ✅ 視覚的妥当性の目視確認

## C. スループット最適化(優先度: 中) — [§13](06_throughput.md) で完了
6. ✅ `batched_grounding_batch_size=2` / `3` / `4` が 12 GB に収まるか検証
7. ✅ 2.15 fps からどこまで伸ばせるかの上限調査(結論: **bs 増加では伸びない**)

## D. 品質回帰の確認(優先度: 中)
8. `batched_grounding_batch_size=1` vs `=16` での**マスク出力同一性**の数値検証
   - Codex はコード上同一性を保証しているが実測確認が理想(大容量 GPU 環境が必要)

## E. 他フェーズ(優先度: 低) — [§14](07_phases.md) で完了
9. ✅ `verify_sam3p1_phase2.py` が同修正で動くか — PASSED
10. ✅ `verify_sam3p1_smoke.py` (build のみ、影響なし見込み) — PASSED

## F. AIxSuture 本体への統合(優先度: プロジェクト依存)
11. 画像単位 (`build_sam3_image_model`) / 動画単位 (修正済 multiplex) / 非-multiplex (`build_sam3_predictor(version="sam3")`) の使い分け設計
12. 上位 API 設計

---

## 未完了タスクの概要

| 章 | 項目 | 優先度 | 状態 |
|---|---|---|---|
| D | 品質回帰の確認 (bs=1 vs bs=16 マスク出力同一性) | 中 | 未着手(大容量 GPU 要) |
| F | AIxSuture 本体への統合 — build variant 使い分け | プロジェクト依存 | 未着手 |
| F | AIxSuture 本体への統合 — 上位 API 設計 | プロジェクト依存 | 未着手 |

### 将来の追加調査候補([§13.5](06_throughput.md#135-多-obj-シーンでの再評価が必要) / [§13.6](06_throughput.md#136-運用ガイドライン更新版) 由来)

- 多 obj シーン(obj=16)での bs=2/3 スループット再測
- 高解像度(1280×720)での bs スイープ OOM 上限の確認
- `compile=True` + warm-up 組み合わせによる propagation 段高速化
