# §14 他フェーズ動作検証

→ [目次に戻る](README.md)

## 14.1 検証目的

[§8.E](remaining_tasks.md#e-他フェーズ優先度-低--§14-で完了) の積み残しに対応:`batched_grounding_batch_size=1` 修正が Phase 1 (smoke) / Phase 2
両スクリプトでも **(F1) 正常動作** するか、**(F2) 既存の VRAM プロファイルに整合** するかを確認する。
品質回帰([§8.D](remaining_tasks.md#d-品質回帰の確認優先度-中))はスキップ。

## 14.2 試験条件

- 入力: `assets/videos/0001` (270 frames, 1280×720) — デフォルト
- `max_num_objects=16` (デフォルト), `batched_grounding_batch_size=1`
- Phase 2: `prompt="person"`, `frame_index=0`, overlay PNG 保存
- ログ: `tmp/log/run_F_smoke_*.log`, `tmp/log/run_F_phase2_*.log`
- overlay: `tmp/phase2_overlay/frame0_person.png` (1280×720, 962 KiB)

## 14.3 結果一覧

| 試行 | スクリプト | build 時間 | after-build alloc/res (GiB) | 追加動作 alloc/res (GiB) | 検出 obj | 結果 |
|---|---|---|---|---|---|---|
| F1 | verify_sam3p1_smoke.py | 7.33 s | 3.65 / 3.75 | start_session: 3.67 / 3.77 | — | ✓ |
| F2 | verify_sam3p1_phase2.py | 7.38 s | 3.65 / 3.75 | add_prompt: **4.36 / 5.24** | **4 (probs 0.94–0.97)** | ✓ |

## 14.4 主要な知見

### F1: Phase 1 smoke は想定通り軽量
- `start_session` 後の VRAM 増分はわずか +0.02 GiB — frame tensor の async loader は未発火、ほぼ predictor 本体のみ常駐
- `close_session` 後も 3.66 GiB が常駐(=モデル重み)
- bs=1 修正の影響なし(build 時点で detector 段の buffer は未確保)

### F2: Phase 2 add_prompt は [§2](01_problem.md#2-発生した問題) の再現と整合
- add_prompt 直後 alloc=4.36 / reserved=5.24 GiB は [§2](01_problem.md#2-発生した問題) の「add_prompt 後 4.34 / 5.27 GiB」と完全一致(±0.03 GiB)
- → bs=1 修正は **add_prompt 段の挙動を変えない**(detector のスパイクは `propagate_in_video` 開始時に発生するため)
- 4 人全員を obj_id=[0,1,2,3] で確実に検出、confidence 0.94–0.97 と高い
- overlay PNG で 4 人のシルエットに配色マスクが正しく乗ることを目視確認(`tmp/phase2_overlay/frame0_person.png`)

## 14.5 統合的な結論

- **`batched_grounding_batch_size=1` は全 3 フェーズ (smoke / phase2 / phase3) で回帰なく動作**
- Phase 1/2 は元々 detector 段を単発実行するのみで、bs 変更の影響を受けない
- Phase 3 のみが detector 段をフレームチャンクで呼ぶため、bs が VRAM ピークを律速する([§13](06_throughput.md) で定量化済)
- → AIxSuture 本体への統合時も、build パラメータとして `batched_grounding_batch_size` を RTX 3060 プリセットで 1 に固定すれば全フェーズ互換
