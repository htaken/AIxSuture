# Phase 2 計画(Codex レビュー反映版)

- 日付: 2026-04-20
- 位置付け: Phase 1 の結果([phase1_results.md](phase1_results.md) / [data_capability_matrix.md](data_capability_matrix.md))を受けた動画伝播段階
- レビュー履歴: Codex レビューで MUST 3 / SHOULD 4 / NICE 2 の指摘 → 下記はすべて反映済み
- 実行着手前に本ドキュメントを参照して計画変更点を確認すること

→ [README に戻る](README.md)

## Phase 2 のゴール

1. Phase 1 で有望と確定した語(`hand`, `forceps`, `scissors`)が **300 秒の動画全体で時間的に持続** するか検証
2. **hand⇄tool の相互作用情報**(どの手が何を持つか)を mask/bbox から導出
3. Phase 1 で不可だった **針/糸を手動 geometric prompt + video propagation** で救う別ルート検証
4. AIxSuture 下流解析用の出力 API スキーマ確定(2e、任意)

## サブフェーズ構造

| Sub | 主題 | 特徴(Codex 指摘反映) | 所要 | 依存 |
|---|---|---|---|---|
| **2a** | 単一動画 baseline(A31H) | **3 prompts を 3 separate sessions に分解** | 4–5 時間 | — |
| **2b** | 動画横断 **stress-case** 検証 | K78J は occlusion、X14A は instrument 重複の stress 設計 | 3 時間 | 2a |
| **2c** | hand⇄tool 相互作用の後段解析 | **mask-to-mask 距離 + 膨張接触 + 3-frame persistence** | 2 時間 | 2a |
| **2d** | 針/糸の手動 geometric prompt 伝播 | 別 session(text prompt と共存不可) | 3 時間 | 2a |
| 2e(任意) | 出力 JSON API / バッチ処理 | | 4 時間 | 2a–2d |

---

## Codex レビューで確定した前提

### MUST(計画の根幹を変えたもの)

**M1. `add_prompt` は semantic prompt ごとに `reset_state()` を呼ぶ**
- 根拠: `sam3/sam3/model/sam3_multiplex_tracking.py:1672, 1694`
- 評価コードも prompt ごとに obj_id を手動 offset してマージ(L1772)
- **→ `hand` / `forceps` / `scissors` は独立した 3 run + オフラインマージ必須**。同一 session に相乗りは不可
- 2d の box prompt も独立 session

**M2. smoke は 150–200 frames 必須**
- VRAM 支配項は detector 初期スパイクだけでなく **`outputs_per_frame` の線形累積**
- §10 の安全圏: 4 obj で ~310f、8 obj で ~250f
- 30 frames smoke では後半 OOM を検知できない
- **→ smoke は heaviest case で 180 frames**

**M3. 実行ログに固定情報を必ず残す**
- Phase 1 の image model(`sam3.pt`)と Phase 2 の multiplex(`sam3.1_multiplex.pt`)で系統が違う
- builder は未指定なら自動 download 。毎 run で下記を記録:
  - checkpoint path / hash
  - `batched_grounding_batch_size` の実値
  - 入力形式(JPG or MP4)
  - prompt 文字列
  - frame count
  - device / autocast 設定 / seed

### SHOULD

**S1. 評価指標の強化**
- per-object lifespan / gap & fragmentation 回数
- **prompt 間 overlap 行列**(forceps × scissors の重複定量)
- run-to-run determinism(同条件 2 回走らせての差分)

**S2. 2b は 2a のコピーではなく stress-case 設計**
- K78J: hand mask 面積 CV 80% → **occlusion / 画面外出 stress**
- X14A / K78J: forceps vs scissors が同一オブジェクト群 → **重複除去 stress**

**S3. 2c のグリップ判定は素朴すぎ**
- 器具は細長 → bbox center が手の外に出やすい
- hand_width 正規化はズームと左右手向きに弱い
- **→ mask-to-mask 最短距離 + 手 mask 膨張(dilation)後の接触判定 + 3-frame persistence**
- forceps/scissors は「instrument」に統合してから hand-association を計算

**S4. 解像度 sweep は優先度低**
- 入力は multiplex 内部で 1008 resize → VRAM 軸では差が出ない
- **→ 代わりに prompt order / 1 fps aliasing / run-to-run determinism を 1 クリップで確認**

### NICE(運用ルール化)

- `tmp/monitor.sh` は無限ループで PID 管理・trap なし → **多重走行・orphan リスク**
- run 単位で別ログ + PID 保存 + 明示 kill を **ラッパースクリプト化**
- 1 本の continuous log より run 単位分離が peak 帰属を明確化

---

## OOM 定量予測(Codex 試算、RTX 3060 実容量 11.63 GiB)

| 条件 | 予測 peak | 判定 |
|---|---|---|
| 302f JPG × **3** actual objects | 適合 | **安全圏** |
| 302f JPG × **4** actual objects | ~11.88 GiB | **赤信号**(残 ~27 MiB) |
| 302f JPG × 8 actual objects | ~13.35 GiB | OOM +1.44 GiB |
| 302f MP4 × 4 objects | ~12.93 GiB | OOM |
| 302f MP4 × 8 objects | ~14.40 GiB | OOM |

**判断**:
- JPG 入力は確定(MP4 は全条件 OOM)
- **1 session で actual objects を 3 以下に抑える**
- `max_num_objects=4`(上限)で `hand` / `forceps` / `scissors` を **各独立 session** で回せば安全圏
- 各プロンプトで検出されうる実対象数は Phase 1 実測: hand 1–2, forceps 3–4, scissors 3–4
  - forceps/scissors はやや超過の懸念 → 各 session で `max_num_objects=4` まで下げる

---

## Phase 2a 改訂版(単一動画 baseline)

### 2a.0 前提確定

- **入力**: A31H.mp4 → **1 fps ダウンサンプル × 302 JPG**(元 1920×1080)
- **autocast**: `torch.autocast("cuda", dtype=torch.bfloat16)` を実行時に enter
- **batched_grounding_batch_size**: 1(fork 済み修正)
- **checkpoint**: `sam3.1_multiplex.pt`(HF default)
- **seed**: 0(PyTorch 全般 + numpy 乱数 fix)

### 2a.1 JPG 列抽出(新スクリプト `scripts/extract_video_jpgs.py`)

```
入力: --video MP4, --out-dir, --fps 1
出力: {out_dir}/00000.jpg 〜 {out_dir}/00301.jpg(5 桁ゼロパディング)
```

verify_sam3p1_phase3.py のフレーム命名規則に準拠。

### 2a.2 伝播スクリプト(新スクリプト `scripts/propagate_video.py`)

```
引数:
  --video-path DIR       JPG ディレクトリ or MP4
  --prompt-text STR      単一 text prompt(プロンプト並列実行は行わない)
  --max-num-objects N    4(安全上限)
  --output-dir DIR       CSV + mask + overlay MP4
  --overlay-mp4 PATH     overlay 動画出力
  --seed INT             0
  --run-tag STR          ログ/CSV の run 識別子(例: A31H_hand)

処理:
  1. torch.autocast bfloat16 enter
  2. run_meta.json に checkpoint path/hash, bs, input format, prompt, frame count,
     device, seed を書き出し
  3. build_sam3_multiplex_video_predictor(max_num_objects=N, ...) 1 回
  4. start_session(resource_path=JPG-dir)
  5. add_prompt(text=PROMPT, frame_index=0)
  6. propagate + per-frame で CSV 1 行/object 記録
  7. mask/overlay 保存
  8. reset_state() 後にプロセス終了
```

M1 反映: **1 run = 1 prompt**。3 prompts は 3 回起動。

### 2a.3 実行順序

1. **smoke**: A31H × `hand` × **先頭 180 frames**(§10 の 4 obj/310f 安全圏を踏まえた負荷試験)
2. smoke 通過後、**3 run を順次実行**(hand / forceps / scissors × 全 302f)
3. 各 run の CSV・mask・overlay をまとめてオフラインマージ

### 2a.4 CSV スキーマ(per-frame × per-object)

| 列 | 型 | 備考 |
|---|---|---|
| video_id | str | A31H |
| run_tag | str | `A31H_hand_run1` |
| frame_idx | int | 0–301 |
| frame_t_s | float | = frame_idx |
| prompt | str | `hand` |
| prompt_obj_id | str | `hand:3`(プロンプト prefix 付き、グローバル ID 衝突回避) |
| local_obj_id | int | multiplex 内 ID |
| score | float | |
| bbox_xyxy | str | `x1,y1,x2,y2` |
| bbox_center_xy | str | |
| mask_area_ratio | float | |
| mask_png_path | str | `masks/{frame_idx}_{local_obj_id}.png`(任意) |

### 2a.5 評価指標(Codex S1 反映)

1. **検出継続率**: プロンプトごとの n_detections > 0 フレーム数 / 302
2. **per-object lifespan**: 各 `prompt_obj_id` の最初〜最後フレーム差
3. **gap & fragmentation 回数**: 1 object が途中消失した回数、断片化のヒストグラム
4. **prompt 間 overlap 行列** — 同フレームで `forceps` と `scissors` の bbox IoU の平均
5. **hand–tool 共在率**: hand run と forceps/scissors run で同 frame に両方あった割合
6. **VRAM peak / 総実行時間 / 実 fps**
7. **determinism**: hand run を 2 回実施して mask Hamming 距離

### 2a.6 run_meta.json スキーマ(M3 反映)

```json
{
  "run_tag": "A31H_hand_run1",
  "started_at": "2026-04-20T17:00:00+09:00",
  "checkpoint_path": "/path/to/sam3.1_multiplex.pt",
  "checkpoint_sha256": "...",
  "batched_grounding_batch_size": 1,
  "input_format": "jpg",
  "prompt": "hand",
  "frame_count": 302,
  "max_num_objects": 4,
  "device": "cuda:0",
  "autocast_dtype": "bfloat16",
  "seed": 0,
  "python_version": "3.12.x",
  "torch_version": "2.x.x",
  "sam3_git_sha": "..."
}
```

### 2a.7 完了条件

- [ ] smoke(hand × 180f)が OOM なし完走、peak VRAM < 11.0 GiB
- [ ] 3 run 全 302f 通過、各 `run_meta.json` が完全
- [ ] `summary.csv`(マージ済)が ~2400–2800 行
- [ ] overlay MP4 3 本
- [ ] `phase2a_results.md` に評価指標 7 項目すべて記録
- [ ] 検出継続率 ≥ 80%(満たさない場合は要因分析を明記)

---

## Phase 2b 改訂版(stress-case 設計)

2a のコピーではなく、**取れにくいケースを意図的に突く**:

| 動画 | 主題 stress | 測る指標 |
|---|---|---|
| K78J | **hand occlusion / 画面外出**(Phase 1 で CV 80%) | hand の gap 回数、lifespan 分布、画面外フレーム率 |
| X14A | **forceps vs scissors 重複**(Phase 1 で bbox center 同一) | overlap 行列、同一 obj_id を指しているかの追跡 |

新 run は **各動画 × 2 prompt**(K78J は hand + forceps、X14A は forceps + scissors の 2 prompt ずつ)= **4 run**。

結果は `phase2b_stress.md` に:
- 2a と比較した検出継続率の差
- 各 stress 指標の定量値
- 実運用で耐えられるか(pass/fail 判定)

---

## Phase 2c 改訂版(mask ベースグリップ解析)

### 入力
2a/2b の `summary.csv` + `mask_png_path` に保存された per-frame mask PNG。

### パイプライン(S3 反映)

1. **器具統合**: 各フレームで `forceps` と `scissors` の mask を論理和 → `instrument` mask
2. **hand mask の膨張**: `scipy.ndimage.binary_dilation`(kernel 半径 = hand bbox 短辺 × 0.15 程度)
3. **接触判定**: 膨張 hand mask と instrument mask の論理積 > 0 なら接触
4. **3-frame persistence**: 接触が連続 3 フレーム以上続いた場合のみ「把持」と認定
5. **左右判定**: hand obj_id ごとに初期フレームの bbox_center_x で左右を割当(低 x = 左手)

### 成果物
- `hand_instrument_timeline.csv`(frame × (left_gripping, right_gripping))
- `grip_timeline.png`(matplotlib 時系列、ガントチャート風)
- `phase2c_grip_analysis.md`(ルールの閾値検討、失敗ケース列挙)

### 新スクリプト `scripts/analyze_grips.py`
SAM 無関係。pandas + numpy + scipy + matplotlib のみ。

---

## Phase 2d 改訂版(針/糸の手動 geometric prompt)

### 入力準備
1. Phase 1 overlay から針/糸が明確に写るフレームを特定(`A31H_tmid.jpg` 周辺)
2. **Jupyter or 専用 UI スクリプトで box を描き、座標を JSON 保存**
3. 該当フレームを含む JPG 列を ±30 frames で抽出(61 frames)

### 実行
- **独立 session**(text prompt と共存不可、M1 反映)
- `add_prompt(type='box_prompt_on_frame', box=[x1,y1,x2,y2], frame_index=起点)` で投入
- propagate(前方 30f + 後方 30f)

### 成果物
- `phase2d_needle_box_prompt.md`:
  - 追跡可能フレーム数(score > 0.3 & nonempty mask が続いた範囲)
  - 破綻タイミングとその画像解析(遮蔽/高速移動/画面外)
  - 糸にも同様実験(point prompt で細尺対応試行)
- overlay MP4 2 本(針、糸)

### 新スクリプト `scripts/propagate_from_geometric.py`

---

## モニタ運用ルール(NICE 反映)

### ラッパースクリプト `scripts/run_with_monitor.sh`(新設)

```bash
#!/usr/bin/env bash
# usage: run_with_monitor.sh <run_tag> <cmd...>
set -euo pipefail
RUN_TAG="$1"; shift
LOG_DIR="/home/takenouchi/AIxSuture/tmp/aixsuture/phase2/logs"
mkdir -p "$LOG_DIR"
MON_LOG="$LOG_DIR/${RUN_TAG}_monitor.log"
RUN_LOG="$LOG_DIR/${RUN_TAG}_run.log"

# monitor 起動 + PID 記録
/home/takenouchi/AIxSuture/tmp/monitor.sh > "$MON_LOG" 2>&1 &
MON_PID=$!
echo "$MON_PID" > "$LOG_DIR/${RUN_TAG}_monitor.pid"

# trap で終了時に必ず kill
trap 'kill "$MON_PID" 2>/dev/null || true; rm -f "$LOG_DIR/${RUN_TAG}_monitor.pid"' EXIT INT TERM

# 本体実行
"$@" 2>&1 | tee "$RUN_LOG"
```

### 運用ルール
- Phase 2 の全 run は `run_with_monitor.sh <tag> uv run python ...` 経由で起動
- smoke と full で **別 run_tag**(例: `A31H_hand_smoke`, `A31H_hand_full`)→ 別ログ
- 以前の orphan を確認する ritual: `pgrep -f monitor.sh | xargs -r ps -o pid,etime,cmd`

---

## Codex 推奨の実行順序(このまま採用)

1. **A31H で hand 単体 × 180f smoke**(heaviest case 相当で早期検知)
2. A31H 302f full、**1 prompt × 3 separate runs**
3. **K78J / X14A は stress case のみ**(全動画 × 全プロンプトのフル sweep は不要)

---

## 横断リスクと未解決事項

| 項目 | 状態 | アクション |
|---|---|---|
| 3 run の obj_id マージ戦略 | 方針は prefix 付与で確定 | 実装時に衝突なしを unit test |
| 1 fps aliasing で速い動きが飛ぶ | 2a trend 把握と割り切り | 精密解析は 2d / 将来の 2 fps 実験 |
| determinism(同条件 2 回実行の差) | 未検証 | 2a の最後に 1 プロンプトだけ再実行して mask Hamming 距離を測定 |
| JPG 抽出時の lossy 圧縮影響 | 既存 Phase 1 で quality 95 | そのまま踏襲、問題があれば PNG 化検討 |
| `max_num_objects=4` でも forceps が 4 超える稀ケース | Phase 1 で最大 4 観測 | 超過時は top-k で切捨て、CSV に切捨てフラグ |
| checkpoint_sha256 計算コスト | 300+ MB で ~数秒 | run_meta.json に 1 回書くのみ、許容 |

---

## ファイル構成(Phase 2 生成物)

```
/home/takenouchi/AIxSuture/
├── scripts/
│   ├── extract_video_jpgs.py          # 2a.1
│   ├── propagate_video.py             # 2a.2
│   ├── analyze_grips.py               # 2c
│   ├── propagate_from_geometric.py    # 2d
│   └── run_with_monitor.sh            # 運用
├── tmp/aixsuture/phase2/
│   ├── {A31H,K78J,X14A}_jpg/          # 1 fps JPG 列
│   ├── runs/
│   │   └── {run_tag}/
│   │       ├── run_meta.json
│   │       ├── summary.csv
│   │       ├── masks/
│   │       └── overlay.mp4
│   ├── logs/                          # run_with_monitor.sh 出力
│   └── merged/
│       ├── phase2a_merged.csv         # hand + forceps + scissors
│       ├── phase2b_merged.csv
│       └── hand_instrument_timeline.csv
└── docs/2026-04-20_aixsuture_integration/
    ├── phase2_plan.md                 # 本ファイル
    ├── phase2a_results.md             # 2a 完了後
    ├── phase2b_stress.md              # 2b 完了後
    ├── phase2c_grip_analysis.md       # 2c 完了後
    └── phase2d_needle_box_prompt.md   # 2d 完了後
```

---

## 推奨着手ステップ

| # | 内容 | 想定時間 |
|---|---|---|
| 1 | `scripts/run_with_monitor.sh` + `extract_video_jpgs.py` + `propagate_video.py` の骨組み実装 | 1 時間 |
| 2 | A31H 1fps JPG 抽出 + **smoke(hand × 180f)** | 30 分 |
| 3 | smoke OOM/挙動レビュー、必要に応じて `max_num_objects` 調整 | 15 分 |
| 4 | A31H × 3 prompts full run | 1 時間 |
| 5 | 2a 評価 + `phase2a_results.md` | 1 時間 |
| 6 | K78J / X14A stress case | 2 時間 |
| 7 | 2c / 2d | 5 時間 |

**Phase 2a の smoke 着手** から始めるのが推奨。
