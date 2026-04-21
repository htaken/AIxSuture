# Phase 1 — プロンプト語彙探索 結果

- 日付: 2026-04-20
- 入力: 3 動画 × 3 フレーム = 9 frames(1920×1080)
- プロンプト: [prompts_v1](README.md#プロンプト語彙-v1) 14 語
- 閾値: `confidence_threshold=0.3`
- 成果物: [`summary.csv`](../../tmp/aixsuture/probe_results/summary.csv)(126 行)、[`overlays/`](../../tmp/aixsuture/probe_results/overlays/)(126 PNG)

## 全体サマリ(プロンプト別)

9 フレーム中の検出成功フレーム数 = `hits`、平均検出数 = `avg_n`、平均 max_score = `avg_max`、平均マスク面積比 = `avg_area`:

| prompt | hits | avg_n | avg_max | avg_area | 判定 |
|---|---:|---:|---:|---:|---|
| **hand** | 9/9 | 1.78 | 0.932 | 0.1748 | ✅ 有望 |
| **glove** | 9/9 | 1.78 | 0.942 | 0.1751 | ✅ 有望(hand と同義視) |
| **forceps** | 9/9 | 3.11 | 0.908 | 0.0373 | ✅ 有望 |
| **scissors** | 9/9 | 3.33 | 0.952 | 0.0424 | ✅ 有望 |
| finger | 8/9 | 7.22 | 0.763 | 0.1068 | ⚠️ 準有望(検出数が多過ぎ) |
| skin | 2/9 | 0.56 | 0.138 | 0.0122 | ✗ 不安定 |
| suture thread | 3/9 | 1.00 | 0.188 | 0.0023 | ✗ 低信頼 |
| thread | 2/9 | 0.33 | 0.123 | 0.0009 | ✗ 低信頼 |
| surgical forceps | 2/9 | 0.56 | 0.083 | 0.0091 | ✗ 修飾語で悪化 |
| surgical needle | 1/9 | 0.22 | 0.052 | 0.0006 | ✗ |
| **needle** | 0/9 | 0 | 0 | 0 | ✗ 不発 |
| **needle holder** | 0/9 | 0 | 0 | 0 | ✗ 不発 |
| **tissue** | 0/9 | 0 | 0 | 0 | ✗ 不発 |
| **practice pad** | 0/9 | 0 | 0 | 0 | ✗ 不発 |

## 動画横断性(動画別 hits / 3 frames)

| prompt | A31H | K78J | X14A | 安定性 |
|---|:---:|:---:|:---:|---|
| hand | 3/3 | 3/3 | 3/3 | ★ 全動画で安定 |
| glove | 3/3 | 3/3 | 3/3 | ★ 全動画で安定 |
| forceps | 3/3 | 3/3 | 3/3 | ★ 全動画で安定 |
| scissors | 3/3 | 3/3 | 3/3 | ★ 全動画で安定 |
| finger | 3/3 | 2/3 | 3/3 | 準安定 |
| surgical needle | 0/3 | 0/3 | 1/3 | 動画依存・低 |
| surgical forceps | 0/3 | 1/3 | 1/3 | 動画依存・低 |
| thread | 1/3 | 1/3 | 0/3 | 動画依存・低 |
| suture thread | 0/3 | 2/3 | 1/3 | 動画依存・低 |
| skin | 2/3 | 0/3 | 0/3 | A31H のみ |
| needle / needle holder / tissue / practice pad | 0/3 | 0/3 | 0/3 | 全不発 |

## 有望語彙(Phase 2 入力候補)

3 動画 × 3 フレームすべてで nonempty 検出 + max_score ≥ 0.9 を満たす **4 語**:

1. **`hand`** — max=0.93、1–2 検出(左右の手を分離)
2. **`glove`** — max=0.94、`hand` とほぼ同一の結果。open vocabulary 上 `hand`/`glove` は実質同義
3. **`forceps`** — max=0.91、3 検出前後(シーン内のピンセット類を広く検出)
4. **`scissors`** — max=0.95、3 検出前後(剪刀)

準有望枠:
- **`finger`** — 8/9 で検出。ただし平均 7 検出は過剰で、指本数を超えるケースあり。用途次第(指先位置が欲しいなら使える)

## 同義語対比(重要な発見)

| 対比 | 単純語 hits / avg_max | 修飾語 hits / avg_max | 傾向 |
|---|---|---|---|
| `needle` vs `surgical needle` | 0/9, 0.00 | 1/9, 0.05 | どちらも不発、修飾で微改善 |
| `forceps` vs `surgical forceps` | 9/9, 0.91 | 2/9, 0.08 | **単純語が圧勝**、"surgical" 付加で精度激減 |
| `thread` vs `suture thread` | 2/9, 0.12 | 3/9, 0.19 | どちらも低信頼、修飾で僅差 |

結論: **SAM 3 の学習語彙は一般 common noun に偏る**。"surgical" 等の修飾語を付けると scene 一致率が落ちるケースが多い。Phase 2 では単純語を優先。

## 定性評価(overlay 目視チェック)

### 成功例
- `A31H_t5_hand.png`: 左手(グローブ)を高精度でセグメント化、背景・器具と明確に分離
- `A31H_tmid_forceps.png`: シーンの剪刀・ピンセット類を 3 つ検出、top_box は剪刀
- overlay の赤枠(top score)が視覚的に妥当

### 問題点
- **`forceps` vs `scissors` の重複**: 両プロンプトでシーン内の「剪刀」を検出。SAM 3 は器具の細分類ができていない(surgical tool としては取れるが種別分離不可)
- **`finger` が過剰検出**: 指が 7 本検出されるフレームあり。指関節ごとに分割しているか、他の細長い物体を誤検出
- **縫合針が全く取れない**: 画面内に縫合針が明確に写っているフレーム(`A31H_tmid`)でも `needle` は 0 検出。SAM 3 の学習セットに外科用縫合針が十分含まれていない可能性
- **糸(thread/suture thread)が取れない**: 縫合糸は極細・長尺でエッジ検出が難しい。 Phase 2 で video propagation + 初期フレームマニュアルプロンプトが必要な可能性

## VRAM / 推論性能

| 指標 | 値 |
|---|---|
| Model build | 6.1s(HF cache hit 後) |
| set_image(初回) | 0.40s |
| set_image(2 回目以降) | **0.01s**(キャッシュが効いている) |
| set_text_prompt 平均 | 0.06s / prompt |
| VRAM(peak reserved) | **4.31 GiB**(12 GB 上限の 36%) |
| Phase 1 全体所要 | 約 7 秒(126 runs、モデルロード除く) |

→ **RTX 3060 12GB で十分に余裕**。image model は multiplex video predictor より遥かに軽い。

## Phase 2 への申し送り

1. **動画伝播対象の第一候補**: `hand`, `glove`, `forceps`, `scissors`(4 語)。高信頼・全動画安定。
2. **`glove` と `hand` を両方使う意味は薄い** — 出力が同等なので、Phase 2 では片方に絞る(hand を推奨、glove はラテックス色の変化に敏感な可能性があるため)
3. **Phase 2 の multiplex では `max_num_objects` = 4〜8 で十分**(シーンあたりの有望対象が 4 語 × 数個)
4. **縫合針/糸は別アプローチ必要**: 初期フレームに box prompt か point prompt を手動で与え、video propagation で追跡する戦略を検討
5. **surgical-prefixed 語は原則不採用**、除外で OK

## 次アクション候補

- **Phase 2(video propagation)**: `hand`/`forceps`/`scissors` で 1 動画フル propagate、`max_num_objects` スイープ
- **Phase 1.5(オプション)**: 針/糸専用に点/ボックスプロンプトで image API(`add_geometric_prompt`)を試す
- **V2 語彙**: `instrument`, `tool`, `surgical instrument`, `tweezers`, `clamp` などの代替語を追加 sweep
