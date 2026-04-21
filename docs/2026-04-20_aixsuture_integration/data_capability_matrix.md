# データ取得可否マトリクス — SAM 3 image model × 縫合動画

- 日付: 2026-04-20
- 出典: [Phase 1 結果](phase1_results.md)(`summary.csv` 126 行、3 動画 × 3 フレーム × 14 prompts)
- 目的: Phase 2 以降の設計判断のための「何が取れ / 何が取れないか」の線引き

→ [README に戻る](README.md)

## 分析の視点

Phase 1 の `summary.csv` を 4 つの軸で再集計:

1. **検出成功率**: プロンプト × フレーム(9 runs)で n_detections > 0 の割合
2. **スコア分布**: max_score を `≥0.9 / 0.7–0.9 / 0.5–0.7 / 0.3–0.5` のバケットに分類
3. **時間安定性**: 同動画内 3 フレーム(t5, tmid, tend)の n_detections 変動(CV = stddev/mean)
4. **空間情報**: `top_box_xyxy`、`top_box_center_xy` の分布

## 1. max_score 分布(検出があったケースのみ)

| prompt | ≥0.9 | 0.7–0.9 | 0.5–0.7 | 0.3–0.5 |
|---|---:|---:|---:|---:|
| hand | **8** | 1 | 0 | 0 |
| glove | **8** | 1 | 0 | 0 |
| forceps | **8** | 0 | 0 | 1 |
| scissors | **8** | 1 | 0 | 0 |
| finger | 2 | 6 | 0 | 0 |
| surgical forceps | 0 | 0 | 0 | 2 |
| surgical needle | 0 | 0 | 0 | 1 |
| thread | 0 | 0 | 2 | 0 |
| suture thread | 0 | 0 | 2 | 1 |
| skin | 0 | 0 | 2 | 0 |

**観察**: 上位 4 プロンプト(hand/glove/forceps/scissors)は 9/9 検出中 8 件で `≥0.9` に集中。その他は全件 `<0.9` でノイズ圏。**`0.9` を実質的な信頼境界線**として使える。

## 2. 時間安定性(同動画内 3 フレームの n_detections 推移)

| prompt | A31H (t5,tmid,tend) | K78J | X14A |
|---|:---:|:---:|:---:|
| hand | 2, 2, 2 | 2, 2, 1 | 2, 1, 2 |
| glove | 2, 2, 2 | 2, 2, 1 | 2, 1, 2 |
| forceps | 3, 3, 3 | 4, 3, 3 | 3, 3, 3 |
| scissors | 3, 3, 3 | 4, 3, 4 | 3, 3, 4 |
| finger | 5, 10, 8 | 9, 9, 0 | 11, 7, 6 |

**観察**:
- hand/glove: 常に 2(稀に片手画面外で 1)→ **両手トラッキングに使える**
- forceps/scissors: 3–4 で安定 → **器具の個数カウントに使える**
- finger: 0〜11 で激しく変動 → **個数としては使えない**

## 3. 空間情報(top_box 中心の分布、pixel)

対象: 9 件すべて検出成功の hand/forceps/scissors。

| prompt | cx 範囲 | cx 平均 | cy 範囲 | cy 平均 |
|---|---|---:|---|---:|
| hand | 146–1659 | 715 | 350–952 | 771 |
| forceps | 357–1582 | 1079 | 147–580 | **345** |
| scissors | 357–1582 | 1008 | 147–575 | **344** |

**重要な観察**:
- 手は画面全域(cx 146–1659, cy 350–952)に分布 → 動きが大きい
- 器具は画面上半(cy 147–580、平均 345)に集中
- **forceps と scissors の bbox 中心がほぼ同一**(cx 1079 vs 1008, cy 344 vs 345)→ SAM 3 は両プロンプトで同じ器具群を検出している

## 4. マスク面積比の時間安定性(動画内 3 フレームの CV)

| prompt | A31H | K78J | X14A |
|---|---:|---:|---:|
| hand | CV 35% | CV **80%** | CV 21% |
| forceps | CV 26% | CV 11% | CV 12% |
| scissors | CV 25% | CV 18% | CV 19% |

**観察**:
- 器具のマスク面積は CV 10–26% で安定 → **サイズ計測可**
- 手のマスク面積は動画で大きく変動(K78J は CV 80% = 1 フレームで手が画面外)→ オクルージョンに弱い

---

## データ取得可否マトリクス

### ✅ 取得できる(高信頼)

| データ項目 | 対象プロンプト | 精度 | 用途 |
|---|---|---|---|
| **両手の検出(左右 2 つ)** | `hand` / `glove` | 9/9 フレーム、max≥0.9 | 手のトラッキング |
| **器具群の検出(3–4 個)** | `forceps` / `scissors` | 9/9 フレーム、max≥0.9 | 器具在/不在、使用頻度 |
| **バウンディングボックス(xyxy, pixel)** | 上記 4 語 | 原解像度精度 | 位置追跡、ROI 定義 |
| **画面内位置のヒートマップ** | 同上 | bbox 中心の累積可 | 作業領域解析 |
| **マスク面積比(器具)** | `forceps` / `scissors` | CV 10–26% | サイズ・深度推定 |
| **ピクセル精度セグメンテーション** | 同上 | 原解像度(1920×1080) | 精細な可視化、領域解析 |

### ⚠️ 取得できるが注意が必要

| 項目 | 問題 | 回避策 |
|---|---|---|
| `finger` の指本数 | 0〜11 で暴れる | 個数ではなく位置 heatmap として使う |
| 手のマスク(K78J など) | CV 80%、画面外出で破綻 | 動画では前後フレームで補間、画像単独では不可 |
| `forceps` と `scissors` の区別 | bbox 中心ほぼ同一、同一オブジェクトを検出 | **区別せず「器具」として扱う**。細分類は別アプローチ |

### ❌ 取得できない

| 項目 | 試したプロンプト | 理由 | 代替案 |
|---|---|---|---|
| **縫合針** | `needle`, `surgical needle`, `needle holder` | 全 0 / score<0.5。学習語彙不足 | **初期フレーム手動 box prompt → video propagation** |
| **縫合糸** | `thread`, `suture thread` | 全 score<0.7。細長物体でエッジ検出破綻 | **point prompt 手動 + propagation**、または別モデル(Hough 線検出等) |
| **皮膚・組織・パッド** | `skin`, `tissue`, `practice pad` | 2/9 以下 + score<0.5 | **固定 ROI** を人手定義、または `surface` 等の別語彙を V2 で探索 |
| **器具の個別識別**(forceps vs scissors vs needle holder 等) | 修飾語系 | single common noun が最強、修飾で悪化 | 細分類は **学習済み医療器具分類器**(例: YOLOv8 + 手術器具 dataset)を併用 |
| **動作ラベル**(tying, passing, pulling) | Phase 1 未検証 | image model は動詞プロンプト不向き | **行動認識モデル**(TSN/SlowFast 系)を別経路で |

---

## Phase 2 設計への示唆

### 取得可能な解析タスク
- 両手の軌跡プロット(時系列 bbox center)
- 器具の在/不在タイムライン
- **器具を持つ手の特定**(hand bbox と forceps/scissors bbox の IoU/重なりで導出)
- 作業領域ヒートマップ(bbox 中心の累積)
- 器具使用頻度(フレーム単位の in-scene 率)
- 手⇔器具の距離プロファイル(接近/離脱パターン)

### 取得不可能タスクへの代替設計
- **針/糸の追跡**: SAM 3.1 multiplex の `add_prompt` に **box / point で手動指定 → propagation**
- **縫合面(パッド)領域**: 起動フレームで人手 polygon 定義 → 動画全体で固定 ROI として扱う
- **動作粒度の解析**: SAM 3 は "who / where / what-tool" までに留め、"what-action" は別モデルと pipeline

### SAM 3 のスコープ定義
> **SAM 3 は縫合シーンの "who" と "where" を取れる。"what-object-in-detail" や "what-action" は取れない。**

Phase 2 以降はこの線引きに沿って「取れるものは SAM 3、取れないものは別経路」と pipeline を分割する方針が合理的。

---

## V2 語彙候補(将来の追加 sweep)

Phase 1 で不発だった領域を補うための候補語(未検証):

- 器具系: `instrument`, `tool`, `surgical instrument`, `tweezers`, `clamp`, `hemostat`
- 対象面系: `surface`, `mat`, `fabric`, `cloth`, `green mat`
- 針/糸系: `curved needle`, `metal needle`, `string`, `wire`, `filament`
- 動作関連(試験的): `cutting`, `grasping`, `piercing`

実施時はこの文書の表に同じフォーマットで結果追記。
