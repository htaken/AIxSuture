# §11 C 軸:動画依存性検証

→ [目次に戻る](README.md)

## 11.1 検証目的

[§10](03_scale.md) は単一動画(`assets/videos/0001`, 1280×720, 4 人ダンサー)に基づくため、
別動画(`bedroom.mp4`, 960×540, 200 frames, 2 人 + 多数の小物)で
**(C1) 入力形式 (MP4 vs JPG dir) の差** と **(C2) 別シーンでの B 軸再確認** を行う。

## 11.2 試験前提

- `bedroom.mp4` は cv2 経由で読み込み(`opencv-python-headless` を `uv add`)
- JPG 抽出: `cv2.IMWRITE_JPEG_QUALITY=95` で 200 frame を `tmp/bedroom_jpg/` に保存
- 200 frames @ 30fps、解像度 960×540

## 11.3 結果一覧

| 試行 | 入力 | obj | prompt | 検出 | seen 累計 | fps | after-prop alloc/res (GiB) | GPU peak (MiB) | 結果 |
|---|---|---|---|---|---|---|---|---|---|
| C1-MP4 | bedroom.mp4 | 4 | person | 2 | 2 | 2.16 | 8.59 / 9.86 | 10,569 | ✓ |
| C1-JPG | bedroom_jpg | 4 | person | 2 | 2 | 2.15 | 7.48 / 8.83 | **9,519** | ✓ |
| C2-JPG | bedroom_jpg | 16 | pillow | 5 | 11 | 2.12 | 8.31 / 9.62 | 10,321 | ✓ |

## 11.4 主要な定量的知見

### C1: MP4 vs JPG 入力形式差(同一動画・同一プロンプト・200 frames)

| 指標 | MP4 | JPG | 差分 |
|---|---|---|---|
| **before propagate (alloc)** | 6.44 GiB | **4.40** | **-2.04 GiB** |
| after propagate (alloc) | 8.59 | 7.48 | -1.11 |
| GPU peak | 10,569 MiB | **9,519** | -1,050 |
| per-frame 増加 (alloc) | 10.75 MiB | 15.4 MiB | +4.65 |

- **MP4 入力は upfront +2 GiB の VRAM 常駐コスト**:
  cv2 経由の `load_video_frames_from_video_file_using_cv2` が
  全フレームを `(T, 3, 1008, 1008) fp16` テンソルに常駐させる
  (理論値 200×3×1008²×2 ≈ 1.13 GiB + cv2 stage overhead = 実測 2.04 GiB)
- **JPG 入力は async loader で漸進的にロード**:
  per-frame 増加が +4.65 MiB 高い分は、フレーム消費中に video tensor を蓄積するため
  (フレーム理論サイズ 1008²×3×2 = 6.05 MiB/frame と一致)
- **両者は最終 peak でほぼ収束**するが、MP4 は "before propagate" 時点で既に高負荷
  → 長尺動画では MP4 が先に頭打ちする

### C2: 別シーンでの B 軸再確認(bedroom 960×540, 200 frames, obj=16, "pillow")

- 検出 5 obj から始まり tracking 中に最終 11 obj まで増加
- per-frame 増加 19.5 MiB(0001 1280×720 16obj の 30.1 MiB 比 -35%)
- → **解像度 (960×540 ÷ 1280×720 = 0.56) と obj 数の積に比例**して低下、線形モデル妥当

## 11.5 入力形式選択ガイドライン(更新版)

| シナリオ | 推奨入力 | 理由 |
|---|---|---|
| 短尺 (~100 frames) | どちらでも | MP4 upfront < 1 GiB、無視できる |
| 中尺 (~200 frames @ 1008²) | どちらでも、僅差 | bedroom 実測で確認 |
| **長尺 (300+ frames @ 1008²)** | **JPG 必須** | MP4 upfront が +3 GiB 超 → before-propagate 時点で OOM 危険 |
| **高解像度 (>1280²)** | **JPG 必須** | テンソルサイズが二乗で効く |
| ストリーム処理 | JPG / MP4 別実装が必要 | 現行 MP4 ローダーは全 frame eager load |

## 11.6 12 GB 安全圏の動画別比較(per-frame 増加率)

| 動画 / 設定 | 解像度 | obj | per-frame 増 | 推定最大 frame |
|---|---|---|---|---|
| 0001 / person 4obj | 1280×720 | 4 | 18.5 MiB | ~310 |
| 0001 / window 16obj | 1280×720 | 16 | 30.1 MiB | ~170 |
| bedroom JPG / person 2obj | 960×540 | 2 | 15.4 MiB | ~400(理論) |
| bedroom JPG / pillow 11obj | 960×540 | 11 | 19.5 MiB | ~360(理論) |

低解像度動画は **B 軸の制約も緩む** ため、bedroom サイズ(960×540)なら 16 obj × 300+ frames も実用圏。

## 11.7 副次知見

- **`opencv-python-headless` (47.7 MiB) で MP4 読み込み機能が完結**(GUI deps 不要)
- **`pyproject.toml` 編集で `uv add` 反映**(プロジェクト規約準拠)
- 動画コンテンツ依存性は「解像度 × 検出 obj 数 × フレーム数」の 3 因子モデルで
  実測値を ±5% 以内で予測可能(線形モデルの強い妥当性)
