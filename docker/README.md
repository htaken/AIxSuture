# AIxSuture Docker 開発環境

`aixsuture/` と `sam3/` を別コンテナで動かすための設計。ホスト側の `sudo docker` 経由で
system daemon を使い、rootless Docker は採用しない。

---

## 方針サマリ

- **system Docker + `sudo docker`**: rootless は採用せず、既に nvidia runtime がデフォルト
  登録されている system daemon を `sudo` 経由で使う。docker グループには参加しない。
- **uv 単一ツールチェーン**: 両コンテナとも uv が依存管理。ホストと同一コマンド規約 (`uv run`)。
- **bind-mount による dev loop**: ソースは `..` から bind-mount。venv は `/opt/venv-<svc>`
  にビルド時焼き込み（`UV_PROJECT_ENVIRONMENT` でプロジェクトディレクトリ外に配置）。
- **GPU は UUID 指定**: `NVIDIA_VISIBLE_DEVICES=${GPU_UUID}` でドライバ更新時のインデックス
  入れ替わり事故を回避。共用サーバーでの排他運用前提（並列制御は無し）。
- **排他起動**: `profiles: ["aixsuture"|"sam3"]` でサービスをグループ化。

---

## 前提（ホスト側の確認項目）

### Preflight チェック

```bash
# 1. docker / compose プラグイン (既に入っている想定)
sudo docker version
sudo docker compose version      # v2.x 必須

# 2. nvidia runtime が default に登録されているか
sudo cat /etc/docker/daemon.json
# -> "default-runtime": "nvidia" または runtimes.nvidia が存在すること

# 3. GPU UUID 取得（.env の GPU_UUID に書き写す）
nvidia-smi -L

# 4. ホストデータディレクトリが takenouchi (UID 1002) 所有か
stat -c '%U %G %a %n' /mnt/ssd2/takenouchi/aixsuture/datasets \
  /home/takenouchi/AIxSuture/{weights,runs,runs/aixsuture,runs/sam3}
# datasets: SATA 側、未作成 or root 所有の場合は sudo で作成・chown
sudo mkdir -p /mnt/ssd2/takenouchi/aixsuture/datasets
sudo chown -R 1002:1002 /mnt/ssd2/takenouchi/aixsuture
# weights/runs: NVMe 側 (repo 直下), sudo 不要
mkdir -p /home/takenouchi/AIxSuture/{weights,runs/aixsuture,runs/sam3}
```

### ホストディレクトリ構成（bind-mount 先）

```
/mnt/ssd2/takenouchi/aixsuture/
└── datasets/          # video, OSATS.xlsx 等 (SATA, read-only マウント)

/home/takenouchi/AIxSuture/          # このリポジトリ直下 (NVMe)
├── weights/           # rgb_imagenet.pt 等の pretrain weights (read-only マウント)
└── runs/              # 全サービスの実験出力先 (read-write)
    ├── aixsuture/<exp>/...
    └── sam3/<task>/...
```

weights/ と runs/ は `.gitignore` と `.dockerignore` で除外されるため、
repo 直下にあっても commit や build context には含まれない。

---

## 初期セットアップ

```bash
cd docker/
cp .env.example .env
$EDITOR .env                  # GPU_UUID 等を書き込む

# 初回ビルド（sudo が必要）
sudo docker compose --profile aixsuture build
sudo docker compose --profile sam3 build
```

ビルドには aixsuture 側で ~5 分、sam3 側で ~10 分（PyTorch cu128 wheel が重い）程度。
以後 `pyproject.toml` / `uv.lock` を変更したときだけ再ビルドが必要。

---

## 実行の規約

### 基本形

```bash
sudo docker compose --profile <svc> run --rm <svc> uv run python3 <script.py> [args...]
```

サービス起動はプロファイル単位。同時起動は避ける（共用サーバー排他運用）。

### aixsuture 例

```bash
# 前処理（フレーム抽出）
sudo docker compose --profile aixsuture run --rm aixsuture \
  uv run python3 preprocessing.py \
    --data_root /workspace/aixsuture/data \
    --out_dir /workspace/runs/aixsuture/frames \
    --frame_rate 1

# 学習
sudo docker compose --profile aixsuture run --rm aixsuture \
  uv run python3 train.py \
    --exp swin_tiny \
    --data_path /workspace/runs/aixsuture/frames \
    --out /workspace/runs/aixsuture/exp_swin_tiny \
    --split 70_15_15 --snippet_length 64 --num_segments 12 \
    --arch SWINTransformer_T
```

### sam3 例（推論のみ）

```bash
sudo docker compose --profile sam3 run --rm sam3 \
  uv run python3 -c "import sam3; print(sam3.__version__, sam3.__file__)"

# bundled script 例（出力先は明示指定すること）
sudo docker compose --profile sam3 run --rm sam3 \
  uv run python3 scripts/qualitative_test.py \
    --output_dir /workspace/runs/sam3/qualitative_<date>
```

### 出力先の規約

**重要**: 各スクリプトのデフォルト出力先はバラバラなので、**呼び出し側で必ず
`/workspace/runs/<svc>/<task>/...` を明示指定する**こと。

| script | 出力先指定方法 |
|---|---|
| `aixsuture/train.py` | `--out /workspace/runs/aixsuture/<exp>` |
| `aixsuture/preprocessing.py` | `--out_dir /workspace/runs/aixsuture/<name>` |
| `sam3/scripts/qualitative_test.py` | `--output_dir /workspace/runs/sam3/<name>` |
| `sam3/sam3/train/train.py` 等 | 引数の出力先を `/workspace/runs/sam3/...` に |

デフォルトのまま叩くと `/tmp` や `~/traces` に書かれ、コンテナ削除と共に消える。

---

## uv 関連の挙動

### 環境変数（Dockerfile で設定済）

| 変数 | 値 | 意味 |
|---|---|---|
| `UV_PROJECT_ENVIRONMENT` | `/opt/venv-<svc>` | bind-mount の `/workspace/<svc>/.venv` を避け、image 焼き込みの venv を使う |
| `UV_CACHE_DIR` | `/opt/uv-cache` | named volume `uv-cache` を指す。wheel ダウンロードキャッシュ共有 |
| `UV_PYTHON_INSTALL_DIR` | `/opt/uv-python` | uv 配布の CPython バイナリ置き場 |
| `UV_LINK_MODE` | `copy` | cache からの hardlink を避ける（volume 跨ぎで失敗するため） |
| `UV_NO_SYNC` | `1` | `uv run` 時に environment の自動更新を行わない |
| `UV_FROZEN` | `1` | `uv.lock` を書き換えない |
| `HF_HOME` (sam3 のみ) | `/opt/hf-cache` | Hugging Face checkpoint cache を named volume に永続化 |
| `PYTHONPATH` (sam3 のみ) | `/workspace/sam3` | bind-mount したソースを `import sam3` で解決 |

### `UV_NO_SYNC` と `UV_FROZEN` の意味

- `UV_NO_SYNC=1`: `uv run` は project environment の更新（sync）をスキップ。
  **PATH 上の `python3` を直叩きする意味ではない**。あくまで uv 経由で project env
  （= `/opt/venv-<svc>`）の python を使いつつ、自動同期を止める。
- `UV_FROZEN=1`: `uv.lock` を更新しない（lock が古いことを静かに見逃さないために
  build 時は `--locked` を使っているので、runtime の保険）。

### 依存更新フロー

```bash
# 1. ホスト側で編集
cd aixsuture/   # または sam3/
$EDITOR pyproject.toml
uv lock          # ホストで uv.lock 更新

# 2. image 再ビルド
cd ../docker/
sudo docker compose --profile <svc> build <svc>
```

ホストで `uv sync` を走らせる必要はない（コンテナ側が source of truth）。
ただし IDE 補完のためにホスト `.venv` を更新したければ `uv sync` を別途実行。

---

## 名前付きボリュームと所有権

`uv-cache`, `uv-python`, `hf-cache` は system daemon 管理の named volume。
`/var/lib/docker/volumes/<name>/_data` に置かれる。

**初回マウント時の挙動**: Docker は image 内のマウント先ディレクトリの内容を
空 volume にコピーする。Dockerfile で `chown -R developer:developer /opt/uv-cache`
等を実施済みなので、通常は developer ownership が継承される。

**所有権エラーが出た場合**（稀だが defensive に）:

```bash
# root で入って chown し直す
sudo docker compose --profile <svc> run --rm --user root <svc> \
  chown -R 1002:1002 /opt/uv-cache /opt/venv-<svc> /opt/uv-python
```

---

## トラブルシュート

### build 時 `uv sync --locked` が失敗

lock と `pyproject.toml` が不整合。ホストで `uv lock` を再実行。

### GPU が見えない

```bash
# コンテナ内で
sudo docker compose --profile aixsuture run --rm aixsuture nvidia-smi
```

失敗する場合:
- `.env` の `GPU_UUID` が正しいか (`nvidia-smi -L` で再取得)
- ホスト側 `/etc/docker/daemon.json` に `"default-runtime": "nvidia"` があるか

### bind-mount ファイルの所有権がおかしい

コンテナ内 developer = UID 1002。ホスト takenouchi = UID 1002。
`.env` の `UID`/`GID` が 1002/1002 になっているか確認。

### ホスト `.venv` とコンテナ側の干渉

`aixsuture/.venv` はホスト IDE (Zed/LSP) 用に残している。
`UV_PROJECT_ENVIRONMENT=/opt/venv-aixsuture` が効くのでコンテナから参照されないが、
万一混乱したら `.venv` を一時的にリネームして動作確認する。

### build context が巨大

`.dockerignore` で `aixsuture/.venv` (5.6GB) 等を除外済み。
それでも遅ければ `du -sh */` で膨張元を特定。

---

## 設計上の意図（要点）

- rootless Docker 撤回の理由: ホストに uidmap 未導入・`~/.config/docker` 未作成・
  NVIDIA の `no-cgroups` 未設定で、セットアップコストに対して得られる isolation
  (「sudo の speedbump 程度の行儀」) が見合わない。構造的隔離が必要になったら再検討。
- sam3 を editable install しない理由: 現状は推論のみで、`PYTHONPATH` 経由の import で
  十分。training に拡張する場合は `--extra train` を足し、editable install するか
  `python -m sam3.train.train` 規約に切り替えるかを別途決める。
- sam3 の Python pin (`3.12`) は `docker/sam3/.python-version` で管理。
  submodule 側 `sam3/.gitignore` が `.python-version` を除外しているため、fork に
  差分を入れずに親 repo 側で pin を追跡する構成。aixsuture は自前管理なので
  `aixsuture/.python-version` を直接追跡（対称ではないが、ownership の違いを反映）。
- `--extra notebooks` を sam3 に入れた理由: bundled script (`qualitative_test.py` 等)
  が `cv2`/`matplotlib` を要求するため。default deps だけでは初日に import エラー。
