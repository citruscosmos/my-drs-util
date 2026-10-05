# t4dataset-v1-shard-ext

T4(NuScenes 形式)データセットを、tar シャードと parquet index で保持するためのフォーマット規定。

- 目的: セグメント単位の tar を NAS 上に置き、parquet の offset index でランダムアクセスする。
- 背景: `jt10/20260930_jt10_utsunomiya/t4dataset`(以下 t4 版)と `datasets/tlr/fastlabel_data_2410`(以下 fastlabel 版)の
  構成差を調査し、両者の良い点を統合した(2026-10-05)。付録 A に調査結果を残す。
- 用語: 「規定」は本書が内容を決めるもの。「非規定」は意図的に決めないもの(用途ごとの自由度を残す)。

## 1. 用語

| 用語 | 意味 |
|---|---|
| corpus | データセット 1 つ。ルートディレクトリ 1 つに対応する(例: `fastlabel_data_2410`) |
| session | 1 回の録画。corpus 直下のディレクトリ 1 つ(例: `HRdqo3pf_2026-09-30T11-02-12+0900`) |
| segment | session を約 30 秒ごとに区切った単位。`segment_id` で識別する |
| layer | segment に付く tar の種類。`data`、`annotation`、およびそれ以外(追加 tar) |

## 2. フォルダ構成(規定)

```
<corpus>/
  _manifest.json
  _index/part-00000.parquet, part-00001.parquet, ...
  <session>/
    <segment_id>_data.tar
    <segment_id>_annotation.tar
    <segment_id>_<layer>[_<version>].tar    # 追加 tar(0 個以上、随時追加)
    <segment_id>_index.parquet
```

- segment_id のフォルダ階層は作らない。session の下にファイルを直接置く。
  - 根拠: segment は約 30 秒で、3 時間の session でも 360 segment、ファイルは数千に収まる。
- session フォルダは作る(録画単位の一覧・移動・削除のため)。
- パスは corpus ルートからの相対で扱う。絶対パスは index にも manifest にも書かない。

### 2.1 命名

- `segment_id` は `<session>_<連番>` 形式とする。例: `HRdqo3pf_2026-09-30T11-02-12+0900_0`
  (録画名が `<車両ID>_<開始時刻>` であり、`<session>` はこの部分に当たる)。
  - 連番は 0 始まりの 10 進数で、ゼロ埋めしない(`_071` ではなく `_71`)。
- tar のファイル名は `<segment_id>_<layer>.tar`、版を持つ場合は `<segment_id>_<layer>_<version>.tar`。
  - `layer` は英小文字で始まり、英小文字・数字・`-`・`_` を使う。例: `data`、`annotation`、`panoptic-segmentation`、`tlr_conditions`。
  - `version` は `v<数字>(.<数字>)*` の形。例: `v0.1.0`。層名の末尾を `_v<数字>` の形にしてはならない(版と区別できなくなるため)。
  - ファイル名には、英数字と `-`、`_`、`.`、`+`(session 名の `+0900`)以外を使わない。`@` や `:` は使わない(`rsync`/`scp` や URL で別の意味に解釈されるため)。
  - 解析規則(一意に決まる): 拡張子 `.tar` を除いた名前を、次の正規表現で分ける。

    ```
    ^(?P<segment_id>.+_\d+)_(?P<layer>[a-z][a-z0-9_-]*?)(?:_(?P<version>v\d+(?:\.\d+)*))?$
    ```

    segment_id は `_<数字>` で終わり、layer は英小文字で始まるため、境目は 1 通りに決まる。
    例: `..._103_tlr_conditions_v0.1.0.tar` → segment `..._103`、layer `tlr_conditions`、version `v0.1.0`。
- 空の tar を置かない。ごみファイル(session に属さない tar など)を置かない。

### 2.2 レガシーの tar

- 命名規則(§2.1)に合わない既存の tar は、**改名せずレガシーとして残す**。対象は次の 2 種類。
  - segment の連番がゼロ埋めされている tar(例: `..._071_data.tar`)。
  - `segment_id` が `_<数字>` で終わらない tar(例: `SemiAnnotation_*`、`models_annotation.tar` のようなグループ名のみの tar)。
- レガシー tar は、次のように扱う。
  - `_manifest.json` の `legacy` に、corpus ルートからの相対パスで列挙する(§6)。
  - 命名規則(§2.1)と、検査項目 1・2・6(§7)の対象外とする。名前は変えないので、index の `segment_id` も、レガシーの名前から読み取ったままとする(`_071` のまま)。
  - 置き場所は §2 に従う。`segment_id` から session が決まるもの(ゼロ埋めの tar)は `<session>/` の下へ置く。
    session が決まらないもの(グループ名のみの tar など)は、corpus ルート直下に残す。
  - 新しく作る tar は、レガシーの名前に合わせず、必ず §2.1 に従う。
- レガシーかどうかの判定は、`legacy` への記載だけで行う(名前の形では判定しない)。

### 2.3 tar の場所の解決

- 利用側は、`segment_id` と層から、tar の場所を次の順で探す。
  1. `<corpus ルート>/<session>/<segment_id>_<layer>[_<version>].tar`。`session` は、`segment_id` の末尾の `_<数字>` を取り除いた部分。
  2. 1 になければ、`<corpus ルート>/<segment_id>_<layer>[_<version>].tar`(`segment_id` が `_<数字>` で終わらないレガシー tar は、ルート直下にある)。
- 迷う場合は、index の `tar_file` を正とする。`tar_file` は corpus ルートからの相対パス。

### 2.4 corpus 名と置き場所

- 生データ(録画)から作る corpus の名前は、録画グループ名 `<日付>_<車両>_<場所>` とする。
  例: `20260930_jt10_utsunomiya`(日付は `yyyymmdd`)。
  - 名前に**車両名を含める**。車両が違っても名前が衝突しないようにするため。
- 置き場所は `datasets/<corpus>/`。この `<corpus>/` が corpus ルートになる。
- 生データ(`ecu0`、`ecu1`)と、そこから作る `gps-route/` は、corpus ルートの外(`raw/<車両>/<group>/`)に置く。

## 3. 必須の層(規定)

segment は `data` と `annotation` の 2 つの tar を必ず持つ。

### 3.1 `<segment_id>_data.tar`

- メンバー名は `<センサチャンネル>/<フレーム番号>.<拡張子>`。プレフィックスなしの相対名。
  - 画像: `CAM_FRONT_NARROW/00000.jpg`
  - 点群: `LIDAR_CONCAT/00000.pcd.bin`
- フレーム番号は 5 桁ゼロ埋めの 10 進数。

### 3.2 `<segment_id>_annotation.tar`

- メンバーは JSON テーブルをそのまま、ディレクトリなしの平置きで入れる。
- 必須のメンバーは、T4 の次の 13 個。
  `attribute.json`、`calibrated_sensor.json`、`category.json`、`ego_pose.json`、`instance.json`、`log.json`、
  `map.json`、`sample.json`、`sample_annotation.json`、`sample_data.json`、`scene.json`、`sensor.json`、
  `visibility.json`。
- `vehicle_status.json` は任意とする(fastlabel_data_2410 の annotation tar 9,889 個のうち 8,683 個に存在しないため)。
- 必須以外の JSON テーブルを追加してよい。追加するメンバーの内容は規定しない
  (既存の例: `vehicle_status.json`、`object_ann.json`、`surface_ann.json`、`autolabel_model.json`)。
- 内容の整合(prev/next リンク、トークンの一意性)は、tar にする前に検証する。

## 4. 追加 tar(非規定)

- 追加 tar(例: `panoptic-segmentation`、`traffic-light-recognition`、`tlr_conditions_v0.1.0`)は、
  **中身を規定しない**。メンバー名・形式は用途に任せる。
- 追加 tar は随時追加される。segment ごとに、あるものとないものが混在してよい。
- 「tar がない」は「その層は未提供」を意味する。「空である」という宣言は行わない(manifest にも書かない)。
- 追加した tar は index に載せる(§5)。tar の中身を規定しなくても、index はメンバー名・offset・size だけで作れる。

## 5. parquet index(規定)

### 5.1 スキーマ

| 列 | 型 | 内容 |
|---|---|---|
| `corpus` | string(辞書) | corpus 名 |
| `session` | string(辞書) | session 名 |
| `segment_id` | string | segment_id |
| `layer` | string(辞書) | `data`、`annotation`、追加 tar の層名 |
| `layer_version` | string(辞書)、null 可 | 版。版がなければ null |
| `tar_file` | string | corpus ルートからの相対パス(例: `<session>/<segment_id>_data.tar`) |
| `member` | string | tar 内のメンバー名 |
| `offset_data` | int64 | tar ファイル先頭からメンバーの**データ**開始位置までのバイト数 |
| `size` | int64 | メンバーのデータサイズ(バイト) |
| `offset_header` | int64、null 可 | メンバーのヘッダ開始位置。PAX ヘッダ付きの tar を `tarfile` で読み直すために持つ |
| `sha256` | string、null 可 | メンバーのハッシュ。任意 |

- ランダムアクセスに必要なのは `tar_file`、`offset_data`、`size` だけ。
  `offset_data` はヘッダの長さ(PAX ヘッダ有無)に依存しない値とする。

### 5.2 物理形式

- 行は `(segment_id, layer, member)` の順に並べる。
- 圧縮は zstd、行グループは約 10 万行。行グループの最小・最大値で `segment_id` の絞り込みができるようにする。
- 1 つの segment の行は 1 つの index ファイルにまとめる(segment を複数パートにまたがせない)。

### 5.3 2 種類の index

| index | 置き場所 | 役割 |
|---|---|---|
| segment index | `<session>/<segment_id>_index.parquet` | 書き込み単位。その segment の全 tar の行を持つ。アトミックに(一時名→リネーム)書く |
| corpus index | `_index/part-*.parquet` | 検索用。segment index をまとめたもの(パートは約 30 万行ごと) |

- 追加 tar を足したら、その segment の segment index を作り直し、該当する corpus index のパートを更新する。
- corpus index を作り直す前でも、segment index だけで読める。

## 6. manifest(規定)

`_manifest.json` に、以下を持たせる。

```json
{
  "format": "t4dataset-v1-shard-ext",
  "schema_version": 1,
  "corpus": "<corpus 名>",
  "updated": "<ISO 8601>",
  "sessions": 34,
  "segments": 771,
  "tars": [
    {"path": "<session>/<segment_id>_data.tar", "size": 8145766400, "sha256": null}
  ],
  "legacy": ["models_annotation.tar"]
}
```

- `format` は、`t4dataset-v1-shard-ext`(本書)か、`t4dataset-v1`(展開済み形式、§9)。
- `source`(任意)は、生データへの対応。例: `{"vehicle": "jt10", "group": "20260930_jt10_utsunomiya"}`。
  生データのない corpus(fastlabel など)では書かなくてよい。
- `tars` は、存在する tar の一覧。**ないことの宣言は含めない**(§4)。
- `sha256` は任意。
- `legacy` は、§2.2 のレガシー tar の相対パス一覧。なければ空配列。`tars` にも同じものを載せる。

## 7. 検査項目(規定)

corpus を公開する前に、次を検査する。

1. すべての segment に `data` と `annotation` の tar がある(レガシーを除く)。
2. 存在するすべての tar が、index に載っている(tar と index の差分が 0。レガシーを除く)。
3. index の `tar_file` がすべて実在し、`offset_data + size` が tar のサイズ以内に収まる。
4. 無作為抽出した行で、`offset_data` から読んだ先頭バイトが拡張子と一致する(JPEG の `FFD8`、JSON の先頭文字など)。
5. `annotation` の JSON が有効で、トークンが重複せず、sample_data の prev/next が整合する。
6. 空の tar、命名規則に反するファイルがない(レガシーを除く)。

## 8. 現状との差

2026-10-05: `fastlabel_data_2410`、`Kognic-2D-Annotation-adapted`、および `jt10` の 13 グループの T4 データセット
(`datasets/<group>/`)は、本書の配置・manifest へ移行済み。tar 形式の 2 グループ(`20260930_jt10_utsunomiya`、
`20260910_jt10_todenarakawa_verify`)は index も本書の形に変換済みで、展開済みの 11 グループは `t4dataset-v1`(§9)として移動のみ。
パイプライン(`rosbag-data-pipeline`)の出力も、この配置・index・manifest になった。
以下の表は、移行前の差と、今も残っている項目(空の tar、旧 t4 版の記述)を示す。

| 項目 | 現状 | 本書 |
|---|---|---|
| t4 版の置き場所 | `jt10/<group>/t4dataset/` | `datasets/<group>/`(§2.4) |
| t4 版のパス | `<session>/<segment_id>/` の下に 3 ファイル | segment フォルダなし |
| t4 版の index 列 | `category`、`member`、`tar_file`(ファイル名のみ)、`offset`、`size` | §5.1 の列 |
| t4 版の corpus index | なし | `_index/part-*.parquet` |
| t4 版の manifest | `.chunked_complete`(録画ごと) | `_manifest.json` |
| fastlabel 版 | フラット(49,529 tar)、絶対パス、`tlr_conditions` が index にない、空 tar あり | §2〜§7 |
| fastlabel 版のゼロ埋め | `segment_id` の連番がゼロ埋めの tar が 1,767 セグメント(46 session) | レガシーとして残す(§2.2)。`<session>/` へ移し、`legacy` に載せる |
| Kognic-2D-Annotation-adapted | 絶対パスのシンボリックリンク 744 個(元 corpus `Kognic-2D-Annotation` を指す)と、実体 296 個。index なし | リンクを実体のコピーに置き換えて自己完結させる。`<session>/` へ移し、index と manifest を作る |
| fastlabel 版の既存名 | 命名規則(§2.1)の正規表現で、`.tar` 49,529 個中 86 個が解析できない: `SemiAnnotation_*_<時刻>_<時刻>_<層>.tar` 50 個(segment_id に `_<数字>` がない)、`models_*`、`tfl_*`、`rain_*` などグループ名のみの tar 36 個 | レガシーとして残す(§2.2)。改名しない |

## 9. 展開済み形式 `t4dataset-v1`

tar にしていない既存の T4 データセットは、変換せずに、次の形式として扱う。

```
<corpus>/
  _manifest.json
  <session>/<segment_id>/
    annotation/<テーブル>.json      # §3.2 と同じ(必須の 13 個 + 任意)
    data/<センサチャンネル>/<フレーム番号>.<拡張子>     # §3.1 と同じ
```

- `_manifest.json` は、`format` を `t4dataset-v1` とする。`tars`、`legacy`、`_index/` は持たない。
  `corpus`、`updated`、`sessions`、`segments`、任意の `source` を書く。
- 配置は規定の `<session>/` の下を踏襲する(segment は、segment_id のフォルダ 1 つ)。
- メンバー名(`annotation/` と `data/` の下の相対パス)は、tar 形式のメンバー名と同じ。
  後から `t4dataset-v1-shard-ext` へ変換できる。
- 規定の検査項目のうち、tar と index に関するもの(§7 の 2・3・4)は対象外。

## 付録 A. 調査結果(2026-10-05)

| 観点 | t4 版(`20260930_jt10_utsunomiya`) | fastlabel 版(`fastlabel_data_2410`) |
|---|---|---|
| 規模 | 34 session、771 segment、index 291 万行 | 49,529 tar、9,889 segment、index 113.9 万行 |
| フォルダ | `<session>/<segment>/` に 3 ファイル | フラット + `_offset_index/` |
| segment あたり | `data`、`annotation`、index | `data`、`annotation`、`panoptic-segmentation`、`traffic-light-recognition`、`tlr_conditions_v0.1.0` |
| index の列 | `category`、`member`、`tar_file`、`offset`、`size` | `corpus`、`segment_id`、`shard_type`、`tar_path`、`member_name`、`offset_header`、`offset_data`、`size` |
| パス | 相対(ファイル名のみ) | 絶対パス |
| index の網羅性 | 全 tar が対象 | tar 13,547 個が対象外(層別: `tlr_conditions` 9,889、`traffic-light-recognition` 3,092(すべて空 tar)、`panoptic-segmentation` 522、`data` 23、`annotation` 21)。空 tar は全体で 3,661 個 |

offset の正しさは、両方でランダムに 300 行ずつ実際に読んで確認した(不一致 0)。
