# annotation-qc

`tools/t4dataset-webdataset/` の出力(sensor/anno tar + parquet index)を読み、画像へ
LiDAR点群投影・アノテーション投影(3Dボックスワイヤーフレーム、2Dbbox、セグメンテーション
マスク)を行った可視化画像を生成し、それをローカル/オンラインのLLM・VLMに渡してアノテーション
品質を自動チェックするツール。

データセット形式非依存の `DatasetAdapter` インターフェース(`adapter_base.py`)を挟んでいるため、
別のデータセット形式を追加する場合は新しいアダプタクラスを実装するだけでよい。現在
`t4dataset-webdataset`(点群+2D/3Dアノテーション)と `tlr-crops`(信号検出用クロップ、
2D画像のみ)の2アダプタに対応。

`run_aug_qc.py`(後述)は別ツールで、アノテーションではなく**データ拡張・合成画像そのもの**の
自然さ/破綻をチェックする。

## セットアップ

```bash
pip install -r requirements.txt
```

## 使い方

```bash
# 可視化のみ(LLMなし)のスモークテスト
python3 run_qc.py \
  --adapter t4dataset-webdataset --input /path/to/webdataset --out-dir ./qc-out \
  --clips <clip_id> --every-n 30 --channels CAM_FRONT_WIDE --max-frames 5 --skip-llm

# Ollamaでのフル実行、歩行者だけを対象にカテゴリ別QA観点を適用
python3 run_qc.py \
  --adapter t4dataset-webdataset --input /path/to/webdataset --out-dir ./qc-out \
  --categories pedestrian --llm-base-url http://localhost:11434/v1 --llm-model qwen3.8:27b

# オンラインモデルへの切り替え(コード変更不要、base-url/api-key/modelを変えるだけ)
python3 run_qc.py \
  --adapter t4dataset-webdataset --input /path/to/webdataset --out-dir ./qc-out \
  --llm-base-url https://api.openai.com/v1 --llm-api-key-env OPENAI_API_KEY --llm-model gpt-4o

# 信号検出(TLR)クロップデータセット (Annotations/JPEGImages/labels レイアウト)
python3 run_qc.py \
  --adapter tlr-crops --input /path/to/baseline3/train_dataset --out-dir ./qc-out \
  --clips <clip_id> --max-frames 20
```

主なオプション:

| オプション | 説明 |
|---|---|
| `--clips` | カンマ区切りのclip_id。省略時は`--input`配下の全クリップ |
| `--every-n` | Nフレームごとにサンプリング |
| `--channels` | カンマ区切りのカメラチャンネル。省略時は全カメラ |
| `--categories` | カンマ区切りの対象カテゴリ(例: `car,pedestrian,traffic_light`)。省略時は全カテゴリ |
| `--only-target-categories` | 対象カテゴリ以外は可視化に描画しない(見やすさ優先の減量モード) |
| `--qc-prompts-file` | カテゴリ別QA観点YAML。デフォルトは同梱の`qc_prompts.yaml` |
| `--ann3d-frame` | 3Dアノテーションの座標系。`world`(デフォルト)/`ego` |
| `--skip-llm` | 可視化のみ実行し、LLM呼び出しをしない(ドライラン) |
| `--max-frames` | 処理する(frame, channel)組の総数上限(デバッグ用) |
| `--jobs` | LLM呼び出しの並列数 |
| `--llm-base-url` | カンマ区切りで複数指定すると、タスクをラウンドロビンで振り分ける(例: GPUごとに立てた複数Ollamaインスタンス。`--jobs`もエンドポイント数以上にすること) |
| `--min-display-size` | 合成画像の短辺がこれ未満なら、全レイヤー描画完了後にアップスケール(デフォルト320px、TLRクロップ等の極小画像向け) |

## カテゴリ別QA観点の運用 (`qc_prompts.yaml`)

QAで見るべき観点はPythonコードに埋め込まず、`qc_prompts.yaml` に外出ししてある。運用者は
このYAMLを直接編集するだけで、コード変更なしに観点を追記・改善できる。`categories:` に
無いカテゴリは `default:` の汎用観点にフォールバックする。

`--categories` で絞り込んだ場合、実際にそのフレームで検出されているカテゴリとの積集合だけが
プロンプトに組み込まれる(無関係な観点文でプロンプトを膨らませない)。

## 出力

```
<out-dir>/
  viz/<clip_id>/<key>.<channel_lower>.png   # 可視化画像
  report.jsonl                               # (clip_id, frame_index, channel)単位の結果
  report.parquet                             # 同上、pandas/DuckDB等で集計しやすい形式
  report_summary.json                        # カテゴリ/深刻度別集計、frame_ok率
```

## 座標系に関する注意

`ann3d.json` の `translation` は nuScenes公式スキーマに従い **world(global)フレーム**として
扱っている(`--ann3d-frame ego` で切り替え可能)。また `ego_pose.rotation` が実データで長時間
恒等回転のまま変化しないケースを確認しており、プレースホルダの可能性がある(要確認)。

## アダプタ: `tlr-crops` (信号検出用クロップデータセット)

`Annotations/*.json` + `JPEGImages/*.jpg` + `labels/*.txt`(Darknet分類器用、未使用)という
レイアウトの信号機クロップデータセット向け。クロップは1つの信号灯パネル全体(複合信号なら
複数灯を含む)を切り出したもので、点群・3Dアノテーションは持たない。

- `category_name` は常に `"traffic_light"` に統一し、色状態(red/green/...)・`lit_frac`
  (点灯割合)・`overexposed_frac`(白とび割合)・`orientation`・`type` は `attribute_names`
  に載せる。こうすることで t4dataset-webdataset 側の `traffic_light` カテゴリ向けQA観点が
  そのまま流用でき、`--categories traffic_light` の絞り込みも同じ感覚で使える。
- データ拡張で生成された `_whole_*`/`_flicker`/`_ph0X`/`_lr0X` サフィックス付き画像は対応する
  `Annotations/*.json` を持たないため、`Annotations/*.json` を正として自然に除外される。
  ただし `_rcn0X`/`_rcw0X`(ランダムクロップの狭め/広げ)は例外で、実写真を別の窓枠で
  再クロップしただけ(ピクセルの合成はしていない)のため、bbox_relative等を再計算した
  **独自のAnnotations JSONを持つ**。つまり8,125件(baseline2/val_datasetの実測)は
  「ユニークな実写真の数」ではなく「ラベル付きクロップ・インスタンスの数」であり、
  同じ実写真・同じ信号が複数のクロップ窓枠違いで重複カウントされている。
- クロップは数十px四方など極小のことが多く(`--min-display-size` で全レイヤー描画後に
  アップスケールされる)、`lit_frac`/`overexposed_frac` の数値をプロンプト本文にも
  明示的に埋め込む(`qc_schema.describe_frame_annotations`)ことで、画像だけでは
  判別しづらい「点灯しているはずなのに実際は暗い/白とびしている」ようなケースを
  LLMがクロスチェックできるようにしている。

## `run_aug_qc.py`: データ拡張・合成画像そのものの自然さチェック

`run_qc.py --adapter tlr-crops` は「アノテーション(信号の状態ラベル等)が実物と一致しているか」
をチェックするが、`run_aug_qc.py` は目的が異なり、**アノテーションは見ずに「拡張・合成処理
そのものが視覚的に破綻していない・不自然でないか」だけ**を別のプロンプト・別の出力スキーマ
(`aug_qc_schema.py`, `aug_prompts.yaml`)でチェックする。意図的に `qc_schema.py`/
`qc_prompts.yaml` とは完全に分離してある。

```bash
# 可視化のみ(LLMなし)のスモークテスト
python3 run_aug_qc.py --input /path/to/baseline2/val_dataset --out-dir ./aug-qc-out \
  --clips <clip_id> --max-frames 5 --skip-llm

# Ollamaでのフル実行、GPUごとに立てた複数インスタンスに振り分け
python3 run_aug_qc.py --input /path/to/baseline2/val_dataset --out-dir ./aug-qc-out \
  --jobs 3 --llm-base-url http://127.0.0.1:11435/v1,http://127.0.0.1:11436/v1,http://127.0.0.1:11437/v1
```

**仕組み**:
- `aug_pairs.py` が、拡張済み画像のファイル名から「そのクリップの実アノテーションstem集合の
  うち最長一致する接頭辞」を探すことで、どの実クロップ(または`_rcn0X`/`_rcw0X`の再クロップ)
  から派生した拡張かを特定する(bbox数値の位置を正規表現で当てにいく方式より頑健)。
  残りの部分(タグチェイン、例: `rcn00_whole_green_deg015`)を構造化してパースする。
- 各タグの意味は生成ログ(`train_gen.log`/`val_gen.log`の`Aug:`行)の設定値から確認したもの:
  `whole_rot`→`_whole_<color>_deg<NNN>`(色相回転によるレンズ全体の合成塗り替え)、
  `photo`→`_ph0X`(明るさ/コントラスト等のジッター)、
  `random_crop`→`_rcn0X`(narrower)/`_rcw0X`(wider)、`flicker`→`_flicker`。
  `_lr0X`(`lowres`設定に対応すると推定)だけは生成ツールのソース未確認のため
  正確な変換内容が断定できておらず、`aug_prompts.yaml`にもその旨を明記している。
- LLMには **[1]元画像 と [2]拡張後画像のペア** + 適用された変換の説明文を渡し、
  「合成そのものが破綻/不自然でないか(色が不自然、継ぎ目が見える、周囲の照明と矛盾、
  クロップで構造が欠けている、劣化のかけ方が非現実的、等)」だけを判定させる。
- 出力スキーマは `{"aug_ok": bool, "issues": [...]}` で `frame_ok`/`issues`とは別カテゴリ体系
  (`unnatural_color`/`visible_artifact`/`implausible_lighting`/`structure_loss`/
  `unrealistic_degradation`/`other`)。`report.py`のReportWriter/ReadWriteは共通利用しており、
  `report.jsonl`の`frame_ok`フィールドには`aug_ok`の値が入る。

実データでの検証例: `_rcn00_whole_red_deg000`(緑信号を赤に色相回転で合成)のサンプルで、
「赤というよりピンクに寄っている」「周囲の発光にじみ(グロー)が元の緑のまま更新されていない」
という2点を実際に検出。一方、目視で自然に見えた`_flicker`サンプルは正しく`aug_ok: true`と
判定し、何でも不合格にするバイアスは無いことも確認済み。

## 既知の制約 / 今後の拡張候補

- マルチカメラの合成グリッド表示は未対応(`--panel`フラグ候補)
- Anthropic Claude APIなど、OpenAI互換でないオンラインAPI用のアダプタは未実装
