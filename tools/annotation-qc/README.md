# annotation-qc

`tools/t4dataset-webdataset/` の出力(sensor/anno tar + parquet index)を読み、画像へ
LiDAR点群投影・アノテーション投影(3Dボックスワイヤーフレーム、2Dbbox、セグメンテーション
マスク)を行った可視化画像を生成し、それをローカル/オンラインのLLM・VLMに渡してアノテーション
品質を自動チェックするツール。

データセット形式非依存の `DatasetAdapter` インターフェース(`adapter_base.py`)を挟んでいるため、
将来別のデータセット形式(例: 信号検出用train/val、2D画像のみで点群・3Dアノテーションなし)を
追加する場合は、新しいアダプタクラスを実装するだけでよい。

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

## 既知の制約 / 今後の拡張候補

- マルチカメラの合成グリッド表示は未対応(`--panel`フラグ候補)
- Anthropic Claude APIなど、OpenAI互換でないオンラインAPI用のアダプタは未実装
- 信号検出用train/val等、2D専用データセットのアダプタは未実装(`DatasetAdapter`を実装するだけで追加可能)
