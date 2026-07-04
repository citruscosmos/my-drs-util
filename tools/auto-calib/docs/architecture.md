# Auto-Calibration Pipeline — Architecture Design

## 目的

RTK/INS 付き車両に搭載された複数 LiDAR・カメラの外部パラメータ（TF）を、
8の字走行データから自動キャリブレーションするオフラインパイプライン。

---

## センサー構成

| センサー | 数 | トピック | 周波数 |
|---|---|---|---|
| LiDAR (Seyond) | 4 | `/sensing/lidar/{front,rear,left,right}/seyond_points` | 10 Hz |
| カメラ 通常 | 8 | `/sensing/camera/camera{0-7}/image_raw/compressed` | 20 Hz |
| カメラ 魚眼 | 4 | `/sensing/camera/camera{8-11}/image_raw/compressed` | 20 Hz |
| camera_info | 12 | `/sensing/camera/camera{0-11}/camera_info` | 20 Hz |
| RTK/INS odometry | 1 | `/sensing/ins/oxts/odometry` | 50 Hz |
| INS IMU | 1 | `/sensing/ins/oxts/imu` | 100 Hz |

### カメラ-LiDAR 対応表

| LiDAR | 通常カメラ | 魚眼カメラ |
|---|---|---|
| `lidar_front` | camera0, camera1 | camera8 (前方) |
| `lidar_right` | camera2, camera3 | camera9 (右方) |
| `lidar_rear`  | camera4, camera5 | camera10 (後方) |
| `lidar_left`  | camera6, camera7 | camera11 (左方) |

### 魚眼カメラの歪みモデル

camera8–11 は equidistant モデル (`camera_info.distortion_model = "equidistant"`)。
投影・undistort には `cv2.fisheye.*` を使用する。

---

## TF ツリー（rosbag より確認: HRdqo3pf_2026-06-25T16-34-04_pc.mcap）

```
map (= ECEF, 地心座標)                     ← odometry の位置は 1e6 m オーダーの ECEF
└── oxts_link          (/sensing/ins/oxts/odometry  50 Hz, dynamic: map → oxts_link)
    │                   ↑ この動的 TF は odometry から得る（tf ツリー上の静的親は下記 drs_base_link）
    │
base_link
└── drs_base_link  (static: base_link→drs_base_link t=(0.69,0,1.97))
        ├── oxts_link  ← static: drs_base_link→oxts_link t=(0,0,0), q_xyzw=(1,0,0,0)
        │              = **X軸まわり180°回転（identity ではない！）**。oxts は FRD 系、drs は FLU 系
        │   └── imu_link (identity)
        ├── lidar_front   ← 基準フレーム (変更しない)
        │   ├── camera0/camera_link → camera0/camera_optical_link
        │   ├── camera1/camera_link → camera1/camera_optical_link
        │   └── camera8/camera_link → camera8/camera_optical_link
        ├── lidar_rear
        │   ├── camera4/camera_link → camera4/camera_optical_link
        │   ├── camera5/camera_link → camera5/camera_optical_link
        │   └── camera10/camera_link → camera10/camera_optical_link
        ├── lidar_left
        │   ├── camera6/camera_link → camera6/camera_optical_link
        │   ├── camera7/camera_link → camera7/camera_optical_link
        │   └── camera11/camera_link → camera11/camera_optical_link
        └── lidar_right
            ├── camera2/camera_link → camera2/camera_optical_link
            ├── camera3/camera_link → camera3/camera_optical_link
            └── camera9/camera_link → camera9/camera_optical_link
```

`camera_link → camera_optical_link` は固定 (`q = (0.5, -0.5, 0.5, -0.5)`)。

> 上記の `drs_base_link → lidar_{front,rear,left,right}` および全カメラの初期外部パラは
> **すべて rosbag の `/tf_static`（13 transforms）に含まれる**ことを実バッグで確認済み。
> Step 3/4 の初期 TF はここから読み込む（別途 `multi_tf_static.yaml` を用意しても可）。
> なお `/sensing/ins/oxts/lever_arm` トピックが存在し、INS 基準点のレバーアーム値を持つ。

### LiDAR PointCloud2 フィールド（キャリブ対象バッグの実測フィールド）

キャリブは Seyond ドライバが復号した **`seyond_points`（PointCloud2）** を入力とする。実バッグ
（`HRdqo3pf_2026-06-25T17-37-26_pc_replaced.mcap`）で確認したフィールドは:

`x, y, z, intensity, time_stamp`（point_step=32、offset: x0 y4 z8 intensity16 time_stamp20、他はパディング）

- **`time_stamp` (uint32)**: **スキャン先頭からの相対経過時間 [ns]**（実測 50,000〜92,451,500 ns = 0〜約92.45ms、点ごとに単調増加）。**単位は ns**（μs ではない点に注意）
- ある点の絶対取得時刻 = `header.stamp + time_stamp[ns]`。1スキャンは約 **92.4 ms** に渡る（周期100msのほぼ全域）→ undistortion の効果大

> ⚠ **フィールド名/単位に関する重要な注意（実データで確定）**:
> - 点ごとの時刻情報の一次ソースは生 LiDAR の `seyond_packets`（型 `seyond/msg/SeyondScan`）だが、
>   そこでは各点時刻は不透明な `uint8[] data`（Seyond 独自エンコード）に埋め込まれており**直接読めない**。
>   Seyond ドライバが `seyond_points` へ復号する際に点単位の相対時刻を **`time_stamp` (ns)** として露出する。
>   → **キャリブに生 `seyond_packets` は不要。`seyond_points.time_stamp` をそのまま使う。**
> - バッグ由来により時刻フィールド名/単位は揺れる（別バッグでは `t_us`(μs, offset20)＋`timestamp`(ns, offset24) の2フィールド構成だった）。
>   実装では**フィールド名を固定せず**、存在する時刻フィールドを検出し、値域から単位（μs/ns）を判定して
>   「スキャン先頭からの相対時間」に正規化すること。
> - undistortion は `t0 = header.stamp` を基準に、各点時刻 `header.stamp + Δt` で補間する。

---

## 推定する TF（計 15 本）

### LiDAR-to-LiDAR（3 本）

**出力フレーム: `drs_base_link` → `lidar_{rear, left, right}`**

| TF | 説明 |
|---|---|
| `drs_base_link → lidar_rear`  | 車体基準→後 LiDAR |
| `drs_base_link → lidar_left`  | 車体基準→左 LiDAR |
| `drs_base_link → lidar_right` | 車体基準→右 LiDAR |

> ICP は `T_front_to_other` を直接算出するが、既存 TF ツリーの親フレームが
> `drs_base_link` であるため、出力時に
> `T_drs_to_other = T_drs_to_front @ T_front_to_other` に変換して保存する。
> (`drs_base_link → lidar_front` は固定値として保持)

### Camera-to-LiDAR（12 本）

出力フレーム: `lidar_*` → `camera*/camera_link`

| カメラ | 対応 LiDAR | 種別 | 優先度 |
|---|---|---|---|
| camera0, camera1 | lidar_front | 通常 | 高（初期値良好） |
| camera2, camera3 | lidar_right | 通常 | 高 |
| camera4, camera5 | lidar_rear  | 通常 | 高 |
| camera6, camera7 | lidar_left  | 通常 | 高 |
| camera8  | lidar_front | 魚眼・前方 | 中（初期値誤差 10cm 未満前提・equidistant 投影） |
| camera9  | lidar_right | 魚眼・右方 | 中 |
| camera10 | lidar_rear  | 魚眼・後方 | 中 |
| camera11 | lidar_left  | 魚眼・左方 | 中 |

---

## キャリブレーションパイプライン

### データ要件

- 走行パターン: **累計 yaw 変化 360° 以上**の右左折を含む走行（全6自由度の励起に必要）
- トピック構成: 上記センサー構成と同一
- キャリブ専用 rosbag を別途用意（本参照 rosbag とは別）

#### 推奨環境と代替環境

| 優先度 | 環境 | 走行パターン | 速度 | 時間 | 備考 |
|---|---|---|---|---|---|
| **第1** | 駐車場（30m 四方程度） | 8の字走行 | < 10 km/h | 2 分 | 静止構造物に囲まれた理想環境 |
| **第2** | 工業地帯・埠頭などオープンスカイの広場 | 右左折を含む周回 | ≤ 20 km/h | 10 分 | RTK cm 級精度・構造物あり・動体少ない |
| **第3** | 動体の少ない路地（交差点あり） | 右左折を繰り返す往復 | ≤ 20 km/h | 5〜10 分 | **走行前に RTK Fix 状態を確認すること** |

#### 環境選定の判断基準

```
以下の 4 条件をすべて満たす環境が適切:

  ✓ オープンスカイ   → RTK が cm 級精度を維持（最重要）
  ✓ 静止構造物あり   → ICP の対応点と Canny エッジ最適化に有効な特徴量
  ✓ 動体が少ない     → 投票フィルタで動体を除去できる
  ✓ 右左折が可能     → 累計 yaw 360° 以上の 6DOF 励起を確保
```

#### 不適な環境・走行条件

| 条件 | 理由 |
|---|---|
| 高層ビルに囲まれた路地 | GNSS マルチパスで RTK 精度が 10〜50 cm に劣化 → 時間を増やしても改善しない |
| 動体（走行車・歩行者）が多い公道 | 信号待ち停車車両が投票フィルタを通過し LiDAR マップを汚染 |
| 30 km/h 以上の高速走行 | カメラ動きブレ増大 → Canny エッジ検出精度が低下 |
| 直線のみ（右左折なし） | yaw 励起不足 → LiDAR-to-LiDAR TF の回転成分が不正確になる |

### 全体データフロー

```
rosbag（単一 .mcap または分割 rosbag2 ディレクトリ）
  │
  ├─ Step 0: データ検証 / 品質ゲート  ★以降の全 Step の前提
  │     トピック存在・tf_static 完全性・RTK 物理妥当性・LiDAR 時刻フィールドを検査
  │     致命的問題があれば非ゼロ終了して**以降の Step を実行しない**（ハード停止）
  │     covariance 等は警告のみ（物理妥当性を主ゲートとする）
  │     → validation_report.json + 標準出力サマリ（PASS / FAIL）
  │
  ├─ Step 1: RTK 軌道抽出
  │     /sensing/ins/oxts/odometry → 姿勢列 (map → oxts_link, 50 Hz)
  │     → 各センサーフレームへの補間 (平行移動: 線形、回転: slerp)
  │     → rtk_poses.npy
  │
  ├─ Step 2: LiDAR スキャン歪み補正
  │     t_us フィールドを使い、スキャン内各点の位置を RTK 補間で補正
  │     → undistorted 点群ファイル群
  │
  ├─ Step 3: Front LiDAR マップ構築 (map 座標系)
  │     map_T_front(t) = oxts_pose(t) ⊕ T_oxts_drs ⊕ T_drs_frontLiDAR
  │     map += map_T_front(t) ⊕ scan_undistorted(t)
  │     → ボクセル投票フィルタ: 各ボクセルが FOV/レンジ内に入っていたスキャン数を分母とし
  │       その N% 以上で占有されたボクセルのみ残す (動体除去、visibility 正規化)
  │     → ボクセルダウンサンプリング (0.1 m)
  │     → front_lidar_map.pcd
  │
  ├─ Step 4: LiDAR-to-LiDAR TF 推定 (3 本) — 結合最適化（front 固定・E_left/right/rear 同時）
  │     ※ 素朴な「other_map 蓄積 → 単一 ICP」は 8の字で並進 δt を復元不可
  │        （R_i·δt が full-yaw で平均≈0）。また front マップ照合単独では
  │        地面優勢で X/Y が不可観測（実測: Z のみ拘束）。→ 2 項を結合:
  │     項B: 各 LiDAR vs front マップ point-to-plane（Z・大域を拘束）
  │     項A: 隣接 LiDAR の同時刻重畳 point-to-plane（X/Y を拘束、
  │          front-right/front-left/rear-right/rear-left, rear はチェーン）
  │     Huber ロバスト + LM（ステップ棄却）+ 軸別観測性（不可観測軸は tf_static 据え置き）
  │     出力: drs_base_link → lidar_{rear, left, right}（＋軸別観測性レポート）
  │
  ├─ Step 5a: Camera-to-LiDAR TF 精密化 — 通常カメラ (8 本)
  │     各フレーム: CompressedImage decode → undistort → Canny → Distance Transform
  │     対応 LiDAR マップを TF で投影 → 距離場コストを最小化 (L-BFGS-B, 6DOF)
  │     出力: lidar_* → camera{0-7}/camera_link
  │
  ├─ Step 5b: Camera-to-LiDAR TF 推定 — 魚眼カメラ (4 本)
  │     Step 5a と同じ枠組みだが equidistant 投影モデルを使用
  │     初期値誤差は 10cm 未満を前提とし、tf_static 初期値をそのまま最適化開始点に使用
  │     出力: lidar_* → camera{8-11}/camera_link
  │
  └─ Step 6: 全 TF 同時精密化 (任意)
        全 15 本の TF を変数として全センサー残差を同時最小化
        → 個別ステップで残った相互依存誤差を除去
```

---

## 時刻同期と誤差吸収のメカニズム

カメラ（50ms周期）と LiDAR（100ms周期）は PTP で時刻同期されているが、
トリガタイミングは非同期であり重畳領域に誤差が生じる。
本パイプラインがこの誤差をどのように扱うかを整理する。

### LiDAR：点単位の相対時刻（`time_stamp`）による時刻補正

`seyond_points` には各点に **`time_stamp`（スキャン先頭からの相対経過時間、ns単位、実測 0〜約92.4ms）** が付与されている。
点の絶対時刻は `header.stamp + time_stamp[ns]`。1スキャン（≈92.4ms）の間も車両は移動しているため、スキャン内の先頭点と末尾点では
取得位置が異なる（10km/h で最大 ≈ 0.26m のずれ）。

Step 2 では `header.stamp + time_stamp` で各点の取得時刻に対応する RTK 姿勢を補間し、
スキャン先頭時刻（`t0 = header.stamp`）基準の座標系に変換する（スキャン内歪み補正）。

```
スキャンヘッダ header.stamp (t0) ← スキャン先頭の絶対時刻・補正の基準
各点の time_stamp                ← 先頭からの相対経過時間 [ns]（絶対時刻 = header.stamp + time_stamp）

補正式: p_corrected = T_body(t0)^{-1} @ T_body(t0 + time_stamp) @ p_raw
```

これにより、**LiDAR のスキャン内歪みは点単位で吸収される。**
（注: `time_stamp` は絶対エポックではなくスキャン先頭からの相対時間。したがってヘッダ自体にドライバ遅延がある場合、
その遅延は `header.stamp` にのみ乗るため RTK 補間位置に反映される。カメラほど深刻ではないが、
LiDAR ヘッダの遅延が既知なら Step 2 で `t0` に補正を加える余地がある。）

### カメラ：蓄積マップ + RTK 補間による吸収

カメラはフレーム全体が単一タイムスタンプであり、per-pixel の時刻情報はない。
しかし本パイプラインは**単一スキャンとカメラフレームを直接マッチングしない**。

```
Step 3/4: LiDAR 全スキャンを RTK 姿勢で map 座標系に蓄積
            → 静止環境の「世界地図」（時刻に依存しない）を生成

Step 5:   カメラフレームのタイムスタンプ t_cam で RTK 姿勢を補間
            → map 座標系の点群を t_cam 時点のカメラ視点に投影
```

カメラと LiDAR のトリガが非同期であっても、
**共通の map 座標系と RTK の精密な時刻補間**が橋渡しをするため誤差は吸収される。

| 誤差源 | 規模 | 対処 |
|---|---|---|
| PTP クロック同期誤差 | < 1 μs → 位置誤差 < 3 μm | 実用上ゼロ |
| カメラ・LiDAR 非同期トリガ（最大 100 ms） | 10km/h → 最大 278 mm のずれ | 蓄積マップ + RTK 補間で吸収 |
| RTK 50 Hz 補間誤差 | サンプル間隔 20ms の最大移動量は 10km/h で ≈ 56 mm だが、実際の線形補間残差は軌道の曲率に依存する 2 次量で 8の字（旋回半径 ≈7m・2.78m/s）でも ≪ 1 mm | キャリブ精度目標内 |
| LiDAR スキャン内歪み | 最大 0.28 m / scan | Step 2 で `t_us` 補正済み |

### 残留する系統誤差：カメラドライバ遅延

カメラドライバが露光完了後にタイムスタンプを付与する場合、
実際の露光時刻より数 ms〜数十 ms 遅れた値が記録される。
LiDAR の `t_us` とは異なり、カメラには per-frame の実露光時刻情報がないため、
この遅延はアーキテクチャで自動吸収できない系統誤差として残る。

```
10km/h = 2.78 m/s で Δt = 10 ms の遅延 → 位置誤差 ≈ 28 mm
```

対処として `camera_timestamp_offset_sec` パラメータ（後述）を用意する。
値が不明な場合はデフォルト 0.0 のまま使用する（多くのケースで影響は軽微）。

---

## 各ステップの詳細

### Step 0: データ検証 / 品質ゲート

パイプラインの最初に実行し、**入力 rosbag がキャリブに使える品質かを判定する**。
致命的問題が 1 つでもあれば**非ゼロ終了して以降の Step を実行しない**（ハード停止）。
covariance など精度に関わる指標は**警告のみ**とし、処理は止めない（物理妥当性を主ゲートとする方針）。

- **入力**: 単一 `.mcap` または分割 rosbag2 ディレクトリ（`metadata.yaml` + `*_N.mcap`）。
  ディレクトリ指定時はファイル名順（= 記録順）に連結して読む。
- **致命的チェック（FAIL → ハード停止）**:
  1. **必須トピック欠落**: `/sensing/lidar/{front,left,rear,right}/seyond_points`・`/sensing/ins/oxts/odometry`・`/tf_static` のいずれか欠落。
     （カメラトピックはカメラ Step でのみ必須。`--require-cameras` 指定時のみ致命に格上げ）
  2. **tf_static 不完全**: 初期外部パラメータが揃っていない。`drs_base_link→{lidar_front,left,rear,right}`・`drs_base_link→oxts_link`・`base_link→drs_base_link` のいずれか欠落。
  3. **RTK 物理破綻**: odometry フレーム未検出、位置に NaN/Inf、または速度が `rtk_max_speed_mps` 超の区間比率が `rtk_max_reject_ratio` 超。
     （無効バッグ `..._16-34-04_pc` の 1200m/s 級はここで確実に落ちる）
  4. **LiDAR 時刻フィールド無し**: `seyond_points` に `time_stamp`／`t_us`／`timestamp` のいずれも無く、undistortion 不能。
  5. **odometry 時間被覆不足**: odometry の時間範囲が LiDAR スキャン時刻範囲を覆っていない、または補間不能な大ギャップ（`rtk_max_gap_sec` 超）が存在。
- **警告チェック（WARN → 継続、レポートに記載）**:
  - **RTK covariance が粗い**: `pose.covariance` の位置分散中央値が `rtk_cov_warn_m2` 超（実測: 図8データは ~0.2 m²/std≈0.46m で、基準バッグの cm 級 1e-4〜0.097 m² より粗いのが常態。到達精度の上限になり得るため警告に留める）。
  - **NavSatFix.status が常に 0**: status はバッグ依存で当てにならない（無効バッグでも 0=FIX を返す）ため情報表示のみ。
  - **カメラ本数 < 12** / camera_info 欠落: 該当カメラの Step 5 をスキップする旨を警告。
  - **旋回励起不足**: 累計 \|yaw 変化\| が `min_yaw_coverage_deg`（デフォルト 360）未満、または軌道フットプリントが極端に小さい → LiDAR-to-LiDAR の回転成分観測性が低い旨を警告。
  - **map 座標系種別**（ECEF/局所）と検出した LiDAR 時刻フィールド名・単位（μs/ns）を情報表示。
- **出力**:
  - `validation_report.json` — 各チェックの `pass/warn/fail` と実測値（トピック件数・Hz、フットプリント、速度統計、covariance 統計、tf_static 変換一覧、LiDAR フィールド）。
  - 標準出力に PASS/FAIL サマリ。FAIL 時は exit code 1（`run_pipeline.py` はここで停止）。
- **チューニング**:
  - `rtk_cov_warn_m2`: covariance 位置分散の警告閾値 [m²]（デフォルト: `0.1`。超過は警告のみ・停止しない）
  - `rtk_max_gap_sec`: odometry の許容最大ギャップ [s]（デフォルト: `0.5`。超過は致命）
  - `min_yaw_coverage_deg`: 旋回励起の警告閾値 [deg]（デフォルト: `360`）
  - `require_cameras`: カメラトピック欠落を致命に格上げ（デフォルト: `false`）
  - RTK 物理ゲートは Step 1 と共通の `rtk_max_speed_mps` / `rtk_max_reject_ratio` を使用。

### Step 1: RTK 軌道抽出

- **入力**: `/sensing/ins/oxts/odometry` (nav_msgs/Odometry, 50 Hz)
  - frame_id: `map`, child_frame_id: `oxts_link`
- **座標系の注意（実バッグで確認）**:
  - `map` 座標系は**バッグにより異なる**（実測）: 生バッグは **ECEF（地心座標、位置 1e6 m オーダー）**、
    処理済み `_replaced` バッグは **局所メートル座標（原点付近 ~数百 m）** だった。
    どちらでも安全になるよう、**位置の大きさを検出して最初の有効姿勢を局所原点に平行移動し、内部計算は float64** で行う
    （ECEF を float32 で蓄積すると分解能が 0.25m 級に劣化する。Step 2/3 も同様）。
  - odometry は `map → oxts_link`。車体 `drs_base_link` 姿勢は `map_T_drs = map_T_oxts @ (drs_T_oxts)^{-1}` で求める。
    `drs_T_oxts` は **X軸まわり180°回転**（`t=0, q_xyzw=(1,0,0,0)`）であり identity ではない。この回転の適用漏れは全体を上下反転させる。
- **RTK 品質ゲート（重要な訂正）**: `pose.covariance` と `NavSatFix.status` の信頼性はバッグ依存で不安定（実測）:
  - 無効バッグ（`..._16-34-04_pc`）では `pose.covariance` が全フレーム一定のプレースホルダ（4294.7 m²、姿勢分散0）、
    `NavSatFix.status` は常に `0=FIX` を返し、**軌道が完全に無効（位置 482m/20ms・速度 1200m/s）でも異常を示さなかった**。
  - 有効バッグ（`..._17-37-26_pc_replaced`）では covariance は実値（位置分散 1e-4〜0.097 m²）で機能していた。
  - → covariance/status は**参考程度**にとどめ、**物理的妥当性を主ゲート**とする:
    - 連続姿勢のステップ距離・速度が上限（`rtk_max_speed_mps`）を超える区間・位置ジャンプ・姿勢不連続を外れ値除外
    - `/sensing/ins/oxts/velocity` と位置差分の整合性チェック
    - 除外区間が `rtk_max_reject_ratio` を超えたら**「この rosbag の INS 解は無効」として中断・警告**
- **補間**: 任意タイムスタンプ t に対し前後フレームを線形補間 (位置) + slerp (姿勢)
- **出力**: `rtk_poses.npy` — shape `(N, 8)` = `[timestamp_ns, x, y, z, qx, qy, qz, qw]`（局所原点基準・drs_base_link 姿勢）
- **チューニング**:
  - `rtk_smoothing_window`: RTK 軌道の平滑化ウィンドウ幅。ノイズが多い場合に増やす (デフォルト: 0 = 無効)
  - `rtk_max_speed_mps`: 物理妥当性ゲートの上限速度 [m/s]（超過区間を外れ値除外、デフォルト: 30 ≈ 108km/h）
  - `rtk_max_reject_ratio`: 除外区間がこの割合を超えたら INS 無効として中断 (デフォルト: 0.1)

### Step 2: LiDAR スキャン歪み補正

- **入力**: 各 LiDAR の PointCloud2、`rtk_poses.npy`
- **処理**:
  1. スキャン先頭時刻 `t0 = header.stamp` の車体姿勢 `T_world_body(t0)` を基準に設定
  2. 各点の絶対時刻 `t_point = header.stamp + time_stamp[ns]`（`time_stamp` はスキャン先頭からの相対時間[ns]）から `T_world_body(t_point)` を補間
  3. 各点を `T_world_body(t0)^{-1} @ T_world_body(t_point) @ p` で変換（1スキャン ≈ 92.4ms に渡るため効果は大きい）
- **出力**: 補正済み点群ファイル群
- **チューニング**:
  - `undistort_enabled`: 歪み補正の有効/無効。速度が十分に低い場合はスキップ可 (デフォルト: true)
  - `lidar_scan_stride`: 処理するスキャンを N 枚に 1 枚に間引く (デフォルト: 1 = 全スキャン)

### Step 3: Front LiDAR マップ構築

- **入力**: 補正済み front_lidar 点群、`rtk_poses.npy`、`T_drs_to_frontLiDAR` (`/tf_static` 初期値)
- **処理**: 各スキャンを **局所原点基準の map 座標系**（ECEF から局所原点を差し引いた float64）に変換して蓄積し VoxelGrid でダウンサンプリング
- **出力**: `front_lidar_map.pcd`（局所原点・原点値をメタとして保存）
- **チューニング**:
  - `map_voxel_size`: ボクセルサイズ (デフォルト: 0.1 m)。大きくするとマップ構築・ICP が速くなるが精度低下
  - `vote_threshold_ratio`: 投票フィルタの占有率閾値。**各ボクセルが FOV/レンジ内に入っていたスキャン数**を分母とし、その割合以上で占有されたボクセルのみ残す (デフォルト: 0.3)。
    分母を「全スキャン数」にすると、長距離走行（第2/第3環境）で車両近傍にある時間が短い静止構造物まで除去されてしまうため、必ず visibility 正規化を行う
  - `map_scan_stride`: マップ蓄積に使うスキャンを N 枚に 1 枚に間引く (デフォルト: 3)
  - **visibility モデル**（投票の分母 = 各スキャンの range+FOV コーン内にボクセル中心が入るか、かつ遮蔽されていないか）。Seyond の実 FOV は近似値のため、過剰除去を避けるよう既定は広めに設定:
    - `vote_max_range` / `vote_min_range`: 可視判定のレンジ [m] (デフォルト: 60 / 1.5)
    - `vote_hfov_deg` / `vote_vfov_deg`: 可視判定の水平/垂直 FOV [deg] (デフォルト: 120 / 40)
    - `vote_occlusion_enabled` / `vote_occlusion_footprint_m` / `vote_occlusion_tol`: 遮蔽考慮（2026-07-04 追加、詳細は後述の「Step 3: 近傍ボクセル欠落問題の調査と修正」節）。各スキャン自身の実点群からアーク長（角度×その点自身のレンジ）ビンで最小レンジの「レンジイメージ」を作り、候補ボクセルより手前に実際の返り値がある場合のみ「遮蔽」として visible から除外する。単純な range+FOV コーンだけだと、8の字などタイトなループでは近傍ボクセルが実際には遮蔽されている多数の他ループ地点から「可視」と誤判定され、投票分母が実際の occ 数に対して過大評価されて近傍構造がほぼ全滅する不具合があったための対応
    - 実装は各ボクセルが「それを占有したスキャン」には必ず可視となるよう分母を `max(visible, occupied)` でクランプする（分母過小で ratio>1 になるのを防止）

### Step 4: LiDAR-to-LiDAR TF 推定（結合最適化：項A LiDAR間重畳 ＋ 項B front マップ）

- **入力**: `front_lidar_map.npy`（無ければ `.pcd`）、全 LiDAR の補正済み点群（Step 2）、`rtk_poses.npy`、tf_static 初期外部パラ
- **出力**: `lidar_to_lidar_tf.yaml`（`drs_base_link → lidar_{rear,left,right}` を x,y,z,roll,pitch,yaw で）
  ＋ `lidar_to_lidar_result.json`（4x4・RMSE・inliers・**軸別観測性**・初期比 補正量）
- **基本方針**: `drs_T_front` を tf_static で固定した基準とし、`E_left, E_right, E_rear` を **同時最適化**する。
  2 種類の残差を重み付き結合する:
  - **項B（各 LiDAR vs front マップ, point-to-plane）**: `r = n·(M_i E p − y)`（`M_i=map_T_drs(t_i)`）。
    大域自由度と **Z を拘束**。RTK 姿勢で全区間を蓄積した front マップが基準。
  - **項A（LiDAR間 同時重畳, point-to-plane）**: 近同時刻（Δt≤~60ms）のスキャン対で、隣接 LiDAR A の局所点群と
    LiDAR B の点群を共通 `drs@t_B` 系で照合。**X/Y を拘束**（同時共観測の縦構造＝静止車両・建物が法線多様性を与える）。
    隣接ペア = **front-right / front-left / rear-right / rear-left**。**rear は front と重畳しない**（対向）ため、
    rear-right・rear-left 経由で **チェーン**して拘束する。
- **手法（設計変更・実装で確定。理由は下の「検証」参照）**:
  素朴な「other_map を蓄積 → 単一 Open3D ICP で `ΔT_world`」は **8の字では並進 `δt` を復元できない**
  （lidar 系並進誤差 `δt` の map 系変位 `R_i·δt` が full-yaw で平均≈0 になり単一剛体変換に見えない）。
  そこで **各スキャンを自姿勢 `M_i` に固定したまま共有外部パラ `E` を最適化する multi-scan point-to-plane** とし、
  さらに front マップ照合だけでは **X/Y が地面優勢で不可観測**になるため、項A（LiDAR間同時重畳）を加えて X/Y を拘束する。
- **処理**:
  1. 全 LiDAR の補正済みスキャン（lidar 系）と姿勢 `M_i` を用意。front マップは Open3D で voxel ダウンサンプル＋法線推定（粗→精）、scipy cKDTree で最近傍対応。
  2. 項A: 各隣接ペアで近同時刻スキャン対を作り、A の局所参照（近傍数スキャンを `drs@t_B` に集約＋法線）に対する B の点面残差を構築。
  3. 項B: 各 other LiDAR の全スキャンを front マップに対して点面残差を構築。
  4. `E_left,E_right,E_rear` を変数に、項A＋項B の **Huber ロバスト重み付き Gauss-Newton / Levenberg-Marquardt**（ステップ棄却付き）で同時最小化。粗→精スケール、ステップ収束（`icp_step_tol`）で終了。
  5. **軸別観測性**: Hessian の各自由度方向の曲率を評価し、`eig_max·icp_degeneracy_rel` 未満の不可観測軸は更新せず tf_static 据え置き（スカラの最小固有値では地面優勢の X/Y 平坦を見抜けないため軸別で判定・レポート）。

- **⚠ データ要件（実測で判明・重要）**:
  - **front マップは full-yaw 全区間で構築必須**（項B）。短窓（12〜15s・部分8の字）だと角度被覆不足で rear/side の初期対応率が 13〜30% しか無く、誤構造にエイリアスして数 m の偽補正になる。全区間なら車体が 360°+ 回り front LiDAR が周囲全体を掃引するため被覆確保。
  - **投票フィルタ（Step 3）は疎サンプリングで過剰除去に注意**（下記検証参照）。項B の対応 target には十分密なマップを使うこと。
  - **隣接 LiDAR に重畳が必要**（項A）。本データでは front-right ≈339 点/scan、front-left ≈206 点/scan の重畳を確認済み。
  - **⚠ 固定基準 front は正しい値を使うこと**。実データでバッグの `/tf_static` front（y=0.132）が受け入れ済み校正値（y=0.000）と **13cm 相違**していた。front を誤った値で固定するとマップが偏り、他 LiDAR の絶対値がその分バイアスする。→ **Step 3・Step 4 の両方に同じ `--tf-override <multi_tf_static.yaml>` を渡して正しい front（および必要なら各 LiDAR 初期値）を指定する**。実装済み・GT で検証済み（下記）。

- **検証（2026-07-03 実施・本設計の根拠）**（bag: `HRdqo3pf_2026-06-10T15-22-51.mcap`, 全区間 stride10 → front マップ）:
  1. **最適化器の正しさ**: front を自マップに最適化 → 補正 ≈0（0.7cm）。既知 **12cm/4.6° 摂動 → 0.7cm/0.15°** に復元（perturbation-recovery）。RMSE 下限はマップ固有スミア（0.1m voxel＋RTK ジッタ）で ~8–9.5cm のため収束はステップ幅で判定。
  2. **投票フィルタの過剰除去**: 疎な全区間マップ（151 scan）で vote_threshold_ratio=0.3 は voxel の 83% を削除 → rear/side の対応率が 1–5%@10cm に低下し数 m の偽補正。未フィルタ密マップにすると **50–59%@10cm**（中央値残差 ~5.5cm）に回復。
  3. **X/Y 不可観測（マップ照合単独の限界）**: 密マップでも rear/left/right は tf_static から 20–64cm ドリフト。
     - Test A（偶奇独立マップの再現性）: rear は even 63.6cm / odd 64.3cm と **0.9cm 差で再現**（ノイズではなく系統的）。
     - Test B（tf_static 周りのコスト断面）: **X/Y はほぼ平坦（±0.6m で ~1cm しか変化せず不可観測）、Z のみ鋭い最小**（rear/left の Z 最小は tf_static 位置＝Z は校正済みで正しい。right は Z が −20cm ずれ＝要確認）。
     - 原因: rear/side と front マップの重畳が**地面優勢**で、点面拘束は Z のみ。X/Y を拘束する縦構造が単独 LiDAR 視野では法線多様性不足。
  4. **LiDAR間同時重畳の有効性（項A の根拠）**: front-right/front-left の直接重畳でコスト断面を測定 →
     **X/Y の曲率がマップ方式の 3〜5倍**（front-right: X Δ3.2cm・Y Δ5.4cm、front-left: X Δ1.9cm・Y Δ2.4cm。対してマップ方式は ~0.5–1cm で平坦）。
     同時共観測は (a) 構造の確実な共観測、(b) RTK 累積誤差ゼロ（50ms 差のみ）、(c) 異視点で法線多様性増、により **X/Y を拘束できる**ことを実証。
     現状は疎（16 ペア・5000 点間引き・単一スキャン参照）で床が高い（14–18cm）が、密度を上げれば鋭くなる → 本実装で全ペア・局所蓄積参照を用いる。
  5. **結合最適化（項A＋項B・項別正規化）の GT 照合**: 受け入れ済み校正値と照合。**right が直接比較で 2.7cm / 0.37° で GT 再現**（rear 9.1cm/0.42°）。X/Y 観測性獲得（±15cm 摂動→0.0cm 収束）。バッグ front の 13cm 誤りは `--tf-override` で補正（Step 3/4 両方に渡す）。
  6. **left の未達（20.6cm・z+15.7・roll 1.26°）は項B改良では直らない**（地面除外→right も悪化、近距離マップ→不変、マップ重み 0〜1 スイープ→**項A のみ(RTK 免疫)でも z+12.8cm**）。真因は **left の真 roll=−1.45°（右 −0.19°・後 +0.24° より突出）が本8の字で不可観測 → roll 誤差が z に結合**。側方 LiDAR の z は地面でのみ可観測（縦構造は z 平坦）。→ **left はデータ/幾何の制約**（改善: left 視野の構造が豊富な走行 / left-rear を加えた結合18変数同時最適化 / Step 6 カメラ拘束）。
  7. **RTK 〜0.5m の影響**: LiDAR-LiDAR は剛体相対校正なので絶対 RTK には非依存。項A（X/Y）は 50ms 相対のみ使用で **RTK ほぼ免疫**。項B（Z/大域）は相対平滑性（実測 ~8cm）に依存し 0.5m 絶対値には支配されない（right 2.7cm が証拠）。RTK/姿勢の軌道規模不整合は項B の Z に二次的に効くが left の主因ではない。

- **チューニング**:
  - `icp_multiscale_voxels`: マルチスケールの voxel サイズ列 (デフォルト: `[0.5, 0.2, 0.1]`)
  - `icp_corr_dist_scale` / `icp_max_correspondence_dist`: 対応ゲート = `voxel×scale`（上限 cap） (デフォルト: 3.0 / 2.0 m)
  - `icp_huber_scale`: Huber デルタ = `voxel×this` (デフォルト: 1.0)
  - `icp_degeneracy_rel`: 不可観測軸の除去閾値（`eig_max×this`。**軸別**に適用） (デフォルト: 1e-4)
  - `icp_max_iterations` / `icp_step_tol`: 最大反復 / ステップ収束閾値 (デフォルト: 30 / 1e-4)
  - `icp_rmse_threshold`: 早期終了 RMSE（多くの場合マップ下限未満で到達しない） (デフォルト: 0.02 m)
  - `l2l_scan_stride` / `l2l_points_per_scan` / `l2l_max_range`: スキャン/点の間引きとクロップ (デフォルト: 12 / 1000 / 60m。
    2026-07-04: 高速化検証の結果、旧デフォルト 3 / 3000 から変更。詳細は「処理高速化」節参照)
  - `l2l_pair_dt_max`: 項A の同時刻ペア許容時間差 [s] (デフォルト: 0.06)
  - `l2l_pair_ref_scans`: 項A で局所参照に集約する隣接スキャン数 (デフォルト: 3)
  - `l2l_interlidar_weight` / `l2l_map_weight`: 項A / 項B の相対重み（無次元化後） (デフォルト: 1.0 / 1.0)
  - `icp_method`: `point_to_plane` のみ実装（`point_to_point` 指定時は警告して point_to_plane 使用）

### Step 5a: Camera-to-LiDAR TF 精密化 (通常カメラ)

> **実装メモ（2026-07-03, `step5_cam_to_lidar.py`, 通常・魚眼 共通実装）**:
> - 投影は **生（歪み）画像**上で行う（`cv2.projectPoints`／魚眼は `cv2.fisheye.projectPoints`）。Canny/DT も生画像で計算＝一貫。魚眼を pinhole に undistort する歪みを回避。
> - 連鎖: `map → RTK(map_T_drs(t_cam)) → drs_T_lidar(Step4) → lidar→camera_link(最適化) → camera_link→optical(固定 q=0.5,-0.5,0.5,-0.5) → 投影`。map 点の lidar フレーム化はカメラ外部パラに依存しないので**フレーム毎に前計算**（frustum crop＋サブサンプル `cam_proj_max_points`）。
> - **コストは固定分母**：フレーム毎の点集合を固定し、枠外/背面点は `dist_transform_max` でペナルティ。可変分母（枠外に押し出して点数を減らす）で L-BFGS-B が発散するのを防止。探索は init 近傍に bound（`cam_trans_bound_m`/`cam_rot_bound_rad`）。
> - `--tf-override` 対応（Step 3/4 と同じ YAML を渡し drs→front を正しく）。
> - 並列化: `multiprocessing.Pool`＋ワーカー初期化で traj/p_map を各プロセスに一度ロード（`cam_parallel_workers`／`--workers`）。12カメラ/6ワーカー ~50秒。
> - 検証①（投影連鎖）: cam0(plumb_bob) 残差 1.34→1.30px・cam8(equidistant) 0.98→0.95px。バッグの tf_static カメラ外部パラは元々良好（~1px）で最適化が微修正＋確認。**投影連鎖の正しさのみを確認**。
>
> **⚠ 検証②（摂動回復テスト）で判明した本質的限界（2026-07-03・v1 方式は棄却）**:
> 全12カメラの外部パラを既知の 11.2cm/3.54° だけ摂動させて回復を検証したところ、**回復に失敗**（中央値回復残差 21cm/5.08°、摂動より悪化）。cam0/cam9 は勾配ゼロで動かず、cam2/3/5/11 は残差を下げつつ真値から 50〜74cm 離れた偽極小へドリフト。
> 決定的証拠: **cam0 は 11cm/3.5° ずらしても残差 1.58px**（真値 ~1.3px とほぼ同じ）＝**DT コストが外部パラに平坦**（勾配ゼロ）。
>
> 対策として以下を実装・検証したが**いずれも失敗**（同日）:
> 1. **エッジ点化**（固有値ベース面変化 + 強度エッジ、`cam_use_edges` / `cam_edge_ratio` / `cam_edge_knn`）→ 改善せず。
> 2. **強 Canny (150/300) + dt_max 拡大 (60px)** → コストは鋭くなるが真値から 44〜56cm 離れた**偽の大域最小**へ移動（エイリアシング）。
> 3. **coarse-to-fine 多段 DT ぼかし**（`cam_dt_blur_sigmas=[16,8,4,0]`）→ 偽の大域最小へ**滑らかに**収束（cam4: 摂動 11.2cm → 回復残差 44.4cm と悪化）。
>    局所極小の粗さの問題ではなく**大域最小自体が真値から変位**していることを実証（coarse-to-fine では原理的に直らない）。
>
> **根本原因は 2 つ**:
> - **(a) 点集合**: 固有値ベースの面変化エッジ（面の交線・植生等）は画像の**輝度エッジと物理的に一致しない**。一致するのは**遮蔽境界（奥行き不連続＝シルエット）**のみ。
> - **(b) 連鎖**: `map→RTK→drs_T_lidar` 経由のマップ投影は **Step 4 誤差（rear ~24cm）と RTK 誤差（cov ~0.2m²）を継承**する。カメラをいくら動かしても消せないバイアスが入る。
>
> → **下記「Step 5 v2 設計」で全面再設計**（v1 の DT 最小化方式は棄却）。

#### Step 5 v2 設計: scan-to-image 遮蔽エッジ整合（採用方針・実装予定）

v1 の失敗分析から、変更は「**連鎖・点集合・コスト・探索/合否判定**」の 4 点セット。1 つだけ直しても機能しない（エッジ点化のみ・coarse-to-fine のみは検証済みで失敗）。

**1. 連鎖の変更: 蓄積マップ投影 → 単一スキャン直接投影（scan-to-image）**

- 画像タイムスタンプに最も近い**対応 LiDAR の 1 スキャン**を、その LiDAR フレームのまま投影する:
  `lidar点(t_point) → [RTK 相対運動で t_img へ ≤50ms 補正] → lidar_T_camera_link（最適化変数）→ camera_link→optical（固定）→ 生画像へ投影`
- **map / RTK 絶対精度 / Step 4 の drs_T_lidar に一切依存しない**。未知数は求める lidar_T_camera のみ。
  （運動補正は短時間の相対運動のみで、Step 4 項A と同じ理由で RTK 免疫）
- 単一スキャンは疎（frustum 内 数千点）なので、**多数の scan-image ペア（30〜50、視点多様）を合算**して拘束。
- ペア選抜: 角速度の小さいフレーム優先（同期誤差最小化）、Laplacian 分散でブレ画像を除外、ヨー方向に多様な位置から選ぶ。

**2. 点集合の変更: 遮蔽境界（奥行き不連続）エッジ + 強度エッジ**

- スキャンを球面投影（方位角×仰角ビン）して**レンジ画像**を構築（Seyond はリング構造が保証されないため構造非依存の方法を採る）。
- レンジ画像上で走査方向の奥行き不連続 `Δr > 閾値` を検出し、**手前側（foreground）の点のみ**採用。
  これがカメラ画像のシルエット＝輝度エッジと物理的に一致する唯一の点クラス（Levinson & Thrun, 2013）。
- 補助として LiDAR 反射強度の急変点（路面標示・標識）を第 2 クラスで追加。
- 各点に重み `w_i = 奥行き不連続量`（深度差が大きいほど画像に強いエッジが立つ）。
- 遮蔽エッジ点は同一スキャン由来なので LiDAR からは自明に可視。カメラ視点との視差分は近傍ピクセルビンの最近傍デプス比較（簡易 z-buffer）で遮蔽チェック。
- v1 の固有値ベース面変化エッジ（`extract_edge_points`）は**棄却**。

**3. コストの変更: DT 最小化 → エッジ強度相関の最大化（Levinson 型）**

- 画像側: 2値 Canny + DT ではなく**勾配強度マップ `E = blur(|∇I|)`**（エッジらしさの連続値、正規化）。
- 目的関数: `J(T) = Σ_frames Σ_i w_i · E(π(p_i; T))` を**最大化**（枠外/背面点は寄与 0、分母は固定点数）。
- 棄却理由の整理: DT 最小化は「どのエッジでもいいから近くにいる」姿勢を**すべて**低コストにする（雑然画像で平坦化 or エイリアス）。
  相関最大化は「不連続量の大きいシルエット点が強い輝度エッジに乗る」姿勢にのみ鋭いピークを作る。
- **補完コスト（クロスチェック用）: MI（相互情報量）** — frustum 内全点の LiDAR 反射強度と画像輝度の MI を最大化（Pandey et al., 2012）。
  エッジ相関と故障モードが独立なので、**両コストの最適解が一致＝高信頼**の判定材料になる。

**4. 探索と合否判定: 局所降下 1 発 → グリッド探索 + 識別可能性ゲート**

- L-BFGS-B 単発をやめ、init（tf_static）近傍 ±10cm/±2° の粗グリッド → ステップ縮小の座標降下（Levinson 方式）で精密化。
- **識別可能性ゲート（v1 失敗の再発防止・必須仕様）**:
  - **M0: 1D スイープ事前検査（go/no-go）**: 最適化の前に tf_static 姿勢周りで各軸 ±20cm / ±3° をスイープし、
    目的関数が真値位置に**単峰ピーク**を持つかを可視化・定量化。平坦またはピーク変位ならそのカメラは**最適化せず tf_static を採用**して警告。
  - **split-half 一致検査**: フレーム集合を 2 分して独立に最適化し、結果乖離が閾値（3cm/0.3°）超なら不合格 → tf_static 採用 + 警告。
  - 「黙って発散した値を出力しない」ことを Step 5 の仕様とする（v1 は摂動回復失敗を残差だけでは検知できなかった）。

**マイルストーンと受入基準**

| M | 内容 | 受入基準 / go-no-go |
|---|---|---|
| M0 | 遮蔽エッジ抽出 + tf_static 姿勢での重畳可視化 + 目的関数 1D スイープ | 過半のカメラで単峰ピーク（変位 <3cm/0.3°）。**不成立なら以降の実装工数を使わず方針転換**（tf_static 採用＝検証のみ運用） |
| M1 | cam0（front/pinhole）・cam4（rear/pinhole）の摂動回復 ±11.2cm/±3.54° | 回復残差 <3cm / <0.5° |
| M2 | 全 12 カメラ + 識別可能性ゲート + 並列化 | カメラ毎に 合格 / 不合格（tf_static fallback）を明示レポート |
| M3 | `camera_timestamp_offset_sec` の同時推定（1D 探索） | オフセット推定が split-half で一致 |

**v1 から流用する資産**: 生画像投影（pinhole `cv2.projectPoints` / 魚眼 `cv2.fisheye.projectPoints`）、`lidar_to_optical` 連鎖、bounds、multiprocessing 並列、`--tf-override`、フレームデコード。
**新規実装**: レンジ画像遮蔽エッジ抽出、scan-image ペアリング（運動補正込み）、勾配強度マップ + 相関目的関数、MI コスト、グリッド探索器、識別可能性ゲート。

> **M0 実装・初回実行結果（2026-07-03, `step5_m0_check.py`）**: 8の字バッグ（15-22-51）で camera0/2/4/6（front/right/rear/left の代表）を実行。
> **camera0 (front) のみ 3/6 軸合格**（occlusion クラス: z, roll, pitch が tf_static 位置に鋭い単峰ピーク）。**camera2/4/6 (right/rear/left) は 0〜1/6 で NO-GO**。ペア数を 8→30 に増やしても傾向は不変（ノイズではなく構造的）。
>
> **原因を投影可視化で特定（実装の欠陥ではなくシーン内容の非対称性）**:
> - front: 立体駐車場が画面のほぼ全体を占め、梁・柱・フェンスの明瞭な直線構造が豊富 → 強く曖昧性のないエッジ対応。
> - right: 画面の大半が特徴のない舗装面。**平面エッジ（緑）が地面に大量に誤検出**＝Step 4 で既知の地面縮退問題と同型のノイズ源。
> - rear: 画面上半分が**樹木の葉**（LiDAR・画像双方で微細テクスチャは大量だが点対応が不安定＝古典的エイリアシング要因）＋車カバー（布・非剛体）。
> - left: **メッシュフェンスの周期構造**（フェンスの目の間隔でコストがほぼ同値になるアパーチャ問題＝偽の複数極小）＋葉の写り込み。
> - また TF 誤差伝播が主因ではないことも確認済み: Step4 で最良（GT 2.7cm）の right が rear（9.1cm）と同等以下の M0 結果。
>
> → v2 のコスト設計自体は front で機能実証できたが、**この経路でのカメラ視野内容に強く依存**する。次の一手候補: (1) 8の字の別ヘディングで各カメラが駐車場構造を捉える瞬間を狙ったペア選抜、(2) 地面/葉の自動除外（強度・リターン数・法線ヒューリスティック）、(3) 周期構造への対処（広域スイープ・事前分布）、(4) 上記が奏功しないカメラは tf_static 採用＋検証のみで運用（M0 設計時から想定済みのフォールバック）。

**リスク**:
- 単一スキャンの frustum 内遮蔽エッジ点が少ないカメラ（特に魚眼の車体近傍視野）→ ペア数増・レンジ画像分解能の調整で対応。
- カメラ露光タイミング vs スキャン時刻の同期誤差 → 低速フレーム選抜 + M3 のオフセット推定で吸収。
- M0 不成立の可能性は残る（データ起因）。その場合は定量的根拠を持って「カメラは tf_static 採用（Step 5 は検証のみ）」へ転換する。M0 を先頭に置くのはこのため。

**論文的根拠と適用上の注意**:

| v2 要素 | 出典 | 備考 |
|---|---|---|
| 遮蔽エッジ + エッジ強度相関最大化 + グリッド/座標降下 + 誤校正検知 | Levinson & Thrun, RSS 2013 "Automatic Online Calibration of Cameras and Lasers" | Velodyne HDL-64＋雑然市街地で検証。M0 ゲートは彼らの miscalibration detection（近傍姿勢のコスト悪化割合 F 指標）が元ネタ。**注意: オンライン微修正が主用途で、大摂動回復の保証はない**（→ M1 で自データ検証） |
| MI コスト（反射強度 vs 画像輝度） | Pandey et al., AAAI 2012 / JFR 2015 | Ford キャンパスデータで検証。多フレーム合算で成立 |
| scan-to-image（単一スキャン投影） | 上記 2 論文とも単一スキャン〜短時間蓄積で実施 | v1 の蓄積マップ投影連鎖の方が非標準だった |

- **⚠ 反例的知見（要 M0 での実測判定）**: Yuan et al., RA-L 2021（HKU-MARS `livox_camera_calib`）は、Livox 級の**高密度ソリッドステート LiDAR では奥行き不連続エッジは「にじみ点（foreground bleeding）」で精度を落とす**とし、**depth-continuous エッジ（平面交線）**を推奨。Seyond は同クラスのため、v2 では**両エッジクラスを抽出し、M0 の 1D スイープでどちらが単峰ピークを作るかを実測して選択**する（設計に組み込み）。
- **既存 OSS `direct_visual_lidar_calibration`（Koide et al., ICRA 2023）の先行評価 → 見送り（2026-07-03）**: Docker（ROS2 Humble イメージあり、本環境で動作確認済み）・NID（相互情報量）ベース直接整合・pinhole/fisheye 両対応と、条件は良好だった。
  しかし公式データ収集プロトコル（`docs/collection.md`）が**センサー静止状態での 10〜15 秒録画**を必須前提としており（非回転式 LiDAR＝Seyond と同クラスの場合）、「各バッグの最初の点群とフレームが同一姿勢であること」も明記された制約。
  **8の字走行データはこの前提と根本的に非適合**（連続移動中のため静止蓄積ができない）。既存バッグに静止区間もなし。専用の静止バッグを新規取得すれば正式評価は可能だが、データ取得（車両停止）が前提になるため今回は保留し、**自前 v2 設計（scan-to-image・走行データ前提）を優先実装**する方針とした。SuperGlue 自動初期化は商用利用不可ライセンスの点も要留意（手動初期化なら制限なし）。静止バッグを別途取得できる状況になれば再評価の価値あり。

---

### Step 5 v3 設計: 全軌道蓄積カメラ点群 → LiDAR マップ GICP（RTK fix 解前提・採用方針）

**前提（重要）**: このアーキテクチャは **RTK が integer（fix）解**であることを必須前提とする。校正走行の前に準備走行してアンビギュイティを解消すれば容易に達成でき（位置 1〜2cm、INS 姿勢 sub-0.1°）、これにより下記「なぜ v3 か」の①が成立する。既存バッグは float 解（cov ~0.2m²）なので v3 の実測検証には **fix 解での再取得が必要**。

**なぜ v3 か（v1/v2 の失敗と M0 検証を踏まえた転換）**:
- v1（LiDAR→画像 DT 最小化）・v2（scan-to-image エッジ相関）はいずれも **2D-3D 対応**。雑然/無地/周期/植生シーンで画像エッジと LiDAR 構造が一意対応せず、front 以外は識別不能（M0 実測で確定）。
- v3 は **3D-3D 位置合わせ**に変える: 各カメラの多視点画像から**メトリックなカメラ点群**を復元し、LiDAR マップに GICP 登録する。同じ幾何量同士の照合は 2D-3D より格段に条件が良く、Step 4 で実証済みの point-to-plane 最適化基盤をそのまま流用できる。
- ① RTK fix 解ならベースライン（数m 視差）付き三角測量が cm 級に成立 → v2 が回避していた「並進はベースライン必須」問題が**利点に反転**。② 全軌道で各カメラ点群を蓄積でき、疎な構造しか写らないカメラ（right 等）も多角度集約で密化できる。

**パイプライン（カメラ毎、tf_static を初期値に）**:

1. **M0 前段ゲート（`step5_m0_check.py` を流用・拡張）**: 本実装前に、各カメラが復元可能な剛体構造を十分持つか（三角測量カバレッジ・視差角分布・非縮退性）を安価に判定。NO-GO のカメラは即 tf_static 採用＋検証のみに落とし、SfM 実装工数を投じない。
2. **特徴抽出・追跡**: ORB/SIFT（cv2）または SuperPoint 等でフレーム間追跡。空・無地地面・植生・動体は簡易マスク（強度/セマンティック/再投影不整合）で除外。
3. **メトリック三角測量**: 既知カメラ姿勢 `W_T_cam(t) = W_T_drs(t)(fix RTK) ∘ drs_T_cam(現在値)` で多視点 DLT + 非線形精密化。**視差角・再投影誤差・cheirality でフィルタ**（低視差点は並進を拘束しないため除外）→ world 系の疎メトリック点群。
4. **全軌道蓄積**: 150s 全区間の三角測量点を集約しボクセル間引き → 各カメラの密な構造点群（v3 の情報量の要）。
5. **LiDAR マップへ登録＝外部パラ解**: Step 3 の LiDAR マップ（fix 解で先鋭）を対象に、**Step 4 の point-to-plane GN + Huber-LM + 縮退方向射影 + 軸別可観測性**をそのまま適用し、6-DOF `drs_T_cam` 補正を解く。
6. **再構成↔登録の反復（2〜3回）**: 三角測量は `drs_T_cam` に依存するが、tf_static 初期値が良好（~1px）なため誤差は微小で、1〜2 反復で収束（良初期値ゆえ循環依存は無害）。
7. **識別可能性の事後ゲート**: 軌道を2分割して独立に解き乖離 <3cm/0.3° を要求（split-half）。GN ヘッセ固有値から軸別可観測性を報告し、縮退軸は tf_static にアンカー。**黙って発散値を出力しない**（v1 の教訓）。
8. **タイムスタンプオフセット**: `camera_timestamp_offset_sec` を登録スコア最大化の 1D 探索で同時推定（v2 M3 と同思想）。

**再構成手段の選択（主・副）**:
- **主: 既知姿勢つき三角測量（SfM）** — fix 解でメトリックかつ高精度。GPU 不要。上記循環依存は良初期値で無害。
- **副: 単眼メトリック深度（Depth Anything V2 metric / Metric3D / UniDepth）** — フレーム毎に深度→点群化するので**ベースライン不要＝RTK 相対姿勢に非依存**（fix 解が得られない状況の保険）。GPU とモデル依存・絶対スケール偏差が難点。②のシーン限界は同様に残る。

**v3 が流用する資産（新規実装は最小）**:

| 流用元 | 使う部分 |
|---|---|
| Step 4 `step4_lidar_to_lidar.py` | point-to-plane GN + Huber-LM + 縮退射影 + 軸別可観測性（→ カメラ点群↔LiDAR マップ登録にそのまま） |
| Step 3 `step3_build_front_map.py` | 登録対象の LiDAR マップ（fix 解で先鋭化） |
| v1/v2 投影関数 | 三角測量の再投影・魚眼 unproject・特徴フィルタ |
| `common.py` | 軌道補間で各フレームのカメラ姿勢生成 |
| `step5_m0_check.py` | 前段ゲート＋ split-half 事後ゲート |

**新規実装**: 特徴抽出・追跡、既知姿勢メトリック三角測量（魚眼対応 unproject 込み）、動体/無地/植生マスク、再構成↔登録の反復ループ。

**マイルストーン・受入基準（v3）**:

| M | 内容 | go-no-go |
|---|---|---|
| v3-M0 | 各カメラの再構成カバレッジ + 識別可能性の前段判定 | 過半カメラで十分な視差付き剛体点群。不成立カメラは tf_static 採用（検証のみ） |
| v3-M1 | front + right の摂動回復 ±11.2cm/±3.54° | 回復 <3cm / <0.5° |
| v3-M2 | 全 12 カメラ + 事後ゲート + 並列化 | カメラ毎に合格/不合格（tf_static fallback）を明示レポート |
| v3-M3 | `camera_timestamp_offset_sec` 同時推定 | split-half で一致 |

**論文的根拠**: Pandey/Levinson 系（2D-3D）ではなく **3D-3D クロスモーダル登録**（Scaramuzza 系の hand-eye/odometry-based extrinsic、および LiDAR-visual マップ登録）に立脚。fix 解による metric SfM は測量分野で確立。

**残存リスク（②＝シーン観測性は RTK で解決しない）**:
- 無地地面（right の大半）・植生（rear）・周期構造（left）は fix 解でも復元が疎/曖昧。**3D でもアパーチャ問題（周期フェンスは面内に滑る）は残る**。→ マスク・高信頼点限定・全軌道蓄積で緩和し、最終ゲートで捕捉。根本解決は下記「推奨走行データ」による。
- 魚眼の三角測量は fisheye unproject が必要（実装可）。

#### v3 先行実装・事前検証まとめ（2026-07-03〜04）

**実装**: `step5_v3_cam_cloud.py`（KLT前後方向フローチェック付き追跡 + 多視点三角測量、視差角/再投影/正面性フィルタ）・`step5_v3_register.py`（三角測量点を LiDAR マップへ point-to-plane スナップ + **観測（画素・時刻）毎の reprojection 残差**で `lidar_T_camera_link` を Huber-robust `least_squares` 精密化、Step4 の `build_target` を再利用、reconstruct↔register を outer loop で反復、`--self-test` で既知摂動からの回復を検証）。

**設計の要点**: 登録は「蓄積点群への事後剛体ICP」ではなく「観測毎の reprojection 残差」で解く。Step4 で判明済みの「並進誤差が全周走行で平均化され消える」問題が、事後剛体ICPだと再発するため（誤差が車両ヘディングに依存し単一の剛体オフセットに縮退しない）。LiDAR マップは登録対象そのものではなく、三角測量点を独立検証データにスナップする「アンカー」として機能する。

**検証方法**: 既存の float RTK 8の字バッグ（`HRdqo3pf_2026-06-10T15-22-51.mcap`）で、(1) 未摂動 tf_static からの挙動（ドリフト量・残差・視差角・LiDARマップへのスナップ率）、(2) `--self-test`（±10cm/3° 摂動からの回復）の2段階を、**FOV・向きの異なる5カメラ**で実施（narrow-front / wide-front / wide-right / fisheye-right の組み合わせで、視野方向とFOVの効果を分離する狙い）。

**結果一覧**（全カメラ 500フレーム、`v3_outer_iters=3`）:

| camera | LiDAR | 投影モデル | FOV(参考) | 平均視差角 | LiDARマップへのスナップ率 | 未摂動ドリフト | 摂動回復（±10cm/3°） |
|---|---|---|---|---|---|---|---|
| 0 | front | plumb_bob | 30° | 2.8° | 1.4〜1.6%（28/1791〜86/6122） | 40〜45cm/1.4〜2.8° | 未実施（未摂動で既に発散的） |
| 1 | front | rational_polynomial | 120° | 5.1° | 84%（2228/2644） | 4.4cm/0.02° | **回復せず**: 9.4cm/2.74°（摂動位置に留まる） |
| 2 | right | rational_polynomial | 120° | 14.8〜14.9° | 91〜93%（5773/6216） | 4.9cm/0.66° | **回復せず**: 11.4cm/2.85° |
| 3 | right | rational_polynomial | 120° | 15.6° | 96%（5394/5609） | 11.3cm/0.46° | **悪化**: 16.9cm/3.33° |
| 9 | right | equidistant（魚眼） | 198° | 36.8〜37.1° | 97〜98%（2927/2993） | 11.2cm/0.95° | **回復せず**: 12.2cm/3.52°（残差は反復毎に単調改善: 22.7→15.6px、最良の収束） |

**分かったこと（幾何学的条件）**:
1. **FOV が広いほど視差角・LiDARマップ対応点数とも大きく改善する**（camera0 30°→camera1 120°で視差 2.8°→5.1°・スナップ率 1.5%→84% と劇的改善。camera1 120°→camera9 198°(魚眼)で視差 5.1°→37°とさらに拡大）。視野方向（前方/右方）以上に **FOV の広さが支配的な要因**。
2. **前方視野は進行方向ゆえに視差が本質的に低い**（単眼SfMの古典的な focus-of-expansion 問題）。同じ FOV でも front(camera1, 5.1°) < right(camera2/3, 14.8〜15.6°) となり、走行方向に対する視野の向きも独立に効く。
3. **M0（v2, 2D画像エッジ）とは逆の傾向**: M0 は無地の舗装が大半を占める right で失敗（偽エッジ）、front で成功した。v3 は特徴点追跡ベースのため無地領域では自然に特徴点が取れず、実構造（建物・バス等）だけが三角測量される — **2つの手法は相補的な弱点を持つ**。

**分かったこと（float RTK の限界、5カメラ共通）**:
4. **未摂動時の小さいドリフトだけでは「正しい解に収束した」ことの証拠にならない**。camera1/2/9 は未摂動で 4〜11cm と一見良好だったが、**摂動回復テストでは全カメラが失敗**（摂動位置からほぼ動かないか、むしろ悪化）。最良の幾何学的条件（camera9, 視差37°・スナップ率98%）でも同様に回復しなかった。
5. 診断: `huber_px=3.0` に対し残差が終始 15〜27px（v1/v2 の 1〜2px 台と比べ桁違い）。**大半の観測がロバスト損失で減衰され、実質的に勾配へ寄与する低ノイズ点が少ない** — float RTK による三角測量深度ノイズが支配的という設計時の懸念と整合。
6. **結論: 幾何学的条件（FOV・視差・LiDAR対応点数）は改善の必要条件だが十分条件ではない。float RTK 下ではどのカメラも摂動回復を実証できず、「fix 解が必須」という結論は変わらない。**

**次に活きる知見（fix 解データ取得後の優先順位）**:
- **camera9 系（広FOV・魚眼）が最有力候補**: 視差・スナップ率が最良で、残差の反復収束も唯一単調改善しており、fix 解でノイズが下がれば最初に成果が出る可能性が高い。
- **camera0 のような狭FOV前方カメラは v3 単独では最も困難**（視差不足がFOVでも向きでも救われない）。M0（v2）が front で機能した実績と併せ、**カメラごとに v2/v3 を使い分ける**運用が現実的（狭FOV前方系は v2、広FOV/側方/魚眼系は v3、という住み分け）。
- 摂動回復に必要な「鋭い勾配」を得るには、fix 解に加えて **v3_outer_iters・v3_ls_max_nfev の増強**、または **huber_px の見直し**（現状 3px は float RTK 由来の大きな残差の下では大半を外れ値化してしまう）も検討候補。

---

---

### 推奨キャリブレーション走行データ（RTK int 解前提・v3 で校正が成立する条件）

v3 でも解決しないのは**シーン観測性（②）**であり、これは幾何情報量の問題なので**走行データ設計で担保する**。校正の可否は「各カメラが 6-DOF を拘束できる幾何を、軌道のどこかで観測したか」で決まる（全軌道蓄積するので**全カメラが同時に構造を見る必要はなく、各カメラが軌道中のどこかで見れば足りる**）。

**必須要件（各カメラについて）**:
1. **視野を埋める剛体構造**: 建物・壁・柱・縁石・コンテナ・駐車車両など、**明瞭な角と多方向を向いた面**。無地地面・空・植生・単一大平面が視野を占有する状況を避ける。**経路は 12 カメラ全て（前後左右＋魚眼）の前に構造が来るよう配置**する。
2. **奥行き多様性**: 同一視野内に**近景（2〜5m）と遠景（15〜40m）が同時**に入ること。並進 3-DOF は視差＝奥行き差でのみ可観測。近い柱＋遠い建物のような配置が理想（無地地面が失敗したのは単一平面ゆえ）。
3. **並進による視差（三角測量用）**: 構造を視野に保ったまま数m の**横方向並進**。純回転は視差ゼロで復元不能。8の字の曲線経路は横移動＋ヨーを与え好適。
4. **回転軸の多様性（DOF 分離・hand-eye 併用時は必須）**: ヨーは8の字で十分。**ピッチ/ロールは平地では励起不足**（Step 4 で left の roll が不可観測だった原因と同型）→ **緩い勾配・段差・スロープ区間**を含めてピッチ/ロールを励起し、z/pitch 結合を可観測にする。
5. **低速・低ブレ**: ~1〜3 m/s。動きブレとスキャン-画像同期誤差を抑える（M0 のブレフィルタと整合）。
6. **静的シーン**: 撮影中は移動体（人・車）を最小化。
7. **RTK fix 維持**: 準備走行で fix 化し、上空視界を確保して fix を保つ。

**8の字に加えて推奨する要素**:
- **四方を構造物に囲まれた環境で8の字**（構造が一部にしかない開けた駐車場を避ける）。または**複数地点で8の字**を行い、全カメラが順に豊富な構造を捉える。
- **各サイドカメラが建物ファサードと平行に通る直線区間**を最低1本ずつ（サイドカメラの清浄な視差源）。
- **緩勾配/段差の1区間**（ピッチ/ロール励起）。
- **両回転方向の周回**を数ループ（平均化と対称性）。

**最小構成（実務目安）**:
> RTK fix（準備走行）＋ INS 姿勢 sub-0.1°。四方を剛体構造（建物/壁/柱/駐車車両）に囲まれ、近景<5m と遠景>15m が併存する環境。軌道は8の字（両回転方向）を ~1〜3 m/s、＋緩勾配/段差1区間、＋各サイドカメラ用の構造沿い直線パス各1本。静的シーン。数分。

これが揃えば front は確実、right は成立見込み、**rear/left も「植生/周期構造に視野を占有されない構造豊富な区間」を経路に含めることで可観測化**できる。逆にこれらが欠けるカメラは、どの手法でも（v3 でも）自動校正は原理的に困難で、tf_static 採用＋検証のみが妥当。

---

以下は **v1（DT 最小化）の記録**（検証②で棄却。受入基準・パラメータは実装済みコードの現状）。

- **コスト関数**: `C(T) = Σ_{p ∈ lidar_map} dist_transform[project(p, T)]²`
- **処理**:
  1. カメラ画像 Canny エッジ → Distance Transform (距離場)
  2. LiDAR マップ点群を TF で投影 → 距離場でコスト評価
  3. `scipy.optimize.minimize(L-BFGS-B)` で 6DOF 最適化
  4. 通常カメラ 8 台を multiprocessing で並列実行（魚眼 4 台は Step 5b で処理）
- **投影関数**: 既存 `project_lidar_to_cam.py` の関数群を流用
- **出力**: `lidar_* → camera{0-7}/camera_link` の 6DOF TF (8 本)
- **受け入れ基準**: 収束後、投影エッジ点の距離場残差の中央値が `cam_reproj_accept_px` 以下なら合格。
  超過したカメラは `verify_result.py` で警告し、初期値・特徴量不足・タイムスタンプ遅延を疑う
- **チューニング**:
  - `cam_reproj_accept_px`: 合格とみなす投影エッジ残差の中央値 [px] (デフォルト: 2.0)
  - `camera_timestamp_offset_sec`: カメラドライバの系統的タイムスタンプ遅延補正。カメラごとに個別設定 (デフォルト: 全カメラ 0.0)
    ```yaml
    camera_timestamp_offset_sec:
      camera0: 0.0
      camera1: 0.0
      camera2: 0.0
      camera3: 0.0
      camera4: 0.0
      camera5: 0.0
      camera6: 0.0
      camera7: 0.0
      camera8: 0.0   # 魚眼
      camera9: 0.0   # 魚眼
      camera10: 0.0  # 魚眼
      camera11: 0.0  # 魚眼
    ```
  - `cam_frame_stride`: 最適化に使う画像フレームを N 枚に 1 枚に間引く (デフォルト: 5)
  - `cam_parallel_workers`: カメラ並列処理数 (デフォルト: 4、CPU コア数に応じて調整)
  - `canny_low` / `canny_high`: Canny エッジ検出の閾値 (デフォルト: 50 / 150)
  - `dist_transform_max`: 距離場の上限クランプ値 [px] (デフォルト: 20)
  - `lbfgsb_max_iter`: L-BFGS-B 最大反復回数 (デフォルト: 200)
  - `lbfgsb_ftol`: L-BFGS-B 関数値収束閾値 (デフォルト: 1e-6)
  - `image_scale`: 最適化時の画像縮小率。0.5 にすると 4× 高速化 (デフォルト: 1.0)
  - `proj_point_stride`: 投影する LiDAR マップ点を N 点に 1 点間引く (デフォルト: 1)

### Step 5b: Camera-to-LiDAR TF 推定 (魚眼カメラ)

- Step 5a と同じ枠組みで **equidistant 歪みモデル** (`cv2.fisheye.projectPoints`) を使用
- 初期値誤差は 10 cm 未満を前提とし、tf_static 初期値をそのまま最適化の初期値として使用
- **チューニング**: Step 5a と共通（`cam_frame_stride`、`image_scale` 等）
- **v2 も共通**: scan-to-image 遮蔽エッジ整合は投影関数の分岐のみで魚眼に適用（実装は `step5_cam_to_lidar.py` 1 ファイルのまま）。
  魚眼は視野に車体が写り込むため frustum 内エッジ点が減るリスクあり → M2 で個別確認。

### Step 5 カメラ手法ルーティング（camera 毎 v2 / v3 の固定割り当て・2026-07-04 決定）

> **決定**: カメラIDによる**固定ルーティング**とする（M0 の結果だけに基づく動的判定は不採用）。
> - **v2（DT/エッジ相関、`step5_cam_to_lidar.py`）**: `camera0`（front narrow）, `camera4`（rear narrow）
> - **v3（全軌道蓄積カメラ点群→LiDAR登録、`step5_v3_*`）**: それ以外（1,2,3,5,6,7,8,9,10,11＝ワイド/魚眼）
>
> **背景**: `camera0/2/4/6` の M0 検証（3〜30ペア、全区間から均等サンプリング）では `camera4` が 0/6 で不合格だった。
> ただし M0 のサンプル数は少数（最大30ペア）であり、8の字走行は全周を回るため、**front が捉えた良好な構造を rear も走行中の別区間で捉えている可能性がある**（M0 が単にその瞬間を拾えなかっただけの可能性）。
> 固定ルーティングを採用し、**v2 実行時は M0 のサンプルではなく Step 5 の全フレーム（`cam_frame_stride` 間引きのみ）を使う**ことで、rear が全区間のどこかで機能する構造を捉えていれば拾える。
> **既存の受け入れ基準（`cam_reproj_accept_px` 等）はルーティングと独立に有効**なので、camera4 が実際には機能しない場合でも「HIGH-RESIDUAL」として検知され、`verify_result.py` で警告される（サイレント発散にはならない）。

### Step 6: 全 TF 同時精密化

> **前提条件（2026-07-03）**: カメラ残差は Step 5（ルーティングに従い v2 または v3）と同一のコストを使うため、
> **各カメラの Step 5 コストが単体で機能する（M0 合格 or v3 self-test 合格）ことが前提**。Step 5 が機能しない状態で
> Step 6 に進んでもそのカメラの TF は拘束されない（同時最適化は魔法ではない）。

Step 6 は 15 本の TF を変数として、**3種類の残差項**の無次元化・重み付き和を最小化する:

1. **LiDAR-LiDAR 残差**（Step 4 と同じ: 項A 近時刻重畳 + 項B マップ）
2. **Camera-LiDAR 残差**（Step 5、カメラ毎にルーティングされた v2 or v3 のコスト）
3. **Camera-Camera クロスビュー残差（新規、2026-07-04 設計）**: 同一搭載クラスタ内（`front={0,1,8}`, `right={2,3,9}`, `rear={4,5,10}`, `left={6,7,11}`）で、**入れ子になった FOV の重畳**を利用する

#### Camera-Camera クロスビュー残差の設計

**根拠**: `camera0(30°) ⊂ camera1(120°) ⊂ camera8(魚眼~200°)` のように、同一クラスタ内のカメラは搭載位置がほぼ共通（数十cmオフセットはあるが魚眼のため重畳域は十分）で FOV が入れ子になっている。これは Step 4 の**項A（LiDAR間の近時刻同時重畳）と同型の構造**であり、同じ実世界構造を複数センサーが異なる角度・解像度で見ることで、**RTK/三角測量の奥行きノイズに免疫な相対拘束**を追加できる。

**メカニズム**:
1. ワイド/魚眼カメラ（v3 ルーティング）は既に自身の v3 パイプラインで **LiDAR マップに point-to-plane スナップ済みの高信頼 3D 点群**を持つ（`step5_v3_register.py` の `pts_final`）。
2. その3D点群のうち、**同クラスタの狭FOVカメラ（v2 ルーティング）の視野角内に落ちる点だけ**を抽出する。
3. 抽出した点を狭FOVカメラの v2 コスト（画像エッジとの相関）に**追加の投影点集合として供給**する。狭FOVカメラが単独で LiDAR マップから抽出する幾何/強度エッジより、**他カメラで三角測量・LiDAR照合済みの高信頼点**なので、密マップ投影のエイリアシング問題を緩和できる可能性がある。
4. クラスタ内の全カメラの 6-DOF（`{E_0, E_1, E_8}` 等）を Step 6 の結合最適化の中で**同時に**解く。狭FOVカメラの観測がワイド/魚眼カメラの拘束を補強し、逆にワイド/魚眼カメラの観測が狭FOVカメラのシーン限界（M0 で発見した「front以外は識別不能」問題）を緩和する。

**位置づけ**: 独立の「Step 5 v4」ファイルではなく、**Step 6 の第3残差項として実装**する（Step 6 は元々複数残差項の重み付き和という設計であり、クロスビュー項もその一種として自然に収まるため）。

**チューニング（追加）**:
- `joint_crossview_weight`: 無次元化後のクロスビュー残差の相対重み
- `joint_crossview_fov_margin_deg`: 狭FOVカメラの視野角にマージンを持たせて点を抽出する角度

**実装状況**: 設計のみ（未実装）。Step 6 全体が未着手のため、fix 解データ取得後、Step 6 本体の実装と合わせて着手する。

- 実装: `scipy.optimize.minimize` または Ceres Solver (Python バインディング)
- **残差の無次元化（必須）**: LiDAR-LiDAR 残差は **メートル**、Camera-LiDAR / クロスビュー残差は **ピクセル** と
  単位が異なるため、単純に加算できない。各残差項をその項の初期 RMS で割って無次元化してから
  重み付き和をとる（または px→m 換算にカメラの瞬時 IFOV と対象距離を用いる）。
  `joint_*_weight` はこの無次元化後の相対重みを指す。
- **チューニング**:
  - `joint_lidar_weight` / `joint_cam_weight` / `joint_crossview_weight`: 無次元化後の各残差項の相対重み (デフォルト: 1.0 / 1.0 / 1.0)
  - `joint_max_iter`: 最大反復回数 (デフォルト: 500)

---

## 処理時間の見積もり

> **下記は実装前の見積もり**（2分bag・RTX3060 前提の推測値、Step 5 以降は未実装のため実測なし）。
> Step 2-4 の実測値・実際のボトルネックは「処理高速化: 実測プロファイリングと対応」節を参照
> （検証bag は 150秒・8の字全区間で条件が異なるため、下表と直接比較しないこと）。

### 前提条件

- rosbag: 2 分間（≈ 20〜28 GB on disk）
- ハードウェア: CPU Core i7-12700 / GPU RTX 3060 12GB
- デフォルトパラメータ使用（`map_scan_stride=3`、`cam_frame_stride=5` など）

### データ量

| センサー | 件数 | 備考 |
|---|---|---|
| LiDAR スキャン（4台合計） | 4,800 スキャン | 各 ≈ 60,000点 |
| カメラフレーム（12台合計） | 28,800 フレーム | compressed JPEG |
| RTK ポーズ | 6,000 件 | 50 Hz |

### ステップ別処理時間

| Step | 内容 | CPU のみ | GPU (RTX3060) |
|---|---|---|---|
| Step 1 | RTK 軌道抽出 | < 1 min | < 1 min |
| Step 2 | LiDAR undistortion（288M 点） | 5〜8 min | 30〜60 sec |
| Step 3 | Front LiDAR マップ構築 + 投票フィルタ | 2〜4 min | 30〜60 sec |
| Step 4 | 他 3 台マップ構築 + マルチスケール ICP × 3 | 8〜15 min | 1〜3 min |
| Step 5a | 通常カメラ最適化 × 8 台（並列） | 15〜30 min | 4〜8 min |
| Step 5b | 魚眼カメラ最適化 × 4 台（並列） | 8〜15 min | 2〜5 min |
| Step 6 | 全 TF 同時精密化（任意） | 10〜20 min | 3〜6 min |
| I/O | rosbag 読み込み（NVMe SSD 前提） | 1〜2 min | 1〜2 min（GPU 非依存） |
| **合計（Step 1〜5b）** | | **40〜75 min** | **10〜18 min** |
| **合計（Step 1〜6）** | | **50〜95 min** | **13〜24 min** |

> HDD 使用時は I/O だけで +5〜10 min 追加。NVMe SSD 推奨。

### ボトルネック

```
CPU 版: Step 5a/5b が全体の 60〜70%（L-BFGS-B 反復 × 12 カメラが律速）
GPU 版: Step 5a/5b が依然として最大 (30〜40%)
        Distance Transform は cv2.cuda 非対応のため CPU のまま
```

---

## チューニングパラメータ一覧

実行時に `config.yaml` で指定する。全パラメータにデフォルト値を持ち、
デフォルトのまま実行可能。**精度重視** / **速度重視** の 2 プリセットを用意する。

### パラメータ一覧

| パラメータ | デフォルト | 精度重視 | 速度重視 | 説明 |
|---|---|---|---|---|
| **共通** | | | | |
| `gpu_enabled` | `true` | `true` | `true` | GPU 使用フラグ（現状 Step 3 の `_visibility_counts` のみ対応。CUDA 未検出時は自動で CPU フォールバック） |
| `gpu_visibility_chunk_scans` | `32` | `32` | `32` | Step 3 GPU 経路でバッチ処理する scan 数（VRAM 使用量の上限を決める） |
| `cpu_workers` | `4` | `4` | `8` | multiprocessing ワーカー数 |
| **Step 0** | | | | |
| `rtk_cov_warn_m2` | `0.1` | `0.1` | `0.1` | RTK covariance 位置分散の警告閾値 [m²]（超過は警告のみ・停止しない） |
| `rtk_max_gap_sec` | `0.5` | `0.5` | `0.5` | odometry の許容最大ギャップ [s]（超過は致命） |
| `min_yaw_coverage_deg` | `360` | `360` | `360` | 旋回励起の警告閾値 [deg]（未満は観測性低下の警告） |
| `require_cameras` | `false` | `false` | `false` | カメラトピック欠落を致命に格上げ |
| **Step 1** | | | | |
| `rtk_smoothing_window` | `0` | `0` | `0` | RTK 平滑化ウィンドウ（0 = 無効） |
| `rtk_max_speed_mps` | `30` | `30` | `30` | 物理妥当性ゲートの上限速度 [m/s]（covariance/status は placeholder で使えないため速度で判定） |
| `rtk_max_reject_ratio` | `0.1` | `0.1` | `0.1` | 除外区間がこの割合超で INS 無効として中断 |
| **Step 2** | | | | |
| `undistort_enabled` | `true` | `true` | `true` | LiDAR スキャン歪み補正の有効/無効 |
| `lidar_scan_stride` | `1` | `1` | `2` | スキャン間引き率（1 = 全スキャン） |
| **Step 3** | | | | |
| `map_voxel_size` | `0.1` | `0.05` | `0.2` | マップボクセルサイズ [m] |
| `vote_threshold_ratio` | `0.3` | `0.4` | `0.2` | 動体除去の占有率閾値 |
| `map_scan_stride` | `3` | `1` | `5` | マップ蓄積スキャン間引き率 |
| `vote_max_range` / `vote_min_range` | `60` / `1.5` | `60` / `1.5` | `60` / `1.5` | visibility 判定レンジ [m] |
| `vote_hfov_deg` / `vote_vfov_deg` | `120` / `40` | `120` / `40` | `120` / `40` | visibility 判定 FOV [deg] |
| `vote_occlusion_enabled` | `true` | `true` | `true` | 遮蔽考慮 visibility の有効/無効（2026-07-04 追加。無効時は旧来の range+FOV コーンのみの判定にフォールバック） |
| `vote_occlusion_footprint_m` | `0.15` | `0.15` | `0.3` | 遮蔽判定レンジイメージのビン物理サイズ [m]（アーク長 = 角度×レンジ でビニングし距離非依存にする） |
| `vote_occlusion_tol` | `0.5` | `0.3` | `0.5` | 遮蔽判定の許容誤差 [m]（実測レンジ ≥ 候補レンジ - tol なら非遮蔽） |
| **Step 4** | | | | |
| `icp_multiscale_voxels` | `[0.5, 0.2, 0.1]` | `[0.4, 0.15, 0.05]` | `[1.0, 0.5, 0.2]` | マルチスケール voxel サイズ列 [m] |
| `icp_corr_dist_scale` | `3.0` | `3.0` | `3.0` | 対応ゲート = voxel × this |
| `icp_max_correspondence_dist` | `2.0` | `2.0` | `2.0` | 対応ゲートの絶対上限 [m] |
| `icp_huber_scale` | `1.0` | `1.0` | `1.0` | Huber デルタ = voxel × this |
| `icp_degeneracy_rel` | `1e-4` | `1e-4` | `1e-4` | 不可観測方向除去（eig_max×this 未満） |
| `icp_max_iterations` | `30` | `50` | `20` | スケール毎の最大反復回数 |
| `icp_step_tol` | `1e-4` | `1e-5` | `1e-3` | ステップ収束閾値 |
| `icp_rmse_threshold` | `0.02` | `0.01` | `0.05` | 早期終了 RMSE [m]（下限未満で通常到達せず） |
| `l2l_scan_stride` | `12` | `1` | `24` | スキャン間引き率（2026-07-04: 旧デフォルト 3 → 12。フル503→126scanでrmse差0.05cm以内・8の字全区間で検証済み） |
| `l2l_points_per_scan` | `1000` | `5000` | `500` | スキャン毎の点サブサンプル数（2026-07-04: 旧デフォルト 3000 → 1000。3000→750までrmse差0.1cm以内を確認） |
| `l2l_max_range` | `60` | `60` | `60` | 点のクロップ距離 [m] |
| `l2l_pair_dt_max` | `0.06` | `0.06` | `0.06` | 項A 同時刻ペアの許容時間差 [s] |
| `l2l_pair_ref_scans` | `3` | `5` | `1` | 項A 局所参照に集約する隣接スキャン数 |
| `l2l_interlidar_weight` / `l2l_map_weight` | `1.0` / `1.0` | `1.0` / `1.0` | `1.0` / `1.0` | 項A / 項B の相対重み |
| `icp_method` | `point_to_plane` | `point_to_plane` | `point_to_plane` | 方式（point_to_plane のみ実装） |
| **Step 5a/5b** | | | | |
| `camera_timestamp_offset_sec` | 下記参照 | — | — | カメラごとのドライバ系統遅延補正 [sec]。カメラ・LiDAR 非同期トリガ誤差は蓄積マップ+RTK補間で自動吸収されるため通常は全て 0.0。ドライバ遅延が既知のカメラのみ設定する |
| `cam_frame_stride` | `5` | `2` | `10` | 最適化フレーム間引き率 |
| `cam_parallel_workers` | `4` | `4` | `8` | カメラ並列処理数 |
| `image_scale` | `1.0` | `1.0` | `0.5` | 最適化時の画像縮小率（0.5 で 4× 高速化） |
| `canny_low` / `canny_high` | `50` / `150` | `30` / `100` | `50` / `150` | Canny エッジ閾値 |
| `dist_transform_max` | `20` | `30` | `15` | 距離場の上限クランプ [px] |
| `lbfgsb_max_iter` | `200` | `500` | `100` | L-BFGS-B 最大反復回数 |
| `lbfgsb_ftol` | `1e-6` | `1e-8` | `1e-5` | L-BFGS-B 収束閾値 |
| `proj_point_stride` | `1` | `1` | `3` | 投影 LiDAR 点間引き率 |
| `cam_proj_max_points` | `8000` | `12000` | `4000` | フレーム毎に投影する map 点のサブサンプル上限 |
| `cam_trans_bound_m` | `0.5` | `0.5` | `0.5` | L-BFGS-B 並進探索範囲（init 近傍）[m] |
| `cam_rot_bound_rad` | `0.3` | `0.3` | `0.3` | L-BFGS-B 回転探索範囲（init 近傍）[rad] |
| `cam_reproj_accept_px` | `2.0` | `1.5` | `3.0` | 合格とみなす投影エッジ残差の中央値 [px] |
| **Step 6** | | | | |
| `joint_lidar_weight` | `1.0` | `1.0` | `1.0` | 無次元化後の LiDAR-LiDAR 残差の重み |
| `joint_cam_weight` | `1.0` | `1.0` | `1.0` | 無次元化後の Camera-LiDAR 残差の重み |
| `joint_max_iter` | `500` | `1000` | `200` | Step 6 最大反復回数 |

### プリセット別処理時間（参考）

| プリセット | CPU 時間 | GPU 時間 |
|---|---|---|
| **精度重視** | 80〜120 min | 15〜30 min |
| **デフォルト** | 40〜75 min | 10〜18 min |
| **速度重視** | 15〜25 min | 4〜8 min |

---

## 出力フォーマット

既存の `multi_tf_static.yaml` 形式で出力する。

```yaml
# LiDAR-to-LiDAR (drs_base_link を親フレームとする)
drs_base_link:
  lidar_rear:
    x: 0.0
    y: 0.0
    z: 0.0
    roll: 0.0
    pitch: 0.0
    yaw: 3.14159
  lidar_left:
    x: ...
  lidar_right:
    x: ...

# Camera-to-LiDAR
lidar_front:
  camera0/camera_link:
    x: ...
  camera1/camera_link:
    x: ...
  camera8/camera_link:
    x: ...
```

---

## 既存コードの再利用

| 既存モジュール | 流用箇所 |
|---|---|
| `tools/lidar-camera/project_lidar_to_cam.py` の `_load_camera_info()` | Step 5a/5b のカメラ内部パラ読み込み |
| `tools/lidar-camera/project_lidar_to_cam.py` の `build_undistort()` | Step 5a の画像 undistort |
| `tools/lidar-camera/project_lidar_to_cam.py` の `extrinsic_lr_to_optical()` | TF 座標系変換 |
| `tools/lidar-camera/tune_extrinsic.py` の `project()` | Step 5a/5b の点群投影 |
| `tools/lidar-camera/tune_extrinsic.py` の `euler_to_R()` / `R_to_euler()` | 6DOF パラメータ変換 |

---

## 依存ライブラリ

| ライブラリ | 用途 |
|---|---|
| `open3d` | ICP (Step 4)、点群処理 |
| `scipy` | L-BFGS-B 最適化 (Step 5/6) |
| `opencv-python` | Canny、Distance Transform、fisheye 投影 |
| `mcap-ros2-support` | rosbag 読み込み |
| `numpy` | 全般 |
| `pyyaml` | TF YAML 入出力 |

> **実行環境（2026-07-03 決定・構築済み）**: マシン素の `~/.local` は numpy 2.2.6 + opencv(numpy≥2必須) + system scipy(numpy<1.25必須) で自己矛盾しており scipy が壊れていた。
> このため **専用の clean venv（`<repo>/.venv`, `--system-site-packages` なし）に整合バージョン一式を導入**して全 Step を単一環境で動かす方針に決定・構築済み:
> `numpy==1.24.4` / `scipy==1.10.1` / `open3d==0.18.0` / `opencv-python==4.9.0.80` / `mcap` / `pyyaml`（`tools/auto-calib/requirements.txt`）。
> rclpy・ROS メッセージ型は pip 不可のため `/opt/ros/humble` を source して PYTHONPATH 経由で供給（venv の numpy 1.24 は Humble の rclpy と ABI 整合、実バッグの Odometry/PointCloud2 deserialize を検証済み）。
> セットアップ: `tools/auto-calib/setup.sh`。実行: `tools/auto-calib/run.sh <step>.py ...`（ROS source + venv python を内包）。
> **全 Step が同一 venv で動くため `run_pipeline.py` はインタプリタ切替不要**。scipy.optimize(L-BFGS-B) / Open3D ICP / cv2 いずれも本 venv で import 可能を確認済み。

---

## 処理高速化: 実測プロファイリングと対応（Step 2-4・2026-07-04）

> 下記「GPU 高速化」節（旧版）は実装前の**推測**（torch/cupy 前提、Step 4 は Open3D ICP を使う想定）だったが、
> 実データでの `cProfile` 実測の結果、想定と異なるボトルネックが判明した。本節が現状の正しい記述で、
> 旧節の想定（Step 2/3 の点群変換、Step 4 の Open3D ICP）は**外れ**（詳細は下記）。以降は実測ベースで更新する。

**検証環境**: bag `HRdqo3pf_2026-06-10T15-22-51.mcap`（8の字・全区間、51GB、1507 scans/lidar）、
CPU 22 コア、GPU RTX 4070 Laptop (8GB, CUDA 12.9)、`torch`/`cupy` は未導入・`open3d==0.18.0`（CUDA tensor 利用可）。

### 実測ボトルネック（`cProfile`、100-150 scan サブセットで測定）

| Step | 支配的コスト | 実測内訳 | GPU の可否 |
|---|---|---|---|
| 2 (undistort) | bag I/O（mcap 読込 + zstd 展開）が最大 | 200 scan/25s 中: I/O 34%、姿勢補間(quat slerp, numpy既にベクトル化済み) 30% | **不向き**: I/O が支配的で GPU 化しても math の 30% しか削れない |
| 3 (build_front_map) | ①`_VoxelAccumulator` の逐次 dict マージ ②可視性判定 `_visibility_counts` | 修正前: 100 scan/12.5s 中 ① 62%（Python `dict.get()` を voxel 毎に呼ぶ generator、100 scan で 480 万回） ② 32%（Python for ループで n_voxels×n_scans の距離+FOV判定） | ①は GPU 無関係（純アルゴリズム） ②は embarrassingly parallel で GPU 向き |
| 4 (lidar_to_lidar) | `map_blocks`（同一 front map の木を共有する 503 個）を**個別に** `cKDTree.query(workers=-1)` していたことによる**スレッドプール生成オーバーヘッド** | 150 scan/777s 中: `cKDTree.query` 79%（うち大半がPythonの`threading`機構、`_thread.lock.acquire`だけで416s） | GPU 化ではなく**呼び出し回数を減らす**のが正解。GPU 化するなら Open3D tensor 最近傍探索への全面書き換えが必要（自前の Huber-LM・縮退方向射影ロジックは移植対象外＝リスク大） |

### 実施した対応と実測効果（フル 503 scan、同一 bag・同一 tf で検証、全て結果の数値一致を確認済み）

| 対応 | 内容 | Before → After | 検証方法 |
|---|---|---|---|
| Step 4: クエリ一括化 | `map_blocks` は全員同じ `tree` を参照しているため、`id(tree)` でグループ化し**同じ木への複数ブロックのクエリを 1 回にまとめる**（`inter_blocks` は木がブロック毎に異なるため対象外、従来通り個別クエリ） | **1549.6s → 1210.2s（-22%）** | `lidar_to_lidar_result.json` が変更前と bit 完全一致 |
| Step 3-1: `add_scan` の numpy 一括化 | 各 scan は従来通り `np.unique` でローカル重複排除するが、**scan 間のマージを逐次 dict ではなく全 scan 分をリストに貯めて最後に 1 回だけグローバル `np.unique`** で行う（`occ` は「各 scan が自身の voxel key をちょうど 1 回だけ寄与する」性質を使い、グローバル unique 後の出現回数でそのまま計算） | **209.0s → 180.5s（-14%）** | 合成データでブルートフォース dict 参照実装と centers/intensity/occ 完全一致を確認。実データでも raw/kept voxel 数が変更前と完全一致 |
| Step 3-2: `_visibility_counts` の GPU 化 | 距離+レンジ判定のみを Open3D CUDA tensor でスキャン `gpu_visibility_chunk_scans`（既定 32）件ずつバッチ化。**boolean mask だけを CPU に転送**し、その後の FOV角度判定(`arctan2`)は従来通り CPU/numpy のまま（対象点数が既にレンジフィルタ後の小さい部分集合のため）。`gpu_enabled`（既定 true）で有効・CUDA 無ければ自動で CPU 経路にフォールバック | **180.5s → 167.5s（-7%）** | raw/kept voxel 数完全一致。生成点群もソート後に完全一致（行順序のみ非決定的に入れ替わるが下流はKDTree構築にしか使わないため無害） |
| viz.py: 凡例マーカーサイズ | `ax.legend(markerscale=10)` が疎点群(s=0.15-3)向けの拡大率を、start/end 等の大きい注釈マーカー(s=80)にも一律適用し、凡例アイコンが肥大化してテキストと重なる不具合。**凡例マーカーサイズを一律 50 に正規化**する方式に変更（全 Step 共通のプロット凡例に反映） | — | Step1 trajectory 等で目視確認 |
| Step 2: 単一読み込み + lidar 別並列処理 | 従来は 4 lidar 分、`process_lidar()` が独立して `common.iter_messages(files, [topic])` を呼び、**51GB の bag 全体を実質 4 回、逐次的に読み直していた**（mcap の `SeekingReader` はチャンク単位の索引を持つが、4 lidar は同時録画のため大半のチャンクに全 lidar のメッセージが混在しており、チャンクスキップによる I/O 削減は効かない）。`read_lidar_buffers()` で **bag を 1 回だけ走査**し（`log_time_order=True` により topics を跨いでも各 lidar 自身の相対順序は従来と不変）、4 lidar 分の生メッセージ（未デシリアライズ）を `(idx, raw_bytes)` としてバッファ、その後 `multiprocessing.get_context("fork")` で **4 lidar を並列プロセスとして** undistort + 書き出し（fork なのでバッファは copy-on-write 継承・pickle 転送なし） | **796.3s → 272.1s（-66%）** | フルスケール（1507 scan×4 lidar）で `undistorted/<lidar>/*.npy` + `manifest.json` が変更前と完全一致（`diff -rq` 差分ゼロ）。中間スケール（400 scan）・部分 lidar+stride 指定でも別途完全一致を確認 |

**Step 2 合計: 796.3s → 272.1s（-66%）、Step 3 合計: 209.0s → 167.5s（-20%）、Step 4: 1549.6s → 1210.2s（-22%、クエリ一括化のみ。間引き込みの最終結果は下記参照）。フルパイプライン（Step0-4合計 2710.6s）換算で約 994秒（約37%）短縮**。
Step 3/4 のクエリ一括化・GPU化はいずれも cProfile 実測（100-150 scan サブセット）が示す割合ほど劇的な改善にはならなかった（Step 3 add_scan 62%→実質-14%、Step 4 スレッド管理 79%→実質-22%）。
これは cProfile 自体のオーバーヘッドが「呼び出し回数の多い処理」を計測上過大評価するため（154万〜1.5億回規模の関数呼び出しがあるコードでは特に顕著）。Step 2 の I/O 削減・下記の Step 4 間引きは実測どおり倍率に近い改善が出た。

### Step 4: スキャン間引き（`l2l_scan_stride` / `l2l_points_per_scan`、2026-07-04 実施・新デフォルト採用）

対応点クエリは呼び出し回数（scipy `workers=-1` のスレッドプール生成）が支配的なため、点を減らすよりスキャン数を減らす方が効くという見立てのもとスイープした。

- **`l2l_scan_stride` 3→12**（503→126 scan/lidar）: **1210.2s → 290.6s（-76%）**。rmse: right 12.47→12.45cm、left 9.58→9.64cm、rear 11.31→11.36cm（全て0.06cm以内）。left の `converged` フラグのみ true→false（rmse自体はほぼ同じで境界的な違い）。
- **`l2l_points_per_scan`（stride=12固定でスイープ）**: range-filter後の実点数は1scanあたり約118k〜120kあり、既定の3000でも既に約2.5%しか使っていないことを確認した上で検証。

  | l2l_points_per_scan | 所要時間 | right rmse | left rmse | rear rmse |
  |---|---|---|---|---|
  | 3000 | 290.6s | 12.45cm (not-conv) | 9.64cm (not-conv) | 11.36cm (not-conv) |
  | 1500 | 209.5s | 12.42cm (not-conv) | 9.62cm (not-conv) | 11.38cm (not-conv) |
  | **1000（新デフォルト）** | **176.8s** | 12.46cm (not-conv) | 9.71cm (not-conv) | 11.37cm (not-conv) |
  | 750 | 148.6s | 12.28cm (converged) | 9.62cm (not-conv) | 11.42cm (not-conv) |

  rmse は全設定で0.2cm以内に収まり、精度への影響は誤差レベル。以前の「503 scanの方が151 scanより悪化していた」知見（データを増やせば精度が上がるとは限らない）と整合。

**結論・採用値**: `l2l_scan_stride=12` / `l2l_points_per_scan=1000` を `common.py` の新デフォルトに採用。**Step 4: 1210.2s → 176.8s（クエリ一括化後比 -85%、旧オリジナル1549.6s比では -89%）**。

`icp_max_iterations` を上げても改善しないことも確認済み: right/left/rear いずれも line search 失敗（8回の λ バックオフ全滅）により上限30回よりずっと早い10〜14回で打ち切られており、反復回数ではなく対応点/front マップ側のノイズ床が rmse を決めている。

### 今後の検討事項（未着手・2026-07-04 時点）

- Step 4 の rmse 10〜12cm 台の床は、間引き量ではなく front マップ側のスミア（Step 3 の voxel サイズ・deskew 精度）に起因する可能性が高い。→ 下記「Step 3: 近傍ボクセル欠落問題の調査と修正」で近傍データ復活には着手・解消したが、Step 4 への実測検証の結果 rmse は改善せず（right改善/left・rear微悪化、3台とも引き続き非収束）、rmse 床自体はこの修正だけでは解消しないことを確認した。→「Front map スミア改善策の検討」節でスミアの定量化を含む対策候補を整理（未実装）。

## Step 3: 近傍ボクセル欠落問題の調査と修正（2026-07-04）

**症状**: front map の可視化（`step3_front_removed_overlay.png`）で、8の字ループの**内側（＝車両が繰り返し近傍を走行したエリア）がほぼ全域 vote filter で除去**されていた。8の字の外周部（遠方の静止構造物）は概ね正常に残る。

**診断**（走行経路からの最近傍距離 [m] 別に `occ`/`visible`/`ratio` を集計、front, 503 scans, 旧実装）:

| range[m] | n_vox | kept% | occ_med | vis_med | ratio_med |
|---|---|---|---|---|---|
| 0-2 | 155892 | 0.1% | 7.0 | 121.0 | 0.054 |
| 2-5 | 325860 | 1.2% | 8.0 | 135.0 | 0.056 |
| 5-10 | 617256 | 1.9% | 7.0 | 144.0 | 0.043 |
| 10-15 | 633282 | 0.8% | 6.0 | 144.0 | 0.034 |
| 15-40 | 1554213 | 0.1-0.2% | 3.0 | 137-148 | 0.020-0.021 |
| 40-60 | 388606 | 9.1% | 2.0 | 50.0 | 0.045 |
| >60 | 851975 | 100.0% | 2.0 | 0.0 | 1.000 |

**根本原因**: `_visibility_counts()` の可視性モデルは「距離 `vote_max_range` 以内・水平/垂直 FOV コーン内に入っていれば見えたはず」という**単純な幾何チェックのみで、実際の遮蔽（line-of-sight occlusion）を一切考慮していない**。8の字ループの内側にある近傍点は、ループ上のかなり多数の他地点（最大 60m 以内ならどこでも）から見て「コーンには入っている」と判定されるが、実際にはその大半の視点からは地面や他の構造物に遮られてヒットしない。結果として `visible`（分母）が実際の `occ`（分子）に対して大きく過大評価され、比率が閾値 0.3 を大きく下回り、経路付近がほぼ全滅していた。

**却下案（`vote_max_range` を可視判定専用に 15m へ縮小）**: 近傍は 0.1%→5.1-63.5% と改善したが、副作用として **15m 以遠は `visible` が構造的に常にゼロになり**（`ratio = occ/max(visible,occ) = occ/occ = 1.0` に自動収束）、投票フィルタが 15m 以遠で事実上無効化された（vote 比率ヒストグラムの66%が ratio=1.0 に集中、kept_ratio 21.1%→75.9%）。中遠方の移動物体除去能力を犠牲にする trade-off だったため不採用。

**採用した修正: 遮蔽考慮 visibility モデル**。各スキャン自身の実点群（変換前・センサー座標系）から、方位角/仰角の**アーク長（角度×その点自身のレンジ）**でビニングした最小レンジの「レンジイメージ」を構築する（`_range_image()`、int64 キーによる疎な `np.unique`/`np.minimum.at` 実装、`_VoxelAccumulator` と同じパターン）。候補ボクセルは、そのスキャンの同じビンに**実際に手前で返り値がある場合のみ「遮蔽」として visible から除外**する（`measured < 候補レンジ - vote_occlusion_tol` で遮蔽、ビンに返り値が一切ない場合は「証拠なし」として非可視扱い＝過大評価を避ける保守的な選択）。

固定角度ビン（`az_bin_deg`/`el_bin_deg`）でまず実装したが、スイープ（0.4/0.6/0.8/1.2度、tol 0.2/0.5/1.0）の結果、**ビンの物理フットプリントは距離に比例して粗くなる**ため、近傍を改善する設定は必ず 15-40m 帯の判別力も同方向に緩めてしまうことが判明（例: 0.8度で近傍 kept% 改善するも overall p50=0.563 まで上昇＝過半数が自動通過）。→ ビンを角度ではなく**アーク長（角度×レンジ）で固定物理サイズ (`vote_occlusion_footprint_m`, 既定0.15m) にビニング**することで、この距離依存性を解消した。

**結果比較（front, 503 scans）**:

| | 旧実装 | 15m円錐(却下) | 固定角0.4° | **アーク長適応(採用)** |
|---|---|---|---|---|
| kept_ratio | 21.1% | 75.9% | 48.9% | **42.4%** |
| vote比率 p50 | 0.03 | 1.00(自動通過) | 0.28 | **0.21** |
| 0-2m kept% | 0.1% | 5.1% | 10.5% | **14.1%** |
| 10-15m kept% | 0.8% | 63.5% | 30.6% | **27.5%** |
| 15-20m kept% | 0.1% | 100.0% | 43.8% | **29.1%** |
| 20-30m kept% | 0.1% | 100.0% | 45.3% | **23.3%** |
| 30-40m kept% | 0.2% | 100.0% | 38.1% | **25.4%** |
| Step 3 所要時間 | ~209s | - | ~217s | **~235s**（誤差数十秒程度、GPU レンジフィルタが支配的costのため影響小） |

固定角ビンでは 15-40m 帯の kept% が近傍より不自然に高く出ていた（ビンが距離とともに粗くなる副作用）が、アーク長適応では 0-40m 帯の vote 比率中央値が 0.12〜0.15 でほぼ均一になり、「近傍だけ構造的に不利」というバイアスが解消された。

**残課題（未解決）**: `occ` の中央値自体が全レンジで 2〜8 と低い（ボクセルサイズ 0.1m に対しビーム密度が疎で、静止構造物でも同じ 10cm ボクセルに複数スキャンでヒットしにくいことが根本原因。移動物体とは無関係の構造的特性）。このため近傍〜中遠方は軒並み比率 0.12〜0.15 程度で頭打ちになり、閾値 0.3 に対してまだ多くが除去されたままである。これは `vote_occlusion_*` のチューニングでは解決できない別軸の話で、`vote_threshold_ratio` 自体の見直し、または投票ロジックの再設計（絶対 occ 数ベースの閾値など）が必要になる可能性がある。

**関連コード**: `step3_build_front_map.py` の `_range_image()` / `_arc_length_keys()` / `_pack_occ_keys()` / `_visibility_counts()`、`common.py` の `vote_occlusion_enabled` / `vote_occlusion_footprint_m` / `vote_occlusion_tol`。Step 4 は `step3.load_scan_manifest()` のみ使用し `build_voxel_map()`/`_visibility_counts()` は呼ばないため、本修正の影響範囲は Step 3 の front map 生成のみ。

**Step 4 への影響検証（2026-07-04）**: 新旧 front map を、それ以外の条件（`l2l_scan_stride=12` / `l2l_points_per_scan=1000`、undistorted scan、rtk_poses）を完全に揃えた状態で Step 4 に投入し比較した。

| lidar | rmse (旧 front map) | rmse (新 front map) | 差分 | inliers 旧→新 |
|---|---|---|---|---|
| right | 12.46cm (not-conv) | **10.86cm** (not-conv) | -1.6cm 改善 | 28185 → 39172 |
| left | 9.71cm (not-conv) | 10.57cm (not-conv) | +0.9cm 悪化 | 17657 → 38207 |
| rear | 11.37cm (not-conv) | 12.79cm (not-conv) | +1.4cm 悪化 | 55318 → 59221 |

結果は混在（right 改善、left/rear 微悪化）で、3台とも非収束のまま変わらない。inliers（ICP 対応点として実採用された点数）は3台とも大幅増加しており、近傍復活の効果自体は Step 4 に反映されている。rmse が一様に改善しなかったのは、復活した近傍点群が完全にクリーンな静止構造ではなく（8の字交差点付近など、occ 中央値の低さに由来するノイズをまだ含む領域がある）、点対平面残差にとって局所曲率が高い/やや雑音を含む対応点が増えた可能性がある。**結論**: Step 3 の近傍フィルタ修正だけでは Step 4 の rmse 10〜12cm 台の床は解消しない。front map のスミア自体への対策が別途必要（詳細は次節）。

---

## Front map スミア改善策の検討（2026-07-04・設計のみ、未実装）

Step 4 の rmse が 10〜12cm 台で頭打ちになる根本原因として、front map 自体の「スミア」（本来 1 点であるべき静止構造が、複数スキャンの位置合わせ誤差により空間的にぼやけて広がる現象）を疑っている。考えられる寄与要因と対策候補を整理する。

### スミアの想定要因

1. **RTK 内挿誤差の蓄積**: 各スキャンは `traj.interpolate()` で RTK 軌道から姿勢を内挿して world 変換している。RTK 自体の precision（[[auto-calib-bag-facts]] 記載の通り cov ~0.2m² が正常範囲）や内挿誤差が、各スキャンごとに数cm単位で異なる姿勢誤差を生み、同一の物理点が voxel 境界を跨いで複数の異なる 0.1m voxel に分散する。
2. **Deskew（歪み補正）精度**: Step 2 の per-point 時刻補正が RTK 内挿ベースのため、(1) と同じ誤差源に依存し、高速走行区間・急旋回区間ほど deskew 誤差が拡大しやすい。
3. **ボクセルサイズと centroid 平均化**: `map_voxel_size=0.1m` の centroid は、voxel 内に落ちた全観測の単純平均であり、上記 (1)(2) の誤差がランダムでなく系統的（例: 旋回区間で常に同方向にバイアス）である場合、centroid 自体が真の表面位置からずれる。
4. **drs_T_lidar 初期外挿の誤差**: Step 3 は `/tf_static` の初期値（または `--tf-override`）をそのまま使うため、これ自体に既知の誤差があると、真の外挿を推定する対象であるはずの Step 4 が「歪んだ地図」に対して最適化することになり、収束しても真値からずれる。

### 対策候補（優先度順、未実装）

1. **スミア量の直接定量化（最優先・低コスト）**: 現状「rmse 10-12cm はスミアが原因では」という仮説止まりで、スミアそのものを測定していない。`_VoxelAccumulator` は centroid だけでなく各 voxel 内の観測位置ばらつき（分散・最大偏差）を計算できる（`sum_xyz`/`cnt` に加えて `sum_xyz²` を蓄積すれば追加コストはほぼゼロ）。この voxel 内分散を可視化・集計し、(a) 分散が本当に rmse 床（10-12cm）と同オーダーか、(b) 8の字の直線区間 vs 旋回区間で分散が有意に変わるか（RTK/deskew 起因なら旋回区間で悪化するはず）を検証すべき。仮説検証が先で、対策の要否・優先度はここで決まる。
2. **旋回区間の重み低減 or 除外**: (1) で旋回区間の分散悪化が確認できれば、`map_scan_stride` 適用前にヨーレート閾値でスキャンを除外する、または蓄積時に旋回中のスキャンを低ウェイトにする対応が考えられる。Step 1 に `min_yaw_coverage_deg` 等の姿勢情報は既にあるため、必要な角速度情報の追加コストは小さい。
3. **Voxel centroid → ロバスト中心への変更**: 単純平均ではなく、median やロバスト推定（例: RANSAC 平面フィッティング後の平面上への射影）に変えることで、外れ値観測の影響を抑える。ただし (1) の分散が主に系統誤差（旋回区間バイアス等）由来であれば、ロバスト推定だけでは解決しない。
4. **map_voxel_size を小さくして deskew 誤差の平均化幅を削る**: 現在 0.1m。精度重視プリセットは既に 0.05m を使用しているが、スミアの空間スケールが 0.1m 相当なら voxel を細かくしてもスミアの「量」自体は変わらず、単に voxel 単位から漏れて隣接 voxel に分散するだけの可能性がある。(1) の分散測定と合わせて評価しないと有効性が判断できない。

**次のアクション**: 対策候補を選ぶ前に、(1) のスミア定量化を先に行うべきという結論。これは Step 3 に軽微な追加集計を加えるだけで実施可能で、以降の対応要否・優先順位を実データで判断できる。

### スミア定量化の実施結果（2026-07-04・仮説は否定）

`_VoxelAccumulator` に `sum_xyz²` の追加集計（`add_scan`/`centroids()`）を実装し、kept voxel（occ≥2）の RMS radial spread（voxel 内観測ばらつきの分散平方根、centroid からの実質的なブレ量）を算出できるようにした（`build_voxel_map()` の戻り値 `spread`、`plot_map()` にヒストグラム/topdown 可視化を追加、`_run()` で percentile を出力・meta.json に記録）。あわせて `build_voxel_map()` に `scan_filter` 引数（`callable(scan_index, scan_dict, t0_ns) -> bool`）を追加し、スキャン集合を任意条件で絞り込めるようにした。

front lidar の初期 `drs_T_lidar` はユーザー指示により、rosbag の `/tf_static` 値（`y=0.132`）ではなく、以前 GT 検証で展開した `y=0` 版を `--tf-override` で使用（`x=0.299, y=0.0, z=0.037`, 回転なし）。RTK 軌道の四元数からヨーレート [deg/s] を計算し、閾値 5 deg/s で「直進」「旋回」区間に分けて front map を別々に構築・比較した（503 scans, `map_scan_stride=3` 適用後の対象 scan 数ベース）。

| | 全体 | 直進 (yaw rate ≤ 5°/s) | 旋回 (> 5°/s) |
|---|---|---|---|
| scan数 | 503 | 269 | 234 |
| kept voxel(occ≥2) spread p10 [cm] | 2.67 | 2.56 | 2.63 |
| spread p50 [cm] | 4.33 | **4.20** | **4.23** |
| spread p90 [cm] | 5.04 | 5.01 | 5.00 |
| spread p99 [cm] | 5.64 | 5.62 | 5.60 |

**結論（要因1・2の仮説を否定）**: 直進区間と旋回区間でスミア量に有意差なし（p50 4.20cm vs 4.23cm）。RTK 内挿/deskew 誤差が旋回区間で悪化してスミアを生む、という仮説は支持されなかった。

**より重要な発見**: front map 自体のスミア（p50=4.3cm、p99でも5.6cm）は、Step 4 の rmse 床（10〜12cm）よりずっと小さい。front map の内部一貫性（同一静止構造への複数スキャンの位置合わせ）はむしろ良好であり、**front map のスミア単体では Step 4 の rmse 床を説明できない**。

**示唆される今後の方向性**（未着手）: rmse 床の主因は front map のスミアではなく、以下のいずれか（または複合）である可能性が高い:
- 他 LiDAR（right/left/rear）側の `drs_T_lidar` 初期値・deskew 精度（front だけでなく全 LiDAR で同様の検証が必要）
- Step 4 の点対平面残差が捉えている法線方向の誤差が、実際のシーン形状（縁石・植生・フェンス等、非平面構造）の cm 級テクスチャそのものを反映している（front map 側のノイズではなく、対象シーンの実際の凹凸）
- 法線推定パラメータ（`l2l_normal_radius_scale` 等）や `icp_multiscale_voxels` の粗いスケールでの推定品質
- Step 4 が真に収束していない（3 lidar とも `converged: false`）ことそのものが rmse 高止まりの説明である可能性（スミアや法線品質以前に、最適化自体が局所解/ライン探索失敗で止まっている）

対策候補 2〜4（旋回区間の重み低減、ロバスト中心化、voxel サイズ変更）は、いずれも「旋回起因のスミア」を前提にしていたため、この否定結果を受けて**優先度を下げる**。次に着手すべきは、他 LiDAR 側の同様のスミア検証、または Step 4 の非収束原因そのものの調査。

**方針判断（2026-07-04）**: right/left/rear は front と同クラスの LiDAR かつ同一 RTK 情報を使うため、同様のスミア検証をしても front と有意差は出ない可能性が高いと判断し、優先度を下げた。代わりに「実際のシーン非平面性」を先に検証する。

### 実際のシーン非平面性の検証（2026-07-04・仮説を支持する結果）

Step 4 の点対平面 ICP は、front map 側の法線を `build_target()`（`estimate_normals`, radius = `map_voxel_size × l2l_normal_radius_scale` = 0.1m×3.0 = **0.3m**, max_nn=30）で推定している。この半径内で対象面が真に平面なら残差は理論上ゼロに近いはずだが、実シーン（縁石・植生・フェンス等）が非平面なら、**完璧な位置合わせでも半径内の法線方向のばらつき（局所曲率）分だけ残差の下限が生じる**。この下限を直接測定した。

**手法**: front map（y=0 tf-override 使用、`vote_occlusion_*` 修正後、503 scans）を Step 4 と同じ `voxel_down_sample(0.1m)` で間引いた後（1,830,056 → 1,571,682 点）、ランダム5万点サンプルに対して半径 0.3m（Step 4 と同一設定）で近傍点を集め、共分散行列の最小固有値の平方根（法線方向の局所的な「厚み」＝理論上の残差下限）を算出した。

**結果（半径 0.3m、Step 4 と同一設定）**:

| percentile | 局所面厚み [cm] |
|---|---|
| p10 | 2.38 |
| p25 | 5.00 |
| **p50** | **6.91** |
| p75 | 8.11 |
| **p90** | **9.01** |
| p95 | 9.61 |
| p99 | 10.75 |
| mean | 6.34 |

高さ (z) 別に見ても地面付近〜高所まで大差なく（p50 6.4〜7.2cm）、特定の高さ帯（植生等）に極端に偏ってはいない。

**半径依存性の確認**（スケール依存＝真のテクスチャ、スケール非依存＝登録誤差由来のノイズ床、を切り分けるため半径 0.15m でも実施）: p50 が 6.91cm→3.59cm、p90 が 9.01cm→5.63cm と、**半径にほぼ比例して縮小**した。これは登録誤差やスミアのような固定的なノイズ床ではなく、**スケールに応じた真の局所形状（非平面テクスチャ）**であることを示す。

**結論**: 半径0.3m（Step 4 が実際に使う設定）での局所面厚み p50=6.9cm・p90=9.0cm・p99=10.8cm は、Step 4 の実測 rmse（10〜12cm、新 front map で right 10.86cm / left 10.57cm / rear 12.79cm）と**極めて近い**。front map のスミア（p50=4.3cm）よりもこちらの方が rmse 床への説明力が強く、**Step 4 の rmse 10〜12cm 台の床は、front map 側の誤差というより、シーン自体が `l2l_normal_radius_scale` の空間スケールで真に非平面であることに大きく起因している**と考えられる。

**示唆**: この結論が正しければ、rmse 自体をさらに下げるには front map 側の改善（スミア対策等）ではなく、(a) 法線推定半径を小さくする（テクスチャの影響を減らせるが、対応点の安定性とのトレードオフ）、(b) 点対平面ではなく点対点や別のロバスト対応方式を検討する、(c) そもそも rmse 10cm 前後を「この環境・このセンサー幾何では妥当な下限」として受け入れる、のいずれかが選択肢になる。いずれも未着手。

**方針決定（2026-07-04）**: (c) を採用し、rmse 10cm 前後をこの環境・センサー幾何での妥当な下限として受け入れる。front map 側の追加改善（法線半径縮小・点対点方式への変更等）はこれ以上追求しない。

### Step 4 結果妥当性の数値化: `overlay_gap_stats`（2026-07-04・パイプラインに統合済み）

Step 4 の結果は従来、`step4_{name}_overlay_topdown.png`（各 lidar 単独）と `step4_combined_overlay_topdown.png`（全 lidar 同時）という**目視のオーバーレイ画像**でのみ妥当性を確認していた。既存の `rmse`（`lidar_to_lidar_result.json`）はソルバー自身の Huber 重み付き点対平面残差で、ソルバーが使った対応点のサブサンプル上での値であり、オーバーレイ画像が実際に描画している点群そのものに対する指標ではない。

`step4_lidar_to_lidar.py` に `overlay_gap_stats(front_tree_xy, world_xy)` を追加した: オーバーレイ画像と**全く同じ点群**（`aligned_world_xy()` の出力）に対し、front map への生の 2D 最近傍距離を計算し、p50/p90/p99/max [cm] を返す。`_run()` 内で right/left/rear それぞれについて常時（`--no-viz` でも）計算され、`lidar_to_lidar_result.json` の各 lidar に `overlay_gap_cm` として保存される。**Step 4 はパイプライン（`run_pipeline.py`）から通常呼ばれる標準の処理なので、この数値化は既に本番パイプラインに統合済み。**

さらに、この統計を `step4_{name}_overlay_topdown.png`（該当 lidar 分のみ）と `step4_combined_overlay_topdown.png`（right/left/rear 全て）の左下に直接テキスト表示するようにした（`_gap_text()`/`_add_corner_text()`/`_combine_extra()`）。

**実測結果（front map 修正後、フル 503 scans）**:

| lidar | rmse | gap p50 | gap p90 | gap p99 | gap max |
|---|---|---|---|---|---|
| right | 10.48cm (not-conv) | 4.74cm | 45.54cm | 165.57cm | 288.61cm |
| left | 12.29cm (converged) | 5.12cm | 31.25cm | 132.46cm | 300.46cm |
| rear | 11.58cm (not-conv) | 4.12cm | 31.36cm | 107.80cm | 619.88cm |

中央値（p50）は 4〜5cm と rmse より引き締まっており、大半の点は良く整列している。ただし p90 で 31〜46cm、p99 で 108〜166cm、max は最大 620cm まで裾が伸びており、少数だが明確にズレている点が残っていることが分かる（オーバーレイ画像で目視される散在した外れ点に対応すると考えられる）。この裾の一部は、front map がカバーしていない領域（他 LiDAR の FOV 境界付近等）への最近傍探索によるもので、必ずしも較正誤差とは限らない点に注意が必要（**追跡はしないことを決定** — 較正誤差と断定できないため）。

### Step 0-4 最適化の締めくくり（2026-07-04 時点）

本セッションの Step 3 近傍データ復活・スミア検証・シーン非平面性検証・Step 4 妥当性数値化をもって、Step 0-4 の最適化・検証作業を一区切りとする。現時点の処理時間（同一 bag、フル 503 scans/lidar）:

| Step | 時間 | 備考 |
|---|---|---|
| Step 0+1 | 約155.7s | 本セッションでは未変更（合算値、個別未計測） |
| Step 2 | 272.1s | 変更なし（単一読み込み＋lidar別並列化、既述） |
| Step 3 (front) | 約246s | 遮蔽考慮 visibility モデル＋スミア定量化の追加により 167.5s→246s（+79s）。近傍データ復活という正しさとのトレードオフとして許容 |
| Step 4 | 約160〜170s | `overlay_gap_stats` 追加後も 176.8s から実質変化なし（cKDTree クエリのオーバーヘッドは無視できる） |
| **合計** | **約842s** | 元の未最適化ベースライン（2710.6s）比で**約69%短縮** |

Step 3 は今回の修正で若干増加したが、Step 4 側の大幅な高速化（1549.6s→160s台）がそれを大きく上回るため、パイプライン全体としては引き続き大幅な短縮を維持している。

---

## 実装ファイル構成

```
tools/auto-calib/
├── docs/
│   └── architecture.md          ← 本ドキュメント
├── common.py                    # TF補間・座標変換・mcap読み込み・tf_override 共通処理 ✅
├── viz.py                       # Step0-4 共通ログ/可視化ヘルパー（tee_log, topdown_scatter 等）✅
├── step0_validate.py            # データ検証・品質ゲート（FAIL で非ゼロ終了）✅
├── step1_extract_rtk.py         # RTK軌道を npy に保存 ✅
├── step2_undistort_lidar.py     # time_stamp(ns,相対) で点群歪み補正 ✅
├── step3_build_front_map.py     # Front LiDAR マップ構築（投票フィルタ・--tf-override）✅
├── step4_lidar_to_lidar.py      # 結合最適化 drs→lidar_{rear,left,right}（項A+項B）✅ GT検証済
├── step5_cam_to_lidar.py        # camera0-11 共通（通常+魚眼）✅ v1(DT最小化)棄却 → v2(scan-to-image遮蔽エッジ)へ改修予定
├── step5_m0_check.py            # Step5 v2 M0: 識別可能性スイープ（scan-to-image、遮蔽/平面エッジ+MI）✅ 実装・検証済み
├── step5_v3_cam_cloud.py        # Step5 v3 part1: KLT追跡+多視点三角測量（camera cloud構築）✅ 実装・メカニズム検証済み（fix解での品質検証待ち）
├── step5_v3_register.py         # Step5 v3 part2: LiDARマップへ登録（reprojection GN, Step4流用）✅ 実装・メカニズム検証済み（fix解での品質検証待ち）
├── step6_joint_optimize.py      # 全 TF 同時精密化 (任意) ⬜ 未実装
├── run_pipeline.py              # 全ステップ実行（Step 0 が FAIL なら即中断、各Stepの所要時間を表示）✅ Step 0-4 実装済み
└── verify_result.py             # 投影可視化・残差レポート・合否判定 ⬜ 未実装
```
> Step 5 は通常・魚眼を **1ファイル `step5_cam_to_lidar.py`** に統合（当初案の `step5b_fisheye_to_lidar.py` は投影モデル分岐で吸収したため不要）。

---

## ログ出力・可視化（Step 0-4・2026-07-04 追加）

問題発生時にオフラインで検証できるよう、Step 0-4 に共通のログ/テキスト出力と可視化を追加。共通ヘルパーは `viz.py`（matplotlib Agg、`viz_enabled`/`viz_dpi`/`viz_max_points`/`viz_max_scans_overlay` を `common.py` の DEFAULT_CONFIG に追加）。各ステップ `main()` は `viz.tee_log(out_dir, "stepN_log.txt")` で標準出力全体（Step4 の反復ログ等、途中の print も含む）をファイルにも保存する。`--no-viz` で PNG 生成のみスキップ可能（ログは常時保存）。

| Step | 追加テキスト出力 | 追加可視化 (PNG) |
|---|---|---|
| 0 (validate) | `--out-dir` 追加（`--report` 未指定時はここに report を書く）、`step0_log.txt` | `step0_trajectory.png`（軌跡・速度で色分け、速度reject点を赤Xで表示）、`step0_speed.png`（速度-時間・閾値線） |
| 1 (extract_rtk) | dt パーセンタイル（median/p95/max）を `rtk_meta.json`・stdout に追加、`step1_log.txt` | `step1_trajectory.png`（軌跡・時間で色分け、start/end マーカー）、`step1_z_yaw.png`、`step1_yaw.png` |
| 2 (undistort_lidar) | manifest.json の各 scan に `disp_max_m`/`disp_mean_m` を追加記録（従来は捨てていた）、disp パーセンタイルを stdout に追加、`step2_log.txt` | `step2_<lidar>_disp_timeseries.png`、`step2_<lidar>_disp_hist.png`、`step2_<lidar>_sample_{first,worst}_scan*.png`（deskew後の点群を**変位量でヒートマップ着色**；raw/deskewed の単純オーバーレイは cm 級の差が数十m スケールで見えず不採用） |
| 3 (build_front_map) | vote-ratio パーセンタイル（p10/p50/p90）を stdout に追加、`step3_<lidar>_log.txt` | `step3_<lidar>_map_topdown.png`（高さで色分け＋カラーバー）、`step3_<lidar>_vote_hist.png`（閾値線付き）、`step3_<lidar>_removed_overlay.png`（vote filter で除去された点を赤表示） |
| 4 (lidar_to_lidar) | 反復ごとの収束ログを `step4_log.txt` に保存（従来 stdout のみ）、`optimize()` に `history`（voxel,it,rmse）を追加 | `step4_<lidar>_convergence.png`（coarse→fine の scale 切替を破線で表示）、`step4_<lidar>_residual_hist.png`（init vs final の point-to-plane 残差分布）、`step4_<lidar>_overlay_topdown.png`（front map への位置合わせ結果をtop-downで目視確認、lidar毎に1枚）、`step4_combined_overlay_topdown.png`（front map + right/left/rear を色分けして1枚に重畳、全 lidar が同一構造物に整合しているかを一目で確認）、`step4_<lidar>_axis_curv.png`（x/y/z の Hessian curvature＝観測性の直感的表示） |

**overlay プロットの表示範囲・点サイズ（2026-07-04 調整）**: 当初は front map 全域（270×340m）を等サイズの不透明点で描画していたため、cm 級のアラインメント誤差が視認不能（数百mスケールでは1px未満）で、密な点群が塗り潰されたブロブに見える問題があった。対策として (1) `viz_point_size`/`viz_point_alpha`（`common.py` DEFAULT_CONFIG、既定 0.15pt²/alpha 0.4）でマーカーを縮小・半透明化、(2) overlay 系プロット（`step4_<lidar>_overlay_topdown.png`・`step4_combined_overlay_topdown.png`）は重畳対象の点群のバウンディングボックス（+5m マージン）に自動クロップ（`_bbox_crop`/`_crop_xy`）— front map 全域ではなく実際に lidar データが存在する領域のみ表示することで、個々の壁面・エッジの一致/不一致が視認できるようにした。

実データ（`HRdqo3pf_2026-06-10T15-22-51.mcap`、前回セッションの scratch 出力）で Step 0-4 全て動作確認済み。

---

## 実装状況・検証サマリ（2026-07-03 時点）

bag: `HRdqo3pf_2026-06-10T15-22-51.mcap`（8の字・全区間）で検証。

| Step | 実装 | 検証結果 |
|---|---|---|
| 0–2 | ✅ | 品質ゲート/RTK抽出/de-skew（既存・動作確認済） |
| 3 Front マップ | ✅ | 地面が z≈−1.97m に集積（＝base→drs z=1.97 と一致）で姿勢変換の正しさを実証。visibility 正規化投票フィルタ。`--tf-override` |
| 4 LiDAR-LiDAR | ✅ | **結合最適化（項A LiDAR間重畳＋項B マップ・項別正規化）**。GT 照合で **right 2.7cm/0.37° 再現**、rear 9.1cm。X/Y 観測性獲得（±15cm 摂動→0.0cm）。left は roll 不可観測でデータ制約（後述） |
| 5 Camera-LiDAR | 🔶 v2/v3 併存・固定ルーティング決定済み（fix 解での品質検証待ち） | v1（DT最小化）棄却。v2 M0 で **float 解8の字では front のみ識別可（3/6）、他はシーン内容で NO-GO** と実証。→ v3（RTK fix 前提・全軌道蓄積カメラ点群→LiDAR登録）を実装、5カメラ横断検証で **広FOV・魚眼ほど幾何学的条件が改善**（M0とは逆傾向＝相補的）も**摂動回復は全カメラ未達**。**2026-07-04: camera0,4→v2／それ以外→v3 の固定ルーティングを決定**（camera4 は M0 不合格だが全区間データで再検証予定）。詳細は Step 5 v3 節・カメラ手法ルーティング節参照 |
| 6 / run_pipeline / verify | ⬜ | 未実装。**2026-07-04: Camera-Cameraクロスビュー残差（front={0,1,8}等の入れ子FOV重畳を利用、Step4項Aと同型）を第3残差項として設計に追加**（Step5 v4として独立させず Step6 に統合）。Step 6 は各カメラの Step5（v2/v3）コストが単体で機能することが前提 |

**本セッションで確立した主要知見**:
1. **バッグの `/tf_static` front が誤り**（drs→lidar_front の y=0.132 ≠ 受け入れ値 0.000, 13cm）。固定基準を誤ると全 LiDAR がバイアス → **Step 3/4/5 に同じ `--tf-override` で正しい front を渡す**。
2. **Step 4 の設計変更**: 素朴な「other_map 蓄積＋単一 ICP」は 8の字で並進復元不可、かつマップ照合単独では地面優勢で **X/Y が不可観測**。→ **LiDAR間同時重畳（項A・RTK 免疫）で X/Y を拘束**する結合最適化に変更。項別正規化が必須（項B の多数点が項A を押し潰すため）。
3. **RTK 〜0.5m は LiDAR-LiDAR を 0.5m 級には劣化させない**（剛体相対校正・項A は 50ms 相対のみ使用で RTK 免疫、項B は相対平滑性 ~8cm 依存）。right 2.7cm が証拠。
4. **left の限界**: 真 roll=−1.45°（他 LiDAR より突出）が本8の字で不可観測 → roll 誤差が z に結合し ~15cm。項B改良（地面除外/近距離マップ/重み）では直らずデータ/幾何の制約。改善は left 視野の構造豊富な走行 / 結合18変数同時最適化 / Step 6。
5. **Step 5 v1 の棄却**: 雑然とした 8MP 市街画像では DT 最小化コストが外部パラに平坦（front 系）またはエイリアス（rear/side 系＝偽の大域最小へ変位）。
   面変化エッジ点化・強 Canny・coarse-to-fine 多段ぼかしを**全て実装・検証したが失敗**（coarse-to-fine で偽最小へ滑らかに収束＝大域最小自体の変位を実証）。
   さらにマップ投影連鎖は Step 4 誤差（rear ~24cm）と RTK 誤差を継承する構造的問題あり。
   → **v2 再設計**（①scan-to-image 直接投影で Step4/RTK から切離し、②レンジ画像の遮蔽境界エッジ、③勾配強度相関の最大化＋MI クロスチェック、④M0 1D スイープ go/no-go＋split-half ゲート）。詳細は Step 5a 節。

## 未解決事項 / 注意点

- **⚠ INS 解の有効性検証（最優先・バッグごとに要確認）**: INS 解の有効性はバッグにより大きく異なることを実測で確認:
  - 無効例 `HRdqo3pf_2026-06-25T16-34-04_pc.mcap`（13秒）: 位置 482m/20ms・速度 1200m/s と**完全に無効**。
    `pose.covariance` は一定プレースホルダ、`NavSatFix.status` は常に `0=FIX` で、**フィールド値では異常を検知できなかった**。
  - 有効例 `HRdqo3pf_2026-06-25T17-37-26_pc_replaced.mcap`（386秒・通常走行）: 位置は局所座標で連続、速度 median 6.7km/h・max 19.7km/h、
    covariance も実値（1e-4〜0.097 m²）で**健全**。RTK 検証・Step 1/2 実装の入力として利用可能。
    （フットプリント 240×327m の通常走行データで、駐車場30m想定とは異なるが RTK と時刻補正の検証には十分）
  - → **各バッグで Step 1 前に軌道の物理妥当性を必ず確認**。ゲートは covariance ではなく速度・連続性で行う（`rtk_max_speed_mps` 等、設計反映済み）。
- **座標系・時刻フィールド（実バッグで確認・解決済み）**:
  - `map` はバッグ依存（生=ECEF 1e6m / `_replaced`=局所メートル）→ 大きさ検出して局所原点化＋float64。
  - `drs_base_link → oxts_link` は X軸180°回転で identity ではない。
  - キャリブ入力 `seyond_points` の点時刻は **`time_stamp` (uint32, ns, スキャン先頭相対)**。生 `seyond_packets`(SeyondScan) の
    不透明バイト列を復号せずとも、この `time_stamp` で undistortion 可能。フィールド名/単位はバッグで揺れるため値域から自動判定する。 **→ 設計反映済み**
- **camera8-11 初期値**: 魚眼カメラは LiDAR から離れた車体ボディに取り付けられているため、オフセットが 1〜2.5m になるのは正常。
  初期値誤差は 10cm 未満を前提とし、`tune_extrinsic.py` による手動粗合わせは不要。Step 5b は tf_static の初期値をそのまま使って最適化を開始する。 **→ 解決済み**
- **8の字走行データ（取得・検証済み）**: キャリブ専用 rosbag を 2 本取得し Step 0 相当の検証を実施、いずれも健全:
  - `HRdqo3pf_2026-06-10T15-22-51.mcap`（単一・150.7s）: フットプリント 25.4×27.4m、8の字2ローブ、速度 max 5.0m/s。
  - `HRdqo3pf_2026-06-10T15-20-20+0900/`（4分割・117.7s）: フットプリント 22.4×26.2m、8の字2ローブ、速度 max 1.5m/s、分割跨ぎギャップ無し。
  - 両者とも: map=局所座標、tf_static 31変換完備（drs→4LiDAR・→12カメラ・drs→oxts=180°X）、LiDAR時刻=`t_us`(μs)+`timestamp`(ns) span≈92.4ms。
  - **RTK covariance は両者とも ~0.2 m²（std≈0.46m）** で、基準バッグの cm 級（1e-4〜0.097 m²）より粗い。これが図8データの常態のため、
    Step 0 では **covariance は警告のみ**とし物理妥当性を主ゲートとする（到達精度の上限になり得る点は認識）。
  - **→ 解決済み（Step 0 品質ゲートとして設計反映済み）**
- **LiDAR 間重畳領域**: 隣接 LiDAR の水平 FOV 重畳が十分かどうかは走行データ取得後に確認が必要。
  - 30m 四方の駐車場環境では、8 の字走行中に全 4 LiDAR が周囲の静止構造物を異なる角度から捉えるため、
    同一フレームの同時重畳よりも「蓄積マップ間の共通構造物」が ICP の対応点になる。駐車車両・建物が十分あれば重畳量は確保できる見込み。
  - **→ データ取得後に蓄積マップを可視化して確認**
- **動体**: 車数台（静止）・人数名（移動）が存在する。
  - 静止した駐車車両はマップノイズにならない（むしろ特徴量として有用）。
  - 移動する人は 8 の字走行中に同じ位置に複数回現れないため、**ボクセル投票フィルタ**で除去する。
    「**そのボクセルが FOV/レンジ内に入っていたスキャン数**のうち N % 以上で占有されたボクセルのみ残す」（N = 30〜50 % を目安）。
    分母を「全スキャン数」にしないこと（長距離走行時に静止構造物まで消える）。
  - **→ Step 3 / Step 4 のマップ構築にボクセル投票フィルタを実装する（設計に組み込み済み）**
