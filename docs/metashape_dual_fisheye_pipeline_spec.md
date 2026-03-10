# Metashape内完結型 デュアル魚眼動画 前処理・マスク・アライメント実装仕様書

## 1. 目的

Metashape 上で Python スクリプトを実行し、以下を一連で処理する。

1. OSV コンテナ（内部は 2 video stream: front / back）を入力
2. ffprobe / ffmpeg を用いて front / back の魚眼フレームを抽出
3. OpenCV によりブレ判定を実施
4. 判定条件 `front または back のどちらかが閾値以上` を満たす時刻のみ、front/back の両方を採用
5. YOLO 等で不要領域のマスク画像を生成
6. Metashape に front/back をマルチカメラシステムとして読み込み
7. マスク適用後にアライメントを実施
8. アライメント結果から近接・冗長画像を無効化
9. 必要に応じて再アライメント

本仕様は、**Metashape 内の Python をオーケストレータ**として使用し、外部処理は `subprocess` で `ffprobe` / `ffmpeg` を呼び出す構成を前提とする。

---

## 2. 前提条件

### 2.1 動画仕様

- 入力形式: OSV（内部は2-stream video）
- 動画構造: **2Stream 型**
- Input container rule: use a `.osv` container that contains at least two usable video streams for the front/back pair.
- `front_stream_index` and `back_stream_index` identify the two video streams to extract with `ffmpeg -map 0:v:<index>`.
- Additional video streams may exist in the same `.osv`; they must be logged and ignored unless explicitly selected.
  - `0:v:0` = front または back
  - `0:v:1` = back または front
- 各ストリームは魚眼動画
- front / back の順序は設定値または初回確認で確定する
- `.osv` は ffprobe / ffmpeg にそのまま渡す入力コンテナとして扱う

### 2.2 実行環境

- Metashape Professional
- Metashape 内蔵 Python 3.8 系
- 同一環境から以下を呼び出せること
  - ffprobe
  - ffmpeg
  - OpenCV
  - YOLO 推論環境
- GPU は任意だが、YOLO 使用時は利用可能な構成を推奨

### 2.3 処理方針

- 動画抽出は ffmpeg を使用する
- 画質評価は Python 側のブレ判定 + Metashape の `analyzeImages()` の二段階
- 採用判定は **ペア維持優先** とする
- 画像削減は前処理で過剰に行わず、アライメント後に姿勢情報を用いて整理する

---

## 3. システム全体フロー

```text
[入力OSV]
   ↓
[ffprobeで2ストリーム確認]
   ↓
[ffmpegでfront/back全候補抽出]
   ↓
[OpenCVでブレ判定]
   ↓
[片方でもOKならfront/back両方採用]
   ↓
[YOLOでmask生成]
   ↓
[Metashape chunk作成]
   ↓
[front/backをMultiplaneLayoutで読込]
   ↓
[mask適用]
   ↓
[analyzeImages(filter_mask=True)]
   ↓
[低品質画像の無効化]
   ↓
[matchPhotos → alignCameras]
   ↓
[近接・冗長画像の無効化]
   ↓
[必要なら再アライメント]
```

---

## 4. ディレクトリ仕様

```text
project_root/
├─ input/
│  └─ source.osv
├─ work/
│  ├─ extracted/
│  │  ├─ front_raw/
│  │  └─ back_raw/
│  ├─ selected/
│  │  ├─ images/
│  │  │  ├─ front/
│  │  │  └─ back/
│  │  └─ masks/
│  │     ├─ front/
│  │     └─ back/
│  ├─ logs/
│  │  ├─ ffprobe.json
│  │  ├─ frame_quality.csv
│  │  ├─ mask_summary.csv
│  │  ├─ metashape_quality.csv
│  │  └─ overlap_reduction.csv
│  └─ temp/
├─ project/
│  └─ dual_fisheye_project.psx
└─ scripts/
   └─ metashape_dual_fisheye_pipeline.py
```

### 4.1 命名規則

- front画像: `F_000001.jpg`
- back画像: `B_000001.jpg`
- frontマスク: `F_000001.png`
- backマスク: `B_000001.png`
- フレームIDは同一時刻で front/back 共通番号とする

---

## 5. Metashapeメニュー構成

Metashape 起動後に以下メニューを追加する。

- `Custom/DualFisheye/00 GUIを開く`
- `Custom/DualFisheye/01 フルパイプライン実行`
- `Custom/DualFisheye/02 ストリーム抽出`
- `Custom/DualFisheye/03 フレーム選別`
- `Custom/DualFisheye/04 マスク生成`
- `Custom/DualFisheye/05 Metashapeへ読込`
- `Custom/DualFisheye/06 アライメント`
- `Custom/DualFisheye/07 冗長画像削減`
- `Custom/DualFisheye/08 ログ出力`

### 5.1 推奨方針

初期開発では以下の2段階を推奨する。

- Stage 1: `Run Full Pipeline` のみ実装
- Stage 2: 各処理を個別メニュー化してデバッグしやすくする

---

## 6. 設定パラメータ仕様

```python
CONFIG = {
    "input_mp4": "",
    "work_root": "",
    "front_stream_index": 0,
    "back_stream_index": 1,
    "extract_every_n_frames": 1,
    "jpeg_quality": 2,
    "blur_method": "laplacian_center70",
    "blur_threshold_front": 60.0,
    "blur_threshold_back": 60.0,
    "fft_blur_threshold": None,
    "keep_rule": "either_side_ok_keep_both",
    "mask_model": "yolo",
    "mask_classes": ["person", "car", "truck", "bus", "motorbike"],
    "mask_dilate_px": 8,
    "mask_polarity": "target_black",
    "opencv_backend": "auto",
    "prefer_cuda": True,
    "cuda_device_index": 0,
    "cuda_allow_fallback": True,
    "cuda_log_device_info": True,
    "yolo_device_mode": "auto",
    "prefer_yolo_cuda": True,
    "yolo_allow_fallback": True,
    "yolo_device_index": 0,
    "save_backend_report": True,
    "metashape_image_quality_threshold": 0.5,
    "match_downscale": 1,
    "keypoint_limit": 40000,
    "tiepoint_limit": 10000,
    "filter_stationary_points": True,
    "overlap_target": 3,
    "camera_distance_threshold": 0.15,
    "camera_angle_threshold_deg": 5.0,
    "chunk_size_limit": 250,
    "chunk_overlap_ratio": 0.25,
    "enable_rig_reference": False,
    "rig_relative_location": [0.0, 0.0, 0.0],
    "rig_relative_rotation": [0.0, 0.0, 0.0]
}
```

### 6.1 閾値方針

- 前処理ブレ判定閾値は環境に応じて調整する
- トンネル・暗所では閾値をやや緩めに設定する
- Metashape の `Image/Quality` は初期値として `0.5` を採用
- 近接画像無効化閾値は対象距離・歩行速度・要求精度に応じて調整する

---

## 7. モジュール構成

```text
metashape_dual_fisheye_pipeline.py
├─ class PipelineConfig
├─ class FFmpegExtractor
├─ class BlurEvaluator
├─ class MaskGenerator
├─ class MetashapeImporter
├─ class MetashapeAligner
├─ class OverlapReducer
├─ class LogWriter
└─ menu entry functions
```

---

## 8. 各モジュール仕様

## 8.1 PipelineConfig

### 役割
- 設定値の保持
- GUI入力または固定設定の反映
- パス検証

### 主な関数

```python
class PipelineConfig:
    def validate(self): ...
    def ensure_directories(self): ...
```

---

## 8.2 FFmpegExtractor

### 役割
- ffprobe により動画のストリーム情報を取得
- ffmpeg により front/back を全候補抽出

### 入力
- input OSV パス（内部キー名 `input_mp4` は互換維持のため許容）
- front/back の stream index
- `probe_streams()` must validate that the selected `front_stream_index` / `back_stream_index` exist and point to usable video streams.
- `probe_streams()` must not fail only because the `.osv` contains three or more video streams.

### 出力
- `work/logs/ffprobe.json`
- `work/extracted/front_raw/*.jpg`
- `work/extracted/back_raw/*.jpg`

### 実装方針

1. `ffprobe -show_streams -of json` を実行
2. video stream が2本存在するか確認
3. `-map 0:v:<index>` で front/back を個別抽出
4. front/back の抽出枚数が一致するか確認
5. 不一致時はエラーまたは警告とする

### コマンド例

```bash
ffprobe -v error -show_streams -of json input.osv
ffmpeg -i input.osv -map 0:v:0 -vsync 0 work/extracted/front_raw/F_%06d.jpg
ffmpeg -i input.osv -map 0:v:1 -vsync 0 work/extracted/back_raw/B_%06d.jpg
```

`probe_streams()` logs the detected video streams first, then selects the configured front/back pair, and records any ignored stream indices.

### 関数仕様

```python
class FFmpegExtractor:
    def probe_streams(self, mp4_path) -> dict: ...
    def extract_front_stream(self, mp4_path, out_dir, stream_index): ...
    def extract_back_stream(self, mp4_path, out_dir, stream_index): ...
    def verify_frame_counts(self, front_dir, back_dir): ...
```

### 例外処理

- ffprobe 実行失敗
- ffmpeg 実行失敗
- video stream が2本未満
- 抽出枚数不一致

---

## 8.3 BlurEvaluator

### 役割
- front/back 画像のブレ評価
- ペア採用判定
- 採用画像の selected ディレクトリへのコピー

### 判定ルール

- 各時刻の front/back についてブレ指標を算出
- `front >= th_front or back >= th_back` の場合、**front/back 両方を採用**
- それ以外は不採用

### ブレ評価手法

- 基本: 中央70%円領域のラプラシアン分散
- 任意: FFT ベース補助判定

### 出力
- `work/selected/images/front/*.jpg`
- `work/selected/images/back/*.jpg`
- `work/logs/frame_quality.csv`

### CSV仕様

```csv
frame_id,front_path,back_path,front_score,back_score,keep_pair,better_side
1,F_000001.jpg,B_000001.jpg,72.4,31.8,1,front
2,F_000002.jpg,B_000002.jpg,18.1,22.3,0,back
```

### 関数仕様

```python
class BlurEvaluator:
    def compute_center70_mask(self, image): ...
    def laplacian_score(self, image): ...
    def fft_blur_score(self, image): ...
    def evaluate_pair(self, front_path, back_path) -> dict: ...
    def select_pairs(self, front_dir, back_dir, out_front_dir, out_back_dir): ...
```

---

## 8.4 MaskGenerator

### 役割
- 採用済み画像に対してマスク生成
- 必要に応じて dilation を適用

### 入力
- `work/selected/images/front/*.jpg`
- `work/selected/images/back/*.jpg`

### 出力
- `work/selected/masks/front/*.png`
- `work/selected/masks/back/*.png`
- `work/logs/mask_summary.csv`

### 実装方針

- 初期版は YOLO ベースで人物・車両等を除外対象とする
- マスクは二値 PNG（`uint8` の `0 / 255`）とする
- 検出対象（人物・車両など）は黒 `0`、それ以外は白 `255` とする
- 必要に応じて二値化後に dilation を適用し、最終保存前に必ず `黒=除外対象 / 白=有効領域` へ正規化する
- 設定値 `mask_polarity` の既定値は `"target_black"` とする
- 将来拡張で SAM 系との併用を許容

### 関数仕様

```python
class MaskGenerator:
    def load_model(self): ...
    def infer_mask(self, image_path): ...
    def dilate_mask(self, mask, px): ...
    def save_mask(self, mask, out_path): ...
    def process_directory(self, image_dir, out_mask_dir): ...
```

---

## 8.5 MetashapeImporter

### 役割
- chunk 作成
- front/back をマルチカメラとして読み込み
- sensor type 設定
- マスク適用

### 読込方式

- `Metashape.MultiplaneLayout` を使用
- `filenames` と `filegroups` は **生成関数で構築**する
- front/back は時刻単位でペア管理する
- API 文面だけで最終的な front/back 配列順を断定しない

### 重要注意

実装時は、Metashape の `addPhotos()` に渡す `filenames` と `filegroups` の組を **4〜8枚程度の小規模データで current build 上で検証**すること。マルチプレーン入力は並び順や group 構成の誤りで期待しないセンサー割り当てになる可能性がある。

実装コードには少なくとも次の TODO を残すこと。

```python
# TODO: validate filenames/filegroups layout on current Metashape build
```

### 関数仕様

```python
class MetashapeImporter:
    def create_or_get_chunk(self, doc, name="Chunk 1"): ...
    def build_filename_sequence(self, front_dir, back_dir): ...
    def build_filegroups(self, pair_count): ...
    def import_multiplane_images(self, chunk, filenames, filegroups): ...
    def set_sensor_types(self, chunk): ...
    def apply_rig_reference(self, chunk, config): ...
    def apply_masks_from_disk(self, chunk, mask_front_dir, mask_back_dir): ...
```

### センサー設定

- front/back sensor の `type = Metashape.Sensor.Type.Fisheye`
- 必要に応じてスレーブセンサーに相対位置・回転を設定

### マスク適用方針

初期版は **`Metashape.Mask()` を使った camera 単位の個別適用** を主方式とする。

1. ディスク上の PNG を camera ごとに読み込む
2. `Metashape.Mask()` を生成する
3. `mask.load(path)` または `mask.setImage(...)` を使う
4. `camera.mask = mask` で割り当てる

`importMasks` / task ベース一括取込は optional とし、small sample で current build を検証後に採用可否を決める。古い記事や古いサンプルの記法はそのまま使わない。

---

## 8.6 MetashapeAligner

### 役割
- 画像品質評価
- 低品質画像の無効化
- マッチング
- アライメント

### 実装方針

1. `analyzeImages(filter_mask=True)` を実行
2. `Image/Quality < threshold` の camera を `enabled=False`
3. `matchPhotos()` を現行引数名で実行
4. `alignCameras()` を実行
5. 再実行時は `matchPhotos(reset_matches=True, ...)` と `alignCameras(reset_alignment=True, ...)` を使える構造にする

### 関数仕様

```python
class MetashapeAligner:
    def analyze_image_quality(self, chunk): ...
    def disable_low_quality_cameras(self, chunk, threshold=0.5): ...
    def export_quality_log(self, chunk, csv_path): ...
    def match_photos(self, chunk, config): ...
    def align_cameras(self, chunk): ...
    def realign_after_cleanup(self, chunk, config): ...
```

### matchPhotos 推奨値

現行 2.3.0 系で使う引数名を前提とする。

- `downscale`
- `generic_preselection`
- `reference_preselection`
- `filter_mask=True`
- `mask_tiepoints=True`
- `filter_stationary_points=True`
- `keep_keypoints`
- `keypoint_limit`
- `tiepoint_limit`
- `reset_matches`

古い引数名は使わない。

---

## 8.7 OverlapReducer

### 役割
- アライメント後の camera pose を用いて近接・冗長画像を無効化

### 基本方針

近接判定は**距離のみでなく、回転差と品質値も考慮**する。

### 判定ルール例

隣接する有効カメラペアについて、以下を満たす場合は冗長候補とする。

- 並進距離 < `camera_distance_threshold`
- 回転差 < `camera_angle_threshold_deg`

冗長候補同士では以下優先で残す。

1. `Image/Quality` が高い
2. 前処理ブレスコアが高い
3. front/back の整合性が高い

### 出力
- `work/logs/overlap_reduction.csv`

### 関数仕様

```python
class OverlapReducer:
    def get_enabled_cameras(self, chunk): ...
    def camera_center(self, camera): ...
    def camera_rotation(self, camera): ...
    def distance_between(self, cam_a, cam_b): ...
    def angle_between(self, cam_a, cam_b): ...
    def disable_redundant_cameras(self, chunk, config): ...
    def run_reduce_overlap_builtin(self, chunk, overlap=3): ...
```

### 推奨手順

- 初期版は **自前ロジックを主** とする
- 追加検証として `reduceOverlap()` も試験的に評価する
- 無効化後は必要に応じて `matchPhotos(reset_matches=True)` + `alignCameras(reset_alignment=True)` を実施する

---

## 8.8 LogWriter

### 役割
- CSV / JSON ログの出力
- 実行サマリ出力
- エラー内容保存

### 関数仕様

```python
class LogWriter:
    def write_csv(self, path, rows, headers): ...
    def write_json(self, path, obj): ...
    def write_summary(self, path, summary): ...
```

---

## 9. メイン処理仕様

```python
def run_full_pipeline(config):
    config.validate()
    config.ensure_directories()

    extractor = FFmpegExtractor(config)
    evaluator = BlurEvaluator(config)
    masker = MaskGenerator(config)
    importer = MetashapeImporter(config)
    aligner = MetashapeAligner(config)
    reducer = OverlapReducer(config)

    probe_info = extractor.probe_streams(config.input_mp4)
    extractor.extract_front_stream(...)
    extractor.extract_back_stream(...)
    extractor.verify_frame_counts(...)

    evaluator.select_pairs(...)
    masker.process_directory(... front ...)
    masker.process_directory(... back ...)

    doc = Metashape.app.document
    chunk = importer.create_or_get_chunk(doc)
    filenames = importer.build_filename_sequence(...)
    filegroups = importer.build_filegroups(pair_count=...)
    importer.import_multiplane_images(chunk, filenames, filegroups)
    importer.set_sensor_types(chunk)
    importer.apply_rig_reference(chunk, config)
    importer.apply_masks_from_disk(chunk, ...)

    aligner.analyze_image_quality(chunk)
    aligner.disable_low_quality_cameras(chunk, config.metashape_image_quality_threshold)
    aligner.match_photos(chunk, config)
    aligner.align_cameras(chunk)

    reducer.disable_redundant_cameras(chunk, config)

    if config.overlap_target is not None:
        reducer.run_reduce_overlap_builtin(chunk, config.overlap_target)

    aligner.realign_after_cleanup(chunk, config)
    doc.save()
```

---

## 10. GUI / 入力仕様

初期版は簡易ダイアログまたは設定ファイルで以下を指定する。

- 入力 OSV
- 作業フォルダ
- front stream index
- back stream index
- ブレ閾値
- YOLOモデルパス
- OpenCV 実行方式
- YOLO 実行方式
- Metashape 品質閾値
- 近接距離閾値
- 回転差閾値

GUI / 設定ファイルの入力要件:

- `入力OSV` は未選択状態を許容し、固定ダミーパスを初期値にしない
- Browse または手入力で設定した `.osv` は、実行前だけでなく GUI 内部 config にも同期する
- stale config に旧 `.mp4` が残っていても GUI 起動は継続し、invalid 状態として警告表示する
- If probe results disagree with the configured stream indices, the GUI must surface a clear warning or error message.
- If the `.osv` contains three or more video streams, the GUI must continue when the configured front/back indices are valid.
- GUI のユーザー向け文言は日本語を優先し、OpenCV / YOLO / Metashape GPU 状態を個別表示する
- `save_backend_report=True` の場合は `opencv_backend_report.json`、`yolo_backend_report.json`、`metashape_gpu_report.json`、`gpu_summary_report.json` を出力する

### 将来拡張

- Metashape 内の Qt UI ダイアログ
- 前処理ログの表表示
- 採用 / 不採用フレーム枚数の可視化
- 前処理ブレ分布ヒストグラム表示

---

## 11. エラー処理仕様

### 11.1 停止すべきエラー

- ffprobe / ffmpeg が見つからない
- stream 数が2未満
- 抽出画像枚数が一致しない
- YOLOモデル読込失敗
- selected 画像数が0
- Metashape への読込失敗
- アライメント後の有効 camera が極端に少ない

### 11.2 警告に留めるもの

- 一部マスク生成失敗
- 一部画像で `Image/Quality` 未取得
- overlap reduction 後の残枚数が想定より多い

### 11.3 ログ出力

- 標準出力
- `work/logs/error.log`
- Metashape Console

---

## 12. テスト仕様

## 12.1 単体テスト

- `probe_streams()` が2ストリームを正しく認識する
- `extract_front_stream()` / `extract_back_stream()` で画像枚数が一致する
- `evaluate_pair()` が keep / discard を正しく返す
- `build_filename_sequence()` が front/back の順番を崩さない
- mask 出力が画像枚数と一致する

## 12.2 結合テスト

- 小規模動画 1 本でフルパイプラインを完走できる
- selected 枚数が期待どおりになる
- Metashape で front/back が別 sensor として登録される
- mask が適用されている
- `Image/Quality` が記録される
- 冗長画像無効化後に再アライメントできる

## 12.3 評価観点

- 抽出枚数
- 採用率
- アライメント成功率
- 有効 camera 数
- tie point 数
- 再アライメント前後の安定性

---

## 13. 実装優先順位

## Phase 1

- ffprobe / ffmpeg 抽出
- ブレ判定
- ペア採用
- Metashape 読込
- アライメント

## Phase 2

- YOLO マスク生成
- mask 適用
- `analyzeImages()` による低品質除外

## Phase 3

- 近接・冗長画像除外
- 再アライメント
- ログ整備

## Phase 4

- UI改善
- グラフ表示
- chunk 分割処理
- rig オフセット補正高度化

---

## 14. 実装上の注意点

1. `0:v:0` と `0:v:1` の front/back 対応は固定値にせず設定可能にすること
2. front/back の片方のみを保存する仕様にはしないこと
3. 前処理では削りすぎず、後段で冗長除去すること
4. `MultiplaneLayout` の `filenames` 並びと `filegroups` は必ず小規模サンプルで検証すること
5. マスクを強くかけすぎると特徴点不足になるため、対象クラスは最小構成から始めること
6. 長尺トンネルは 1 chunk 全投入ではなく分割投入を将来考慮すること

---

## 15. 今回の仕様での確定事項

- 実行場所は **すべて Metashape 上の Python**
- 動画抽出は **ffmpeg**
- 入力 OSV は **front/back に使う主要 2 stream を含む OSV コンテナ**
- 採用判定は **片方でも閾値OKなら両方残す**
- front/back は **Metashape へマルチカメラとして読込**
- アライメント後に **近接・冗長画像を無効化** する

---

## 16. 今後の追加仕様候補

- fish-eye の中心寄り重み付き品質スコア
- front/back のどちらが優位かを後段重み付けに反映
- chunk 自動分割
- sparse point cloud 密度を使った冗長除去
- 反射面専用マスクモデルの追加
- rig 相対姿勢の自動推定
