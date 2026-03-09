# STEP6: GUI 操作レイヤー実装プロンプト

前回の続きです。  
まず以下をこの順番で読んでください。

1. `docs/metashape_dual_fisheye_pipeline_spec.md`
2. `docs/reference/metashape_api_checklist_dual_fisheye.md`

今回は **STEP6: GUI 操作レイヤーの実装** を行ってください。

---

## 今回の目的

既存のデュアル魚眼処理ツールを、Metashape 上で **GUI から操作できる構成**に拡張してください。

重要なのは、**既存の処理ロジック（FFmpegExtractor / BlurEvaluator / MaskGenerator / MetashapeImporter / MetashapeAligner / OverlapReducer）を作り直さず、GUI はそれらを呼び出す操作レイヤーにすること**です。

---

## 今回の作業範囲

- `scripts/metashape_dual_fisheye_pipeline.py` の GUI 拡張
- 必要なら GUI 関連の補助クラス追加
- `README.md` の GUI 使用方法追記
- 必要なら設定保存用 JSON の追加
- 無関係なファイルは変更しない

---

## 実装方針

### 基本方針
- GUI は **Metashape 上から起動する**
- GUI から設定値を入力し、各処理を個別実行または一括実行できるようにする
- コア処理は既存クラスを再利用し、GUI はそれらの orchestrator とする
- 既存メニュー `01`〜`08` は残す
- GUI 起動用に新しいメニューを追加する  
  推奨: `Custom/DualFisheye/00 Open GUI`

### GUI の優先方針
- **Qt ベースのダイアログ**を第一候補とする
- ただし current Metashape build / embedded Python 環境で Qt バインディングの扱いが曖昧な場合は、
  - GUI 実装を分離する
  - 明確な `TODO:` を残す
  - 極力既存ロジックを壊さない
- Metashape API や GUI バインディングが不確かな箇所は、推測で断定実装せず `TODO:` を残す

---

## GUI に必須の入力項目

最低限、以下を GUI から指定できるようにしてください。

### 基本入力
- 入力 MP4
- 作業フォルダ
- front stream index
- back stream index

### 前処理設定
- blur threshold front
- blur threshold back
- FFT blur threshold（未使用でも項目は保持可）
- extract every n frames
- JPEG quality

### マスク設定
- YOLO model path
- mask classes
- mask dilate px

### Metashape / 後段設定
- Metashape image quality threshold
- keypoint limit
- tiepoint limit
- overlap target
- camera distance threshold
- camera angle threshold deg

### optional
- rig reference の有効 / 無効
- rig relative location
- rig relative rotation

---

## GUI レイアウト要件

過剰に凝らなくてよいので、まずは **使いやすい最小構成** を優先してください。

推奨構成:

### Tab 1: Basic
- 入力 MP4
- 作業フォルダ
- front/back stream index
- 実行ボタン（Run Full Pipeline）

### Tab 2: Preprocess
- extract every n frames
- blur threshold front/back
- JPEG quality
- `Extract Streams`
- `Select Frames`

### Tab 3: Mask / Import
- YOLO model path
- mask classes
- dilate px
- `Generate Masks`
- `Import to Metashape`

### Tab 4: Align / Reduce
- image quality threshold
- keypoint limit
- tiepoint limit
- distance threshold
- angle threshold
- overlap target
- `Align`
- `Reduce Overlap`

### Tab 5: Logs / Summary
- 実行ログ表示欄
- selected 枚数
- mask 枚数
- 有効 camera 数
- 最終 summary 表示

---

## GUI の操作要件

以下の操作ができるようにしてください。

1. **設定値を GUI で編集**
2. **設定値を config オブジェクトへ反映**
3. **各ステップを個別実行**
4. **フルパイプラインを一括実行**
5. **ログを GUI 上に表示**
6. **エラー時に GUI 上で分かるように表示**
7. **前回設定を保存 / 再読込できるようにする**

---

## 推奨機能

### 必須
- Browse ボタン
  - input MP4 選択
  - work root 選択
  - YOLO model path 選択
- 実行中の status 表示
- GUI 上のテキストログ表示
- 成功 / 警告 / エラーの区別
- 設定値の保存 / 読込

### できれば実装
- `frame_quality.csv` の集計表示
- selected / discarded 数の表示
- `metashape_quality.csv` の件数表示
- `overlap_reduction.csv` の件数表示

### optional
- 前処理ブレスコアの簡易ヒストグラム
- 採用 / 不採用比率の簡易表示
- 小規模サンプル検証モード

---

## 設定保存仕様

設定は JSON で保存 / 読込できるようにしてください。

例:
- `work/config/last_used_config.json`

要件:
- GUI 起動時に前回設定が存在すれば読込可能
- `Save Config`
- `Load Config`
- `Reset to Default`
- config の不足キーは既定値で補完

---

## 実装上の厳守事項

1. **front/back の片側だけを残す仕様にしない**
2. keep rule は必ず維持する  
   `front_score >= threshold_front or back_score >= threshold_back`  
   のとき、その時刻の front/back 両方を採用
3. Metashape 読込は `MultiplaneLayout`
4. sensor は `Metashape.Sensor.Type.Fisheye`
5. mask 適用は初期版では `Metashape.Mask()` + `camera.mask` 優先
6. `filenames` / `filegroups` は GUI 実装でも断定しない  
   current build で検証する前提を維持する
7. `matchPhotos()` は現行引数名を使う
8. 再アライメントは  
   - `matchPhotos(reset_matches=True, ...)`
   - `alignCameras(reset_alignment=True, ...)`  
   を使える構造にする
9. GUI 追加のためにコア処理を巨大な 1 ファイル手続きコードへ崩さない
10. 無関係なファイルは変更しない

---

## 実装したい構造

少なくとも以下のいずれかの構造にしてください。

### 推奨案
```python
class DualFisheyeMainDialog:
    ...
class PipelineConfig:
    ...
class FFmpegExtractor:
    ...
class BlurEvaluator:
    ...
class MaskGenerator:
    ...
class MetashapeImporter:
    ...
class MetashapeAligner:
    ...
class OverlapReducer:
    ...
class LogWriter:
    ...
```

### 補助クラス候補
```python
class GuiLogHandler:
    ...
class ConfigPersistence:
    ...
class PipelineController:
    ...
```

---

## GUI 実行フロー要件

### Run Full Pipeline
GUI から以下を順に呼ぶこと。

1. `config.validate()`
2. `config.ensure_directories()`
3. `extractor.probe_streams(...)`
4. `extractor.extract_front_stream(...)`
5. `extractor.extract_back_stream(...)`
6. `extractor.verify_frame_counts(...)`
7. `evaluator.select_pairs(...)`
8. `masker.process_directory(... front ...)`
9. `masker.process_directory(... back ...)`
10. `importer.create_or_get_chunk(...)`
11. `importer.build_filename_sequence(...)`
12. `importer.build_filegroups(...)`
13. `importer.import_multiplane_images(...)`
14. `importer.set_sensor_types(...)`
15. `importer.apply_rig_reference(...)`
16. `importer.apply_masks_from_disk(...)`
17. `aligner.analyze_image_quality(...)`
18. `aligner.disable_low_quality_cameras(...)`
19. `aligner.match_photos(...)`
20. `aligner.align_cameras(...)`
21. `reducer.disable_redundant_cameras(...)`
22. optional: `reducer.run_reduce_overlap_builtin(...)`
23. `aligner.realign_after_cleanup(...)`
24. `doc.save()`

GUI はこの処理の進捗を分かるように表示してください。

---

## エラー処理要件

GUI 化で特に重要なのは、**失敗時にどこで止まったか見えること**です。

必須:
- 例外を捕捉して GUI に表示
- Metashape Console にも出す
- `work/logs/error.log` に保存
- 失敗した処理ステップ名を表示
- 処理継続不能時はボタン状態を戻す

エラー例:
- ffprobe / ffmpeg が見つからない
- stream 数不足
- 抽出枚数不一致
- selected 0 件
- YOLO モデル読み込み失敗
- Metashape 読込失敗
- アライメント後の有効 camera が極端に少ない

---

## テスト要件

今回の GUI 実装後、最低限確認しやすいようにしてください。

### GUI 単体確認
- ダイアログが開く
- 各入力欄に値を入れられる
- Browse が動く
- Save / Load Config が動く
- ログ欄が更新される

### 処理連携確認
- Extract Streams ボタンで抽出が走る
- Select Frames ボタンで採用判定が走る
- Generate Masks ボタンで mask が出る
- Import to Metashape で chunk に入る
- Align で quality / match / align が走る
- Reduce Overlap で冗長除去が走る
- Run Full Pipeline で一括完走できる

---

## README 更新要件

README に以下を追記してください。

- GUI 起動方法
- GUI 各タブの説明
- Save / Load Config の説明
- macOS 開発時の注意
- Windows 動作確認時の注意
- Qt / GUI バインディングが current build で曖昧な場合の注意
- 既存 CLI / menu 実行との関係

---

## 完了条件

1. GUI 起動用メニューがある
2. GUI から主要設定を編集できる
3. GUI から各処理を個別実行できる
4. GUI からフルパイプラインを実行できる
5. ログが GUI に表示される
6. config 保存 / 読込ができる
7. 既存コア処理を再利用している
8. README が更新されている
9. 不確定な GUI / Metashape API 部分は `TODO:` で残してある

---

## 出力してほしい内容

1. 変更ファイル一覧
2. GUI の構成要約
3. どの既存クラスを再利用したか
4. 追加した GUI クラス / 補助クラスの説明
5. 残っている TODO
6. GUI の起動方法
7. GUI の最低限の検証方法

