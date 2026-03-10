# Metashape API チェックシート（Dual Fisheye Pipeline 用・修正版）

## 1. 目的

このチェックシートは、Metashape 上で実装する以下の処理に必要な **主要 Python API** だけを抜き出して確認用にまとめたもの。

対象処理:

- Metashape メニュー追加
- front / back のマルチカメラ読込
- マスク適用
- 画像品質評価
- 特徴点マッチング / アライメント
- 近接・冗長画像の無効化
- 再アライメント
- 保存 / ログ出力補助

本シートは **実装時の確認用**。正式な根拠は API Reference PDF と実装仕様書を優先する。

---

## 2. 最優先で使う API 一覧

### 2.1 アプリ / メニュー

#### `Metashape.app`
- 用途: 現在のアプリケーションインスタンス取得
- 主な使用先:
  - `Metashape.app.document`
  - `Metashape.app.addMenuItem(...)`
  - `Metashape.app.messageBox(...)`

#### `Application.addMenuItem(label, method)`
- 用途: カスタムメニュー登録
- チェック項目:
  - [ ] Metashape 起動後にメニューが追加される
  - [ ] 同名メニューの重複登録が起きない
  - [ ] クリック時に対象関数が呼ばれる

#### `Metashape.app.document`
- 用途: 現在のドキュメント取得
- チェック項目:
  - [ ] ドキュメント未保存時の扱いを決めている
  - [ ] `doc.save()` の呼び出し位置が適切

---

### 2.2 Chunk / 読込

#### `Document.addChunk()`
- 用途: 新規 chunk 作成
- チェック項目:
  - [ ] 新規 chunk 名の付与方針がある
  - [ ] 既存 chunk 再利用ロジックがある

#### `Chunk.addPhotos(filenames=..., filegroups=..., layout=...)`
- 用途: 画像読込
- 想定設定:
  - `layout=Metashape.MultiplaneLayout`
- 最重要注意:
  - `filenames` の並び順と `filegroups` の整合性確認が必須
  - API 文面だけで最終的な front/back 配列順を断定しない
- チェック項目:
  - [ ] `build_filename_sequence()` と `build_filegroups()` を分離している
  - [ ] `filenames` / `filegroups` の組を 4〜8枚程度の small sample で検証している
  - [ ] front/back が別 sensor として認識される
  - [ ] 実装に `TODO: validate filenames/filegroups layout on current Metashape build` を残している

#### `Metashape.MultiplaneLayout`
- 用途: マルチカメラ / マルチプレーン読込
- チェック項目:
  - [ ] 読込後に 1 時刻あたり 2 camera 構成になる
  - [ ] ペア順序崩れがない

---

### 2.3 Sensor / カメラ種別

#### `chunk.sensors`
- 用途: sensor 一覧アクセス

#### `sensor.type = Metashape.Sensor.Type.Fisheye`
- 用途: 魚眼センサー指定
- チェック項目:
  - [ ] front/back の全 sensor が Fisheye になっている
  - [ ] Spherical など別タイプになっていない

#### `sensor.reference.location`
#### `sensor.reference.rotation`
#### `sensor.reference.location_enabled`
#### `sensor.reference.rotation_enabled`
- 用途: rig 相対位置 / 回転付与
- 注意:
  - 初期導入では optional 扱いでよい
- チェック項目:
  - [ ] rig 補正を使う場合のみ有効化している
  - [ ] front/back のどちらを master / slave にするか整理している

---

### 2.4 Mask 関連

#### `camera.mask`
- 用途: カメラ単位の mask アクセス

#### `Metashape.Mask()`
- 用途: mask オブジェクト生成

#### `mask.load(path)` / `mask.setImage(image)`
- 用途: mask 読込 / セット
- チェック項目:
  - [ ] camera ごとに対応する mask を正しく割当できる
  - [ ] front と back の mask を取り違えていない
  - [ ] mask 適用後に Metashape 上で視覚確認できる

#### `camera.mask = mask`
- 用途: カメラへ mask 適用
- チェック項目:
  - [ ] 全採用画像に対して適用される
  - [ ] mask 未生成時の例外処理がある

#### 注意事項
- `Metashape.Mask.load()` / `setImage()` は 2.3.0 PDF で確認済み
- 初期版は **camera 単位の個別適用** を優先する
- `importMasks` / task ベース一括取込は optional とし、small sample 検証後に採用する
- 旧 `ImportMasks.method` など古い記事の記法はそのまま使わない

---

### 2.5 画像品質評価

#### `Chunk.analyzeImages(filter_mask=True)`
- 用途: 画像品質評価
- チェック項目:
  - [ ] mask 適用後に実行している
  - [ ] `Image/Quality` が camera meta に保存される
  - [ ] 失敗時ログが出る

#### `camera.meta["Image/Quality"]`
- 用途: 品質値取得
- チェック項目:
  - [ ] 値が存在しない場合の分岐がある
  - [ ] 初期閾値 0.5 を設定可能にしている
  - [ ] overlap 除外時の優先順位に使っている

#### `camera.enabled = False`
- 用途: カメラ無効化
- チェック項目:
  - [ ] front/back のペア保存方針と矛盾していない
  - [ ] 無効化理由をログへ出している

---

### 2.6 特徴点マッチング / アライメント

#### `Chunk.matchPhotos(...)`
- 用途: 画像間マッチング
- 現行引数名として扱うもの:
  - `downscale`
  - `generic_preselection`
  - `reference_preselection`
  - `filter_mask`
  - `mask_tiepoints`
  - `filter_stationary_points`
  - `keep_keypoints`
  - `keypoint_limit`
  - `tiepoint_limit`
  - `reset_matches`
- チェック項目:
  - [ ] mask を特徴点抽出に反映している
  - [ ] `generic_preselection` など現行名を使っている
  - [ ] 古い引数名を使っていない
  - [ ] `reset_matches=True` を再実行経路で使える構造になっている

#### `Chunk.alignCameras(...)`
- 用途: カメラアライメント
- チェック項目:
  - [ ] 初回アライメントと再アライメントを分けて呼べる
  - [ ] `reset_alignment=True` を再アライメント経路で使える構造にしている
  - [ ] アライメント後の有効 camera 数を確認している

---

### 2.7 カメラ姿勢 / 重複除外

#### `camera.center`
- 用途: カメラ中心位置取得
- チェック項目:
  - [ ] 未アライメント camera を除外している
  - [ ] `None` 対応がある

#### `camera.transform`
- 用途: カメラ姿勢取得
- チェック項目:
  - [ ] transform 未取得時の分岐がある
  - [ ] 回転差計算関数が分離されている

#### `Chunk.reduceOverlap(overlap=...)`
- 用途: 内蔵の冗長カメラ削減
- チェック項目:
  - [ ] 初期版では optional 扱い
  - [ ] 自前ロジックの後に試せる構造
  - [ ] 実行前後の枚数差をログ出力

#### 重複除外ロジックの実装観点
- [ ] 並進距離閾値を設定化している
- [ ] 回転差閾値を設定化している
- [ ] 残す優先順位に `Image/Quality` を使っている
- [ ] 次点で前処理ブレスコアを使っている
- [ ] 無効化結果を CSV 出力している

---

### 2.8 保存

#### `doc.save()`
- 用途: プロジェクト保存
- チェック項目:
  - [ ] 保存先が存在する
  - [ ] 保存失敗時の例外処理がある
  - [ ] 途中保存の必要性を検討している

---

## 3. 今回の処理フローと API 対応

### 3.1 メニュー起動
- `Metashape.app.addMenuItem()`
- `Metashape.app.document`

### 3.2 front/back 読込前処理
- ffprobe / ffmpeg / OpenCV / YOLO を使用
- 実行制御は Metashape Python 内

### 3.3 Metashape 取込
- `Document.addChunk()`
- `Chunk.addPhotos(..., layout=Metashape.MultiplaneLayout)`
- `sensor.type = Metashape.Sensor.Type.Fisheye`

### 3.4 Mask 適用
- `Metashape.Mask()`
- `mask.load()` または `mask.setImage()`
- `camera.mask = mask`

### 3.5 品質評価
- `Chunk.analyzeImages(filter_mask=True)`
- `camera.meta["Image/Quality"]`
- `camera.enabled = False`

### 3.6 アライメント
- `Chunk.matchPhotos(...)`
- `Chunk.alignCameras()`

### 3.7 近接・冗長除外
- `camera.center`
- `camera.transform`
- `camera.enabled = False`
- `Chunk.reduceOverlap(...)`（補助）

### 3.8 再アライメント
- `Chunk.matchPhotos(reset_matches=True, ...)`
- `Chunk.alignCameras(reset_alignment=True, ...)`

### 3.9 保存
- `doc.save()`

---

## 4. 高リスク項目チェック

### 4.1 最優先で小規模検証すべき項目
- [ ] `MultiplaneLayout` の `filenames` / `filegroups` が正しい
- [ ] front/back が別 sensor として入る
- [ ] `camera.mask` 個別適用が問題なく動く
- [ ] `analyzeImages(filter_mask=True)` 後に `Image/Quality` が取れる
- [ ] `matchPhotos()` の現行引数名が現在の実行環境と一致する
- [ ] `alignCameras(reset_alignment=True)` が現在の実行環境で通る

### 4.2 コード上で TODO 明示すべき項目
- [ ] `filenames` / `filegroups` の最終構成ロジックを current build で確定
- [ ] mask 一括取込 API を採用するか確定
- [ ] rig 相対姿勢を使うかどうか確定
- [ ] `reduceOverlap()` を主運用に入れるか確定
- [ ] chunk 分割を初期版から入れるか確定

---

## 5. 実装時の最終確認

### 5.1 絶対に守る仕様
- [ ] 片方だけ保存する仕様にしていない
- [ ] 「片方でもOKなら両方残す」を守っている
- [ ] 入力は `.osv` コンテナ前提になっている
- [ ] Metashape 読込は MultiplaneLayout
- [ ] sensor は Fisheye
- [ ] 前処理で削りすぎていない
- [ ] 冗長除去はアライメント後

### 5.2 ログ
- [ ] `ffprobe.json`
- [ ] `ffprobe.json` records selected front/back stream indices
- [ ] `ffprobe.json` records ignored stream indices or equivalent stream-selection metadata
- [ ] `.osv` with 3 or more video streams does not fail when valid front/back indices are configured
- [ ] selected `front_stream_index` / `back_stream_index` are visible in the log output
- [ ] ignored video streams can be traced from the log output
- [ ] GUI shows a clear warning or error when probe results do not match the configured stream indices
- [ ] GUI can continue on `.osv` inputs with 3 or more video streams when the selected front/back indices are valid
- [ ] `frame_quality.csv`
- [ ] `mask_summary.csv`
- [ ] `metashape_quality.csv`
- [ ] `overlap_reduction.csv`
- [ ] `error.log`

### 5.3 検証
- [ ] 小規模動画で完走
- [ ] front/back ペア維持
- [ ] mask 適用確認
- [ ] quality 値取得確認
- [ ] アライメント成功確認
- [ ] 冗長除外後の再アライメント確認

### 5.4 GUI / Config 検証
- [ ] GUI で選択した `.osv` が `PipelineConfig.input_mp4` に反映される
- [ ] `validate(require_input=True)` が未選択 / ディレクトリ / `.osv` 以外 / 不存在を区別できる
- [ ] stale `last_used_config.json` に `.mp4` が残っていても GUI 起動が継続し、再選択した `.osv` が優先される

---

## 6. 実装者メモ

- まずは **camera 単位の mask 適用** を優先
- `importMasks` 系の古い記述はそのまま採用しない
- Metashape API の曖昧部分は `TODO:` で残す
- コード生成 AI にはこのチェックシートと実装仕様書を両方読ませる
- 実装後はこのシートのチェックボックスを埋めながら確認する
