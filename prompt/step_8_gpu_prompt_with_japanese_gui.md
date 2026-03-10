# STEP8: GPU活用強化 + GUI日本語化 開発プロンプト

前回の続きです。  
まず以下をこの順番で読んでください。

1. `README.md`
2. `docs/metashape_dual_fisheye_pipeline_spec.md`
3. `docs/reference/metashape_api_checklist_dual_fisheye.md`
4. `AGENTS.md`

今回は、**本ツール全体の GPU 活用を強化し、GUI 文言を日本語化**してください。

---

## 背景

現状の repository では、README から以下が確認できます。

- OpenCV backend は `auto / cpu / cuda` を持つ
- OpenCV の backend report / fallback log を出す構造がある
- GUI には OpenCV backend status、CUDA device count、fallback flag の表示がある
- YOLO ベースの mask 生成がある
- Metashape の importer / aligner / overlap reducer がある

ただし、現状の GPU 活用は部分的であり、
- OpenCV CUDA は前処理の一部のみ
- YOLO の GPU 利用方針が十分明示されていない
- Metashape 本体の GPU 利用状況確認が弱い
- GUI 文言は英語ベースが残っている

そのため、今回は
**「GPU を使える部分は使う」**
**「使えない部分は安全に fallback する」**
**「GUI は日本語で操作しやすくする」**
ことを目的に修正する。

---

## 今回の目的

1. OpenCV CUDA 利用をより安定化する
2. YOLO / PyTorch の GPU 利用を明示化する
3. Metashape 本体の GPU 利用状況を確認・可視化できるようにする
4. GPU が使えない環境でも CPU fallback で動作を維持する
5. GUI の表示文言・ボタン名・ラベル・警告文を日本語化する
6. 既存の keep rule / OSV / MultiplaneLayout / Fisheye / mask / align / overlap reduction の仕様は壊さない

---

## 最重要方針

### GPU 方針
- GPU は **使えるところだけ使う**
- 使えない場合は **必ず CPU に安全フォールバック**
- GPU を使うかどうかを GUI とログで見えるようにする
- OpenCV / YOLO / Metashape を分けて状態表示する
- 不確実な Metashape GPU API は推測で断定実装せず `TODO:` を残す

### GUI 方針
- GUI のユーザー向け文言は **日本語に統一**
- 内部クラス名・関数名・ログキーは英語のままでよい
- ただし GUI に出る文言は日本語にする
- エラー文言も日本語優先にする
- README には英語識別子と日本語UI対応表があってもよい

---

## 今回の作業範囲

- `scripts/metashape_dual_fisheye_pipeline.py` の GPU 活用強化
- GUI 文言の日本語化
- 必要なら補助クラス追加
- `README.md` 更新
- 必要なら `docs/metashape_dual_fisheye_pipeline_spec.md` 更新
- 必要なら `docs/reference/metashape_api_checklist_dual_fisheye.md` 更新
- 無関係なファイルは変更しない

---

## GPU 対応の実装要件

## 1. OpenCV CUDA を強化する

### 維持すること
- `opencv_backend = auto / cpu / cuda`
- `cuda_allow_fallback`
- backend report
- fallback log

### 追加 / 改善すること
- `BlurEvaluator` の CUDA 経路を安定化する
- 画像ごとの backend 使用状況を必要に応じて追跡可能にする
- CPU / CUDA の切り替え理由をログに残す
- GUI に現在の OpenCV backend を日本語表示する

### 望ましい機能
- `OpenCVBackendManager` の report を強化する
- device 名、device count、selected backend を表示する
- benchmark mode が有効なら CPU/CUDA の簡易比較を可能にする

---

## 2. YOLO / PyTorch の GPU 利用を明示化する

### 今回やってほしいこと
- MaskGenerator で使用する device を明示管理する
- `auto / cpu / cuda` または `auto / cpu / gpu` に近い設定を追加する
- CUDA が使える場合は YOLO を GPU で動かす
- CUDA が使えない場合は CPU fallback
- GUI に YOLO device 状態を表示する
- model 読込時に
  - 使用 device
  - fallback の有無
  - local model path
  をログへ出す

### 設定追加候補
```python
{
    "yolo_device_mode": "auto",   # auto / cpu / cuda
    "prefer_yolo_cuda": True,
    "yolo_allow_fallback": True,
    "yolo_device_index": 0
}
```

### 要件
- ローカル `.pt` 指定を優先
- GPU 利用不可でも mask 生成は CPU で継続可能にする
- GUI 上で「YOLO: GPU 使用中 / CPU フォールバック」を日本語表示する

---

## 3. Metashape 本体の GPU 利用状況を確認しやすくする

### 重要
Metashape 本体の GPU 制御 API が current build で曖昧な場合は、推測で危険な実装をしないこと。

### 今回やってほしいこと
- 少なくとも GUI / ログで「Metashape 本体の GPU 利用状況」を確認しやすくする
- 可能なら current build で取得できる GPU 関連情報をログに出す
- 取得できない場合は `TODO:` として残し、GUI では
  - `Metashape GPU 状態: 未確認`
  のように表示する

### 望ましい出力
- `work/logs/metashape_gpu_report.json`
- GUI 上で
  - `Metashape GPU 状態`
  - `OpenCV GPU 状態`
  - `YOLO GPU 状態`
  を分けて表示

---

## 4. GPU 統合ステータス表示を追加する

GUI の Logs / Summary タブに以下を追加してください。

- `OpenCV 状態`
- `YOLO 状態`
- `Metashape GPU 状態`
- `GPU フォールバック有無`
- `使用デバイス番号`
- `backend report 保存先`

表示例（日本語）:
- `OpenCV: CUDA 使用中`
- `OpenCV: CPU フォールバック`
- `YOLO: GPU 使用中`
- `YOLO: CPU 実行`
- `Metashape GPU: 未確認`
- `GPUフォールバック: あり`

---

## GUI 日本語化 要件

## 1. メニュー名の日本語化
既存メニューは内部機能を変えず、日本語ラベルに変更または日本語を併記してください。

推奨例:
- `Custom/DualFisheye/00 GUIを開く`
- `Custom/DualFisheye/01 フルパイプライン実行`
- `Custom/DualFisheye/02 ストリーム抽出`
- `Custom/DualFisheye/03 フレーム選別`
- `Custom/DualFisheye/04 マスク生成`
- `Custom/DualFisheye/05 Metashapeへ読込`
- `Custom/DualFisheye/06 アライメント`
- `Custom/DualFisheye/07 冗長画像削減`
- `Custom/DualFisheye/08 ログ出力`

## 2. GUI のタブ名日本語化
- `Basic` → `基本設定`
- `Preprocess` → `前処理`
- `Mask / Import` → `マスク / 読込`
- `Align / Reduce` → `アライメント / 間引き`
- `Logs / Summary` → `ログ / 状態`

## 3. 入力欄ラベル日本語化
例:
- `Input OSV` → `入力OSV`
- `Work Root` → `作業フォルダ`
- `Front Stream Index` → `前方ストリーム番号`
- `Back Stream Index` → `後方ストリーム番号`
- `YOLO Model Path` → `YOLOモデルパス`
- `OpenCV Backend` → `OpenCV実行方式`
- `Prefer CUDA` → `CUDAを優先`
- `Allow CPU Fallback` → `CPUフォールバックを許可`

## 4. ボタン名日本語化
例:
- `Run Full Pipeline` → `フル実行`
- `Extract Streams` → `ストリーム抽出`
- `Select Frames` → `フレーム選別`
- `Generate Masks` → `マスク生成`
- `Import to Metashape` → `Metashapeへ読込`
- `Align` → `アライメント`
- `Reduce Overlap` → `冗長画像削減`
- `Save Config` → `設定保存`
- `Load Config` → `設定読込`
- `Reset to Default` → `初期値に戻す`
- `Browse` → `参照`

## 5. 警告 / エラー文言日本語化
例:
- `Input OSV is not selected.` → `入力OSVが未選択です。`
- `Input OSV not found.` → `入力OSVが見つかりません。`
- `YOLO model file is not available locally.` → `YOLOモデルファイルがローカルに見つかりません。`
- `CUDA is not available. Fallback to CPU.` → `CUDAが利用できないため、CPUに切り替えました。`

---

## 追加してほしい設定項目

既存 config に加えて、少なくとも以下を追加または整理してください。

```python
CONFIG = {
    "opencv_backend": "auto",
    "prefer_cuda": True,
    "cuda_device_index": 0,
    "cuda_allow_fallback": True,
    "cuda_log_device_info": True,

    "yolo_device_mode": "auto",
    "prefer_yolo_cuda": True,
    "yolo_allow_fallback": True,
    "yolo_device_index": 0,

    "save_backend_report": True
}
```

---

## 追加 / 拡張してほしいクラス

### 新規候補
```python
class GpuStatusAggregator:
    ...
class JapaneseUiText:
    ...
```

### 既存拡張
- `OpenCVBackendManager`
- `MaskGenerator`
- `LogWriter`
- `DualFisheyeMainDialog`
- `PipelineConfig`

### 役割
- `GpuStatusAggregator`
  - OpenCV / YOLO / Metashape の GPU 状態をまとめる
- `JapaneseUiText`
  - GUI ラベル、ボタン名、メッセージの日本語文言を集中管理する

---

## ログ要件

以下を追加または改善してください。

### 既存 / 新規ログ
- `work/logs/opencv_backend_report.json`
- `work/logs/yolo_backend_report.json`
- `work/logs/metashape_gpu_report.json`
- `work/logs/gpu_summary_report.json`
- `work/logs/cuda_fallback.log`

### ログに含めたい項目
- OpenCV backend
- OpenCV device count
- OpenCV fallback reason
- YOLO device
- YOLO fallback reason
- Metashape GPU status（取得できるなら）
- 実行時の総合 GPU summary

---

## README 更新要件

README に以下を追記してください。

1. OpenCV の GPU 利用条件
2. YOLO の GPU 利用条件
3. CPU fallback の挙動
4. Metashape GPU 状態の扱い
5. GUI 日本語化の説明
6. OpenCV / YOLO / Metashape の GPU 状態が別表示であること
7. オフライン環境では YOLO モデルをローカル指定すること

---

## 壊してはいけない仕様

1. keep rule  
   `front_score >= blur_threshold_front or back_score >= blur_threshold_back`  
   のとき、その時刻の front/back 両方を採用
2. front/back の片側だけを残す仕様にしない
3. `.osv` 入力
4. front/back stream index 選択
5. `Metashape.MultiplaneLayout`
6. `Metashape.Sensor.Type.Fisheye`
7. `Metashape.Mask()` + `camera.mask`
8. `matchPhotos(reset_matches=True, ...)`
9. `alignCameras(reset_alignment=True, ...)`
10. `filegroups` を断定しない方針
11. 無関係なファイルは変更しない

---

## 完了条件

1. OpenCV の GPU/CPU 状態が GUI とログで見える
2. YOLO の GPU/CPU 状態が GUI とログで見える
3. Metashape GPU 状態が取得できるなら表示、無理なら未確認として扱える
4. GUI 文言が日本語化されている
5. 主要ボタン・タブ・ラベル・警告文が日本語になっている
6. CPU-only 環境でも動作する
7. GPU がある環境では利用可能な部分で GPU を使う
8. README が更新されている
9. 変更ファイルと TODO が明示される

---

## 出力してほしい内容

1. 変更ファイル一覧
2. GPU 活用をどう強化したか
3. OpenCV / YOLO / Metashape の GPU 状態をどう扱ったか
4. GUI の日本語化で変更した項目一覧
5. fallback の設計
6. 残っている TODO
7. CPU-only / GPU環境それぞれの確認手順