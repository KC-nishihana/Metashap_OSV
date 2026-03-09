# STEP7: OpenCV CUDA 対応 開発プロンプト

前回の続きです。
まず以下をこの順番で読んでください。

1. `docs/metashape_dual_fisheye_pipeline_spec.md`
2. `docs/reference/metashape_api_checklist_dual_fisheye.md`

今回は **STEP7: OpenCV CUDA 対応** を実装してください。

---

## 今回の目的

既存のデュアル魚眼処理ツールに、**OpenCV の CUDA 実行経路**を追加してください。

重要なのは、**CUDA を追加しても既存の CPU 実装を壊さず、CPU-only 環境でも完全に動作すること**です。

今回の主対象は **BlurEvaluator を中心とした前処理の高速化基盤** です。

---

## 今回の作業範囲

* `scripts/metashape_dual_fisheye_pipeline.py` の CUDA 対応
* 必要なら補助クラスの追加
* GUI 実装済みであれば GUI に CUDA 設定を追加
* `README.md` の CUDA 使用方法 / 制約 / fallback 挙動の追記
* 無関係なファイルは変更しない

---

## 実装方針

### 基本方針

* OpenCV backend は **auto / cpu / cuda** の3モードを持つ
* `auto` は CUDA 利用可能なら CUDA、不可なら CPU
* `cpu` は常に CPU
* `cuda` は CUDA 優先

  * `cuda_allow_fallback=True` なら CPU に落とす
  * それ以外はエラーにしてよい

### 最重要要件

* keep rule は変更しない
* front/back の片側だけを残す仕様にしない
* Metashape 読込 / mask / align / reduce overlap の仕様を壊さない
* CUDA が使えなくてもフルパイプラインが動くこと

---

## 追加してほしい設定項目

以下を config に追加してください。

```python
CONFIG = {
    "opencv_backend": "auto",
    "prefer_cuda": True,
    "cuda_device_index": 0,
    "cuda_allow_fallback": True,
    "cuda_log_device_info": True,
    "cuda_use_gaussian_preblur": False,
    "cuda_benchmark_mode": False,
    "save_backend_report": True
}
```

既存 config と統合し、GUI 実装済みなら GUI から編集可能にしてください。

---

## 追加 / 拡張してほしいクラス

### 新規候補

```python
class OpenCVBackendManager:
    ...
class CudaCapabilityReport:
    ...
```

### 既存拡張

* `PipelineConfig`
* `BlurEvaluator`
* `LogWriter`
* `DualFisheyeMainDialog`（存在する場合）

---

## OpenCVBackendManager に必須の役割

実装してください。

* OpenCV CUDA 利用可否判定
* device count 取得
* backend 決定
* fallback 理由保持
* backend report 生成

最低限ほしい関数例:

```python
class OpenCVBackendManager:
    def detect_cuda_support(self) -> dict: ...
    def select_backend(self, config) -> str: ...
    def set_active_device(self, device_index: int) -> None: ...
    def build_backend_report(self) -> dict: ...
    def ensure_backend(self, config) -> str: ...
```

---

## BlurEvaluator の修正要件

### 追加してほしい関数

```python
class BlurEvaluator:
    def laplacian_score_cpu(self, image): ...
    def laplacian_score_cuda(self, image): ...
    def laplacian_score(self, image): ...
```

### 要件

* `laplacian_score()` は backend を見て CPU / CUDA を切り替える
* CUDA 実行失敗時は fallback 条件に従って CPU へ切り替える
* fallback が起きたらログに残す
* keep rule のロジックはそのまま維持する
* `evaluate_pair()` と `select_pairs()` から使えるようにする

### 初期版で十分な範囲

* 単画像単位の CUDA 処理でよい
* 複雑な batch 最適化は不要
* FFT CUDA 化は optional / TODO でよい

---

## GUI 追補要件

GUI 実装済みなら以下を追加してください。

### 追加項目

* OpenCV backend: Auto / CPU / CUDA
* Prefer CUDA
* CUDA device index
* Allow CPU fallback
* Log CUDA device info
* CUDA benchmark mode（optional）

### 表示項目

* current backend
* CUDA device count
* fallback 発生有無
* backend report 保存先

### 挙動要件

* CUDA 利用不可時は GUI に明示する
* `opencv_backend=cuda` かつ fallback 無効で GPU なしなら開始前に止める
* `auto` の場合は CPU へ自動フォールバックする

---

## ログ要件

追加してください。

### 新規ログ

* `work/logs/opencv_backend_report.json`
* optional: `work/logs/opencv_backend_report.csv`
* optional: `work/logs/cuda_fallback.log`

### 既存ログへの追加候補

`frame_quality.csv` に backend 情報を追加できる構造にしてください。

例:

```csv
frame_id,front_score,back_score,keep_pair,backend_front,backend_back,fallback_front,fallback_back
```

---

## テスト要件

### 単体

* CPU-only 環境で backend 判定が壊れない
* CUDA 環境で device count が取れる
* `auto/cpu/cuda` が正しく選ばれる
* CUDA 失敗時に fallback が発動する

### 結合

* CPU-only 環境でフルパイプライン完走
* CUDA 環境で前処理が CUDA backend で動く
* GUI から backend 切替できる
* backend report が保存される

### 比較

* 同一画像で CPU / CUDA の score 差を確認可能にする
* 必要なら README に差異注意を追記する

---

## 実装上の厳守事項

1. CUDA は optional acceleration であること
2. CPU-only 環境でも正常動作すること
3. front/back の片側だけを残す仕様にしないこと
4. keep rule を崩さないこと
5. Metashape 読込 / mask / align / reduce overlap の仕様を崩さないこと
6. `filegroups` を断定しないこと
7. GUI 追加のために既存コア処理を崩さないこと
8. OpenCV CUDA API が曖昧な箇所は `TODO:` を残すこと
9. 無関係なファイルは変更しないこと

---

## 完了条件

1. OpenCV backend を auto / cpu / cuda で選べる
2. CUDA 利用可否を runtime 判定できる
3. CPU-only 環境で既存処理がそのまま動く
4. CUDA 環境で blur 評価に CUDA 経路を使える
5. fallback が機能する
6. backend report が保存される
7. GUI 実装済みなら GUI から設定できる
8. README が更新されている
9. 不確定な CUDA API 部分は `TODO:` で残してある

---

## 出力してほしい内容

1. 変更ファイル一覧
2. CUDA 対応の設計要約
3. どの既存クラスを拡張したか
4. 新規追加クラスの説明
5. fallback 設計の説明
6. GUI で追加した CUDA 項目
7. 残っている TODO
8. CPU-only / CUDA 環境それぞれの検証方法
