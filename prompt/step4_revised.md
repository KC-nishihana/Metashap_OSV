まず以下をこの順番で読んでください。
1. `docs/metashape_dual_fisheye_pipeline_spec.md`
2. `docs/reference/metashape_api_checklist_dual_fisheye.md`

変更は最小限にとどめ、無関係なファイルは編集しないでください。
実装後は、変更ファイルごとの要約と TODO を必ず出してください。

Metashape API の使い方は checklist を優先し、背景説明や古いコード例よりも現行 API 名を優先してください。
曖昧な API は TODO コメントで残してください。

前回の続きです。
`docs/metashape_dual_fisheye_pipeline_spec.md` と `docs/reference/metashape_api_checklist_dual_fisheye.md` を参照し、Phase 3 を実装してください。

今回の作業範囲:
- `MetashapeAligner`
- `OverlapReducer`
- `Export Logs`
- `Align`
- `Reduce Overlap`
- フルパイプラインの結線

実装条件:
- `analyzeImages(filter_mask=True)` を実装
- `Image/Quality < threshold` は `camera.enabled = False`
- `matchPhotos()` では checklist にある現行引数名を使うこと
  - `filter_mask=True`
  - `mask_tiepoints=True`
  - `filter_stationary_points=True`
  - 必要なら `generic_preselection`, `reference_preselection`, `keypoint_limit`, `tiepoint_limit`, `keep_keypoints` を設定から渡す
- 古い引数名は使わない
- `alignCameras()` を実装
- `metashape_quality.csv` を保存
- 近接画像判定は距離だけでなく回転差も使う
- 冗長候補では
  1. `Image/Quality`
  2. 前処理ブレスコア
  の順で残す
- `overlap_reduction.csv` を保存
- 冗長除去後、必要なら再アライメントする
- 再実行経路として `matchPhotos(reset_matches=True, ...)` と `alignCameras(reset_alignment=True, ...)` を使える構造にする
- `reduceOverlap()` は補助的に呼べる構造にする

変更禁止:
- 前処理側で過剰に画像削減すること
- Multiplane 読込構造を崩すこと

完了条件:
- フルパイプラインが 1 本の経路としてつながっている
- 品質 CSV と overlap CSV が出る
- README に Phase 3 の使い方がある
- TODO が整理されている

出力してほしい内容:
1. 変更ファイル一覧
2. 実装内容の要約
3. 主要な設計判断
4. 残っている TODO
5. 最低限の検証手順
