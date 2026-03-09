まず以下をこの順番で読んでください。
1. `docs/metashape_dual_fisheye_pipeline_spec.md`
2. `docs/reference/metashape_api_checklist_dual_fisheye.md`

変更は最小限にとどめ、無関係なファイルは編集しないでください。
実装後は、変更ファイルごとの要約と TODO を必ず出してください。

Metashape API の使い方は checklist を優先し、背景説明や古いコード例よりも現行 API 名を優先してください。
曖昧な API は TODO コメントで残してください。


前回の続きです。
`docs/metashape_dual_fisheye_pipeline_spec.md` と `docs/reference/metashape_api_checklist_dual_fisheye.md` を参照し、Phase 2 を実装してください。

今回の作業範囲:
- `MaskGenerator`
- `MetashapeImporter`
- `Generate Masks`
- `Import to Metashape`

実装条件:
- YOLO ベースで不要領域マスクを生成する
- 初期対象クラスは person, car, truck, bus, motorbike
- マスクは二値 PNG で保存する
- dilation を設定可能にする
- `mask_summary.csv` を保存する
- front/back は同一 frame_id を維持する
- Metashape 取込は `MultiplaneLayout`
- sensor type は `Metashape.Sensor.Type.Fisheye`
- mask は初期版では **camera 単位でディスクから適用**する構造を優先する
- `Metashape.Mask()` + `mask.load()` または `mask.setImage()` + `camera.mask = mask` を主方式にする
- `importMasks` / task ベース一括取込は optional または TODO 扱いにする
- Metashape API の曖昧な箇所は TODO で残す

特に重視すること:
- `filenames` と `filegroups` の整合性
- front/back ペア順序を崩さないこと
- ただし `filenames` / `filegroups` の最終構成は API 文面だけで断定しないこと
- `build_filename_sequence()` と `build_filegroups()` を分離すること
- 4〜8枚程度の small sample で current build 上の挙動を検証しやすい構造にすること
- 実装に `TODO: validate filenames/filegroups layout on current Metashape build` を残すこと
- マスクを強くかけすぎないこと

完了条件:
- mask PNG が front/back 別に出力される
- Metashape 取込処理がコード化されている
- README に Phase 2 の使い方が追記されている

出力してほしい内容:
1. 変更ファイル一覧
2. 実装内容の要約
3. Metashape API で TODO にした箇所
4. 動作確認手順
