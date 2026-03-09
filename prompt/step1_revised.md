まず以下をこの順番で読んでください。
1. `docs/metashape_dual_fisheye_pipeline_spec.md`
2. `docs/reference/metashape_api_checklist_dual_fisheye.md`

変更は最小限にとどめ、無関係なファイルは編集しないでください。
実装後は、変更ファイルごとの要約と TODO を必ず出してください。

Metashape API の使い方は checklist を優先し、背景説明や古いコード例よりも現行 API 名を優先してください。
曖昧な API は TODO コメントで残してください。


`docs/metashape_dual_fisheye_pipeline_spec.md` と `docs/reference/metashape_api_checklist_dual_fisheye.md` を参照して、このリポジトリの初期整備をしてください。

今回の作業範囲:
- `scripts/metashape_dual_fisheye_pipeline.py` を新規作成
- `README.md` を追加または更新
- 必要なら `requirements.txt` を追加
- 必要なら `.gitignore` を更新
- 無関係なファイルは変更しない

今回の実装内容:
- `PipelineConfig` の骨組み
- メニュー登録の骨組み
- `Custom/DualFisheye/01 Run Full Pipeline`
- `Custom/DualFisheye/02 Extract Streams`
- `Custom/DualFisheye/03 Select Frames`
- `Custom/DualFisheye/04 Generate Masks`
- `Custom/DualFisheye/05 Import to Metashape`
- `Custom/DualFisheye/06 Align`
- `Custom/DualFisheye/07 Reduce Overlap`
- `Custom/DualFisheye/08 Export Logs`

要件:
- Python 3.8 互換
- pathlib 優先
- docstring と型ヒントを可能な範囲で付ける
- 設定値は dataclass または設定辞書で一元管理
- ログ出力先ディレクトリ生成を含める
- まだ未実装の処理は TODO で明示する
- spec と checklist の参照先を README に記載する

特に注意すること:
- Metashape API の曖昧な部分は実装を断定せず、TODO に落とす
- `filenames` / `filegroups` は後続フェーズで検証する前提なので、初期整備段階ではヘルパー関数の名前だけ先に確保してよい
- 旧 `importMasks` 前提の設計にしない

出力してほしい内容:
1. 変更ファイル一覧
2. 何を実装したか
3. TODO 一覧
4. 実行方法
