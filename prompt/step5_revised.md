このリポジトリ全体をレビューしてください。
仕様書は `docs/metashape_dual_fisheye_pipeline_spec.md`、API 確認資料は `docs/reference/metashape_api_checklist_dual_fisheye.md` です。

やってほしいこと:
- 仕様とのズレを洗い出す
- 危険な実装を指摘する
- macOS 開発時に壊れやすい箇所を修正する
- Windows での動作確認時に注意すべき点を README に追記する
- Metashape API 呼び出しで怪しい箇所を TODO として明示する
- 可能なら最小限の修正を加える

特に見てほしい点:
- front/back のペア維持
- `either_side_ok_keep_both` のロジック
- `filenames` / `filegroups` の整合性
- ただし `filenames` / `filegroups` は current build で検証前提になっているか
- subprocess のエラー処理
- ログ保存の欠落
- `Mask.load()` / `camera.mask` ベースの実装が優先されているか
- `matchPhotos()` が現行引数名を使っているか
- 再アライメントの流れ
- 無関係なファイル変更の有無

出力してほしい内容:
1. 問題一覧
2. 修正したファイル
3. まだ残るリスク
4. Windows 実行時の確認ポイント
