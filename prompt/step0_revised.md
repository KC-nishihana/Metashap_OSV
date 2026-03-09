このリポジトリでは、Metashape Professional 用の Python スクリプトを実装します。

作業前に、必ず以下をこの順番で読んでください。
1. `docs/metashape_dual_fisheye_pipeline_spec.md`
2. `docs/reference/metashape_api_checklist_dual_fisheye.md`

以後の作業では、実装仕様は spec を、Metashape API の使い方は checklist を優先してください。
背景メモや古いサンプルコードより、上記 2 ファイルを優先してください。

共通ルール:
- 開発環境は macOS + VS Code + Codex を前提にする
- 実行オーケストレーションは Metashape 内蔵 Python
- 動画抽出は ffprobe / ffmpeg を subprocess で呼ぶ
- 入力動画は 2Stream 型 MP4
- front/back の片側だけを残す仕様にしない
- 採用判定は「front または back のどちらかが閾値以上なら、その時刻の front/back 両方を採用」
- Metashape 取込は MultiplaneLayout
- sensor type は Fisheye
- Python 3.8 互換
- macOS 前提で実装するが、README には Windows での動作確認観点も書く
- 無関係なファイルは編集しない
- 変更したファイルごとに要約を出す
- 不確定な Metashape API 部分は TODO コメントで明示する
- まずは動く最小構成を優先し、過剰な最適化はしない

重要な補足:
- `filenames` / `filegroups` の最終構成は、API 文面だけで断定しないこと
- `filenames` / `filegroups` は small sample で current Metashape build 上の挙動を検証して確定すること
- mask 適用の初期版は `Metashape.Mask()` + `mask.load()` / `mask.setImage()` + `camera.mask = mask` を優先すること
- `importMasks` 系の古い記事やサンプルは、そのまま採用しないこと
- `matchPhotos()` は checklist にある現行引数名を使うこと
- 再アライメントは `matchPhotos(reset_matches=True, ...)` と `alignCameras(reset_alignment=True, ...)` の経路を確保すること
