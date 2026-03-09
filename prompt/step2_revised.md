まず以下をこの順番で読んでください。
1. `docs/metashape_dual_fisheye_pipeline_spec.md`
2. `docs/reference/metashape_api_checklist_dual_fisheye.md`

変更は最小限にとどめ、無関係なファイルは編集しないでください。
実装後は、変更ファイルごとの要約と TODO を必ず出してください。

Metashape API の使い方は checklist を優先し、背景説明や古いコード例よりも現行 API 名を優先してください。
曖昧な API は TODO コメントで残してください。

前回の続きです。
`docs/metashape_dual_fisheye_pipeline_spec.md` と `docs/reference/metashape_api_checklist_dual_fisheye.md` を参照し、Phase 1 を実装してください。

今回の作業範囲:
- `PipelineConfig`
- `FFmpegExtractor`
- `BlurEvaluator`
- `LogWriter` の基本部分
- `Run Full Pipeline` のうち、抽出とフレーム選別まで
- `Extract Streams`
- `Select Frames`

実装条件:
- macOS 前提で subprocess から ffprobe / ffmpeg を呼ぶ
- ffprobe は JSON を保存する
- ffmpeg は front/back を別ディレクトリへ抽出する
- front/back の抽出枚数不一致はエラーにする
- ブレ判定は中央70%円領域のラプラシアン分散を基本にする
- 将来 FFT を足せる構造にする
- 判定ルールは必ず
  `front_score >= threshold_front or back_score >= threshold_back`
  のとき、その時刻の front/back 両方を採用
- `frame_quality.csv` を保存
- selected が 0 件なら停止
- 例外処理を丁寧に実装する

変更禁止:
- front/back 片側だけを残す仕様への変更
- Windows 専用コマンドの使用
- 無関係なファイルの変更

完了条件:
- `scripts/metashape_dual_fisheye_pipeline.py` に上記クラスがある
- ffprobe / ffmpeg 呼び出しコードがある
- `frame_quality.csv` の出力がある
- README に Phase 1 の使い方がある

追加の注意:
- `extract_every_n_frames` を将来使えるよう、抽出部を拡張しやすい構造にする
- front/back のストリーム index は固定値ベタ書きにせず設定値から与える
- frame_id の整合性が後続フェーズで使えるよう、front/back の命名規則を厳密にそろえる

出力してほしい内容:
1. 変更ファイル一覧
2. 実装内容の要約
3. 未実装の TODO
4. テスト方法
