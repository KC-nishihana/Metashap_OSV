# Metashape Dual Fisheye Pipeline

Phase 1 implementation for a Metashape Professional Python pipeline that targets a dual-fisheye 2-stream MP4 workflow.

## Source References

Use these documents as the implementation source of truth:

1. `docs/metashape_dual_fisheye_pipeline_spec.md`
2. `docs/reference/metashape_api_checklist_dual_fisheye.md`
3. `docs/reference/metashape_python_api_2_3_0.pdf`

The checklist takes priority over older snippets or background notes when choosing Metashape API names.

## Current Scope

The repository currently provides `scripts/metashape_dual_fisheye_pipeline.py` with:

- `PipelineConfig` as the central dataclass for paths and processing parameters
- `FFmpegExtractor` for `ffprobe` JSON export and front/back JPEG extraction
- `BlurEvaluator` for center-70-percent Laplacian variance scoring and pair-aware selection
- `LogWriter` for CSV / JSON phase outputs
- menu registration for:
  - `Custom/DualFisheye/01 Run Full Pipeline`
  - `Custom/DualFisheye/02 Extract Streams`
  - `Custom/DualFisheye/03 Select Frames`
  - `Custom/DualFisheye/04 Generate Masks`
  - `Custom/DualFisheye/05 Import to Metashape`
  - `Custom/DualFisheye/06 Align`
  - `Custom/DualFisheye/07 Reduce Overlap`
  - `Custom/DualFisheye/08 Export Logs`
- directory creation under `work/` and `project/`
- ffprobe / ffmpeg extraction with configurable stream indexes
- pair selection rule:
  - if `front_score >= blur_threshold_front` or `back_score >= blur_threshold_back`, keep both images for that timestamp
- `frame_quality.csv` output under `work/logs/`
- placeholder logging for mask generation, Metashape import validation, and overlap reduction

## Current Limitations

Later phases still keep unvalidated API behavior behind explicit `TODO` markers:

- `MultiplaneLayout` `filenames` / `filegroups` validation on the current Metashape build
- detector-backed mask generation
- pair-aware low-quality disabling policy
- custom post-alignment overlap reduction
- optional rig reference handling

The initial revision does not assume the older `importMasks` workflow.

## Runtime Requirements

Expected runtime:

- Metashape Professional with Python 3.8
- `ffprobe` and `ffmpeg` available on `PATH`
- OpenCV (`cv2`) and `numpy` available inside the Metashape Python environment
- optional future YOLO runtime for mask generation

No `requirements.txt` is included yet because execution is expected from the Metashape Python runtime.

## Phase 1 Usage

1. Open Metashape Professional.
2. Run `scripts/metashape_dual_fisheye_pipeline.py` from the Metashape Python console or scripts menu.
3. Edit `PipelineConfig` defaults in the script if your project paths differ from the repository layout.
4. Confirm `input/source.mp4` is a 2-stream MP4 and that `front_stream_index` / `back_stream_index` match the file.
5. Use one of these menu entries:
   - `Custom/DualFisheye/01 Run Full Pipeline`
   - `Custom/DualFisheye/02 Extract Streams`
   - `Custom/DualFisheye/03 Select Frames`

`01 Run Full Pipeline` currently runs Phase 1 only:

1. `ffprobe` writes `work/logs/ffprobe.json`
2. `ffmpeg` extracts:
   - `work/extracted/front_raw/F_*.jpg`
   - `work/extracted/back_raw/B_*.jpg`
3. `Select Frames` scores each pair with the center-70-percent Laplacian variance
4. If either side passes threshold, both images are copied to:
   - `work/selected/images/front/`
   - `work/selected/images/back/`
5. Selection results are written to `work/logs/frame_quality.csv`

Expected default paths:

- input MP4: `input/source.mp4`
- work directory: `work/`
- Metashape project path: `project/dual_fisheye_project.psx`
- logs: `work/logs/`

## Logs

Phase 1 produces these log outputs under `work/logs/`:

- `ffprobe.json`
- `frame_quality.csv`
- `pipeline_summary.json`
- `error.log`

Later-phase placeholders may create additional CSVs when those menu items are used.
