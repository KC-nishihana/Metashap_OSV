# Metashape Dual Fisheye Pipeline

Phase 1 and Phase 2 implementation for a Metashape Professional Python pipeline that targets a dual-fisheye 2-stream MP4 workflow.

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
- `MaskGenerator` for YOLO-based binary PNG mask generation with configurable dilation
- `MetashapeImporter` for `MultiplaneLayout` photo import, fisheye sensor assignment, and per-camera mask loading from disk
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
- YOLO target classes initialized to `person`, `car`, `truck`, `bus`, `motorbike`
- binary mask PNG output under:
  - `work/selected/masks/front/`
  - `work/selected/masks/back/`
- `frame_quality.csv` and `mask_summary.csv` output under `work/logs/`
- `MultiplaneLayout` import planning split into `build_filename_sequence()` and `build_filegroups()`
- camera-level mask assignment via `Metashape.Mask()` + `mask.load()` + `camera.mask = mask`

## Current Limitations

Later phases still keep unvalidated API behavior behind explicit `TODO` markers:

- `MultiplaneLayout` `filenames` / `filegroups` validation on the current Metashape build
- `camera.mask` assignment re-check on the current Metashape build with a small sample
- pair-aware low-quality disabling policy
- custom post-alignment overlap reduction
- optional rig reference handling

The current revision does not assume the older `importMasks` workflow and keeps task-based bulk mask import as future optional work.

## Runtime Requirements

Expected runtime:

- Metashape Professional with Python 3.8
- `ffprobe` and `ffmpeg` available on `PATH`
- OpenCV (`cv2`) and `numpy` available inside the Metashape Python environment
- Ultralytics YOLO with segmentation weights available inside the Metashape Python environment

No `requirements.txt` is included yet because execution is expected from the Metashape Python runtime.

Default mask-related configuration in `PipelineConfig`:

- `mask_model_path = "yolo11n-seg.pt"`
- `mask_classes = ("person", "car", "truck", "bus", "motorbike")`
- `mask_dilate_px = 8`
- `mask_confidence_threshold = 0.25`
- `mask_iou_threshold = 0.45`

## Usage

1. Open Metashape Professional.
2. Run `scripts/metashape_dual_fisheye_pipeline.py` from the Metashape Python console or scripts menu.
3. Edit `PipelineConfig` defaults in the script if your project paths differ from the repository layout.
4. Confirm `input/source.mp4` is a 2-stream MP4 and that `front_stream_index` / `back_stream_index` match the file.
5. Use one of these menu entries:
   - `Custom/DualFisheye/01 Run Full Pipeline`
   - `Custom/DualFisheye/02 Extract Streams`
   - `Custom/DualFisheye/03 Select Frames`
   - `Custom/DualFisheye/04 Generate Masks`
   - `Custom/DualFisheye/05 Import to Metashape`

`01 Run Full Pipeline` now runs Phase 1 and Phase 2:

1. `ffprobe` writes `work/logs/ffprobe.json`
2. `ffmpeg` extracts:
   - `work/extracted/front_raw/F_*.jpg`
   - `work/extracted/back_raw/B_*.jpg`
3. `Select Frames` scores each pair with the center-70-percent Laplacian variance
4. If either side passes threshold, both images are copied to:
   - `work/selected/images/front/`
   - `work/selected/images/back/`
5. `Generate Masks` writes binary PNG masks to:
   - `work/selected/masks/front/F_*.png`
   - `work/selected/masks/back/B_*.png`
6. `Generate Masks` writes `work/logs/mask_summary.csv`
7. `Import to Metashape` imports selected front/back pairs using `Metashape.MultiplaneLayout`
8. All imported sensors are set to `Metashape.Sensor.Type.Fisheye`
9. Matching mask PNGs are loaded from disk and assigned camera-by-camera

For small-sample validation of the current Metashape build, first prepare 4 to 8 selected front/back pairs, run `04 Generate Masks`, then run `05 Import to Metashape` and visually confirm:

- front/back ordering is preserved per `frame_id`
- two fisheye sensors are created
- masks appear on the expected cameras

Expected default paths:

- input MP4: `input/source.mp4`
- work directory: `work/`
- Metashape project path: `project/dual_fisheye_project.psx`
- logs: `work/logs/`

## Logs

Current phases produce these log outputs under `work/logs/`:

- `ffprobe.json`
- `frame_quality.csv`
- `mask_summary.csv`
- `pipeline_summary.json`
- `error.log`

Later-phase alignment and overlap-reduction steps may create additional CSVs when those menu items are used.
