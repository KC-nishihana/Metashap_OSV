# Metashape Dual Fisheye Pipeline

Initial scaffold for a Metashape Professional Python pipeline that targets a dual-fisheye 2-stream MP4 workflow.

## Source References

Use these documents as the implementation source of truth:

1. `docs/metashape_dual_fisheye_pipeline_spec.md`
2. `docs/reference/metashape_api_checklist_dual_fisheye.md`
3. `docs/reference/metashape_python_api_2_3_0.pdf`

The checklist takes priority over older snippets or background notes when choosing Metashape API names.

## Current Scope

The repository currently provides `scripts/metashape_dual_fisheye_pipeline.py` with:

- `PipelineConfig` as the central dataclass for paths and processing parameters
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
- ffprobe / ffmpeg extraction wiring
- conservative paired frame selection that keeps both sides until blur scoring is implemented
- placeholder logging for mask generation, Metashape import validation, and overlap reduction

## Current Limitations

The scaffold keeps unvalidated API behavior behind explicit `TODO` markers:

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
- optional future dependencies such as OpenCV or YOLO runtime inside the Metashape Python environment

No `requirements.txt` is included yet because the current scaffold only relies on the standard library plus the Metashape runtime and external executables.

## How To Run

1. Open Metashape Professional.
2. Run `scripts/metashape_dual_fisheye_pipeline.py` from the Metashape Python console or scripts menu.
3. Edit `PipelineConfig` defaults in the script if your project paths differ from the repository layout.
4. Use the `Custom/DualFisheye/...` menu entries.

Expected default paths:

- input MP4: `input/source.mp4`
- work directory: `work/`
- Metashape project path: `project/dual_fisheye_project.psx`
- logs: `work/logs/`

## Logs

The scaffold prepares these log outputs under `work/logs/`:

- `ffprobe.json`
- `frame_quality.csv`
- `mask_summary.csv`
- `metashape_quality.csv`
- `overlap_reduction.csv`
- `pipeline_summary.json`
- `error.log`
