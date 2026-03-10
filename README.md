# Metashape Dual Fisheye Pipeline

Phase 1 through Phase 3 implementation for a Metashape Professional Python pipeline that targets a dual-fisheye OSV container workflow with at least two usable video streams for the front/back pair.

## Source References

Use these documents as the implementation source of truth:

1. `docs/metashape_dual_fisheye_pipeline_spec.md`
2. `docs/reference/metashape_api_checklist_dual_fisheye.md`
3. `docs/reference/metashape_python_api_2_3_0.pdf`

The checklist takes priority over older snippets or background notes when choosing Metashape API names.

## Current Scope

The repository currently provides `scripts/metashape_dual_fisheye_pipeline.py` with:

- `PipelineConfig` as the central dataclass for paths and processing parameters
- `FFmpegExtractor` for `ffprobe` JSON export, front/back stream selection, ignored-stream logging, and JPEG extraction
- `BlurEvaluator` for center-70-percent Laplacian variance scoring and pair-aware selection
- `OpenCVBackendManager` and `CudaCapabilityReport` for `auto` / `cpu` / `cuda` backend selection, runtime CUDA probing, fallback tracking, and backend report export
- `MaskGenerator` and `YoloBackendReport` for YOLO-based binary PNG mask generation, explicit `auto` / `cpu` / `cuda` device selection, and runtime CPU fallback
- `GpuStatusAggregator` and `MetashapeGpuReport` for OpenCV / YOLO / Metashape GPU status aggregation and JSON report export
- `MetashapeImporter` for `MultiplaneLayout` photo import, fisheye sensor assignment, and per-camera mask loading from disk
- `MetashapeAligner` for `analyzeImages(filter_mask=True)`, pair-aware low-quality camera disabling, `matchPhotos(...)`, and `alignCameras(...)`
- `OverlapReducer` for post-alignment redundancy filtering using distance + rotation thresholds with station-level pair preservation
- `LogWriter` for CSV / JSON phase outputs
- `ConfigPersistence`, `GuiLogHandler`, and `PipelineController` as the GUI-facing orchestration layer
- menu registration for:
  - `Custom/DualFisheye/00 GUIを開く`
  - `Custom/DualFisheye/01 フルパイプライン実行`
  - `Custom/DualFisheye/02 ストリーム抽出`
  - `Custom/DualFisheye/03 フレーム選別`
  - `Custom/DualFisheye/04 マスク生成`
  - `Custom/DualFisheye/05 Metashapeへ読込`
  - `Custom/DualFisheye/06 アライメント`
  - `Custom/DualFisheye/07 冗長画像削減`
  - `Custom/DualFisheye/08 ログ出力`
- directory creation under `work/` and `project/`
- ffprobe / ffmpeg extraction with configurable `front_stream_index` / `back_stream_index`
- selection of two front/back streams even when the `.osv` contains three or more video streams
- pair selection rule:
  - if `front_score >= blur_threshold_front` or `back_score >= blur_threshold_back`, keep both images for that timestamp
- YOLO target classes initialized to `person`, `car`, `truck`, `bus`, `motorbike`
- binary mask PNG output under:
  - `work/selected/masks/front/`
  - `work/selected/masks/back/`
- `frame_quality.csv`, `mask_summary.csv`, `metashape_quality.csv`, and `overlap_reduction.csv` output under `work/logs/`
- `opencv_backend_report.json`, `yolo_backend_report.json`, `metashape_gpu_report.json`, and `gpu_summary_report.json` output under `work/logs/` when `save_backend_report=True`
- `cuda_fallback.log` output under `work/logs/` when a CUDA selection or runtime fallback occurs
- `MultiplaneLayout` import planning split into `build_filename_sequence()` and `build_filegroups()`
- camera-level mask assignment via `Metashape.Mask()` + `mask.load()` + `camera.mask = mask`
- reset-aware re-execution hooks for `matchPhotos(reset_matches=True, ...)` and `alignCameras(reset_alignment=True, ...)`
- Unicode-safe OpenCV image IO for non-ASCII project paths on macOS and Windows

## Current Limitations

Later phases still keep unvalidated API behavior behind explicit `TODO` markers:

- `MultiplaneLayout` `filenames` / `filegroups` validation on the current Metashape build
- `camera.mask` assignment re-check on the current Metashape build with a small sample
- whether `alignCameras(reset_alignment=True)` needs additional current-build guards beyond the current implementation
- whether `Chunk.reduceOverlap(...)` should remain optional or become part of the main workflow
- optional rig reference handling
- which Qt binding should be treated as the preferred Metashape GUI binding on the current build

The current revision does not assume the older `importMasks` workflow and keeps task-based bulk mask import as future optional work.

OpenCV CUDA notes for the current implementation:

- preprocessing supports `opencv_backend = "auto" | "cpu" | "cuda"`
- `auto` uses CUDA only when `prefer_cuda=True` and the runtime exposes the required CUDA image-filter APIs
- `cuda` can fall back to CPU when `cuda_allow_fallback=True`; otherwise frame selection stops with an error
- the keep rule remains unchanged: if either side passes, both images for that timestamp are kept
- CUDA blur scoring currently targets single-image Laplacian evaluation only; batch optimization is still future work
- score differences between CPU and CUDA are expected because the CUDA path depends on the OpenCV CUDA Python bindings available in the current build
- if the OpenCV build exposes only `cv2.cuda` core symbols without `createLaplacianFilter`, the pipeline records that in the backend report and stays on CPU

YOLO / PyTorch GPU notes for the current implementation:

- mask generation supports `yolo_device_mode = "auto" | "cpu" | "cuda"`
- `auto` uses CUDA only when `prefer_yolo_cuda=True` and the current PyTorch runtime exposes a usable CUDA device
- `cuda` can fall back to CPU when `yolo_allow_fallback=True`; otherwise mask generation stops with an error
- the GUI and `yolo_backend_report.json` show the selected device, fallback events, and resolved local model path
- offline runtimes must point `mask_model_path` to a local `.pt` file; the pipeline does not rely on automatic model downloads

Metashape GPU notes for the current implementation:

- the pipeline currently treats Metashape GPU control APIs as validation-required and does not force-write GPU settings
- when the current build exposes inspectable fields such as `gpu_mask`, `cpu_enable`, or `enumGPUDevices()`, the pipeline records them in `metashape_gpu_report.json`
- when the current build does not expose a stable inspection path, the GUI shows `Metashape GPU: 未確認` and the report keeps explicit `TODO` notes

## Runtime Requirements

Expected runtime:

- Metashape Professional with Python 3.8
- `ffprobe` and `ffmpeg` available on `PATH`
- OpenCV (`cv2`) and `numpy` available inside the Metashape Python environment
- Ultralytics YOLO with segmentation weights available inside the Metashape Python environment
- Offline runtimes must use a local YOLO `.pt` file selected in the GUI or stored as a valid local path in config

No `requirements.txt` is included yet because execution is expected from the Metashape Python runtime.

Default mask-related configuration in `PipelineConfig`:

- `mask_model_path = "yolo11n-seg.pt"`
- `mask_classes = ("person", "car", "truck", "bus", "motorbike")`
- `mask_dilate_px = 8`
- `mask_confidence_threshold = 0.25`
- `mask_iou_threshold = 0.45`

Default OpenCV backend configuration in `PipelineConfig`:

- `opencv_backend = "auto"`
- `prefer_cuda = True`
- `cuda_device_index = 0`
- `cuda_allow_fallback = True`
- `cuda_log_device_info = True`
- `cuda_use_gaussian_preblur = False`
- `cuda_benchmark_mode = False`
- `yolo_device_mode = "auto"`
- `prefer_yolo_cuda = True`
- `yolo_allow_fallback = True`
- `yolo_device_index = 0`
- `save_backend_report = True`

## Usage

1. Open Metashape Professional.
2. Run `scripts/metashape_dual_fisheye_pipeline.py` from the Metashape Python console or scripts menu.
3. Open `Custom/DualFisheye/00 GUIを開く` and select the input `.osv` container first.
4. Confirm the selected `.osv` contains at least two usable video streams, and set `front_stream_index` / `back_stream_index` to the two streams to extract.
5. Use one of these menu entries:
   - `Custom/DualFisheye/00 GUIを開く`
   - `Custom/DualFisheye/01 フルパイプライン実行`
   - `Custom/DualFisheye/02 ストリーム抽出`
   - `Custom/DualFisheye/03 フレーム選別`
   - `Custom/DualFisheye/04 マスク生成`
   - `Custom/DualFisheye/05 Metashapeへ読込`
   - `Custom/DualFisheye/06 アライメント`
   - `Custom/DualFisheye/07 冗長画像削減`
   - `Custom/DualFisheye/08 ログ出力`

### GUI Usage

`00 GUIを開く` opens a Qt-based dialog that reuses the existing pipeline classes without replacing the phase logic.

Tabs:

- `基本設定`: input OSV, work folder, front/back stream index, `フル実行`
- `前処理`: frame sampling, blur thresholds, JPEG quality, OpenCV backend selection, CUDA toggles, `ストリーム抽出`, `フレーム選別`
- `マスク / 読込`: YOLO model path, mask classes, dilation, YOLO device controls, `マスク生成`, `Metashapeへ読込`
- `アライメント / 間引き`: image quality threshold, keypoint/tiepoint limits, rig reference fields, `アライメント`, `冗長画像削減`, `ログ出力`
- `ログ / 状態`: GUI log view, selected/discarded counts, mask counts, enabled/aligned camera counts, `OpenCV状態`, `YOLO状態`, `Metashape GPU状態`, `GPUフォールバック`, `使用デバイス番号`, `backend report 保存先`, current JSON summary

The GUI supports:

- Browse buttons for input OSV, work folder, and YOLO model path
- bare `mask_model_path` values such as `yolo26x-seg.pt` are resolved locally in this order:
  - absolute path from the current GUI field or loaded config
  - project-root relative path
  - work-root relative path
- if the YOLO model file is not found locally, the GUI warns before inference and does not rely on automatic download retries
- OpenCV / YOLO / Metashape GPU status are displayed separately in the summary tab
- OpenCV and YOLO can fall back independently; the summary tab also shows the aggregated fallback flag
- `front_stream_index` / `back_stream_index` values persisted into `PipelineConfig` and used by `probe_streams()`
- per-phase execution and full-pipeline execution
- colored log messages for info, warning, and error output
- status and step progress updates during execution
- `設定保存`, `設定読込`, and `初期値に戻す`
- start-time validation for strict `opencv_backend="cuda"` runs when `cuda_allow_fallback=False`
- start-time validation for strict `yolo_device_mode="cuda"` runs when `yolo_allow_fallback=False`

Config persistence:

- the GUI saves the last-used config to `work/config/last_used_config.json`
- missing keys are filled from `PipelineConfig` defaults when loading JSON
- `Save Config` writes the current GUI values, and each GUI run also refreshes the last-used file before execution
- the input field now starts empty and no longer assumes `input/source.mp4`
- stale `last_used_config.json` values such as old `.mp4` paths do not block GUI startup; the dialog marks them invalid until a new `.osv` is selected
- on Metashape menu execution, `__file__` may be undefined, so project-root discovery falls back in this order:
  - `__file__`
  - `sys.modules["__main__"].__file__`
  - `Metashape.app.document.path` parent
  - `Path.cwd()`
- if those runtime hints are still unavailable, GUI startup stays alive with a safe fallback path object and defers validation until a run button is pressed

Relationship to the existing menu flow:

- `01` through `08` remain available for direct phase execution without opening the GUI
- the GUI is an additional operation layer on top of the same `FFmpegExtractor`, `BlurEvaluator`, `MaskGenerator`, `MetashapeImporter`, `MetashapeAligner`, and `OverlapReducer` classes
- if Qt is unavailable in the embedded runtime, use `01` through `08` and validate the current Metashape Qt binding before enabling GUI use on that build

`01 Run Full Pipeline` now runs Phase 1 through Phase 3:

1. `ffprobe` writes `work/logs/ffprobe.json`
   - the log includes detected video stream details, selected front/back indices, and ignored stream indices
2. `ffmpeg` extracts:
   - `work/extracted/front_raw/F_*.jpg`
   - `work/extracted/back_raw/B_*.jpg`
   - only the two selected front/back streams are extracted
3. `Select Frames` scores each pair with the center-70-percent Laplacian variance
   - the score path is chosen from `auto` / `cpu` / `cuda`
   - if CUDA runtime selection or execution fails and fallback is allowed, the pipeline records the fallback and continues on CPU
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
10. `Align` runs `analyzeImages(filter_mask=True)`, keeps front/back pairs when either measured side meets `metashape_image_quality_threshold`, disables both sides only when all measured sides for the timestamp fall below threshold, writes `work/logs/metashape_quality.csv`, then calls `matchPhotos(...)` and `alignCameras(...)`
11. `Reduce Overlap` compares adjacent aligned stations using both camera-center distance and rotation delta, disables the lower-priority redundant station pair, writes `work/logs/overlap_reduction.csv`, and can realign with `reset_matches=True` / `reset_alignment=True`
12. `Export Logs` writes `work/logs/pipeline_summary.json` with current log presence and active chunk counts

During preprocessing, the pipeline can also write:

- `work/logs/opencv_backend_report.json`
- `work/logs/cuda_fallback.log`

`frame_quality.csv` now includes per-side backend columns so CPU and CUDA runs can be compared on the same image set:

- `backend_front`
- `backend_back`
- `fallback_front`
- `fallback_back`
- `fallback_reason_front`
- `fallback_reason_back`
- `elapsed_ms_front`
- `elapsed_ms_back`

For small-sample validation of the current Metashape build, first prepare 4 to 8 selected front/back pairs, run `04 Generate Masks`, then run `05 Import to Metashape` and visually confirm:

- front/back ordering is preserved per `frame_id`
- two fisheye sensors are created
- masks appear on the expected cameras

For Phase 3 validation, then run `06 Align` and `07 Reduce Overlap` and confirm:

- `work/logs/metashape_quality.csv` contains `Image/Quality` values and `enabled` flags
- `work/logs/metashape_quality.csv` also records `quality_keep_rule` and any pair-level disable reason
- `work/logs/overlap_reduction.csv` is written even when no stations are disabled
- if redundant stations are disabled, both sides of the losing timestamp are disabled together
- rerunning overlap cleanup can trigger `matchPhotos(reset_matches=True, ...)` and `alignCameras(reset_alignment=True, ...)`

### GUI Validation Checklist

On a small sample, confirm the following in order:

1. Open `Custom/DualFisheye/00 Open GUI`.
2. Edit the required fields on `Basic`, `Preprocess`, `Mask / Import`, and `Align / Reduce`.
3. Use each `Browse` button once and confirm the chosen `.osv` path is reflected in the field and accepted as `Input OSV`.
4. Click `Save Config`, close the dialog, reopen it, and confirm the previous `.osv` value reloads from `work/config/last_used_config.json`.
5. Replace the saved input value with an old `.mp4` path in `work/config/last_used_config.json`, reopen the dialog, and confirm the GUI still opens with a visible warning.
6. Run `Extract Streams`, `Select Frames`, `Generate Masks`, `Import to Metashape`, `Align`, and `Reduce Overlap` individually and confirm the `Logs / Summary` tab updates after each step.
7. If the environment is offline, use `Browse` on `YOLO Model Path` and select a local `.pt` file before `Generate Masks`.
8. Run `Extract Streams` on an `.osv` with three or more video streams and confirm the GUI continues when the configured front/back indices are valid.
9. Confirm the GUI status or log output shows the selected front/back indices and any ignored stream indices.
10. Run `Run Full Pipeline` and confirm the progress label, progress bar, GUI logs, and summary panel all update through completion or a visible error state.
11. Switch `OpenCV Backend` between `Auto`, `CPU`, and `CUDA` and confirm the summary panel updates the backend preview and CUDA device count.
12. Set `OpenCV Backend = CUDA` and `Allow CPU Fallback = off` on a CPU-only runtime and confirm the GUI stops before `Select Frames`.
13. Switch `YOLO Device Mode` between `Auto`, `CPU`, and `CUDA` and confirm the summary panel updates `YOLO状態`.
14. Confirm `ログ / 状態` shows `OpenCV状態`, `YOLO状態`, `Metashape GPU状態`, `GPUフォールバック`, `使用デバイス番号`, and `backend report 保存先` separately.

### OpenCV Backend Validation

CPU-only runtime:

1. Leave `opencv_backend = "auto"` and confirm `Select Frames` completes on CPU.
2. Confirm `work/logs/opencv_backend_report.json` records `active_backend = "cpu"` and `cuda_device_count = 0` or missing CUDA filter support.
3. Set `opencv_backend = "cuda"` and `cuda_allow_fallback = True` and confirm `Select Frames` still completes, with `cuda_fallback.log` written.
4. Set `opencv_backend = "cuda"` and `cuda_allow_fallback = False` and confirm the run stops before frame selection with a CUDA availability error.

CUDA-capable runtime:

1. Confirm the OpenCV build exposes the required CUDA image-filter bindings, not only `cv2.cuda` core symbols.
2. Run `Select Frames` with `opencv_backend = "cuda"` and confirm `work/logs/opencv_backend_report.json` records `active_backend = "cuda"` and the requested `cuda_device_index`.
3. Compare `frame_quality.csv` from a CPU run and a CUDA run on the same extracted frames and inspect `backend_front` / `backend_back` plus score deltas.
4. If `cuda_use_gaussian_preblur=True`, validate it on a small sample first because availability depends on the current OpenCV CUDA build.

### YOLO Backend Validation

CPU-only runtime:

1. Set `yolo_device_mode = "auto"` and confirm `Generate Masks` completes on CPU.
2. Confirm `work/logs/yolo_backend_report.json` records `active_device = "cpu"` and `fallback_events` only when CUDA was requested or preferred.
3. Set `yolo_device_mode = "cuda"` and `yolo_allow_fallback = True` and confirm `Generate Masks` still completes, with `cuda_fallback.log` recording the YOLO fallback event.
4. Set `yolo_device_mode = "cuda"` and `yolo_allow_fallback = False` and confirm mask generation stops with a CUDA availability error.

CUDA-capable runtime:

1. Confirm `torch.cuda.is_available()` is true in the Metashape Python runtime.
2. Run `Generate Masks` with `yolo_device_mode = "cuda"` and confirm `work/logs/yolo_backend_report.json` records `active_device = "cuda:<index>"`, the selected device name, and the resolved local model path.
3. If `yolo_device_mode = "auto"`, confirm the GUI summary shows `YOLO: GPU 使用中` when CUDA is available.
4. In offline mode, confirm the configured local `.pt` path is preserved in `yolo_backend_report.json` and no automatic download is required.

### Metashape GPU Validation

1. Run any pipeline phase or `08 ログ出力` and confirm `work/logs/metashape_gpu_report.json` is written.
2. If the current Metashape build exposes GPU inspection fields, confirm the report records available keys such as `gpu_mask`, `cpu_enable`, or enumerated GPU devices.
3. If the current build does not expose stable GPU inspection fields, confirm the GUI summary shows `Metashape GPU: 未確認` and `metashape_gpu_report.json` includes explicit `TODO` notes instead of guessed state.
4. Confirm `work/logs/gpu_summary_report.json` keeps OpenCV / YOLO / Metashape status in separate fields and records the aggregated fallback flag.

### macOS Development Notes

- The script keeps Unicode-safe OpenCV file IO because macOS project paths may contain spaces and non-ASCII characters.
- Validate that `ffprobe` and `ffmpeg` are visible inside the Metashape Python process launched from macOS, not only in Terminal.
- Confirm the embedded Metashape Python build exposes one of `PySide2`, `PySide6`, `PyQt5`, or `PyQt6` before depending on the GUI workflow for daily use.

## Windows Verification Notes

When validating on Windows, check these points explicitly:

- Confirm `ffprobe.exe` and `ffmpeg.exe` are visible from the Metashape Python process, not just from a separate terminal.
- Re-run a small sample from a path containing spaces or non-ASCII characters and confirm extraction, OpenCV reads, and mask PNG writes still succeed.
- Validate `MultiplaneLayout` with 4 to 8 front/back pairs on the current Windows Metashape build before trusting the final `filenames` / `filegroups` plan.
- Visually confirm `Metashape.Mask()` + `mask.load()` + `camera.mask = mask` still attaches the correct front/back masks on the Windows build.
- Re-run `06 Align` and `07 Reduce Overlap` once to confirm `matchPhotos(...)` current argument names and `alignCameras(reset_alignment=True)` are accepted on that build.
- Verify `work/logs/error.log` captures traceback details for a forced failure, so Windows-specific runtime issues are diagnosable.
- Open `00 Open GUI`, confirm the dialog renders correctly, and verify that `Save Config` / `Load Config` still work with Windows paths.
- If the GUI does not open, identify which Qt binding is available in the current Windows Metashape build before changing the fallback import order.

Expected default paths:

- input OSV: not preset; select a `.osv` file in the GUI
- work directory: `work/`
- Metashape project path: `project/dual_fisheye_project.psx`
- logs: `work/logs/`

## Logs

Current phases produce these log outputs under `work/logs/`:

- `ffprobe.json`
- `frame_quality.csv`
- `opencv_backend_report.json`
- `mask_summary.csv`
- `metashape_quality.csv`
- `overlap_reduction.csv`
- `pipeline_summary.json`
- `error.log`
- `cuda_fallback.log`
