"""Dual fisheye Metashape pipeline scaffold.

This module is intended to be loaded from the Metashape Python runtime.
The initial revision keeps the implementation conservative:

- ffprobe / ffmpeg orchestration is wired and logged.
- frame selection keeps paired frames conservatively until blur scoring is added.
- mask generation, MultiplaneLayout import validation, and overlap reduction keep
  explicit TODO markers instead of assuming unverified API behavior.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import subprocess
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import Metashape  # type: ignore
except ImportError:  # pragma: no cover - Metashape is not available in local linting.
    Metashape = None  # type: ignore


LOGGER = logging.getLogger("dual_fisheye_pipeline")
_MENU_REGISTERED = False


def _default_project_root() -> Path:
    """Return the repository root derived from this script location."""

    return Path(__file__).resolve().parents[1]


def _as_serializable(value: Any) -> Any:
    """Convert Path-heavy structures into JSON-serializable values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


@dataclass
class PhaseResult:
    """Lightweight result object for menu-triggered phases."""

    phase: str
    status: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result for JSON logging."""

        return {
            "phase": self.phase,
            "status": self.status,
            "message": self.message,
            "details": {key: _as_serializable(value) for key, value in self.details.items()},
        }


@dataclass
class PipelineConfig:
    """Central configuration for the dual fisheye pipeline."""

    project_root: Path = field(default_factory=_default_project_root)
    input_mp4: Path = field(default_factory=lambda: _default_project_root() / "input" / "source.mp4")
    work_root: Path = field(default_factory=lambda: _default_project_root() / "work")
    project_path: Path = field(
        default_factory=lambda: _default_project_root() / "project" / "dual_fisheye_project.psx"
    )
    menu_root: str = "Custom/DualFisheye"
    chunk_name: str = "Dual Fisheye"
    front_stream_index: int = 0
    back_stream_index: int = 1
    extract_every_n_frames: int = 1
    jpeg_quality: int = 2
    blur_method: str = "laplacian_center70"
    blur_threshold_front: float = 60.0
    blur_threshold_back: float = 60.0
    fft_blur_threshold: Optional[float] = None
    keep_rule: str = "either_side_ok_keep_both"
    mask_model: str = "yolo"
    mask_classes: Tuple[str, ...] = ("person", "car", "truck", "bus", "motorbike")
    mask_dilate_px: int = 8
    metashape_image_quality_threshold: float = 0.5
    match_downscale: int = 1
    keypoint_limit: int = 40000
    tiepoint_limit: int = 10000
    filter_stationary_points: bool = True
    overlap_target: int = 3
    camera_distance_threshold: float = 0.15
    camera_angle_threshold_deg: float = 5.0
    chunk_size_limit: int = 250
    chunk_overlap_ratio: float = 0.25
    enable_rig_reference: bool = False
    rig_relative_location: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rig_relative_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    provisional_multiplane_import: bool = False
    use_builtin_reduce_overlap: bool = False

    @property
    def extracted_front_dir(self) -> Path:
        return self.work_root / "extracted" / "front_raw"

    @property
    def extracted_back_dir(self) -> Path:
        return self.work_root / "extracted" / "back_raw"

    @property
    def selected_front_dir(self) -> Path:
        return self.work_root / "selected" / "images" / "front"

    @property
    def selected_back_dir(self) -> Path:
        return self.work_root / "selected" / "images" / "back"

    @property
    def mask_front_dir(self) -> Path:
        return self.work_root / "selected" / "masks" / "front"

    @property
    def mask_back_dir(self) -> Path:
        return self.work_root / "selected" / "masks" / "back"

    @property
    def log_dir(self) -> Path:
        return self.work_root / "logs"

    @property
    def temp_dir(self) -> Path:
        return self.work_root / "temp"

    @property
    def ffprobe_log_path(self) -> Path:
        return self.log_dir / "ffprobe.json"

    @property
    def frame_quality_log_path(self) -> Path:
        return self.log_dir / "frame_quality.csv"

    @property
    def mask_summary_log_path(self) -> Path:
        return self.log_dir / "mask_summary.csv"

    @property
    def metashape_quality_log_path(self) -> Path:
        return self.log_dir / "metashape_quality.csv"

    @property
    def overlap_reduction_log_path(self) -> Path:
        return self.log_dir / "overlap_reduction.csv"

    @property
    def summary_log_path(self) -> Path:
        return self.log_dir / "pipeline_summary.json"

    @property
    def error_log_path(self) -> Path:
        return self.log_dir / "error.log"

    def validate(self, require_input: bool = False) -> None:
        """Validate user-editable configuration values."""

        if self.front_stream_index == self.back_stream_index:
            raise ValueError("front_stream_index and back_stream_index must differ.")
        if self.extract_every_n_frames < 1:
            raise ValueError("extract_every_n_frames must be >= 1.")
        if self.jpeg_quality < 1:
            raise ValueError("jpeg_quality must be >= 1.")
        if require_input and not self.input_mp4.exists():
            raise FileNotFoundError("Input MP4 not found: {0}".format(self.input_mp4))

    def ensure_directories(self) -> None:
        """Create the directory layout required by the scaffold."""

        for path in (
            self.extracted_front_dir,
            self.extracted_back_dir,
            self.selected_front_dir,
            self.selected_back_dir,
            self.mask_front_dir,
            self.mask_back_dir,
            self.log_dir,
            self.temp_dir,
            self.project_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config for logs."""

        return {key: _as_serializable(value) for key, value in asdict(self).items()}


class PipelineError(RuntimeError):
    """Domain-specific runtime error."""


def configure_logging(config: PipelineConfig) -> None:
    """Attach a file logger once the work directories exist."""

    config.ensure_directories()
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_path = str(config.error_log_path.resolve())
    has_file_handler = False

    for handler in LOGGER.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == file_path:
            has_file_handler = True
            break

    if not has_file_handler:
        handler = logging.FileHandler(file_path, encoding="utf-8")
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


class LogWriter:
    """CSV / JSON log writer helpers."""

    def write_csv(self, path: Path, rows: Sequence[Mapping[str, Any]], headers: Sequence[str]) -> None:
        """Write rows to a CSV file."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(headers))
            writer.writeheader()
            for row in rows:
                writer.writerow({header: row.get(header, "") for header in headers})

    def write_json(self, path: Path, obj: Any) -> None:
        """Write an object as UTF-8 JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, ensure_ascii=False)

    def write_summary(self, path: Path, summary: Mapping[str, Any]) -> None:
        """Persist a summary document."""

        self.write_json(path, summary)


class FFmpegExtractor:
    """ffprobe / ffmpeg orchestration."""

    def __init__(self, config: PipelineConfig, logs: LogWriter) -> None:
        self.config = config
        self.logs = logs

    def probe_streams(self, mp4_path: Path) -> Dict[str, Any]:
        """Inspect the input video streams via ffprobe."""

        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-of",
            "json",
            str(mp4_path),
        ]
        result = self._run_command(command)
        payload = json.loads(result.stdout or "{}")
        self.logs.write_json(self.config.ffprobe_log_path, payload)

        streams = payload.get("streams", [])
        video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
        if len(video_streams) < 2:
            raise PipelineError("Expected a 2-stream MP4, but found fewer than 2 video streams.")
        return payload

    def extract_front_stream(self, mp4_path: Path, out_dir: Path, stream_index: int) -> None:
        """Extract the configured front stream."""

        self._extract_stream(mp4_path, out_dir, stream_index, "F")

    def extract_back_stream(self, mp4_path: Path, out_dir: Path, stream_index: int) -> None:
        """Extract the configured back stream."""

        self._extract_stream(mp4_path, out_dir, stream_index, "B")

    def verify_frame_counts(self, front_dir: Path, back_dir: Path) -> Tuple[int, int]:
        """Ensure extracted front/back frame counts match."""

        front_count = len(sorted(front_dir.glob("F_*.jpg")))
        back_count = len(sorted(back_dir.glob("B_*.jpg")))
        if front_count != back_count:
            raise PipelineError(
                "Front/back extracted frame count mismatch: front={0}, back={1}".format(
                    front_count, back_count
                )
            )
        return front_count, back_count

    def run(self) -> PhaseResult:
        """Run stream probing and extraction."""

        self.config.validate(require_input=True)
        self.config.ensure_directories()
        self.probe_streams(self.config.input_mp4)
        self.extract_front_stream(
            self.config.input_mp4, self.config.extracted_front_dir, self.config.front_stream_index
        )
        self.extract_back_stream(
            self.config.input_mp4, self.config.extracted_back_dir, self.config.back_stream_index
        )
        front_count, back_count = self.verify_frame_counts(
            self.config.extracted_front_dir, self.config.extracted_back_dir
        )
        return PhaseResult(
            phase="extract_streams",
            status="ok",
            message="Extracted front/back streams with matching frame counts.",
            details={"front_count": front_count, "back_count": back_count},
        )

    def _extract_stream(self, mp4_path: Path, out_dir: Path, stream_index: int, prefix: str) -> None:
        """Run ffmpeg for a single stream."""

        out_dir.mkdir(parents=True, exist_ok=True)
        for old_file in out_dir.glob("{0}_*.jpg".format(prefix)):
            old_file.unlink()

        output_pattern = out_dir / "{0}_%06d.jpg".format(prefix)
        command = ["ffmpeg", "-y", "-i", str(mp4_path), "-map", "0:v:{0}".format(stream_index)]

        if self.config.extract_every_n_frames > 1:
            command.extend(
                [
                    "-vf",
                    "select=not(mod(n\\,{0}))".format(self.config.extract_every_n_frames),
                    "-vsync",
                    "vfr",
                ]
            )
        else:
            command.extend(["-vsync", "0"])

        command.extend(["-q:v", str(self.config.jpeg_quality), str(output_pattern)])
        self._run_command(command)

    @staticmethod
    def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess:
        """Execute a subprocess and convert common failures into pipeline errors."""

        LOGGER.info("Running command: %s", " ".join(command))
        try:
            result = subprocess.run(
                list(command),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise PipelineError("Required executable not found: {0}".format(command[0])) from exc
        except subprocess.CalledProcessError as exc:
            raise PipelineError(
                "Command failed: {0}\n{1}".format(" ".join(command), exc.stderr.strip())
            ) from exc
        return result


class BlurEvaluator:
    """Frame scoring and paired selection scaffold."""

    def __init__(self, config: PipelineConfig, logs: LogWriter) -> None:
        self.config = config
        self.logs = logs

    def compute_center70_mask(self, image: Any) -> Any:
        """Placeholder for the center-70-percent scoring mask."""

        # TODO: implement center-70-percent circular mask with OpenCV in a follow-up phase.
        return image

    def laplacian_score(self, image: Any) -> Optional[float]:
        """Placeholder for the Laplacian blur score."""

        # TODO: implement Laplacian variance scoring for paired blur evaluation.
        return None

    def fft_blur_score(self, image: Any) -> Optional[float]:
        """Placeholder for optional FFT-based blur scoring."""

        # TODO: implement optional FFT blur scoring once OpenCV-based baseline is validated.
        return None

    def evaluate_pair(self, front_path: Path, back_path: Path) -> Dict[str, Any]:
        """Return a conservative placeholder evaluation for a front/back pair."""

        return {
            "frame_id": front_path.stem.split("_")[-1],
            "front_path": front_path.name,
            "back_path": back_path.name,
            "front_score": "",
            "back_score": "",
            "keep_pair": 1,
            "better_side": "TODO",
        }

    def select_pairs(self, front_dir: Path, back_dir: Path, out_front_dir: Path, out_back_dir: Path) -> PhaseResult:
        """Copy all paired frames conservatively until blur scoring is implemented."""

        front_files = sorted(front_dir.glob("F_*.jpg"))
        back_files = sorted(back_dir.glob("B_*.jpg"))
        if not front_files or not back_files:
            raise PipelineError("No extracted frames found. Run stream extraction first.")
        if len(front_files) != len(back_files):
            raise PipelineError("Front/back extracted frame counts do not match.")

        out_front_dir.mkdir(parents=True, exist_ok=True)
        out_back_dir.mkdir(parents=True, exist_ok=True)
        self._clear_directory(out_front_dir, "F_*.jpg")
        self._clear_directory(out_back_dir, "B_*.jpg")

        rows: List[Dict[str, Any]] = []
        for front_path, back_path in zip(front_files, back_files):
            row = self.evaluate_pair(front_path, back_path)
            shutil.copy2(str(front_path), str(out_front_dir / front_path.name))
            shutil.copy2(str(back_path), str(out_back_dir / back_path.name))
            rows.append(row)

        self.logs.write_csv(
            self.config.frame_quality_log_path,
            rows,
            headers=(
                "frame_id",
                "front_path",
                "back_path",
                "front_score",
                "back_score",
                "keep_pair",
                "better_side",
            ),
        )
        return PhaseResult(
            phase="select_frames",
            status="todo",
            message=(
                "Copied all paired frames conservatively. TODO: replace with blur scoring while preserving "
                "the either-side-OK keep-both rule."
            ),
            details={"selected_pairs": len(rows)},
        )

    @staticmethod
    def _clear_directory(directory: Path, pattern: str) -> None:
        """Remove generated files for a clean rerun."""

        for old_file in directory.glob(pattern):
            old_file.unlink()


class MaskGenerator:
    """Mask generation scaffold."""

    def __init__(self, config: PipelineConfig, logs: LogWriter) -> None:
        self.config = config
        self.logs = logs

    def load_model(self) -> None:
        """Placeholder model loader."""

        # TODO: load YOLO or another validated detector from Metashape Python.
        return None

    def infer_mask(self, image_path: Path) -> Optional[Any]:
        """Placeholder mask inference hook."""

        # TODO: run detector inference and return a binary mask image.
        return None

    def dilate_mask(self, mask: Any, px: int) -> Any:
        """Placeholder dilation hook."""

        # TODO: apply dilation after binary mask generation.
        return mask

    def save_mask(self, mask: Any, out_path: Path) -> None:
        """Placeholder mask persistence hook."""

        # TODO: write binary PNG masks to disk once inference output is defined.
        del mask
        del out_path

    def process_directory(self, image_dir: Path, out_mask_dir: Path, camera_side: str) -> List[Dict[str, Any]]:
        """Enumerate images and emit TODO rows without creating masks yet."""

        out_mask_dir.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []
        for image_path in sorted(image_dir.glob("*.jpg")):
            rows.append(
                {
                    "camera_side": camera_side,
                    "image_path": image_path.name,
                    "mask_path": "",
                    "status": "TODO",
                }
            )
        return rows

    def run(self) -> PhaseResult:
        """Log the pending mask-generation workload."""

        rows: List[Dict[str, Any]] = []
        rows.extend(self.process_directory(self.config.selected_front_dir, self.config.mask_front_dir, "front"))
        rows.extend(self.process_directory(self.config.selected_back_dir, self.config.mask_back_dir, "back"))
        self.logs.write_csv(
            self.config.mask_summary_log_path,
            rows,
            headers=("camera_side", "image_path", "mask_path", "status"),
        )
        return PhaseResult(
            phase="generate_masks",
            status="todo",
            message="Mask generation scaffold created. TODO: attach detector-backed mask PNG output.",
            details={"image_count": len(rows)},
        )


class MetashapeImporter:
    """Metashape import scaffold with localized TODOs for unvalidated API behavior."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def create_or_get_chunk(self, doc: Any, name: str = "Dual Fisheye") -> Any:
        """Reuse an existing chunk by label or create a new one."""

        for chunk in getattr(doc, "chunks", []):
            if getattr(chunk, "label", "") == name:
                return chunk
        chunk = doc.addChunk()
        chunk.label = name
        return chunk

    def build_filename_sequence(self, front_dir: Path, back_dir: Path) -> List[str]:
        """Build a provisional front/back filename sequence."""

        front_files = sorted(front_dir.glob("F_*.jpg"))
        back_files = sorted(back_dir.glob("B_*.jpg"))
        if len(front_files) != len(back_files):
            raise PipelineError("Selected front/back frame counts do not match.")

        filenames: List[str] = []
        for front_path, back_path in zip(front_files, back_files):
            filenames.extend([str(front_path), str(back_path)])
        return filenames

    def build_filegroups(self, pair_count: int) -> List[int]:
        """Build provisional filegroups for MultiplaneLayout import."""

        # TODO: validate filenames/filegroups layout on current Metashape build.
        return [2] * pair_count

    def import_multiplane_images(self, chunk: Any, filenames: Sequence[str], filegroups: Sequence[int]) -> None:
        """Run provisional MultiplaneLayout import when explicitly enabled."""

        if not self.config.provisional_multiplane_import:
            raise PipelineError(
                "MultiplaneLayout import scaffold is prepared but disabled. "
                "Set provisional_multiplane_import=True only after validating filenames/filegroups."
            )
        # TODO: validate filenames/filegroups layout on current Metashape build.
        chunk.addPhotos(
            filenames=list(filenames),
            filegroups=list(filegroups),
            layout=Metashape.MultiplaneLayout,
        )

    def set_sensor_types(self, chunk: Any) -> None:
        """Set all sensors to fisheye as required by the spec."""

        for sensor in getattr(chunk, "sensors", []):
            sensor.type = Metashape.Sensor.Type.Fisheye

    def apply_rig_reference(self, chunk: Any, config: PipelineConfig) -> None:
        """Optional rig reference hook."""

        if not config.enable_rig_reference:
            return
        # TODO: validate rig master/slave handling and relative reference assignment on current build.
        del chunk

    def apply_masks_from_disk(self, chunk: Any, mask_front_dir: Path, mask_back_dir: Path) -> int:
        """Attach per-camera masks when matching files exist on disk."""

        applied_count = 0
        for camera in getattr(chunk, "cameras", []):
            if not getattr(camera, "label", ""):
                continue
            mask_path = self._mask_path_for_camera(camera.label, mask_front_dir, mask_back_dir)
            if mask_path is None or not mask_path.exists():
                continue
            # TODO: re-check camera.mask assignment on the current Metashape build with a small sample.
            mask = Metashape.Mask()
            mask.load(str(mask_path))
            camera.mask = mask
            applied_count += 1
        return applied_count

    def run(self) -> PhaseResult:
        """Create or reuse a chunk and keep import-specific uncertainty explicit."""

        self._require_metashape()
        doc = Metashape.app.document
        chunk = self.create_or_get_chunk(doc, name=self.config.chunk_name)

        front_files = sorted(self.config.selected_front_dir.glob("F_*.jpg"))
        back_files = sorted(self.config.selected_back_dir.glob("B_*.jpg"))
        if not front_files or not back_files:
            raise PipelineError("No selected frames found. Run frame selection first.")

        filenames = self.build_filename_sequence(self.config.selected_front_dir, self.config.selected_back_dir)
        filegroups = self.build_filegroups(len(front_files))

        if self.config.provisional_multiplane_import:
            self.import_multiplane_images(chunk, filenames, filegroups)
            self.set_sensor_types(chunk)
            self.apply_rig_reference(chunk, self.config)
            applied_masks = self.apply_masks_from_disk(chunk, self.config.mask_front_dir, self.config.mask_back_dir)
            return PhaseResult(
                phase="import_to_metashape",
                status="todo",
                message="Provisional MultiplaneLayout import executed. Validate sensor grouping on the current build.",
                details={"camera_count": len(getattr(chunk, "cameras", [])), "applied_masks": applied_masks},
            )

        return PhaseResult(
            phase="import_to_metashape",
            status="todo",
            message=(
                "Chunk scaffold is ready. TODO: validate filenames/filegroups on the current Metashape build "
                "before enabling MultiplaneLayout import."
            ),
            details={"chunk_label": getattr(chunk, "label", self.config.chunk_name), "pair_count": len(front_files)},
        )

    @staticmethod
    def _mask_path_for_camera(label: str, mask_front_dir: Path, mask_back_dir: Path) -> Optional[Path]:
        """Map a camera label to the expected mask path."""

        if label.startswith("F_"):
            return mask_front_dir / "{0}.png".format(label)
        if label.startswith("B_"):
            return mask_back_dir / "{0}.png".format(label)
        return None

    @staticmethod
    def _require_metashape() -> None:
        if Metashape is None:
            raise PipelineError("Metashape module is not available in this Python runtime.")


class MetashapeAligner:
    """Alignment hooks using the current API names from the checklist."""

    def __init__(self, config: PipelineConfig, logs: LogWriter) -> None:
        self.config = config
        self.logs = logs

    def analyze_image_quality(self, chunk: Any) -> None:
        """Run Metashape image quality analysis with masks enabled."""

        chunk.analyzeImages(filter_mask=True)

    def disable_low_quality_cameras(self, chunk: Any, threshold: float = 0.5) -> int:
        """Disable cameras below threshold.

        The initial scaffold keeps this method available but does not call it by
        default because pair-aware disable behavior still needs confirmation.
        """

        disabled = 0
        for camera in getattr(chunk, "cameras", []):
            value = camera.meta.get("Image/Quality")
            try:
                quality = float(value)
            except (TypeError, ValueError):
                continue
            if quality < threshold:
                camera.enabled = False
                disabled += 1
        return disabled

    def export_quality_log(self, chunk: Any, csv_path: Path) -> None:
        """Export current Metashape quality metadata."""

        rows: List[Dict[str, Any]] = []
        for camera in getattr(chunk, "cameras", []):
            rows.append(
                {
                    "camera_label": getattr(camera, "label", ""),
                    "enabled": getattr(camera, "enabled", True),
                    "image_quality": camera.meta.get("Image/Quality", ""),
                }
            )
        self.logs.write_csv(csv_path, rows, headers=("camera_label", "enabled", "image_quality"))

    def match_photos(self, chunk: Any, config: PipelineConfig) -> None:
        """Run feature matching with current argument names from the checklist."""

        chunk.matchPhotos(
            downscale=config.match_downscale,
            generic_preselection=True,
            reference_preselection=False,
            filter_mask=True,
            mask_tiepoints=True,
            filter_stationary_points=config.filter_stationary_points,
            keep_keypoints=False,
            keypoint_limit=config.keypoint_limit,
            tiepoint_limit=config.tiepoint_limit,
            reset_matches=False,
        )

    def align_cameras(self, chunk: Any) -> None:
        """Run initial camera alignment."""

        chunk.alignCameras()

    def realign_after_cleanup(self, chunk: Any, config: PipelineConfig) -> None:
        """Rerun alignment after overlap cleanup."""

        chunk.matchPhotos(
            downscale=config.match_downscale,
            generic_preselection=True,
            reference_preselection=False,
            filter_mask=True,
            mask_tiepoints=True,
            filter_stationary_points=config.filter_stationary_points,
            keep_keypoints=False,
            keypoint_limit=config.keypoint_limit,
            tiepoint_limit=config.tiepoint_limit,
            reset_matches=True,
        )
        chunk.alignCameras(reset_alignment=True)

    def run(self) -> PhaseResult:
        """Analyze quality and align the active chunk when cameras are available."""

        self._require_metashape()
        chunk = getattr(Metashape.app.document, "chunk", None)
        if chunk is None:
            return PhaseResult("align", "skipped", "No active chunk is available.", {})
        if not getattr(chunk, "cameras", []):
            return PhaseResult("align", "skipped", "Active chunk has no cameras to align.", {})

        self.analyze_image_quality(chunk)
        self.export_quality_log(chunk, self.config.metashape_quality_log_path)
        # TODO: confirm pair-aware low-quality disable policy before enabling automatic camera disabling.
        self.match_photos(chunk, self.config)
        self.align_cameras(chunk)
        return PhaseResult(
            phase="align",
            status="ok",
            message="Ran image quality analysis and alignment on the active chunk.",
            details={"camera_count": len(getattr(chunk, "cameras", []))},
        )

    @staticmethod
    def _require_metashape() -> None:
        if Metashape is None:
            raise PipelineError("Metashape module is not available in this Python runtime.")


class OverlapReducer:
    """Post-alignment overlap reduction scaffold."""

    def __init__(self, config: PipelineConfig, logs: LogWriter) -> None:
        self.config = config
        self.logs = logs

    def get_enabled_cameras(self, chunk: Any) -> List[Any]:
        """Return enabled cameras only."""

        return [camera for camera in getattr(chunk, "cameras", []) if getattr(camera, "enabled", True)]

    def camera_center(self, camera: Any) -> Any:
        """Return the current camera center."""

        return getattr(camera, "center", None)

    def camera_rotation(self, camera: Any) -> Any:
        """Return the current camera transform."""

        return getattr(camera, "transform", None)

    def distance_between(self, cam_a: Any, cam_b: Any) -> Optional[float]:
        """Placeholder for distance computation."""

        del cam_a
        del cam_b
        # TODO: compute translation distance from aligned camera centers.
        return None

    def angle_between(self, cam_a: Any, cam_b: Any) -> Optional[float]:
        """Placeholder for rotation delta computation."""

        del cam_a
        del cam_b
        # TODO: compute angular delta from aligned camera transforms.
        return None

    def disable_redundant_cameras(self, chunk: Any, config: PipelineConfig) -> List[Dict[str, Any]]:
        """Placeholder for custom overlap-reduction decisions."""

        del chunk
        del config
        # TODO: implement pair-aware redundancy filtering after pose validation.
        return []

    def run_reduce_overlap_builtin(self, chunk: Any, overlap: int = 3) -> Dict[str, Any]:
        """Optionally exercise Metashape's built-in overlap reduction."""

        before_count = len(self.get_enabled_cameras(chunk))
        chunk.reduceOverlap(overlap=overlap)
        after_count = len(self.get_enabled_cameras(chunk))
        return {"before_count": before_count, "after_count": after_count}

    def run(self) -> PhaseResult:
        """Log overlap-reduction status without assuming unvalidated behavior."""

        self._require_metashape()
        chunk = getattr(Metashape.app.document, "chunk", None)
        if chunk is None:
            return PhaseResult("reduce_overlap", "skipped", "No active chunk is available.", {})

        rows = self.disable_redundant_cameras(chunk, self.config)
        details: Dict[str, Any] = {"disabled_count": len(rows)}

        if self.config.use_builtin_reduce_overlap:
            # TODO: decide whether built-in reduceOverlap() should be part of the primary workflow.
            details.update(self.run_reduce_overlap_builtin(chunk, overlap=self.config.overlap_target))

        self.logs.write_csv(
            self.config.overlap_reduction_log_path,
            rows,
            headers=("camera_label", "reason", "kept_camera_label"),
        )
        return PhaseResult(
            phase="reduce_overlap",
            status="todo",
            message="Overlap reduction scaffold logged. TODO: implement custom pose-based redundancy filtering.",
            details=details,
        )

    @staticmethod
    def _require_metashape() -> None:
        if Metashape is None:
            raise PipelineError("Metashape module is not available in this Python runtime.")


class DualFisheyePipeline:
    """High-level orchestration for menu-triggered pipeline phases."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.logs = LogWriter()
        self.extractor = FFmpegExtractor(self.config, self.logs)
        self.blur_evaluator = BlurEvaluator(self.config, self.logs)
        self.mask_generator = MaskGenerator(self.config, self.logs)
        self.importer = MetashapeImporter(self.config)
        self.aligner = MetashapeAligner(self.config, self.logs)
        self.overlap_reducer = OverlapReducer(self.config, self.logs)
        configure_logging(self.config)

    def run_full_pipeline(self) -> PhaseResult:
        """Execute the current scaffold phases in sequence."""

        phase_results = [
            self.run_extract_streams(),
            self.run_select_frames(),
            self.run_generate_masks(),
            self.run_import_to_metashape(),
            self.run_align(),
            self.run_reduce_overlap(),
            self.run_export_logs(),
        ]
        overall_status = "ok"
        if any(result.status == "todo" for result in phase_results):
            overall_status = "todo"
        if any(result.status == "error" for result in phase_results):
            overall_status = "error"
        summary = PhaseResult(
            phase="run_full_pipeline",
            status=overall_status,
            message="Pipeline scaffold completed with per-phase details logged to work/logs.",
            details={"phases": [result.to_dict() for result in phase_results]},
        )
        self.logs.write_summary(self.config.summary_log_path, summary.to_dict())
        return summary

    def run_extract_streams(self) -> PhaseResult:
        return self._run_phase(self.extractor.run, "extract_streams")

    def run_select_frames(self) -> PhaseResult:
        return self._run_phase(
            lambda: self.blur_evaluator.select_pairs(
                self.config.extracted_front_dir,
                self.config.extracted_back_dir,
                self.config.selected_front_dir,
                self.config.selected_back_dir,
            ),
            "select_frames",
        )

    def run_generate_masks(self) -> PhaseResult:
        return self._run_phase(self.mask_generator.run, "generate_masks")

    def run_import_to_metashape(self) -> PhaseResult:
        return self._run_phase(self.importer.run, "import_to_metashape")

    def run_align(self) -> PhaseResult:
        return self._run_phase(self.aligner.run, "align")

    def run_reduce_overlap(self) -> PhaseResult:
        return self._run_phase(self.overlap_reducer.run, "reduce_overlap")

    def run_export_logs(self) -> PhaseResult:
        return self._run_phase(self._export_logs, "export_logs")

    def _export_logs(self) -> PhaseResult:
        """Write a simple summary snapshot of available logs."""

        summary = {
            "config": self.config.to_dict(),
            "existing_logs": sorted(path.name for path in self.config.log_dir.glob("*") if path.is_file()),
        }
        self.logs.write_summary(self.config.summary_log_path, summary)
        return PhaseResult(
            phase="export_logs",
            status="ok",
            message="Exported the current pipeline log summary.",
            details={"summary_path": self.config.summary_log_path},
        )

    def _run_phase(self, callback: Any, phase_name: str) -> PhaseResult:
        """Wrap each phase with common validation and error logging."""

        self.config.ensure_directories()
        try:
            self.config.validate(require_input=phase_name in ("extract_streams", "run_full_pipeline"))
            result = callback()
            LOGGER.info("%s: %s", result.phase, result.message)
            return result
        except Exception as exc:  # pragma: no cover - exercised mainly inside Metashape.
            LOGGER.error("%s failed: %s", phase_name, exc)
            LOGGER.debug("%s", traceback.format_exc())
            return PhaseResult(
                phase=phase_name,
                status="error",
                message=str(exc),
                details={"exception_type": exc.__class__.__name__},
            )


def _show_result(result: PhaseResult) -> None:
    """Display a compact phase summary."""

    message = "[{0}] {1}".format(result.status.upper(), result.message)
    LOGGER.info(message)
    if Metashape is not None:
        Metashape.app.messageBox(message)
    else:
        print(message)


def get_pipeline() -> DualFisheyePipeline:
    """Create a fresh pipeline instance for each menu invocation."""

    return DualFisheyePipeline()


def menu_run_full_pipeline() -> None:
    """Menu callback for the full pipeline."""

    _show_result(get_pipeline().run_full_pipeline())


def menu_extract_streams() -> None:
    """Menu callback for stream extraction."""

    _show_result(get_pipeline().run_extract_streams())


def menu_select_frames() -> None:
    """Menu callback for frame selection."""

    _show_result(get_pipeline().run_select_frames())


def menu_generate_masks() -> None:
    """Menu callback for mask generation."""

    _show_result(get_pipeline().run_generate_masks())


def menu_import_to_metashape() -> None:
    """Menu callback for Metashape import."""

    _show_result(get_pipeline().run_import_to_metashape())


def menu_align() -> None:
    """Menu callback for alignment."""

    _show_result(get_pipeline().run_align())


def menu_reduce_overlap() -> None:
    """Menu callback for overlap reduction."""

    _show_result(get_pipeline().run_reduce_overlap())


def menu_export_logs() -> None:
    """Menu callback for log export."""

    _show_result(get_pipeline().run_export_logs())


def register_menu_items() -> None:
    """Register the Custom/DualFisheye menu tree."""

    global _MENU_REGISTERED
    if Metashape is None or _MENU_REGISTERED:
        return

    menu_items = (
        ("01 Run Full Pipeline", menu_run_full_pipeline),
        ("02 Extract Streams", menu_extract_streams),
        ("03 Select Frames", menu_select_frames),
        ("04 Generate Masks", menu_generate_masks),
        ("05 Import to Metashape", menu_import_to_metashape),
        ("06 Align", menu_align),
        ("07 Reduce Overlap", menu_reduce_overlap),
        ("08 Export Logs", menu_export_logs),
    )
    pipeline = get_pipeline()
    for suffix, callback in menu_items:
        Metashape.app.addMenuItem("{0}/{1}".format(pipeline.config.menu_root, suffix), callback)
    _MENU_REGISTERED = True
    LOGGER.info("Registered Dual Fisheye menu items.")


if Metashape is not None:
    register_menu_items()
