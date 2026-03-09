"""Dual fisheye Metashape pipeline.

This module is intended to be loaded from the Metashape Python runtime.
Current implementation scope:

- Phase 1: ffprobe / ffmpeg extraction and paired frame selection.
- Phase 2: YOLO-based mask PNG generation and MultiplaneLayout import scaffolding.
- Phase 3: image quality analysis, alignment, overlap reduction, and phase summaries.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import shutil
import subprocess
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - depends on the Metashape runtime environment.
    cv2 = None  # type: ignore

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - depends on the Metashape runtime environment.
    np = None  # type: ignore

try:
    import Metashape  # type: ignore
except ImportError:  # pragma: no cover - Metashape is not available in local linting.
    Metashape = None  # type: ignore

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:  # pragma: no cover - depends on the Metashape runtime environment.
    YOLO = None  # type: ignore


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
    mask_model_path: str = "yolo11n-seg.pt"
    mask_classes: Tuple[str, ...] = ("person", "car", "truck", "bus", "motorbike")
    mask_dilate_px: int = 8
    mask_confidence_threshold: float = 0.25
    mask_iou_threshold: float = 0.45
    mask_device: Optional[str] = None
    metashape_image_quality_threshold: float = 0.5
    match_downscale: int = 1
    generic_preselection: bool = True
    reference_preselection: bool = False
    keep_keypoints: bool = False
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
    use_builtin_reduce_overlap: bool = False
    realign_after_overlap_reduction: bool = True

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

        if self.front_stream_index < 0 or self.back_stream_index < 0:
            raise ValueError("front_stream_index and back_stream_index must be >= 0.")
        if self.front_stream_index == self.back_stream_index:
            raise ValueError("front_stream_index and back_stream_index must differ.")
        if self.extract_every_n_frames < 1:
            raise ValueError("extract_every_n_frames must be >= 1.")
        if self.jpeg_quality < 1:
            raise ValueError("jpeg_quality must be >= 1.")
        if self.blur_method != "laplacian_center70":
            raise ValueError("Only blur_method='laplacian_center70' is supported in Phase 1.")
        if self.blur_threshold_front < 0 or self.blur_threshold_back < 0:
            raise ValueError("blur thresholds must be >= 0.")
        if self.keep_rule != "either_side_ok_keep_both":
            raise ValueError("keep_rule must remain 'either_side_ok_keep_both'.")
        if self.mask_model != "yolo":
            raise ValueError("mask_model must remain 'yolo' for the current implementation.")
        if not self.mask_model_path:
            raise ValueError("mask_model_path must be configured for YOLO-based mask generation.")
        if not self.mask_classes:
            raise ValueError("mask_classes must not be empty.")
        if self.mask_dilate_px < 0:
            raise ValueError("mask_dilate_px must be >= 0.")
        if not 0.0 <= self.mask_confidence_threshold <= 1.0:
            raise ValueError("mask_confidence_threshold must be between 0 and 1.")
        if not 0.0 <= self.mask_iou_threshold <= 1.0:
            raise ValueError("mask_iou_threshold must be between 0 and 1.")
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


def extract_frame_id_from_path(image_path: Path, prefix: str) -> int:
    """Extract the numeric frame ID from a prefixed filename."""

    stem = image_path.stem
    expected_prefix = "{0}_".format(prefix)
    if not stem.startswith(expected_prefix):
        raise PipelineError("Unexpected frame naming: {0}".format(image_path.name))
    try:
        return int(stem[len(expected_prefix) :])
    except ValueError as exc:
        raise PipelineError("Invalid frame ID in filename: {0}".format(image_path.name)) from exc


def extract_frame_id_from_label(label: str) -> int:
    """Extract the numeric frame ID from an imported camera label."""

    if not (label.startswith("F_") or label.startswith("B_")):
        raise PipelineError("Unexpected camera label: {0}".format(label))
    try:
        return int(label[2:])
    except ValueError as exc:
        raise PipelineError("Invalid camera label frame ID: {0}".format(label)) from exc


def camera_side_from_label(label: str) -> Optional[str]:
    """Resolve front/back side from an imported camera label."""

    if label.startswith("F_"):
        return "front"
    if label.startswith("B_"):
        return "back"
    return None


def index_frame_paths(directory: Path, prefix: str, suffix: str) -> Dict[int, Path]:
    """Index files by frame ID for pair-aware processing."""

    pattern = "{0}_*{1}".format(prefix, suffix)
    frames: Dict[int, Path] = {}
    for image_path in sorted(directory.glob(pattern)):
        frame_id = extract_frame_id_from_path(image_path, prefix)
        if frame_id in frames:
            raise PipelineError("Duplicate frame ID detected: {0}".format(image_path.name))
        frames[frame_id] = image_path
    return frames


def collect_frame_pairs(
    front_dir: Path,
    back_dir: Path,
    front_suffix: str = ".jpg",
    back_suffix: str = ".jpg",
) -> List[Tuple[int, Path, Path]]:
    """Collect front/back files by shared frame ID while preserving pair order."""

    front_files = index_frame_paths(front_dir, "F", front_suffix)
    back_files = index_frame_paths(back_dir, "B", back_suffix)
    if not front_files or not back_files:
        raise PipelineError("No paired frame files found for front/back processing.")

    front_ids = set(front_files)
    back_ids = set(back_files)
    if front_ids != back_ids:
        missing_front = sorted(back_ids - front_ids)
        missing_back = sorted(front_ids - back_ids)
        raise PipelineError(
            "Front/back frame IDs do not match. missing_front={0}, missing_back={1}".format(
                missing_front[:5], missing_back[:5]
            )
        )

    return [
        (frame_id, front_files[frame_id], back_files[frame_id])
        for frame_id in sorted(front_files)
    ]


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


def save_metashape_document(config: PipelineConfig) -> None:
    """Persist the active Metashape document to the configured project path."""

    if Metashape is None:
        raise PipelineError("Metashape module is not available in this Python runtime.")
    doc = Metashape.app.document
    config.project_path.parent.mkdir(parents=True, exist_ok=True)
    if getattr(doc, "path", ""):
        doc.save()
        return
    doc.save(str(config.project_path))


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


@dataclass
class CameraStation:
    """Grouped front/back cameras for a single timestamp."""

    frame_id: int
    cameras: Dict[str, Any] = field(default_factory=dict)
    blur_scores: Dict[str, Optional[float]] = field(default_factory=dict)

    def station_label(self) -> str:
        return "station_{0:06d}".format(self.frame_id)


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
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise PipelineError("ffprobe did not return valid JSON output.") from exc
        self.logs.write_json(self.config.ffprobe_log_path, payload)

        streams = payload.get("streams", [])
        video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
        if len(video_streams) < 2:
            raise PipelineError("Expected a 2-stream MP4, but found fewer than 2 video streams.")
        for stream_index, side in (
            (self.config.front_stream_index, "front"),
            (self.config.back_stream_index, "back"),
        ):
            if stream_index >= len(video_streams):
                raise PipelineError(
                    "{0}_stream_index={1} is out of range for {2} detected video streams.".format(
                        side, stream_index, len(video_streams)
                    )
                )
        return payload

    def extract_front_stream(self, mp4_path: Path, out_dir: Path, stream_index: int) -> None:
        """Extract the configured front stream."""

        self._extract_stream(mp4_path, out_dir, stream_index, "F")

    def extract_back_stream(self, mp4_path: Path, out_dir: Path, stream_index: int) -> None:
        """Extract the configured back stream."""

        self._extract_stream(mp4_path, out_dir, stream_index, "B")

    def verify_frame_counts(self, front_dir: Path, back_dir: Path) -> Tuple[int, int]:
        """Ensure extracted front/back frame counts match."""

        front_count = len(self._list_frame_files(front_dir, "F"))
        back_count = len(self._list_frame_files(back_dir, "B"))
        if front_count == 0 or back_count == 0:
            raise PipelineError("ffmpeg extraction produced no JPEG frames.")
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
        self._clear_directory(out_dir, "{0}_*.jpg".format(prefix))

        output_pattern = out_dir / "{0}_%06d.jpg".format(prefix)
        command = ["ffmpeg", "-y", "-i", str(mp4_path), "-map", "0:v:{0}".format(stream_index)]
        command.extend(self._build_frame_sampling_args())

        command.extend(["-q:v", str(self.config.jpeg_quality), str(output_pattern)])
        self._run_command(command)

        if not self._list_frame_files(out_dir, prefix):
            raise PipelineError(
                "ffmpeg did not produce extracted frames for stream index {0}.".format(stream_index)
            )

    def _build_frame_sampling_args(self) -> List[str]:
        """Build ffmpeg arguments for frame sampling.

        This is intentionally isolated so extract_every_n_frames can grow later
        without rewriting the front/back extraction code paths.
        """

        if self.config.extract_every_n_frames > 1:
            return [
                "-vf",
                "select=not(mod(n\\,{0}))".format(self.config.extract_every_n_frames),
                "-vsync",
                "vfr",
            ]
        return ["-vsync", "0"]

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

    @staticmethod
    def _clear_directory(directory: Path, pattern: str) -> None:
        for old_file in directory.glob(pattern):
            old_file.unlink()

    @staticmethod
    def _list_frame_files(directory: Path, prefix: str) -> List[Path]:
        return sorted(directory.glob("{0}_*.jpg".format(prefix)))


class BlurEvaluator:
    """Frame scoring and paired selection for Phase 1."""

    def __init__(self, config: PipelineConfig, logs: LogWriter) -> None:
        self.config = config
        self.logs = logs

    def compute_center70_mask(self, image: Any) -> Any:
        """Build a circular mask centered in the image with a 70% diameter."""

        self._require_cv_runtime()
        if image is None or getattr(image, "size", 0) == 0:
            raise PipelineError("Cannot compute a blur mask for an empty image.")

        height, width = image.shape[:2]
        radius = max(1, int(min(height, width) * 0.35))
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (width // 2, height // 2), radius, 255, thickness=-1)
        return mask

    def laplacian_score(self, image: Any) -> Optional[float]:
        """Compute Laplacian variance within the center-70-percent circle."""

        self._require_cv_runtime()
        mask = self.compute_center70_mask(image)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        masked_values = laplacian[mask > 0]
        if masked_values.size == 0:
            raise PipelineError("Center-70-percent mask produced no pixels for blur evaluation.")
        return float(masked_values.var())

    def fft_blur_score(self, image: Any) -> Optional[float]:
        """Placeholder for optional FFT-based blur scoring."""

        # TODO: implement optional FFT blur scoring once the Laplacian baseline is validated.
        del image
        return None

    def evaluate_pair(self, front_path: Path, back_path: Path) -> Dict[str, Any]:
        """Evaluate a front/back frame pair and apply the paired keep rule."""

        front_frame_id = extract_frame_id_from_path(front_path, "F")
        back_frame_id = extract_frame_id_from_path(back_path, "B")
        if front_frame_id != back_frame_id:
            raise PipelineError(
                "Frame ID mismatch between paired images: {0} vs {1}".format(
                    front_path.name, back_path.name
                )
            )

        front_score = self._score_image(front_path)
        back_score = self._score_image(back_path)
        keep_pair = int(
            front_score >= self.config.blur_threshold_front
            or back_score >= self.config.blur_threshold_back
        )
        return {
            "frame_id": front_frame_id,
            "front_path": front_path.name,
            "back_path": back_path.name,
            "front_score": round(front_score, 4),
            "back_score": round(back_score, 4),
            "keep_pair": keep_pair,
            "better_side": self._better_side(front_score, back_score),
        }

    def select_pairs(self, front_dir: Path, back_dir: Path, out_front_dir: Path, out_back_dir: Path) -> PhaseResult:
        """Evaluate extracted pairs, keep both sides when either side passes, and log the results."""

        pairs = self._collect_pairs(front_dir, back_dir)

        out_front_dir.mkdir(parents=True, exist_ok=True)
        out_back_dir.mkdir(parents=True, exist_ok=True)
        self._clear_directory(out_front_dir, "F_*.jpg")
        self._clear_directory(out_back_dir, "B_*.jpg")

        rows: List[Dict[str, Any]] = []
        selected_pairs = 0
        for front_path, back_path in pairs:
            row = self.evaluate_pair(front_path, back_path)
            if row["keep_pair"]:
                shutil.copy2(str(front_path), str(out_front_dir / front_path.name))
                shutil.copy2(str(back_path), str(out_back_dir / back_path.name))
                selected_pairs += 1
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
        if selected_pairs == 0:
            raise PipelineError(
                "No frame pairs met the blur thresholds. See {0}.".format(
                    self.config.frame_quality_log_path
                )
            )
        return PhaseResult(
            phase="select_frames",
            status="ok",
            message="Selected paired frames using the either-side-OK keep-both rule.",
            details={
                "total_pairs": len(rows),
                "selected_pairs": selected_pairs,
                "rejected_pairs": len(rows) - selected_pairs,
            },
        )

    @staticmethod
    def _clear_directory(directory: Path, pattern: str) -> None:
        """Remove generated files for a clean rerun."""

        for old_file in directory.glob(pattern):
            old_file.unlink()

    def _collect_pairs(self, front_dir: Path, back_dir: Path) -> List[Tuple[Path, Path]]:
        """Collect and validate front/back pairs by frame ID."""

        return [(front_path, back_path) for _, front_path, back_path in collect_frame_pairs(front_dir, back_dir)]

    def _index_frames(self, directory: Path, prefix: str) -> Dict[int, Path]:
        """Index extracted JPEGs by numeric frame ID."""

        return index_frame_paths(directory, prefix, ".jpg")

    def _score_image(self, image_path: Path) -> float:
        """Read an image and compute the configured primary blur score."""

        image = self._read_grayscale_image(image_path)
        if self.config.blur_method == "laplacian_center70":
            score = self.laplacian_score(image)
        else:
            raise PipelineError("Unsupported blur_method: {0}".format(self.config.blur_method))
        if score is None:
            raise PipelineError("Blur score could not be computed for {0}".format(image_path))
        return float(score)

    def _read_grayscale_image(self, image_path: Path) -> Any:
        """Read an image from disk as grayscale."""

        self._require_cv_runtime()
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise PipelineError("Failed to read image for blur evaluation: {0}".format(image_path))
        return image

    @staticmethod
    def _better_side(front_score: float, back_score: float) -> str:
        if front_score > back_score:
            return "front"
        if back_score > front_score:
            return "back"
        return "tie"

    @staticmethod
    def _extract_frame_id(image_path: Path, prefix: str) -> int:
        return extract_frame_id_from_path(image_path, prefix)

    @staticmethod
    def _require_cv_runtime() -> None:
        if cv2 is None or np is None:
            raise PipelineError("OpenCV and numpy are required for Phase 1 blur evaluation.")


class MaskGenerator:
    """Generate binary mask PNGs from YOLO segmentation output."""

    CLASS_NAME_ALIASES = {
        "person": ("person",),
        "car": ("car",),
        "truck": ("truck",),
        "bus": ("bus",),
        "motorbike": ("motorbike", "motorcycle"),
    }

    def __init__(self, config: PipelineConfig, logs: LogWriter) -> None:
        self.config = config
        self.logs = logs
        self._model: Optional[Any] = None

    def load_model(self) -> Any:
        """Load the configured YOLO segmentation model once per run."""

        self._require_yolo_runtime()
        if self._model is None:
            try:
                self._model = YOLO(self.config.mask_model_path)
            except Exception as exc:
                raise PipelineError(
                    "Failed to load YOLO model from '{0}'.".format(self.config.mask_model_path)
                ) from exc
        return self._model

    def infer_mask(self, image_path: Path) -> Dict[str, Any]:
        """Run YOLO segmentation and return a binary mask summary."""

        self._require_cv_runtime()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise PipelineError("Failed to read image for mask generation: {0}".format(image_path))

        model = self.load_model()
        predict_kwargs: Dict[str, Any] = {
            "source": str(image_path),
            "verbose": False,
            "conf": self.config.mask_confidence_threshold,
            "iou": self.config.mask_iou_threshold,
        }
        if self.config.mask_device:
            predict_kwargs["device"] = self.config.mask_device

        try:
            results = model.predict(**predict_kwargs)
        except Exception as exc:
            raise PipelineError("YOLO inference failed for {0}.".format(image_path.name)) from exc
        if not results:
            raise PipelineError("YOLO did not return a result for {0}.".format(image_path.name))

        result = results[0]
        class_names = self._class_name_map(getattr(result, "names", {}))
        target_class_ids = self._target_class_ids(class_names)
        binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        target_classes: List[str] = []
        detection_count = 0

        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "cls", None) is None:
            return self._mask_result(binary_mask, detection_count, target_classes)

        selected_indexes = self._selected_detection_indexes(boxes.cls, target_class_ids)
        masks = getattr(result, "masks", None)
        if selected_indexes and masks is None:
            raise PipelineError(
                "YOLO result for {0} does not include segmentation masks. Use segmentation weights.".format(
                    image_path.name
                )
            )

        mask_data = getattr(masks, "data", None) if masks is not None else None
        if mask_data is not None:
            class_ids = self._to_list(boxes.cls)
            for detection_index in selected_indexes:
                detection_mask = self._mask_array(mask_data[detection_index], image.shape[:2])
                binary_mask = np.maximum(binary_mask, detection_mask)
                class_id = int(class_ids[detection_index])
                resolved_name = target_class_ids.get(class_id, "")
                if resolved_name:
                    target_classes.append(resolved_name)
                detection_count += 1

        if np.any(binary_mask):
            binary_mask = self.dilate_mask(binary_mask, self.config.mask_dilate_px)
        return self._mask_result(binary_mask, detection_count, target_classes)

    def dilate_mask(self, mask: Any, px: int) -> Any:
        """Dilate a binary mask with an elliptical kernel."""

        self._require_cv_runtime()
        if px <= 0:
            return mask
        kernel_size = px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate(mask, kernel, iterations=1)

    def save_mask(self, mask: Any, out_path: Path) -> None:
        """Persist a binary PNG mask to disk."""

        self._require_cv_runtime()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mask_to_save = np.where(mask > 0, 255, 0).astype(np.uint8)
        if not cv2.imwrite(str(out_path), mask_to_save):
            raise PipelineError("Failed to save mask PNG: {0}".format(out_path))

    def process_image(
        self,
        frame_id: int,
        image_path: Path,
        out_mask_dir: Path,
        camera_side: str,
    ) -> Dict[str, Any]:
        """Generate and save the mask for a single selected image."""

        result = self.infer_mask(image_path)
        mask_path = out_mask_dir / "{0}.png".format(image_path.stem)
        self.save_mask(result["mask"], mask_path)
        return {
            "frame_id": frame_id,
            "camera_side": camera_side,
            "image_path": image_path.name,
            "mask_path": str(mask_path),
            "status": "ok" if result["detection_count"] else "ok_blank",
            "target_detections": result["detection_count"],
            "target_classes": ",".join(result["target_classes"]),
            "masked_pixels": result["masked_pixels"],
            "dilate_px": self.config.mask_dilate_px,
        }

    def run(self) -> PhaseResult:
        """Generate binary mask PNGs for each selected front/back pair."""

        pairs = collect_frame_pairs(self.config.selected_front_dir, self.config.selected_back_dir)
        self._clear_directory(self.config.mask_front_dir, "F_*.png")
        self._clear_directory(self.config.mask_back_dir, "B_*.png")

        rows: List[Dict[str, Any]] = []
        masked_images = 0
        for frame_id, front_path, back_path in pairs:
            front_row = self.process_image(frame_id, front_path, self.config.mask_front_dir, "front")
            back_row = self.process_image(frame_id, back_path, self.config.mask_back_dir, "back")
            rows.extend((front_row, back_row))
            if front_row["masked_pixels"] > 0:
                masked_images += 1
            if back_row["masked_pixels"] > 0:
                masked_images += 1

        self.logs.write_csv(
            self.config.mask_summary_log_path,
            rows,
            headers=(
                "frame_id",
                "camera_side",
                "image_path",
                "mask_path",
                "status",
                "target_detections",
                "target_classes",
                "masked_pixels",
                "dilate_px",
            ),
        )
        return PhaseResult(
            phase="generate_masks",
            status="ok",
            message="Generated binary mask PNGs for selected front/back frame pairs.",
            details={
                "pair_count": len(pairs),
                "image_count": len(rows),
                "masked_image_count": masked_images,
                "mask_summary_csv": self.config.mask_summary_log_path,
            },
        )

    @staticmethod
    def _clear_directory(directory: Path, pattern: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for old_file in directory.glob(pattern):
            old_file.unlink()

    @classmethod
    def _canonical_mask_class(cls, class_name: str) -> str:
        normalized_name = str(class_name).strip().lower()
        for canonical_name, aliases in cls.CLASS_NAME_ALIASES.items():
            if normalized_name in aliases:
                return canonical_name
        return normalized_name

    def _normalized_target_classes(self) -> Set[str]:
        return {self._canonical_mask_class(name) for name in self.config.mask_classes}

    @staticmethod
    def _class_name_map(names: Any) -> Dict[int, str]:
        if isinstance(names, dict):
            return {int(index): str(name) for index, name in names.items()}
        if isinstance(names, (list, tuple)):
            return {index: str(name) for index, name in enumerate(names)}
        return {}

    def _target_class_ids(self, class_names: Mapping[int, str]) -> Dict[int, str]:
        selected_classes = self._normalized_target_classes()
        resolved: Dict[int, str] = {}
        for class_id, class_name in class_names.items():
            canonical_name = self._canonical_mask_class(class_name)
            if canonical_name in selected_classes:
                resolved[class_id] = canonical_name
        return resolved

    def _selected_detection_indexes(self, class_ids_tensor: Any, target_class_ids: Mapping[int, str]) -> List[int]:
        selected_indexes: List[int] = []
        for detection_index, class_id in enumerate(self._to_list(class_ids_tensor)):
            if int(class_id) in target_class_ids:
                selected_indexes.append(detection_index)
        return selected_indexes

    @staticmethod
    def _to_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            return list(value.tolist())
        return list(value)

    @staticmethod
    def _mask_array(mask_tensor: Any, shape: Tuple[int, int]) -> Any:
        mask_array = mask_tensor
        if hasattr(mask_array, "cpu"):
            mask_array = mask_array.cpu()
        if hasattr(mask_array, "numpy"):
            mask_array = mask_array.numpy()
        mask_array = np.where(mask_array > 0.5, 255, 0).astype(np.uint8)
        if mask_array.shape != shape:
            mask_array = cv2.resize(mask_array, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask_array

    @staticmethod
    def _mask_result(mask: Any, detection_count: int, target_classes: Sequence[str]) -> Dict[str, Any]:
        unique_classes = sorted(set(class_name for class_name in target_classes if class_name))
        return {
            "mask": np.where(mask > 0, 255, 0).astype(np.uint8),
            "detection_count": detection_count,
            "target_classes": unique_classes,
            "masked_pixels": int(np.count_nonzero(mask)),
        }

    @staticmethod
    def _require_cv_runtime() -> None:
        if cv2 is None or np is None:
            raise PipelineError("OpenCV and numpy are required for mask generation.")

    @staticmethod
    def _require_yolo_runtime() -> None:
        if YOLO is None:
            raise PipelineError(
                "Ultralytics YOLO is not available in this Python runtime. Install it in Metashape Python."
            )


class MetashapeImporter:
    """Import paired front/back images into Metashape and apply camera-level masks."""

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
        """Build an interleaved front/back filename sequence for MultiplaneLayout."""

        filenames: List[str] = []
        for _, front_path, back_path in collect_frame_pairs(front_dir, back_dir):
            filenames.extend([str(front_path), str(back_path)])
        return filenames

    def build_filegroups(self, pair_count: int) -> List[int]:
        """Build filegroups aligned to the interleaved filename sequence."""

        # TODO: validate filenames/filegroups layout on current Metashape build.
        return [2] * pair_count

    def import_multiplane_images(self, chunk: Any, filenames: Sequence[str], filegroups: Sequence[int]) -> None:
        """Import paired images using the current MultiplaneLayout plan."""

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

    def apply_masks_from_disk(self, chunk: Any, mask_front_dir: Path, mask_back_dir: Path) -> Tuple[int, List[str]]:
        """Attach per-camera masks from disk using camera labels."""

        applied_count = 0
        missing_labels: List[str] = []
        for camera in getattr(chunk, "cameras", []):
            camera_label = getattr(camera, "label", "")
            if not camera_label:
                continue
            mask_path = self._mask_path_for_camera(camera_label, mask_front_dir, mask_back_dir)
            if mask_path is None:
                continue
            if not mask_path.exists():
                missing_labels.append(camera_label)
                continue
            # TODO: re-check camera.mask assignment on the current Metashape build with a small sample.
            mask = Metashape.Mask()
            mask.load(str(mask_path))
            camera.mask = mask
            applied_count += 1
        return applied_count, missing_labels

    def build_small_sample(self, front_dir: Path, back_dir: Path, sample_pairs: int = 4) -> List[Dict[str, Any]]:
        """Return a compact sample of the import plan for current-build validation."""

        pairs = collect_frame_pairs(front_dir, back_dir)[:sample_pairs]
        sample: List[Dict[str, Any]] = []
        for frame_id, front_path, back_path in pairs:
            sample.append(
                {
                    "frame_id": frame_id,
                    "filenames": [str(front_path), str(back_path)],
                    "filegroup": 2,
                }
            )
        return sample

    def validate_import_plan(self, filenames: Sequence[str], filegroups: Sequence[int]) -> None:
        """Validate local consistency before calling addPhotos."""

        if not filenames:
            raise PipelineError("No filenames were built for Metashape import.")
        if not filegroups:
            raise PipelineError("No filegroups were built for Metashape import.")
        if sum(filegroups) != len(filenames):
            raise PipelineError(
                "filenames/filegroups mismatch: len(filenames)={0}, sum(filegroups)={1}".format(
                    len(filenames), sum(filegroups)
                )
            )

    def expected_camera_labels(self, front_dir: Path, back_dir: Path) -> List[str]:
        """Return the expected camera labels after import."""

        labels: List[str] = []
        for _, front_path, back_path in collect_frame_pairs(front_dir, back_dir):
            labels.extend([front_path.stem, back_path.stem])
        return labels

    def detect_existing_import_state(self, chunk: Any, expected_labels: Sequence[str]) -> str:
        """Protect against accidental duplicate or partial re-imports."""

        if not getattr(chunk, "cameras", []):
            return "empty"
        existing_labels = {getattr(camera, "label", "") for camera in getattr(chunk, "cameras", [])}
        expected_label_set = set(expected_labels)
        if expected_label_set.issubset(existing_labels):
            return "complete"
        if existing_labels & expected_label_set:
            return "partial"
        return "other"

    def save_document(self, doc: Any) -> None:
        """Save the project after import."""

        del doc
        save_metashape_document(self.config)

    def run(self) -> PhaseResult:
        """Create or reuse a chunk, import paired images, and apply per-camera masks."""

        self._require_metashape()
        doc = Metashape.app.document
        chunk = self.create_or_get_chunk(doc, name=self.config.chunk_name)
        pairs = collect_frame_pairs(self.config.selected_front_dir, self.config.selected_back_dir)

        filenames = self.build_filename_sequence(self.config.selected_front_dir, self.config.selected_back_dir)
        filegroups = self.build_filegroups(len(pairs))
        self.validate_import_plan(filenames, filegroups)

        expected_labels = self.expected_camera_labels(self.config.selected_front_dir, self.config.selected_back_dir)
        import_state = self.detect_existing_import_state(chunk, expected_labels)
        if import_state == "other":
            raise PipelineError(
                "Chunk '{0}' already contains cameras unrelated to the current selected frame set. "
                "Use a clean chunk label before importing.".format(getattr(chunk, "label", self.config.chunk_name))
            )
        if import_state == "partial":
            raise PipelineError(
                "Chunk '{0}' already contains a partial subset of the selected camera labels. "
                "Use a clean chunk label before re-importing.".format(getattr(chunk, "label", self.config.chunk_name))
            )

        imported_now = False
        if import_state != "complete":
            self.import_multiplane_images(chunk, filenames, filegroups)
            imported_now = True

        self.set_sensor_types(chunk)
        self.apply_rig_reference(chunk, self.config)
        applied_masks, missing_labels = self.apply_masks_from_disk(
            chunk, self.config.mask_front_dir, self.config.mask_back_dir
        )
        if missing_labels:
            raise PipelineError(
                "Mask PNGs are missing for {0} camera(s). Run Generate Masks first.".format(len(missing_labels))
            )

        self.save_document(doc)
        return PhaseResult(
            phase="import_to_metashape",
            status="ok",
            message="Imported selected front/back pairs using MultiplaneLayout and applied per-camera masks.",
            details={
                "chunk_label": getattr(chunk, "label", self.config.chunk_name),
                "pair_count": len(pairs),
                "camera_count": len(getattr(chunk, "cameras", [])),
                "imported_now": imported_now,
                "applied_masks": applied_masks,
                "sample_pairs": self.build_small_sample(
                    self.config.selected_front_dir,
                    self.config.selected_back_dir,
                    sample_pairs=min(4, len(pairs)),
                ),
            },
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
        """Disable cameras whose Image/Quality falls below threshold."""

        disabled = 0
        for camera in getattr(chunk, "cameras", []):
            quality = self.camera_quality(camera)
            if quality is None:
                continue
            if quality < threshold:
                if not getattr(camera, "enabled", True):
                    continue
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
                    "frame_id": self._frame_id_for_camera(camera),
                    "camera_side": self._side_for_camera(camera),
                    "enabled": getattr(camera, "enabled", True),
                    "aligned": self.camera_is_aligned(camera),
                    "image_quality": getattr(camera, "meta", {}).get("Image/Quality", ""),
                }
            )
        self.logs.write_csv(
            csv_path,
            rows,
            headers=("camera_label", "frame_id", "camera_side", "enabled", "aligned", "image_quality"),
        )

    def match_photos(self, chunk: Any, config: PipelineConfig, reset_matches: bool = False) -> None:
        """Run feature matching with current argument names from the checklist."""

        chunk.matchPhotos(
            downscale=config.match_downscale,
            generic_preselection=config.generic_preselection,
            reference_preselection=config.reference_preselection,
            filter_mask=True,
            mask_tiepoints=True,
            filter_stationary_points=config.filter_stationary_points,
            keep_keypoints=config.keep_keypoints,
            keypoint_limit=config.keypoint_limit,
            tiepoint_limit=config.tiepoint_limit,
            reset_matches=reset_matches,
        )

    def align_cameras(self, chunk: Any, reset_alignment: bool = False) -> None:
        """Run initial camera alignment."""

        chunk.alignCameras(reset_alignment=reset_alignment)

    def realign_after_cleanup(self, chunk: Any, config: PipelineConfig) -> None:
        """Rerun alignment after overlap cleanup."""

        if sum(1 for camera in getattr(chunk, "cameras", []) if getattr(camera, "enabled", True)) < 2:
            return
        self.match_photos(chunk, config, reset_matches=True)
        # TODO: validate alignCameras(reset_alignment=True) behavior on the current Metashape build.
        self.align_cameras(chunk, reset_alignment=True)

    def run(self) -> PhaseResult:
        """Analyze quality and align the active chunk when cameras are available."""

        self._require_metashape()
        chunk = getattr(Metashape.app.document, "chunk", None)
        if chunk is None:
            return PhaseResult("align", "skipped", "No active chunk is available.", {})
        if not getattr(chunk, "cameras", []):
            return PhaseResult("align", "skipped", "Active chunk has no cameras to align.", {})

        self.analyze_image_quality(chunk)
        disabled_count = self.disable_low_quality_cameras(chunk, threshold=self.config.metashape_image_quality_threshold)
        self.export_quality_log(chunk, self.config.metashape_quality_log_path)
        enabled_camera_count = sum(1 for camera in getattr(chunk, "cameras", []) if getattr(camera, "enabled", True))
        if enabled_camera_count < 2:
            raise PipelineError("Fewer than two enabled cameras remain after Image/Quality filtering.")
        self.match_photos(chunk, self.config)
        self.align_cameras(chunk)
        self.export_quality_log(chunk, self.config.metashape_quality_log_path)
        save_metashape_document(self.config)
        return PhaseResult(
            phase="align",
            status="ok",
            message="Ran image quality analysis, disabled low-quality cameras, and aligned the active chunk.",
            details={
                "camera_count": len(getattr(chunk, "cameras", [])),
                "enabled_camera_count": enabled_camera_count,
                "disabled_low_quality_count": disabled_count,
                "aligned_camera_count": self.aligned_camera_count(chunk),
                "quality_csv": self.config.metashape_quality_log_path,
            },
        )

    @staticmethod
    def camera_quality(camera: Any) -> Optional[float]:
        value = getattr(camera, "meta", {}).get("Image/Quality")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def camera_is_aligned(camera: Any) -> bool:
        return getattr(camera, "center", None) is not None and getattr(camera, "transform", None) is not None

    def aligned_camera_count(self, chunk: Any) -> int:
        return sum(1 for camera in getattr(chunk, "cameras", []) if self.camera_is_aligned(camera))

    @staticmethod
    def _frame_id_for_camera(camera: Any) -> Optional[int]:
        label = getattr(camera, "label", "")
        if not label:
            return None
        side = camera_side_from_label(label)
        if side is None:
            return None
        return extract_frame_id_from_label(label)

    @staticmethod
    def _side_for_camera(camera: Any) -> str:
        side = camera_side_from_label(getattr(camera, "label", ""))
        return side or ""

    @staticmethod
    def _require_metashape() -> None:
        if Metashape is None:
            raise PipelineError("Metashape module is not available in this Python runtime.")


class OverlapReducer:
    """Post-alignment overlap reduction with station-level pair preservation."""

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
        """Return the current camera rotation component."""

        transform = getattr(camera, "transform", None)
        if transform is None:
            return None
        if hasattr(transform, "rotation"):
            return transform.rotation()
        return transform

    def distance_between(self, cam_a: Any, cam_b: Any) -> Optional[float]:
        """Compute translation distance from aligned camera centers."""

        center_a = self.camera_center(cam_a)
        center_b = self.camera_center(cam_b)
        if center_a is None or center_b is None:
            return None
        vec_a = self._vector_xyz(center_a)
        vec_b = self._vector_xyz(center_b)
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))

    def angle_between(self, cam_a: Any, cam_b: Any) -> Optional[float]:
        """Compute angular delta in degrees between aligned camera rotations."""

        rotation_a = self.camera_rotation(cam_a)
        rotation_b = self.camera_rotation(cam_b)
        matrix_a = self._matrix3(rotation_a)
        matrix_b = self._matrix3(rotation_b)
        if matrix_a is None or matrix_b is None:
            return None
        trace = sum(
            sum(matrix_a[row_index][column_index] * matrix_b[row_index][column_index] for column_index in range(3))
            for row_index in range(3)
        )
        cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
        return math.degrees(math.acos(cosine))

    def disable_redundant_cameras(self, chunk: Any, config: PipelineConfig) -> List[Dict[str, Any]]:
        """Disable redundant aligned stations while preserving front/back pairing."""

        stations = self._build_stations(chunk)
        if len(stations) < 2:
            return []

        rows: List[Dict[str, Any]] = []
        current_station = stations[0]
        for next_station in stations[1:]:
            comparison = self._comparison_metrics(current_station, next_station)
            if comparison is None:
                current_station = next_station
                continue
            if not self._is_redundant(comparison, config):
                current_station = next_station
                continue

            kept_station, disabled_station = self._choose_station_to_keep(current_station, next_station)
            rows.extend(self._disable_station(disabled_station, kept_station, comparison))
            current_station = kept_station
        return rows

    def run_reduce_overlap_builtin(self, chunk: Any, overlap: int = 3) -> Dict[str, Any]:
        """Optionally exercise Metashape's built-in overlap reduction."""

        before_count = len(self.get_enabled_cameras(chunk))
        chunk.reduceOverlap(overlap=overlap)
        after_count = len(self.get_enabled_cameras(chunk))
        return {"before_count": before_count, "after_count": after_count}

    def run(self) -> PhaseResult:
        """Disable redundant aligned stations and optionally realign."""

        self._require_metashape()
        chunk = getattr(Metashape.app.document, "chunk", None)
        if chunk is None:
            return PhaseResult("reduce_overlap", "skipped", "No active chunk is available.", {})

        rows = self.disable_redundant_cameras(chunk, self.config)
        disabled_station_count = len({row["disabled_frame_id"] for row in rows}) if rows else 0
        details: Dict[str, Any] = {
            "disabled_camera_count": len(rows),
            "disabled_station_count": disabled_station_count,
            "overlap_csv": self.config.overlap_reduction_log_path,
        }

        realigned = False
        if rows and self.config.realign_after_overlap_reduction and len(self.get_enabled_cameras(chunk)) >= 2:
            aligner = MetashapeAligner(self.config, self.logs)
            aligner.realign_after_cleanup(chunk, self.config)
            realigned = True
            details["aligned_camera_count_after_realign"] = aligner.aligned_camera_count(chunk)

        if self.config.use_builtin_reduce_overlap:
            # TODO: decide whether built-in reduceOverlap() should be part of the primary workflow.
            details.update(self.run_reduce_overlap_builtin(chunk, overlap=self.config.overlap_target))

        self.logs.write_csv(
            self.config.overlap_reduction_log_path,
            rows,
            headers=(
                "disabled_frame_id",
                "disabled_station_label",
                "camera_label",
                "camera_side",
                "previously_enabled",
                "kept_frame_id",
                "kept_station_label",
                "kept_camera_label",
                "distance",
                "angle_deg",
                "disabled_station_quality",
                "kept_station_quality",
                "disabled_station_blur_score",
                "kept_station_blur_score",
                "reason",
            ),
        )
        save_metashape_document(self.config)
        return PhaseResult(
            phase="reduce_overlap",
            status="ok",
            message="Reduced redundant aligned stations using distance and rotation thresholds.",
            details={**details, "realigned": realigned},
        )

    def _build_stations(self, chunk: Any) -> List[CameraStation]:
        blur_scores_by_frame = self._load_blur_scores_by_frame()
        stations: Dict[int, CameraStation] = {}
        for camera in self.get_enabled_cameras(chunk):
            label = getattr(camera, "label", "")
            side = camera_side_from_label(label)
            if side is None:
                continue
            frame_id = extract_frame_id_from_label(label)
            station = stations.setdefault(
                frame_id,
                CameraStation(frame_id=frame_id, blur_scores=blur_scores_by_frame.get(frame_id, {})),
            )
            station.cameras[side] = camera
        return [station for _, station in sorted(stations.items()) if self._station_has_pose(station)]

    def _load_blur_scores_by_frame(self) -> Dict[int, Dict[str, Optional[float]]]:
        if not self.config.frame_quality_log_path.exists():
            return {}
        with self.config.frame_quality_log_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            scores: Dict[int, Dict[str, Optional[float]]] = {}
            for row in reader:
                try:
                    frame_id = int(str(row.get("frame_id", "")).strip())
                except ValueError:
                    continue
                scores[frame_id] = {
                    "front": self._parse_float(row.get("front_score")),
                    "back": self._parse_float(row.get("back_score")),
                }
        return scores

    def _comparison_metrics(self, station_a: CameraStation, station_b: CameraStation) -> Optional[Dict[str, Any]]:
        comparison_pair = self._comparison_camera_pair(station_a, station_b)
        if comparison_pair is None:
            return None
        side_name, cam_a, cam_b = comparison_pair
        distance = self.distance_between(cam_a, cam_b)
        angle = self.angle_between(cam_a, cam_b)
        if distance is None or angle is None:
            return None
        return {
            "side": side_name,
            "distance": distance,
            "angle_deg": angle,
            "camera_a_label": getattr(cam_a, "label", ""),
            "camera_b_label": getattr(cam_b, "label", ""),
        }

    def _comparison_camera_pair(
        self, station_a: CameraStation, station_b: CameraStation
    ) -> Optional[Tuple[str, Any, Any]]:
        for side_name in ("front", "back"):
            camera_a = station_a.cameras.get(side_name)
            camera_b = station_b.cameras.get(side_name)
            if self._camera_has_pose(camera_a) and self._camera_has_pose(camera_b):
                return side_name, camera_a, camera_b

        fallback_a = self._best_pose_camera(station_a)
        fallback_b = self._best_pose_camera(station_b)
        if fallback_a is None or fallback_b is None:
            return None
        return "mixed", fallback_a, fallback_b

    def _is_redundant(self, comparison: Mapping[str, Any], config: PipelineConfig) -> bool:
        return (
            float(comparison["distance"]) < config.camera_distance_threshold
            and float(comparison["angle_deg"]) < config.camera_angle_threshold_deg
        )

    def _choose_station_to_keep(
        self, station_a: CameraStation, station_b: CameraStation
    ) -> Tuple[CameraStation, CameraStation]:
        if self._station_rank(station_a) >= self._station_rank(station_b):
            return station_a, station_b
        return station_b, station_a

    def _station_rank(self, station: CameraStation) -> Tuple[float, float, int, int]:
        return (
            self._station_quality(station),
            self._station_blur_score(station),
            self._station_pose_count(station),
            -station.frame_id,
        )

    def _disable_station(
        self, disabled_station: CameraStation, kept_station: CameraStation, comparison: Mapping[str, Any]
    ) -> List[Dict[str, Any]]:
        kept_camera = self._best_pose_camera(kept_station)
        kept_label = getattr(kept_camera, "label", "") if kept_camera is not None else ""
        rows: List[Dict[str, Any]] = []
        for side_name in ("front", "back"):
            camera = disabled_station.cameras.get(side_name)
            if camera is None:
                continue
            previously_enabled = bool(getattr(camera, "enabled", True))
            camera.enabled = False
            rows.append(
                {
                    "disabled_frame_id": disabled_station.frame_id,
                    "disabled_station_label": disabled_station.station_label(),
                    "camera_label": getattr(camera, "label", ""),
                    "camera_side": side_name,
                    "previously_enabled": previously_enabled,
                    "kept_frame_id": kept_station.frame_id,
                    "kept_station_label": kept_station.station_label(),
                    "kept_camera_label": kept_label,
                    "distance": round(float(comparison["distance"]), 6),
                    "angle_deg": round(float(comparison["angle_deg"]), 6),
                    "disabled_station_quality": self._score_or_blank(self._station_quality(disabled_station)),
                    "kept_station_quality": self._score_or_blank(self._station_quality(kept_station)),
                    "disabled_station_blur_score": self._score_or_blank(self._station_blur_score(disabled_station)),
                    "kept_station_blur_score": self._score_or_blank(self._station_blur_score(kept_station)),
                    "reason": "distance<{0} and angle<{1}".format(
                        self.config.camera_distance_threshold,
                        self.config.camera_angle_threshold_deg,
                    ),
                }
            )
        return rows

    def _station_has_pose(self, station: CameraStation) -> bool:
        return self._best_pose_camera(station) is not None

    def _best_pose_camera(self, station: CameraStation) -> Optional[Any]:
        candidates = [camera for camera in station.cameras.values() if self._camera_has_pose(camera)]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda camera: (
                self._camera_quality(camera),
                1 if camera_side_from_label(getattr(camera, "label", "")) == "front" else 0,
                -extract_frame_id_from_label(getattr(camera, "label", "F_000000")),
            ),
        )

    def _station_quality(self, station: CameraStation) -> float:
        qualities = [self._camera_quality(camera) for camera in station.cameras.values()]
        valid = [quality for quality in qualities if quality is not None]
        if not valid:
            return float("-inf")
        return max(valid)

    def _station_blur_score(self, station: CameraStation) -> float:
        valid = [score for score in station.blur_scores.values() if score is not None]
        if not valid:
            return float("-inf")
        return max(valid)

    def _station_pose_count(self, station: CameraStation) -> int:
        return sum(1 for camera in station.cameras.values() if self._camera_has_pose(camera))

    @staticmethod
    def _camera_quality(camera: Any) -> float:
        value = MetashapeAligner.camera_quality(camera)
        if value is None:
            return float("-inf")
        return value

    @staticmethod
    def _camera_has_pose(camera: Any) -> bool:
        if camera is None:
            return False
        return getattr(camera, "center", None) is not None and getattr(camera, "transform", None) is not None

    @staticmethod
    def _parse_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _score_or_blank(value: float) -> Any:
        if math.isfinite(value):
            return round(value, 6)
        return ""

    @staticmethod
    def _vector_xyz(vector: Any) -> Tuple[float, float, float]:
        if hasattr(vector, "x") and hasattr(vector, "y") and hasattr(vector, "z"):
            return float(vector.x), float(vector.y), float(vector.z)
        if hasattr(vector, "__len__") and len(vector) >= 3:
            return float(vector[0]), float(vector[1]), float(vector[2])
        raise PipelineError("Could not extract XYZ coordinates from vector-like value.")

    @classmethod
    def _matrix3(cls, matrix: Any) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]:
        if matrix is None:
            return None
        rows: List[Tuple[float, float, float]] = []
        for row_index in range(3):
            if hasattr(matrix, "row"):
                row = matrix.row(row_index)
                rows.append(cls._vector_xyz(row))
                continue
            if hasattr(matrix, "__getitem__"):
                rows.append(
                    (
                        float(matrix[row_index, 0]),
                        float(matrix[row_index, 1]),
                        float(matrix[row_index, 2]),
                    )
                )
                continue
            return None
        return rows[0], rows[1], rows[2]

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
        """Execute Phase 1 through Phase 3 as a single menu path."""

        phase_results: List[PhaseResult] = []
        for runner in (
            self.run_extract_streams,
            self.run_select_frames,
            self.run_generate_masks,
            self.run_import_to_metashape,
            self.run_align,
            self.run_reduce_overlap,
            self.run_export_logs,
        ):
            result = runner()
            phase_results.append(result)
            if result.status == "error":
                break

        overall_status = "ok"
        if any(result.status == "error" for result in phase_results):
            overall_status = "error"
        summary_payload = self._build_log_summary({"phases": [result.to_dict() for result in phase_results]})
        summary = PhaseResult(
            phase="run_full_pipeline",
            status=overall_status,
            message="Phase 1 through Phase 3 completed through alignment, overlap reduction, and log export.",
            details={"phases": summary_payload["phases"]},
        )
        self.logs.write_summary(self.config.summary_log_path, summary_payload)
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

        summary = self._build_log_summary()
        self.logs.write_summary(self.config.summary_log_path, summary)
        return PhaseResult(
            phase="export_logs",
            status="ok",
            message="Exported the current pipeline log summary.",
            details={"summary_path": self.config.summary_log_path},
        )

    def _build_log_summary(self, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        required_logs = {
            "ffprobe.json": self.config.ffprobe_log_path.exists(),
            "frame_quality.csv": self.config.frame_quality_log_path.exists(),
            "mask_summary.csv": self.config.mask_summary_log_path.exists(),
            "metashape_quality.csv": self.config.metashape_quality_log_path.exists(),
            "overlap_reduction.csv": self.config.overlap_reduction_log_path.exists(),
            "pipeline_summary.json": True,
            "error.log": self.config.error_log_path.exists(),
        }
        summary: Dict[str, Any] = {
            "config": self.config.to_dict(),
            "required_logs": required_logs,
            "existing_logs": sorted(path.name for path in self.config.log_dir.glob("*") if path.is_file()),
        }
        if Metashape is not None and getattr(Metashape.app.document, "chunk", None) is not None:
            chunk = Metashape.app.document.chunk
            summary["active_chunk"] = {
                "label": getattr(chunk, "label", ""),
                "camera_count": len(getattr(chunk, "cameras", [])),
                "enabled_camera_count": len([camera for camera in getattr(chunk, "cameras", []) if getattr(camera, "enabled", True)]),
            }
        if extra:
            summary.update(extra)
        return summary

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
