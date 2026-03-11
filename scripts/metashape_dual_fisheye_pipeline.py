"""Dual fisheye Metashape pipeline.

This module is intended to be loaded from the Metashape Python runtime.
Current implementation scope:

- Phase 1: ffprobe / ffmpeg extraction and paired frame selection.
- Phase 2: YOLO-based mask PNG generation and MultiplaneLayout import scaffolding.
- Phase 3: image quality analysis, alignment, overlap reduction, and phase summaries.
"""

from __future__ import annotations

import csv
import html
import json
import logging
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

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

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - depends on the Metashape runtime environment.
    torch = None  # type: ignore

QT_BINDING = ""
QtCore = None  # type: ignore
QtGui = None  # type: ignore
QtWidgets = None  # type: ignore

# TODO: validate the preferred Qt binding on the current Metashape build; keep fallback imports localized.
for _qt_binding, _qt_modules in (
    ("PySide2", ("PySide2",)),
    ("PySide6", ("PySide6",)),
    ("PyQt5", ("PyQt5",)),
    ("PyQt6", ("PyQt6",)),
):
    try:
        if _qt_binding == "PySide2":
            from PySide2 import QtCore, QtGui, QtWidgets  # type: ignore
        elif _qt_binding == "PySide6":
            from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
        elif _qt_binding == "PyQt5":
            from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
        else:
            from PyQt6 import QtCore, QtGui, QtWidgets  # type: ignore
    except ImportError:
        continue
    QT_BINDING = _qt_binding
    break


LOGGER = logging.getLogger("dual_fisheye_pipeline")
_MENU_REGISTERED = False
_GUI_DIALOG = None
_APP_SHUTDOWN_CONNECTED = False
_HEADLESS_DOCUMENT = None
_HEADLESS_DOCUMENT_PATH = None
_PIPELINE_SCRIPT_NAME = "metashape_dual_fisheye_pipeline.py"
_INPUT_OSV_SUFFIX = ".osv"
_INPUT_OSV_PLACEHOLDER = "OSV未選択"
_MENU_ROOT = "Custom/DualFisheye"
_LOGGER_HANDLER_MARKER = "_dual_fisheye_handler"
_LOGGER_HANDLER_ID_ATTR = "_dual_fisheye_handler_id"
_LOGGER_FILE_HANDLER_ID = "file"
_LOGGER_STREAM_HANDLER_ID = "stream"
_LOGGER_GUI_HANDLER_ID = "gui"


def _append_unique_message(messages: List[str], message: str) -> None:
    """Append a message once while preserving order."""

    if message not in messages:
        messages.append(message)


def _mark_logger_handler(handler: logging.Handler, handler_id: str) -> logging.Handler:
    """Tag handlers created by this module so they can be cleaned up safely."""

    setattr(handler, _LOGGER_HANDLER_MARKER, True)
    setattr(handler, _LOGGER_HANDLER_ID_ATTR, handler_id)
    return handler


def _is_managed_logger_handler(handler: logging.Handler, handler_id: Optional[str] = None) -> bool:
    """Return True when the handler belongs to this module."""

    if not getattr(handler, _LOGGER_HANDLER_MARKER, False):
        return False
    if handler_id is None:
        return True
    return getattr(handler, _LOGGER_HANDLER_ID_ATTR, "") == handler_id


def shutdown_logging(include_gui_handlers: bool = True) -> None:
    """Remove and close logger handlers created by this module."""

    removable_ids = {_LOGGER_FILE_HANDLER_ID, _LOGGER_STREAM_HANDLER_ID}
    if include_gui_handlers:
        removable_ids.add(_LOGGER_GUI_HANDLER_ID)

    for handler in list(LOGGER.handlers):
        if not _is_managed_logger_handler(handler):
            continue
        if getattr(handler, _LOGGER_HANDLER_ID_ATTR, "") not in removable_ids:
            continue
        LOGGER.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def _default_project_root() -> Path:
    """Return the best-effort project root for Metashape and normal Python execution."""

    resolved_root = _resolve_project_root()
    return resolved_root if resolved_root is not None else _safe_fallback_path()


def _safe_fallback_path() -> Path:
    """Return a non-raising path object when runtime path discovery is unavailable."""

    try:
        return Path.cwd()
    except Exception:
        return Path(".")


def _coerce_runtime_path(value: Any) -> Optional[Path]:
    """Convert a runtime path-like value into a Path without assuming it is valid."""

    if value is None:
        return None
    text = str(value).strip()
    if not text or (text.startswith("<") and text.endswith(">")):
        return None
    try:
        return Path(text).expanduser().resolve(strict=False)
    except Exception:
        try:
            return Path(text).expanduser()
        except Exception:
            return None


def _is_pipeline_script_path(path: Path) -> bool:
    """Return True when the path points to this pipeline module file."""

    if path.name != _PIPELINE_SCRIPT_NAME:
        return False
    return path.parent.name == "scripts"


def _project_root_from_script_path(script_path: Any) -> Optional[Path]:
    """Resolve the repository root from scripts/metashape_dual_fisheye_pipeline.py."""

    resolved_path = _coerce_runtime_path(script_path)
    if resolved_path is None:
        return None
    if not _is_pipeline_script_path(resolved_path):
        return None
    return resolved_path.parent.parent


def _project_root_from_module_code() -> Optional[Path]:
    """Resolve the repository root from this module's compiled code filename."""

    return _project_root_from_script_path(_default_project_root.__code__.co_filename)


def _project_root_from_main_module() -> Optional[Path]:
    """Resolve the repository root from sys.modules['__main__'] when __file__ is absent."""

    main_module = sys.modules.get("__main__")
    if main_module is None:
        return None
    return _project_root_from_script_path(getattr(main_module, "__file__", None))


def _project_root_from_metashape_document() -> Optional[Path]:
    """Resolve the project root from the active Metashape document parent directory."""

    if Metashape is None:
        return None
    try:
        document = getattr(Metashape.app, "document", None)
        document_path = getattr(document, "path", "") if document is not None else ""
    except Exception:
        return None
    resolved_path = _coerce_runtime_path(document_path)
    return resolved_path.parent if resolved_path is not None else None


def _project_root_from_cwd() -> Optional[Path]:
    """Resolve the project root from the current working directory."""

    try:
        return Path.cwd()
    except Exception:
        return None


def _resolve_project_root() -> Optional[Path]:
    """Resolve the project root in Metashape-friendly fallback order."""

    for candidate in (
        _project_root_from_script_path(globals().get("__file__")),
        _project_root_from_module_code(),
        _project_root_from_main_module(),
        _project_root_from_metashape_document(),
        _project_root_from_cwd(),
    ):
        if candidate is not None:
            return candidate
    return None


def _default_input_mp4() -> Optional[Path]:
    """Return an empty default input path so GUI startup never assumes a dummy file."""

    return None


def _default_work_root() -> Path:
    """Return the default work directory without assuming __file__ exists."""

    return _default_project_root() / "work"


def _default_project_path() -> Path:
    """Return the default Metashape project file path without assuming __file__ exists."""

    return _default_project_root() / "project" / "dual_fisheye_project.psx"


def _default_last_used_config_path(project_root: Optional[Path] = None) -> Path:
    """Return the stable launcher-side config location used on GUI startup."""

    root = project_root or _default_project_root()
    return root / "work" / "config" / "last_used_config.json"


def normalize_input_video_path(value: Any) -> Optional[Path]:
    """Normalize a user-provided OSV path while preserving an unselected state."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return Path(text).expanduser()
    except Exception:
        return Path(text)


def _path_text(value: Optional[Path]) -> str:
    """Return a safe string for optional path values."""

    return "" if value is None else str(value)


def _as_serializable(value: Any) -> Any:
    """Convert Path-heavy structures into JSON-serializable values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _read_image_with_unicode_path(image_path: Path, flags: int) -> Any:
    """Read an image without relying on OpenCV path Unicode handling."""

    if cv2 is None or np is None:
        raise PipelineError("OpenCV and numpy are required for image IO.")
    try:
        encoded_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    except OSError as exc:
        raise PipelineError("Failed to read image bytes: {0}".format(image_path)) from exc
    if encoded_bytes.size == 0:
        return None
    return cv2.imdecode(encoded_bytes, flags)


def _write_image_with_unicode_path(image_path: Path, image: Any) -> None:
    """Write an image without relying on OpenCV path Unicode handling."""

    if cv2 is None or np is None:
        raise PipelineError("OpenCV and numpy are required for image IO.")
    suffix = image_path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise PipelineError("Failed to encode image for writing: {0}".format(image_path))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        image_path.write_bytes(encoded.tobytes())
    except OSError as exc:
        raise PipelineError("Failed to write image file: {0}".format(image_path)) from exc


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
    input_mp4: Optional[Path] = field(default_factory=_default_input_mp4)
    work_root: Path = field(default_factory=_default_work_root)
    project_path: Path = field(default_factory=_default_project_path)
    menu_root: str = _MENU_ROOT
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
    mask_model_path: str = "yolo26x-seg.pt"
    mask_classes: Tuple[str, ...] = ("person", "car", "truck", "bus", "motorbike")
    mask_dilate_px: int = 8
    mask_polarity: str = "target_black"
    mask_confidence_threshold: float = 0.25
    mask_iou_threshold: float = 0.45
    mask_device: Optional[str] = None
    metashape_image_quality_threshold: float = 0.5
    match_downscale: int = 1
    generic_preselection: bool = True
    reference_preselection: bool = False
    keep_keypoints: bool = True
    keypoint_limit: int = 40000
    tiepoint_limit: int = 4000
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
    opencv_backend: str = "auto"
    prefer_cuda: bool = True
    cuda_device_index: int = 0
    cuda_allow_fallback: bool = True
    cuda_log_device_info: bool = True
    cuda_use_gaussian_preblur: bool = False
    cuda_benchmark_mode: bool = False
    yolo_device_mode: str = "auto"
    prefer_yolo_cuda: bool = True
    yolo_allow_fallback: bool = True
    yolo_device_index: int = 0
    save_backend_report: bool = True

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
    def config_dir(self) -> Path:
        return self.work_root / "config"

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
    def opencv_backend_report_path(self) -> Path:
        return self.log_dir / "opencv_backend_report.json"

    @property
    def cuda_fallback_log_path(self) -> Path:
        return self.log_dir / "cuda_fallback.log"

    @property
    def yolo_backend_report_path(self) -> Path:
        return self.log_dir / "yolo_backend_report.json"

    @property
    def metashape_gpu_report_path(self) -> Path:
        return self.log_dir / "metashape_gpu_report.json"

    @property
    def gpu_summary_report_path(self) -> Path:
        return self.log_dir / "gpu_summary_report.json"

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

    @property
    def last_used_config_path(self) -> Path:
        return self.config_dir / "last_used_config.json"

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
        if self.opencv_backend not in ("auto", "cpu", "cuda"):
            raise ValueError("opencv_backend must be one of: auto, cpu, cuda.")
        if self.cuda_device_index < 0:
            raise ValueError("cuda_device_index must be >= 0.")
        if self.yolo_device_mode not in ("auto", "cpu", "cuda"):
            raise ValueError("yolo_device_mode must be one of: auto, cpu, cuda.")
        if self.yolo_device_index < 0:
            raise ValueError("yolo_device_index must be >= 0.")
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
        if self.mask_polarity != "target_black":
            raise ValueError("mask_polarity must remain 'target_black' for the current implementation.")
        if not 0.0 <= self.mask_confidence_threshold <= 1.0:
            raise ValueError("mask_confidence_threshold must be between 0 and 1.")
        if not 0.0 <= self.mask_iou_threshold <= 1.0:
            raise ValueError("mask_iou_threshold must be between 0 and 1.")
        if require_input:
            self.require_input_video()

    def input_video_validation_error(self, require_exists: bool = True) -> Optional[Exception]:
        """Return a normalized input validation error for GUI preview and run-time checks."""

        input_path = normalize_input_video_path(self.input_mp4)
        if input_path is None:
            return ValueError("Input OSV is not selected.")
        if input_path.exists() and input_path.is_dir():
            return ValueError("Input OSV must be a file, not a directory: {0}".format(input_path))
        if input_path.suffix.lower() != _INPUT_OSV_SUFFIX:
            return ValueError("Input file must be a .osv file.")
        if require_exists and not input_path.exists():
            return FileNotFoundError("Input OSV not found: {0}".format(input_path))
        return None

    def require_input_video(self) -> Path:
        """Return the validated OSV container path for ffprobe / ffmpeg execution."""

        error = self.input_video_validation_error(require_exists=True)
        if error is not None:
            raise error
        input_path = normalize_input_video_path(self.input_mp4)
        if input_path is None:
            raise ValueError("Input OSV is not selected.")
        self.input_mp4 = input_path
        return input_path

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
            self.config_dir,
            self.temp_dir,
            self.project_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def mask_model_path_text(self) -> str:
        """Return the raw model-path text as stored in config / GUI state."""

        return str(self.mask_model_path).strip()

    def mask_model_candidate_paths(self) -> List[Path]:
        """Build the local YOLO model lookup order without assuming network download."""

        raw_value = self.mask_model_path_text()
        if not raw_value:
            return []

        configured_path = Path(raw_value).expanduser()
        candidates: List[Path] = []
        if configured_path.is_absolute():
            candidates.append(configured_path)
        else:
            candidates.append((self.project_root / configured_path).expanduser())
            candidates.append((self.work_root / configured_path).expanduser())

        unique_candidates: List[Path] = []
        seen: Set[str] = set()
        for candidate in candidates:
            try:
                normalized_candidate = candidate.resolve(strict=False)
            except Exception:
                normalized_candidate = candidate
            key = str(normalized_candidate)
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(normalized_candidate)
        return unique_candidates

    def find_local_mask_model_path(self) -> Optional[Path]:
        """Return the first existing local model path according to the configured lookup order."""

        for candidate in self.mask_model_candidate_paths():
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def mask_model_validation_error(self) -> Optional[Exception]:
        """Return a GUI-friendly model-path validation error without triggering downloads."""

        raw_value = self.mask_model_path_text()
        if not raw_value:
            return ValueError("YOLO model file is not configured. Please select a local .pt file in GUI.")
        if self.find_local_mask_model_path() is not None:
            return None
        return FileNotFoundError(
            "YOLO model file is not available locally. Model file not found locally: {0}. "
            "Please select a local .pt file in GUI.".format(raw_value)
        )

    def resolve_mask_model_path(self) -> Path:
        """Resolve the configured YOLO model to a verified local file path."""

        resolved_path = self.find_local_mask_model_path()
        if resolved_path is not None:
            return resolved_path

        raw_value = self.mask_model_path_text() or "<empty>"
        candidate_paths = [str(path) for path in self.mask_model_candidate_paths()]
        LOGGER.error("Model file not found locally: %s", raw_value)
        LOGGER.error("Checked YOLO model candidates: %s", candidate_paths)
        raise PipelineError(
            "YOLO model file is not available locally. Model file not found locally: {0}. "
            "Please select a local .pt file in GUI.".format(raw_value)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config for logs."""

        return {key: _as_serializable(value) for key, value in asdict(self).items()}

    def update_from_mapping(self, data: Mapping[str, Any]) -> None:
        """Update config values from a partially populated JSON-compatible mapping."""

        normalized_data = dict(data)
        if "input_mp4" not in normalized_data and "input_video" in normalized_data:
            normalized_data["input_mp4"] = normalized_data["input_video"]
        for field_info in fields(self):
            key = field_info.name
            if key not in normalized_data:
                continue
            setattr(self, key, self._coerce_field_value(key, normalized_data[key], getattr(self, key)))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PipelineConfig":
        """Build a config with defaults for any missing keys."""

        config = cls()
        config.update_from_mapping(data)
        return config

    @staticmethod
    def _coerce_field_value(field_name: str, value: Any, default: Any) -> Any:
        if value is None:
            return None if field_name in ("fft_blur_threshold", "mask_device") else default
        if field_name == "input_mp4":
            return normalize_input_video_path(value)
        if field_name in ("project_root", "work_root", "project_path"):
            if isinstance(value, str) and not value.strip():
                return default
            return Path(value)
        if field_name == "mask_classes":
            if isinstance(value, str):
                return tuple(item.strip() for item in value.split(",") if item.strip())
            if isinstance(value, (list, tuple)):
                return tuple(str(item).strip() for item in value if str(item).strip())
            return default
        if field_name in ("rig_relative_location", "rig_relative_rotation"):
            if isinstance(value, (list, tuple)) and len(value) == 3:
                return tuple(float(item) for item in value)
            return default
        if isinstance(default, bool):
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if isinstance(default, int) and not isinstance(default, bool):
            return int(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, Path):
            return Path(value)
        return value


@dataclass
class CudaCapabilityReport:
    """Normalized runtime view of OpenCV CUDA availability and fallback state."""

    opencv_version: str = ""
    cv2_available: bool = False
    numpy_available: bool = False
    cuda_namespace_available: bool = False
    cuda_device_count: int = 0
    requested_backend: str = "auto"
    selected_backend: str = "cpu"
    active_backend: str = "cpu"
    prefer_cuda: bool = True
    cuda_allow_fallback: bool = True
    cuda_device_index: int = 0
    active_device_index: Optional[int] = None
    cuda_api_available: bool = False
    laplacian_filter_available: bool = False
    gaussian_filter_available: bool = False
    mean_stddev_available: bool = False
    set_device_available: bool = False
    get_device_available: bool = False
    device_info_logged: bool = False
    device_names: List[str] = field(default_factory=list)
    selected_device_name: str = ""
    fallback_reasons: List[str] = field(default_factory=list)
    fallback_events: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {key: _as_serializable(value) for key, value in asdict(self).items()}


@dataclass
class YoloBackendReport:
    """Normalized runtime view of YOLO / PyTorch device selection and fallback state."""

    ultralytics_available: bool = False
    torch_available: bool = False
    torch_version: str = ""
    cuda_available: bool = False
    cuda_device_count: int = 0
    requested_mode: str = "auto"
    selected_device: str = "cpu"
    active_device: str = "cpu"
    prefer_cuda: bool = True
    allow_fallback: bool = True
    device_index: int = 0
    device_name: str = ""
    device_names: List[str] = field(default_factory=list)
    model_loaded: bool = False
    local_model_path: str = ""
    fallback_reasons: List[str] = field(default_factory=list)
    fallback_events: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {key: _as_serializable(value) for key, value in asdict(self).items()}


@dataclass
class MetashapeGpuReport:
    """Best-effort runtime view of Metashape GPU availability without enforcing control APIs."""

    metashape_available: bool = False
    app_available: bool = False
    status: str = "unverified"
    gpu_mask: Optional[int] = None
    cpu_enable: Optional[bool] = None
    gpu_devices: List[Dict[str, Any]] = field(default_factory=list)
    gpu_device_count: int = 0
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    todo: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {key: _as_serializable(value) for key, value in asdict(self).items()}


class JapaneseUiText:
    """Centralized Japanese labels and status text for the GUI-facing layer."""

    MENU_SUFFIXES = {
        "00 Open GUI": "00 GUIを開く",
        "01 Run Full Pipeline": "01 フルパイプライン実行",
        "02 Extract Streams": "02 ストリーム抽出",
        "03 Select Frames": "03 フレーム選別",
        "04 Generate Masks": "04 マスク生成",
        "05 Import to Metashape": "05 Metashapeへ読込",
        "06 Align": "06 アライメント",
        "07 Reduce Overlap": "07 冗長画像削減",
        "08 Export Logs": "08 ログ出力",
    }
    ACTION_LABELS = {
        "run_full_pipeline": "フル実行",
        "extract_streams": "ストリーム抽出",
        "select_frames": "フレーム選別",
        "generate_masks": "マスク生成",
        "import_to_metashape": "Metashapeへ読込",
        "align": "アライメント",
        "reduce_overlap": "冗長画像削減",
        "export_logs": "ログ出力",
    }
    PHASE_LABELS = {
        "run_full_pipeline": "フルパイプライン",
        "extract_streams": "ストリーム抽出",
        "select_frames": "フレーム選別",
        "generate_masks": "マスク生成",
        "import_to_metashape": "Metashape読込",
        "align": "アライメント",
        "reduce_overlap": "冗長画像削減",
        "export_logs": "ログ出力",
    }
    STEP_LABELS = {
        "probe_streams": "ストリーム確認",
        "extract_front_stream": "前方ストリーム抽出",
        "extract_back_stream": "後方ストリーム抽出",
        "verify_frame_counts": "抽出枚数確認",
        "complete": "完了",
        "select_pairs": "フレームペア評価",
        "process_directory_front_back": "前後画像を処理",
        "create_or_get_chunk": "chunk準備",
        "build_filename_sequence": "読込順序作成",
        "build_filegroups": "filegroups作成",
        "import_multiplane_images": "MultiplaneLayout読込",
        "set_sensor_types": "魚眼センサー設定",
        "apply_rig_reference": "rig参照設定",
        "apply_masks_from_disk": "マスク適用",
        "save_document": "ドキュメント保存",
        "analyze_image_quality": "画質評価",
        "disable_low_quality_cameras": "低品質画像を無効化",
        "export_quality_log_before_align": "画質ログ出力",
        "match_photos": "matchPhotos実行",
        "align_cameras": "alignCameras実行",
        "export_quality_log_and_save": "画質ログ出力と保存",
        "disable_redundant_cameras": "冗長画像を無効化",
        "realign_after_cleanup": "再アライメント",
        "run_reduce_overlap_builtin": "内蔵reduceOverlap実行",
        "export_overlap_log_and_save": "間引きログ出力と保存",
        "write_summary": "サマリー出力",
    }
    DIRECT_TRANSLATIONS = {
        "Ready": "準備完了",
        "Idle": "待機中",
        "Running...": "実行中...",
        "Input OSV is not selected.": "入力OSVが未選択です。",
        "Input file must be a .osv file.": "入力ファイルは .osv である必要があります。",
        "YOLO model file is not configured. Please select a local .pt file in GUI.": (
            "YOLOモデルファイルが未設定です。GUIでローカルの .pt ファイルを選択してください。"
        ),
        "CUDA is not available for YOLO inference.": "YOLO 推論で CUDA を利用できません。",
        "CUDA is not available for YOLO inference. Fallback to CPU.": (
            "YOLO 推論で CUDA が利用できないため、CPU に切り替えました。"
        ),
        "No active chunk is available.": "アクティブなchunkがありません。",
        "Active chunk has no cameras to align.": "アクティブなchunkにアライメント対象カメラがありません。",
        "Selected paired frames using the either-side-OK keep-both rule.": (
            "片側合格なら両側採用の keep rule でフレームペアを選別しました。"
        ),
        "Generated binary mask PNGs for selected front/back frame pairs.": (
            "選別済み front/back フレームペアのバイナリマスク PNG を生成しました。"
        ),
        "Imported selected front/back pairs using MultiplaneLayout and applied per-camera masks.": (
            "選別済み front/back ペアを MultiplaneLayout で読み込み、カメラ単位マスクを適用しました。"
        ),
        "Ran image quality analysis, disabled low-quality cameras, and aligned the active chunk.": (
            "画質評価、低品質カメラの無効化、アクティブchunkのアライメントを実行しました。"
        ),
        "Reduced redundant aligned stations using distance and rotation thresholds.": (
            "距離閾値と回転閾値に基づいて冗長なアライメント済みステーションを削減しました。"
        ),
        "Exported the current pipeline log summary.": "現在のパイプラインログ要約を出力しました。",
        "Phase 1 through Phase 3 completed through alignment, overlap reduction, and log export.": (
            "Phase 1 から Phase 3 までを実行し、アライメント、冗長画像削減、ログ出力まで完了しました。"
        ),
        "FFT blur threshold must be blank or numeric.": "FFTブレ閾値は空欄または数値で入力してください。",
        "Rig vectors must contain exactly 3 comma-separated numbers.": (
            "rigベクトルはカンマ区切りの3要素で入力してください。"
        ),
        "Rig vectors must contain numeric values.": "rigベクトルには数値を入力してください。",
        "Qt bindings are not available in this Python runtime. Use the existing menu actions and validate the current Metashape Qt binding.": (
            "この Python 実行環境では Qt バインディングが利用できません。既存メニューを使用し、現在の Metashape Qt バインディングを確認してください。"
        ),
    }
    PREFIX_TRANSLATIONS = (
        ("Input OSV not found: ", "入力OSVが見つかりません: "),
        ("Input OSV must be a file, not a directory: ", "入力OSVはディレクトリではなくファイルを指定してください: "),
        (
            "YOLO model file is not available locally. Model file not found locally: ",
            "YOLOモデルファイルがローカルに見つかりません: ",
        ),
        ("Saved config to ", "設定を保存しました: "),
        ("Failed to save config: ", "設定の保存に失敗しました: "),
        ("Loaded previous config from ", "前回の設定を読み込みました: "),
        ("Loaded config from ", "設定を読み込みました: "),
        ("Failed to load config: ", "設定の読込に失敗しました: "),
        ("Updated Input OSV selection.", "入力OSVを更新しました。"),
        ("Updated YOLO model path.", "YOLOモデルパスを更新しました。"),
        (
            "Work root changed. Save or run to persist the new config path.",
            "作業フォルダを変更しました。保存または実行すると新しい設定パスが反映されます。",
        ),
        ("Reset GUI fields to default config values.", "GUIの設定値を初期値に戻しました。"),
        ("Failed to read backend report: ", "backend report の読込に失敗しました: "),
    )

    @classmethod
    def translate(cls, text: str) -> str:
        translated = cls.DIRECT_TRANSLATIONS.get(text, text)
        model_prefix = "YOLO model file is not available locally. Model file not found locally: "
        model_suffix = ". Please select a local .pt file in GUI."
        if translated.startswith(model_prefix):
            model_path = translated[len(model_prefix) :]
            if model_path.endswith(model_suffix):
                model_path = model_path[: -len(model_suffix)]
            return "YOLOモデルファイルがローカルに見つかりません: {0}。GUIでローカルの .pt ファイルを選択してください。".format(
                model_path
            )
        for prefix, replacement in cls.PREFIX_TRANSLATIONS:
            if translated.startswith(prefix):
                return replacement + translated[len(prefix) :]
        if translated.endswith("...") and translated.startswith("Running "):
            action_name = translated[len("Running ") : -3]
            return "{0}を実行中...".format(cls.action_label(action_name))
        return translated

    @classmethod
    def action_label(cls, action_name: str) -> str:
        return cls.ACTION_LABELS.get(action_name, action_name)

    @classmethod
    def phase_label(cls, phase_name: str) -> str:
        return cls.PHASE_LABELS.get(phase_name, phase_name)

    @classmethod
    def step_label(cls, step_name: str) -> str:
        return cls.STEP_LABELS.get(step_name, step_name)

    @classmethod
    def progress_label(cls, phase_name: str, step_name: str) -> str:
        return "{0}: {1}".format(cls.phase_label(phase_name), cls.step_label(step_name))

    @classmethod
    def menu_suffix(cls, suffix: str) -> str:
        return cls.MENU_SUFFIXES.get(suffix, suffix)

    @staticmethod
    def opencv_status(report: Mapping[str, Any]) -> str:
        active_backend = str(report.get("active_backend", "cpu"))
        if active_backend == "cuda":
            return "OpenCV: CUDA 使用中"
        if report.get("fallback_events"):
            return "OpenCV: CPU フォールバック"
        return "OpenCV: CPU 実行"

    @staticmethod
    def yolo_status(report: Mapping[str, Any]) -> str:
        active_device = str(report.get("active_device", "cpu"))
        if active_device.startswith("cuda"):
            return "YOLO: GPU 使用中"
        if report.get("fallback_events"):
            return "YOLO: CPU フォールバック"
        return "YOLO: CPU 実行"

    @staticmethod
    def metashape_status(report: Mapping[str, Any]) -> str:
        status = str(report.get("status", "unverified"))
        device_count = int(report.get("gpu_device_count", 0) or 0)
        if status == "detected" and device_count > 0:
            return "Metashape GPU: 情報取得済み"
        if status == "partial":
            return "Metashape GPU: 一部取得"
        return "Metashape GPU: 未確認"


class GpuStatusAggregator:
    """Aggregate OpenCV, YOLO, and Metashape GPU reports for logs and the GUI."""

    def __init__(self, config: PipelineConfig, logs: LogWriter) -> None:
        self.config = config
        self.logs = logs

    def collect_all(
        self,
        opencv_manager: "OpenCVBackendManager",
        mask_generator: "MaskGenerator",
        save: Optional[bool] = None,
        probe_runtime: bool = True,
    ) -> Dict[str, Any]:
        should_save = self.config.save_backend_report if save is None else save
        opencv_report = dict(opencv_manager.build_backend_report(probe_runtime=probe_runtime))
        yolo_report = dict(mask_generator.build_backend_report(probe_runtime=probe_runtime))
        metashape_report = self.collect_metashape_report(probe_runtime=probe_runtime)
        summary_report = self.build_summary(opencv_report, yolo_report, metashape_report)
        if should_save:
            self.logs.write_json(self.config.opencv_backend_report_path, opencv_report)
            self.logs.write_json(self.config.yolo_backend_report_path, yolo_report)
            self.logs.write_json(self.config.metashape_gpu_report_path, metashape_report)
            self.logs.write_json(self.config.gpu_summary_report_path, summary_report)
        return {
            "opencv": opencv_report,
            "yolo": yolo_report,
            "metashape": metashape_report,
            "summary": summary_report,
        }

    def build_summary(
        self,
        opencv_report: Mapping[str, Any],
        yolo_report: Mapping[str, Any],
        metashape_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        fallback_detected = bool(opencv_report.get("fallback_events")) or bool(yolo_report.get("fallback_events"))
        device_indices: List[str] = []
        opencv_index = opencv_report.get("active_device_index")
        yolo_device = str(yolo_report.get("active_device", "cpu"))
        if opencv_index is not None:
            device_indices.append("OpenCV={0}".format(opencv_index))
        if yolo_device.startswith("cuda:"):
            device_indices.append("YOLO={0}".format(yolo_device.split(":", 1)[1]))
        elif yolo_device.startswith("cuda"):
            device_indices.append("YOLO=0")
        return {
            "opencv_status": JapaneseUiText.opencv_status(opencv_report),
            "yolo_status": JapaneseUiText.yolo_status(yolo_report),
            "metashape_gpu_status": JapaneseUiText.metashape_status(metashape_report),
            "gpu_fallback": "あり" if fallback_detected else "なし",
            "device_indices": ", ".join(device_indices) if device_indices else "-",
            "backend_report_paths": {
                "opencv": str(self.config.opencv_backend_report_path),
                "yolo": str(self.config.yolo_backend_report_path),
                "metashape": str(self.config.metashape_gpu_report_path),
                "summary": str(self.config.gpu_summary_report_path),
            },
        }

    def collect_metashape_report(self, probe_runtime: bool = True) -> Dict[str, Any]:
        report = MetashapeGpuReport(
            metashape_available=Metashape is not None,
            app_available=bool(getattr(Metashape, "app", None)) if Metashape is not None else False,
        )
        if not probe_runtime:
            report.status = "deferred"
            _append_unique_message(
                report.notes,
                "Metashape GPU probe is deferred until explicit status refresh or pipeline execution.",
            )
            return report.to_dict()
        if Metashape is None:
            report.warnings.append("Metashape runtime is not available in this Python environment.")
            return report.to_dict()
        app = getattr(Metashape, "app", None)
        if app is None:
            report.warnings.append("Metashape.app is not available in this runtime.")
            return report.to_dict()

        known_fields = False
        if hasattr(app, "gpu_mask"):
            known_fields = True
            try:
                report.gpu_mask = int(getattr(app, "gpu_mask"))
                report.notes.append("Read Metashape.app.gpu_mask from the current runtime.")
            except Exception as exc:
                report.warnings.append("Failed to read Metashape.app.gpu_mask: {0}".format(exc))
        if hasattr(app, "cpu_enable"):
            known_fields = True
            try:
                report.cpu_enable = bool(getattr(app, "cpu_enable"))
                report.notes.append("Read Metashape.app.cpu_enable from the current runtime.")
            except Exception as exc:
                report.warnings.append("Failed to read Metashape.app.cpu_enable: {0}".format(exc))
        if hasattr(app, "enumGPUDevices"):
            known_fields = True
            try:
                devices = app.enumGPUDevices()
                report.gpu_devices = self._serialize_metashape_devices(devices)
                report.gpu_device_count = len(report.gpu_devices)
                report.status = "detected"
            except Exception as exc:
                report.warnings.append("Failed to call Metashape.app.enumGPUDevices(): {0}".format(exc))
        if not known_fields:
            report.status = "unverified"
            report.todo.append(
                "TODO: validate the current Metashape GPU inspection API on this build before using it for control decisions."
            )
            return report.to_dict()
        if report.status != "detected":
            report.status = "partial"
        # TODO: validate the semantics of gpu_mask / cpu_enable on the current Metashape build before writing settings.
        report.todo.append(
            "TODO: validate the semantics of Metashape.app.gpu_mask and Metashape.app.cpu_enable on the current build."
        )
        return report.to_dict()

    @staticmethod
    def _serialize_metashape_devices(devices: Any) -> List[Dict[str, Any]]:
        serialized: List[Dict[str, Any]] = []
        if devices is None:
            return serialized
        for index, device in enumerate(list(devices)):
            payload: Dict[str, Any] = {"index": index, "repr": str(device)}
            for attr in ("name", "vendor", "memory", "total_memory", "driver"):
                if not hasattr(device, attr):
                    continue
                value = getattr(device, attr)
                try:
                    payload[attr] = value() if callable(value) else value
                except Exception:
                    continue
            serialized.append(payload)
        return serialized


class OpenCVBackendManager:
    """Resolve and report CPU/CUDA backend state for OpenCV preprocessing."""

    def __init__(self, config: PipelineConfig, logs: LogWriter) -> None:
        self.config = config
        self.logs = logs
        self._report = CudaCapabilityReport()
        self._report_cached = False
        self.active_backend = "cpu"
        self.active_device_index: Optional[int] = None
        self._backend_ensured = False

    def detect_cuda_support(self) -> Dict[str, Any]:
        """Inspect the current cv2 runtime for CUDA support and required symbols."""

        if self._report_cached:
            return self._report.to_dict()

        report = CudaCapabilityReport(
            opencv_version=getattr(cv2, "__version__", ""),
            cv2_available=cv2 is not None,
            numpy_available=np is not None,
            requested_backend=self.config.opencv_backend,
            selected_backend=self.active_backend,
            active_backend=self.active_backend,
            prefer_cuda=self.config.prefer_cuda,
            cuda_allow_fallback=self.config.cuda_allow_fallback,
            cuda_device_index=self.config.cuda_device_index,
            active_device_index=self.active_device_index,
        )
        if cv2 is None or np is None:
            report.warnings.append("OpenCV and numpy are required for blur evaluation.")
            self._report = report
            self._report_cached = True
            return report.to_dict()

        cuda_api = getattr(cv2, "cuda", None)
        report.cuda_namespace_available = cuda_api is not None
        if cuda_api is None:
            report.warnings.append("cv2.cuda is not available in the current OpenCV build.")
            self._report = report
            self._report_cached = True
            return report.to_dict()

        report.set_device_available = hasattr(cuda_api, "setDevice")
        report.get_device_available = hasattr(cuda_api, "getDevice")
        report.laplacian_filter_available = hasattr(cuda_api, "createLaplacianFilter")
        report.gaussian_filter_available = hasattr(cuda_api, "createGaussianFilter")
        report.mean_stddev_available = hasattr(cuda_api, "meanStdDev")
        try:
            report.cuda_device_count = int(cuda_api.getCudaEnabledDeviceCount())
        except Exception as exc:
            report.warnings.append("Failed to query CUDA device count: {0}".format(exc))
            report.cuda_device_count = 0
        if report.cuda_device_count > 0 and hasattr(cuda_api, "DeviceInfo"):
            for device_index in range(report.cuda_device_count):
                try:
                    device_info = cuda_api.DeviceInfo(device_index)
                except Exception:
                    continue
                device_name = self._device_name_from_info(device_info)
                if device_name:
                    report.device_names.append(device_name)

        report.cuda_api_available = report.cuda_device_count > 0 and report.laplacian_filter_available
        if report.cuda_device_count <= 0:
            report.warnings.append("OpenCV reports no CUDA-enabled devices.")
        if not report.laplacian_filter_available:
            report.warnings.append(
                "OpenCV CUDA image-filter bindings are unavailable; createLaplacianFilter is missing."
            )
            # TODO: validate whether additional OpenCV CUDA Python modules are exposed on the target Metashape build.
        if self.config.cuda_use_gaussian_preblur and not report.gaussian_filter_available:
            report.warnings.append(
                "cuda_use_gaussian_preblur is enabled, but createGaussianFilter is unavailable."
            )
            # TODO: validate gaussian preblur availability on the current OpenCV CUDA build before enabling it by default.

        self._report = report
        self._report_cached = True
        return report.to_dict()

    def select_backend(self, config: PipelineConfig) -> str:
        """Choose the effective backend from the runtime capabilities and config."""

        self.detect_cuda_support()
        report = self._report
        requested = config.opencv_backend
        report.requested_backend = requested
        report.prefer_cuda = config.prefer_cuda
        report.cuda_allow_fallback = config.cuda_allow_fallback
        report.cuda_device_index = config.cuda_device_index

        if requested == "cpu":
            report.selected_backend = "cpu"
            return "cpu"

        if requested == "auto" and not config.prefer_cuda:
            report.notes.append("prefer_cuda=False forced CPU selection while opencv_backend=auto.")
            report.selected_backend = "cpu"
            return "cpu"

        cuda_ready = report.cuda_api_available and report.cuda_device_count > config.cuda_device_index
        if requested == "cuda":
            if cuda_ready:
                report.selected_backend = "cuda"
                return "cuda"
            reason = self._cuda_unavailable_reason(config, report)
            if config.cuda_allow_fallback:
                self.record_fallback(reason, context="backend_selection", requested_backend="cuda")
                report.selected_backend = "cpu"
                return "cpu"
            raise PipelineError(reason)

        if requested == "auto":
            if cuda_ready:
                report.selected_backend = "cuda"
                return "cuda"
            reason = self._cuda_unavailable_reason(config, report)
            self.record_fallback(reason, context="backend_selection", requested_backend="auto")
            report.selected_backend = "cpu"
            return "cpu"

        raise PipelineError("Unsupported opencv_backend: {0}".format(requested))

    def set_active_device(self, device_index: int) -> None:
        """Select the requested CUDA device when the backend resolves to CUDA."""

        self.detect_cuda_support()
        cuda_api = getattr(cv2, "cuda", None)
        if cuda_api is None or not hasattr(cuda_api, "setDevice"):
            raise PipelineError("OpenCV CUDA device selection is unavailable in this runtime.")
        if self._report.cuda_device_count <= device_index:
            raise PipelineError(
                "cuda_device_index={0} is out of range for {1} detected CUDA device(s).".format(
                    device_index, self._report.cuda_device_count
                )
            )
        try:
            cuda_api.setDevice(device_index)
        except Exception as exc:
            raise PipelineError("Failed to activate CUDA device {0}: {1}".format(device_index, exc)) from exc
        self.active_device_index = device_index
        self._report.active_device_index = device_index
        if len(self._report.device_names) > device_index:
            self._report.selected_device_name = self._report.device_names[device_index]
        if self.config.cuda_log_device_info:
            self._log_device_info(device_index)

    def build_backend_report(self, probe_runtime: bool = True) -> Dict[str, Any]:
        """Return a serializable backend report with current fallback state."""

        if probe_runtime:
            self.detect_cuda_support()
        elif not self._report_cached:
            _append_unique_message(
                self._report.notes,
                "OpenCV CUDA probe is deferred until explicit status refresh or pipeline execution.",
            )
            self._report.requested_backend = self.config.opencv_backend
            self._report.prefer_cuda = self.config.prefer_cuda
            self._report.cuda_allow_fallback = self.config.cuda_allow_fallback
            self._report.cuda_device_index = self.config.cuda_device_index
        self._report.active_backend = self.active_backend
        self._report.selected_backend = self._report.selected_backend or self.active_backend
        self._report.active_device_index = self.active_device_index
        payload = self._report.to_dict()
        payload["backend_report_path"] = str(self.config.opencv_backend_report_path)
        payload["cuda_fallback_log_path"] = str(self.config.cuda_fallback_log_path)
        payload["save_backend_report"] = self.config.save_backend_report
        return payload

    def ensure_backend(self, config: PipelineConfig) -> str:
        """Resolve the backend once and persist a report when configured."""

        if self._backend_ensured:
            return self.active_backend
        backend = self.select_backend(config)
        self.active_backend = backend
        if backend == "cuda":
            self.set_active_device(config.cuda_device_index)
        else:
            self.active_device_index = None
            self._report.active_device_index = None
        self._report.active_backend = self.active_backend
        self._backend_ensured = True
        self.save_backend_report()
        return backend

    def record_fallback(
        self,
        reason: str,
        context: str,
        requested_backend: Optional[str] = None,
        image_path: Optional[Path] = None,
    ) -> None:
        """Capture a fallback event and keep the latest report synchronized."""

        clean_reason = str(reason).strip() or "Unknown CUDA fallback reason."
        if clean_reason not in self._report.fallback_reasons:
            self._report.fallback_reasons.append(clean_reason)
        event = {
            "context": context,
            "reason": clean_reason,
            "requested_backend": requested_backend or self.config.opencv_backend,
            "image_path": str(image_path) if image_path else "",
        }
        self._report.fallback_events.append(event)
        LOGGER.warning("OpenCV backend fallback [%s]: %s", context, clean_reason)
        self.logs.append_line(
            self.config.cuda_fallback_log_path,
            "{0}\t{1}\t{2}\t{3}".format(
                context,
                event["requested_backend"],
                event["image_path"] or "<unknown>",
                clean_reason,
            ),
        )

    def fallback_to_cpu(self, reason: str, context: str, image_path: Optional[Path] = None) -> None:
        """Switch the active runtime backend to CPU after a CUDA failure."""

        self.record_fallback(reason, context=context, image_path=image_path)
        self.active_backend = "cpu"
        self.active_device_index = None
        self._report.active_backend = "cpu"
        self._report.selected_backend = "cpu"
        self._report.active_device_index = None
        self.save_backend_report()

    def save_backend_report(self) -> None:
        if self.config.save_backend_report:
            self.logs.write_json(self.config.opencv_backend_report_path, self.build_backend_report())

    @property
    def fallback_detected(self) -> bool:
        return bool(self._report.fallback_events)

    def cleanup(self) -> None:
        """Drop cached runtime state so GUI teardown does not retain backend objects."""

        self.active_backend = "cpu"
        self.active_device_index = None
        self._backend_ensured = False
        self._report = CudaCapabilityReport()
        self._report_cached = False

    def _cuda_unavailable_reason(self, config: PipelineConfig, report: CudaCapabilityReport) -> str:
        if not report.cv2_available or not report.numpy_available:
            return "OpenCV and numpy are required before CUDA selection can succeed."
        if not report.cuda_namespace_available:
            return "OpenCV was built without cv2.cuda support."
        if report.cuda_device_count <= 0:
            return "OpenCV CUDA backend requested, but no CUDA-enabled device was detected."
        if report.cuda_device_count <= config.cuda_device_index:
            return "cuda_device_index={0} is out of range for {1} detected CUDA device(s).".format(
                config.cuda_device_index, report.cuda_device_count
            )
        if not report.laplacian_filter_available:
            return "OpenCV CUDA backend requested, but createLaplacianFilter is unavailable in this build."
        return "OpenCV CUDA backend requested, but the runtime is not ready for CUDA preprocessing."

    def _log_device_info(self, device_index: int) -> None:
        cuda_api = getattr(cv2, "cuda", None)
        if cuda_api is None:
            return
        device_info = None
        if hasattr(cuda_api, "DeviceInfo"):
            try:
                device_info = cuda_api.DeviceInfo(device_index)
            except Exception:
                device_info = None
        if device_info is not None:
            details: Dict[str, Any] = {"device_index": device_index}
            for attr in ("name", "majorVersion", "minorVersion", "multiProcessorCount", "totalGlobalMem"):
                if not hasattr(device_info, attr):
                    continue
                value = getattr(device_info, attr)
                try:
                    details[attr] = value() if callable(value) else value
                except Exception:
                    continue
            LOGGER.info("OpenCV CUDA device info: %s", details)
            if not self._report.selected_device_name:
                self._report.selected_device_name = str(details.get("name", ""))
            self._report.notes.append("Logged CUDA device info for device {0}.".format(device_index))
            self._report.device_info_logged = True
            return
        if hasattr(cuda_api, "printShortCudaDeviceInfo"):
            try:
                cuda_api.printShortCudaDeviceInfo(device_index)
                self._report.notes.append(
                    "Invoked OpenCV printShortCudaDeviceInfo for device {0}.".format(device_index)
                )
                self._report.device_info_logged = True
            except Exception as exc:
                self._report.warnings.append("Failed to log CUDA device info: {0}".format(exc))

    @staticmethod
    def _device_name_from_info(device_info: Any) -> str:
        if device_info is None or not hasattr(device_info, "name"):
            return ""
        try:
            value = getattr(device_info, "name")
            return str(value() if callable(value) else value)
        except Exception:
            return ""


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
    for handler in list(LOGGER.handlers):
        if not _is_managed_logger_handler(handler, _LOGGER_FILE_HANDLER_ID):
            continue
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == file_path:
            has_file_handler = True
            continue
        LOGGER.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    if not has_file_handler:
        handler = _mark_logger_handler(logging.FileHandler(file_path, encoding="utf-8"), _LOGGER_FILE_HANDLER_ID)
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)

    has_stream_handler = False
    for handler in list(LOGGER.handlers):
        if not _is_managed_logger_handler(handler, _LOGGER_STREAM_HANDLER_ID):
            continue
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            has_stream_handler = True
            continue
        LOGGER.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    if not has_stream_handler:
        stream_handler = _mark_logger_handler(logging.StreamHandler(), _LOGGER_STREAM_HANDLER_ID)
        stream_handler.setFormatter(formatter)
        LOGGER.addHandler(stream_handler)


def save_metashape_document(config: PipelineConfig) -> None:
    """Persist the active Metashape document to the configured project path."""

    if Metashape is None:
        raise PipelineError("Metashape module is not available in this Python runtime.")
    doc = get_or_create_metashape_document(config, create=True)
    save_metashape_document_instance(config, doc)


def get_or_create_metashape_document(config: PipelineConfig, create: bool = False) -> Any:
    """Return the active Metashape document or a headless fallback document."""

    global _HEADLESS_DOCUMENT
    global _HEADLESS_DOCUMENT_PATH
    if Metashape is None:
        raise PipelineError("Metashape module is not available in this Python runtime.")

    requested_project_path = config.project_path.resolve(strict=False)
    active_document = getattr(Metashape.app, "document", None)
    if active_document is not None:
        _HEADLESS_DOCUMENT = None
        _HEADLESS_DOCUMENT_PATH = None
        return active_document

    if _HEADLESS_DOCUMENT is not None:
        if _HEADLESS_DOCUMENT_PATH == requested_project_path:
            return _HEADLESS_DOCUMENT
        _HEADLESS_DOCUMENT = None
        _HEADLESS_DOCUMENT_PATH = None

    if config.project_path.exists():
        headless_document = Metashape.Document()
        headless_document.open(str(config.project_path))
        _HEADLESS_DOCUMENT = headless_document
        _HEADLESS_DOCUMENT_PATH = requested_project_path
        return _HEADLESS_DOCUMENT

    if create:
        _HEADLESS_DOCUMENT = Metashape.Document()
        _HEADLESS_DOCUMENT_PATH = requested_project_path
        return _HEADLESS_DOCUMENT

    return None


def save_metashape_document_instance(config: PipelineConfig, doc: Any) -> None:
    """Persist the provided Metashape document to the configured project path."""

    config.project_path.parent.mkdir(parents=True, exist_ok=True)
    if getattr(doc, "path", ""):
        doc.save()
        return
    doc.save(str(config.project_path))


def metashape_metadata_value(metadata: Any, key: str, default: Any = None) -> Any:
    """Read Metashape metadata without assuming a dict-like .get() method exists."""

    if metadata is None:
        return default
    if hasattr(metadata, "get"):
        try:
            return metadata.get(key, default)
        except TypeError:
            pass
    try:
        value = metadata[key]
    except Exception:
        return default
    return default if value is None else value


def summarize_multicamera_sensor_offsets(chunk: Any) -> Dict[str, Any]:
    """Summarize whether slave-sensor relative rotation was estimated after alignment."""

    sensor_rows: List[Dict[str, Any]] = []
    slave_sensor_rows: List[Dict[str, Any]] = []
    missing_slave_rotation: List[Dict[str, Any]] = []
    for sensor in getattr(chunk, "sensors", []):
        master = getattr(sensor, "master", None)
        sensor_key = getattr(sensor, "key", None)
        master_key = getattr(master, "key", None) if master is not None else None
        is_slave = master_key is not None and sensor_key is not None and master_key != sensor_key
        rotation_available = getattr(sensor, "rotation", None) is not None
        row = {
            "sensor_key": sensor_key,
            "sensor_label": getattr(sensor, "label", ""),
            "master_sensor_key": master_key,
            "is_slave": is_slave,
            "fixed_location": getattr(sensor, "fixed_location", None),
            "fixed_rotation": getattr(sensor, "fixed_rotation", None),
            "rotation_available": rotation_available,
            "rotation_covariance_available": getattr(sensor, "rotation_covariance", None) is not None,
        }
        sensor_rows.append(row)
        if is_slave:
            slave_sensor_rows.append(row)
            if not rotation_available:
                missing_slave_rotation.append(
                    {
                        "sensor_key": sensor_key,
                        "sensor_label": getattr(sensor, "label", ""),
                        "master_sensor_key": master_key,
                    }
                )
    return {
        "sensor_count": len(sensor_rows),
        "slave_sensor_count": len(slave_sensor_rows),
        "slave_rotation_estimated_count": sum(1 for row in slave_sensor_rows if row["rotation_available"]),
        "missing_slave_rotation_count": len(missing_slave_rotation),
        "missing_slave_rotation": missing_slave_rotation,
        "sensors": sensor_rows,
    }


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

    def append_line(self, path: Path, line: str) -> None:
        """Append a UTF-8 text line to a log file."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("{0}\n".format(line.rstrip("\n")))


class ConfigPersistence:
    """JSON persistence for GUI-editable pipeline configuration."""

    def load(self, path: Path) -> PipelineConfig:
        """Load config JSON while preserving defaults for missing keys."""

        if not path.exists():
            raise FileNotFoundError("Config JSON not found: {0}".format(path))
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise PipelineError("Config JSON root must be an object.")
        return PipelineConfig.from_mapping(payload)

    def save(self, config: PipelineConfig, path: Optional[Path] = None) -> Path:
        """Save config JSON to the requested path or the work-local default."""

        target_path = path or config.last_used_config_path
        payload = config.to_dict()
        self._write_json(target_path, payload)
        launcher_path = _default_last_used_config_path(config.project_root)
        if launcher_path.resolve() != target_path.resolve():
            try:
                self._write_json(launcher_path, payload)
            except OSError as exc:
                LOGGER.warning("Failed to update launcher config cache at %s: %s", launcher_path, exc)
        return target_path

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)


class GuiLogHandler(logging.Handler):
    """Forward log records into the GUI without changing core logging paths."""

    def __init__(self, callback: Callable[[int, str], None]) -> None:
        super().__init__(level=logging.INFO)
        self.callback = callback
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        _mark_logger_handler(self, _LOGGER_GUI_HANDLER_ID)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.callback(record.levelno, self.format(record))
        except Exception:
            self.handleError(record)


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

    def probe_streams(self, input_path: Path) -> Dict[str, Any]:
        """Inspect the input OSV container streams via ffprobe."""

        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-of",
            "json",
            str(input_path),
        ]
        result = self._run_command(command)
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise PipelineError("ffprobe did not return valid JSON output.") from exc
        streams = payload.get("streams", [])
        video_streams = self._build_video_stream_records(streams)
        usable_video_streams = [stream for stream in video_streams if stream["is_usable"]]
        self._log_video_stream_details(video_streams)

        enriched_payload = dict(payload)
        enriched_payload["video_streams"] = video_streams
        enriched_payload["video_stream_count"] = len(video_streams)
        enriched_payload["usable_video_stream_count"] = len(usable_video_streams)
        try:
            selection = self._select_stream_pair(video_streams, usable_video_streams)
        except PipelineError as exc:
            enriched_payload["stream_selection_error"] = str(exc)
            self.logs.write_json(self.config.ffprobe_log_path, enriched_payload)
            raise
        enriched_payload["stream_selection"] = selection
        self.logs.write_json(self.config.ffprobe_log_path, enriched_payload)
        self._log_stream_selection(selection)
        return enriched_payload

    def extract_front_stream(self, input_path: Path, out_dir: Path, stream_index: int) -> None:
        """Extract the configured front stream."""

        self._extract_stream(input_path, out_dir, stream_index, "F")

    def extract_back_stream(self, input_path: Path, out_dir: Path, stream_index: int) -> None:
        """Extract the configured back stream."""

        self._extract_stream(input_path, out_dir, stream_index, "B")

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
        input_path = self.config.require_input_video()
        probe_payload = self.probe_streams(input_path)
        front_stream_index = self.selected_stream_index_from_probe(probe_payload, "front")
        back_stream_index = self.selected_stream_index_from_probe(probe_payload, "back")
        self.extract_front_stream(
            input_path, self.config.extracted_front_dir, front_stream_index
        )
        self.extract_back_stream(
            input_path, self.config.extracted_back_dir, back_stream_index
        )
        front_count, back_count = self.verify_frame_counts(
            self.config.extracted_front_dir, self.config.extracted_back_dir
        )
        selection = self.stream_selection_from_probe(probe_payload)
        return PhaseResult(
            phase="extract_streams",
            status="ok",
            message=self.build_selection_message(selection, front_count, back_count),
            details={
                "front_count": front_count,
                "back_count": back_count,
                "video_stream_count": selection["video_stream_count"],
                "usable_video_stream_count": selection["usable_video_stream_count"],
                "selected_front_stream_index": front_stream_index,
                "selected_back_stream_index": back_stream_index,
                "ignored_stream_indices": selection["ignored_stream_indices"],
            },
        )

    @staticmethod
    def stream_selection_from_probe(probe_payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Return the validated stream-selection block from a probe payload."""

        selection = probe_payload.get("stream_selection")
        if not isinstance(selection, Mapping):
            raise PipelineError("probe_streams() did not return stream_selection metadata.")
        return dict(selection)

    @classmethod
    def selected_stream_index_from_probe(cls, probe_payload: Mapping[str, Any], side: str) -> int:
        """Return the selected ffmpeg video-stream index for the requested side."""

        selection = cls.stream_selection_from_probe(probe_payload)
        stream_info = selection.get(side)
        if not isinstance(stream_info, Mapping):
            raise PipelineError("probe_streams() did not return '{0}' stream metadata.".format(side))
        stream_index = stream_info.get("video_stream_index")
        if not isinstance(stream_index, int):
            raise PipelineError("Selected {0} stream is missing a valid video_stream_index.".format(side))
        return stream_index

    @staticmethod
    def build_selection_message(selection: Mapping[str, Any], front_count: int, back_count: int) -> str:
        """Build a concise extraction summary with selected and ignored streams."""

        front_stream = selection.get("front", {})
        back_stream = selection.get("back", {})
        ignored_streams = list(selection.get("ignored_stream_indices", []))
        return (
            "Extracted front/back streams with matching frame counts. "
            "Detected {0} video streams ({1} usable); selected front={2} back={3}; ignored={4}; "
            "front_count={5}; back_count={6}."
        ).format(
            selection.get("video_stream_count", 0),
            selection.get("usable_video_stream_count", 0),
            front_stream.get("video_stream_index"),
            back_stream.get("video_stream_index"),
            ignored_streams,
            front_count,
            back_count,
        )

    def _extract_stream(self, input_path: Path, out_dir: Path, stream_index: int, prefix: str) -> None:
        """Run ffmpeg for a single stream."""

        out_dir.mkdir(parents=True, exist_ok=True)
        self._clear_directory(out_dir, "{0}_*.jpg".format(prefix))

        output_pattern = out_dir / "{0}_%06d.jpg".format(prefix)
        command = ["ffmpeg", "-y", "-i", str(input_path), "-map", "0:v:{0}".format(stream_index)]
        command.extend(self._build_frame_sampling_args())

        command.extend(["-q:v", str(self.config.jpeg_quality), str(output_pattern)])
        self._run_command(command)

        if not self._list_frame_files(out_dir, prefix):
            raise PipelineError(
                "ffmpeg did not produce extracted frames for stream index {0}.".format(stream_index)
            )

    @staticmethod
    def _stream_flag(value: Any) -> bool:
        """Convert ffprobe flag-like values into booleans."""

        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if not text or text in ("0", "false", "no", "off", "none"):
            return False
        return True

    def _build_video_stream_records(self, streams: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """Create detailed records for every ffprobe video stream."""

        video_streams: List[Dict[str, Any]] = []
        for video_stream_index, stream in enumerate(
            stream for stream in streams if stream.get("codec_type") == "video"
        ):
            disposition = dict(stream.get("disposition") or {})
            tags = dict(stream.get("tags") or {})
            unusable_reasons: List[str] = []
            if self._stream_flag(disposition.get("attached_pic")):
                unusable_reasons.append("attached_pic")
            if self._stream_flag(disposition.get("timed_thumbnails")):
                unusable_reasons.append("timed_thumbnails")
            video_streams.append(
                {
                    "video_stream_index": video_stream_index,
                    "ffprobe_stream_index": stream.get("index"),
                    "codec_name": stream.get("codec_name"),
                    "width": stream.get("width"),
                    "height": stream.get("height"),
                    "disposition": disposition,
                    "tags": tags,
                    "is_usable": not unusable_reasons,
                    "unusable_reasons": unusable_reasons,
                }
            )
        return video_streams

    @staticmethod
    def _log_video_stream_details(video_streams: Sequence[Mapping[str, Any]]) -> None:
        """Emit detailed logs for later triage of extra video streams."""

        for stream in video_streams:
            LOGGER.info(
                "Video stream detected: v:%s ffprobe_index=%s codec=%s size=%sx%s usable=%s reasons=%s disposition=%s tags=%s",
                stream.get("video_stream_index"),
                stream.get("ffprobe_stream_index"),
                stream.get("codec_name"),
                stream.get("width"),
                stream.get("height"),
                stream.get("is_usable"),
                list(stream.get("unusable_reasons", [])),
                json.dumps(stream.get("disposition", {}), ensure_ascii=False, sort_keys=True),
                json.dumps(stream.get("tags", {}), ensure_ascii=False, sort_keys=True),
            )

    def _select_stream_pair(
        self,
        video_streams: Sequence[Mapping[str, Any]],
        usable_video_streams: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Validate configured front/back stream indices against the probed video streams."""

        available_video_indices = [int(stream["video_stream_index"]) for stream in video_streams]
        usable_video_indices = [int(stream["video_stream_index"]) for stream in usable_video_streams]
        if len(usable_video_streams) < 2:
            message = (
                "Expected at least 2 usable video streams in the OSV container, but found {0} usable "
                "streams out of {1} detected video streams."
            ).format(len(usable_video_streams), len(video_streams))
            LOGGER.error(
                "%s detected=%s usable=%s",
                message,
                available_video_indices,
                usable_video_indices,
            )
            raise PipelineError(message)

        selected_streams: Dict[str, Dict[str, Any]] = {}
        for side, stream_index in (
            ("front", self.config.front_stream_index),
            ("back", self.config.back_stream_index),
        ):
            stream_info = next(
                (
                    dict(candidate)
                    for candidate in video_streams
                    if int(candidate["video_stream_index"]) == int(stream_index)
                ),
                None,
            )
            if stream_info is None:
                message = (
                    "{0}_stream_index={1} did not match any detected video stream. "
                    "detected={2}, usable={3}"
                ).format(side, stream_index, available_video_indices, usable_video_indices)
                LOGGER.error(message)
                raise PipelineError(message)
            if not stream_info.get("is_usable", False):
                message = (
                    "{0}_stream_index={1} points to a non-usable video stream. "
                    "ffprobe_index={2}, reasons={3}"
                ).format(
                    side,
                    stream_index,
                    stream_info.get("ffprobe_stream_index"),
                    list(stream_info.get("unusable_reasons", [])),
                )
                LOGGER.error(message)
                raise PipelineError(message)
            selected_streams[side] = stream_info

        if selected_streams["front"]["video_stream_index"] == selected_streams["back"]["video_stream_index"]:
            message = "front_stream_index and back_stream_index must select different video streams."
            LOGGER.error(message)
            raise PipelineError(message)

        selected_indices = {
            int(selected_streams["front"]["video_stream_index"]),
            int(selected_streams["back"]["video_stream_index"]),
        }
        ignored_stream_indices = [
            int(stream["video_stream_index"]) for stream in video_streams if int(stream["video_stream_index"]) not in selected_indices
        ]
        ignored_usable_stream_indices = [
            int(stream["video_stream_index"])
            for stream in usable_video_streams
            if int(stream["video_stream_index"]) not in selected_indices
        ]
        ignored_non_usable_stream_indices = [
            int(stream["video_stream_index"])
            for stream in video_streams
            if not stream.get("is_usable", False) and int(stream["video_stream_index"]) not in selected_indices
        ]
        return {
            "video_stream_count": len(video_streams),
            "usable_video_stream_count": len(usable_video_streams),
            "front": selected_streams["front"],
            "back": selected_streams["back"],
            "ignored_stream_indices": ignored_stream_indices,
            "ignored_usable_stream_indices": ignored_usable_stream_indices,
            "ignored_non_usable_stream_indices": ignored_non_usable_stream_indices,
        }

    @staticmethod
    def _log_stream_selection(selection: Mapping[str, Any]) -> None:
        """Emit a compact summary for the selected and ignored streams."""

        front_stream = selection.get("front", {})
        back_stream = selection.get("back", {})
        LOGGER.info(
            "Detected %s video streams; selected front=%s back=%s; ignored=%s",
            selection.get("video_stream_count", 0),
            front_stream.get("video_stream_index"),
            back_stream.get("video_stream_index"),
            list(selection.get("ignored_stream_indices", [])),
        )
        LOGGER.info(
            "Selected stream details: front(v:%s ffprobe_index=%s codec=%s size=%sx%s) "
            "back(v:%s ffprobe_index=%s codec=%s size=%sx%s) usable=%s",
            front_stream.get("video_stream_index"),
            front_stream.get("ffprobe_stream_index"),
            front_stream.get("codec_name"),
            front_stream.get("width"),
            front_stream.get("height"),
            back_stream.get("video_stream_index"),
            back_stream.get("ffprobe_stream_index"),
            back_stream.get("codec_name"),
            back_stream.get("width"),
            back_stream.get("height"),
            selection.get("usable_video_stream_count", 0),
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
        self.backend_manager = OpenCVBackendManager(config, logs)
        self._last_score_metadata: Dict[str, Any] = {
            "backend": "cpu",
            "fallback": False,
            "fallback_reason": "",
            "elapsed_ms": None,
        }
        self._current_image_path: Optional[Path] = None

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

    def laplacian_score_cpu(self, image: Any) -> Optional[float]:
        """Compute Laplacian variance on CPU within the center-70-percent circle."""

        self._require_cv_runtime()
        mask = self.compute_center70_mask(image)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        masked_values = laplacian[mask > 0]
        if masked_values.size == 0:
            raise PipelineError("Center-70-percent mask produced no pixels for blur evaluation.")
        return float(masked_values.var())

    def laplacian_score_cuda(self, image: Any) -> Optional[float]:
        """Compute Laplacian variance using OpenCV CUDA when the build exposes the required APIs."""

        self._require_cv_runtime()
        cuda_api = getattr(cv2, "cuda", None)
        if cuda_api is None:
            raise PipelineError("cv2.cuda is unavailable in the current OpenCV build.")
        if not hasattr(cuda_api, "createLaplacianFilter"):
            raise PipelineError("cv2.cuda.createLaplacianFilter is unavailable in the current OpenCV build.")

        source_image = image
        if getattr(source_image, "dtype", None) != np.uint8:
            source_image = source_image.astype(np.uint8, copy=False)
        gpu_image = cuda_api.GpuMat()
        gpu_image.upload(source_image)
        gpu_working = self._ensure_cuda_grayscale(cuda_api, gpu_image, source_image)

        if self.config.cuda_use_gaussian_preblur:
            if not hasattr(cuda_api, "createGaussianFilter"):
                raise PipelineError("cuda_use_gaussian_preblur requires cv2.cuda.createGaussianFilter.")
            gaussian_filter = cuda_api.createGaussianFilter(
                self._cv8uc1(),
                self._cv8uc1(),
                (3, 3),
                0,
            )
            gpu_working = self._apply_cuda_filter(gaussian_filter, gpu_working)

        # Some OpenCV CUDA Python builds only accept matching source/destination types here.
        gpu_working = self._convert_cuda_mat_type(gpu_working, self._cv32fc1())
        laplacian_filter = cuda_api.createLaplacianFilter(self._cv32fc1(), self._cv32fc1(), 3)
        gpu_laplacian = self._apply_cuda_filter(laplacian_filter, gpu_working)
        mask = self.compute_center70_mask(image)
        laplacian = self._download_cuda_mat(gpu_laplacian)
        if laplacian is None or getattr(laplacian, "size", 0) == 0:
            raise PipelineError("OpenCV CUDA Laplacian download returned no pixels for blur evaluation.")
        if getattr(laplacian, "ndim", 0) > 2:
            laplacian = laplacian[:, :, 0]
        laplacian = np.asarray(laplacian, dtype=np.float64)
        masked_values = laplacian[mask > 0]
        if masked_values.size == 0:
            raise PipelineError("Center-70-percent mask produced no pixels for CUDA blur evaluation.")
        return float(masked_values.var())

    def laplacian_score(self, image: Any) -> Optional[float]:
        """Dispatch the configured Laplacian blur evaluation to CPU or CUDA."""

        backend = self.backend_manager.ensure_backend(self.config)
        start_time = time.perf_counter()
        if backend == "cuda":
            try:
                score = self.laplacian_score_cuda(image)
                self._set_score_metadata("cuda", False, "", start_time)
                return score
            except Exception as exc:
                if not self.config.cuda_allow_fallback:
                    raise PipelineError("CUDA blur evaluation failed with fallback disabled: {0}".format(exc)) from exc
                function_name = "BlurEvaluator.laplacian_score_cuda"
                image_label = str(self._current_image_path) if self._current_image_path else "<unknown>"
                reason = "{0} failed for image {1}; falling back to CPU: {2}".format(
                    function_name,
                    image_label,
                    exc,
                )
                self.backend_manager.fallback_to_cpu(
                    reason,
                    context=function_name,
                    image_path=self._current_image_path,
                )
                self._log_cuda_fallback(function_name, reason)
                score = self.laplacian_score_cpu(image)
                self._set_score_metadata("cpu", True, reason, start_time)
                return score

        score = self.laplacian_score_cpu(image)
        self._set_score_metadata("cpu", False, "", start_time)
        return score

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

        front_result = self._score_image(front_path)
        back_result = self._score_image(back_path)
        front_score = float(front_result["score"])
        back_score = float(back_result["score"])
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
            "backend_front": front_result["backend"],
            "backend_back": back_result["backend"],
            "fallback_front": int(bool(front_result["fallback"])),
            "fallback_back": int(bool(back_result["fallback"])),
            "fallback_reason_front": front_result["fallback_reason"],
            "fallback_reason_back": back_result["fallback_reason"],
            "elapsed_ms_front": front_result["elapsed_ms"],
            "elapsed_ms_back": back_result["elapsed_ms"],
        }

    def select_pairs(self, front_dir: Path, back_dir: Path, out_front_dir: Path, out_back_dir: Path) -> PhaseResult:
        """Evaluate extracted pairs, keep both sides when either side passes, and log the results."""

        self.ensure_backend_ready()
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
                "backend_front",
                "backend_back",
                "fallback_front",
                "fallback_back",
                "fallback_reason_front",
                "fallback_reason_back",
                "elapsed_ms_front",
                "elapsed_ms_back",
            ),
        )
        self.backend_manager.save_backend_report()
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
                "active_backend": self.backend_manager.active_backend,
                "fallback_detected": self.backend_manager.fallback_detected,
                "backend_report": self.config.opencv_backend_report_path,
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

    def ensure_backend_ready(self) -> str:
        """Resolve and persist the preprocessing backend before frame selection starts."""

        return self.backend_manager.ensure_backend(self.config)

    def _score_image(self, image_path: Path) -> Dict[str, Any]:
        """Read an image and compute the configured primary blur score."""

        image = self._read_grayscale_image(image_path)
        self._current_image_path = image_path
        if self.config.blur_method == "laplacian_center70":
            score = self.laplacian_score(image)
        else:
            raise PipelineError("Unsupported blur_method: {0}".format(self.config.blur_method))
        if score is None:
            raise PipelineError("Blur score could not be computed for {0}".format(image_path))
        result = dict(self._last_score_metadata)
        result["score"] = float(score)
        return result

    def _read_grayscale_image(self, image_path: Path) -> Any:
        """Read an image from disk as grayscale."""

        self._require_cv_runtime()
        image = _read_image_with_unicode_path(image_path, cv2.IMREAD_GRAYSCALE)
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

    def _set_score_metadata(self, backend: str, fallback: bool, fallback_reason: str, start_time: float) -> None:
        elapsed_ms: Optional[float] = None
        if self.config.cuda_benchmark_mode:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 4)
        self._last_score_metadata = {
            "backend": backend,
            "fallback": bool(fallback),
            "fallback_reason": fallback_reason,
            "elapsed_ms": elapsed_ms,
        }

    def _log_cuda_fallback(self, context: str, reason: str) -> None:
        image_label = str(self._current_image_path) if self._current_image_path else "<unknown>"
        LOGGER.warning("%s [context=%s image=%s]", reason, context, image_label)

    @staticmethod
    def _cv32fc1() -> int:
        return int(getattr(cv2, "CV_32FC1", cv2.CV_32F))

    @staticmethod
    def _cv8uc1() -> int:
        return int(getattr(cv2, "CV_8UC1", cv2.CV_8U))

    @staticmethod
    def _normalize_cuda_result(result: Any) -> Optional[Any]:
        if result is None:
            return None
        if hasattr(result, "download"):
            return result
        if isinstance(result, (tuple, list)):
            for item in result:
                if hasattr(item, "download"):
                    return item
        return None

    def _ensure_cuda_grayscale(self, cuda_api: Any, gpu_image: Any, source_image: Any) -> Any:
        """Normalize the uploaded image to a single-channel CUDA mat."""

        if getattr(source_image, "ndim", 0) < 3:
            return gpu_image
        if source_image.shape[2] == 1:
            return gpu_image

        if hasattr(cuda_api, "cvtColor") and hasattr(cv2, "COLOR_BGR2GRAY"):
            gpu_gray = self._normalize_cuda_result(cuda_api.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY))
            if gpu_gray is not None:
                return gpu_gray

        if hasattr(cv2, "cvtColor") and hasattr(cv2, "COLOR_BGR2GRAY"):
            gray_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
            gpu_gray = cuda_api.GpuMat()
            gpu_gray.upload(gray_image)
            return gpu_gray

        raise PipelineError("Unable to convert a multi-channel image to grayscale for CUDA blur evaluation.")

    @staticmethod
    def _download_cuda_mat(gpu_mat: Any) -> Any:
        if gpu_mat is None or not hasattr(gpu_mat, "download"):
            raise PipelineError("OpenCV CUDA did not return a downloadable GpuMat result.")
        return gpu_mat.download()

    @staticmethod
    def _convert_cuda_mat_type(gpu_mat: Any, cv_type: int) -> Any:
        """Convert a CUDA mat type while tolerating Python binding signature differences."""

        if gpu_mat is None or not hasattr(gpu_mat, "convertTo"):
            raise PipelineError("OpenCV CUDA did not expose GpuMat.convertTo() for type conversion.")
        try:
            result = gpu_mat.convertTo(cv_type)
            normalized_result = BlurEvaluator._normalize_cuda_result(result)
            if normalized_result is not None:
                return normalized_result
        except TypeError:
            pass
        except Exception as exc:
            raise PipelineError("OpenCV CUDA convertTo() failed: {0}".format(exc)) from exc

        if getattr(cv2, "cuda", None) is None or not hasattr(cv2.cuda, "GpuMat"):
            raise PipelineError("OpenCV CUDA convertTo() requires cv2.cuda.GpuMat in this build.")

        gpu_destination = cv2.cuda.GpuMat()
        for args in ((cv_type, gpu_destination), (cv_type, 1.0, 0.0, gpu_destination)):
            try:
                gpu_mat.convertTo(*args)
                return gpu_destination
            except TypeError:
                continue
            except Exception as exc:
                raise PipelineError("OpenCV CUDA convertTo() failed: {0}".format(exc)) from exc

        raise PipelineError("OpenCV CUDA convertTo() does not support the expected Python signatures.")

    @staticmethod
    def _apply_cuda_filter(cuda_filter: Any, gpu_source: Any) -> Any:
        """Run a CUDA filter while tolerating minor Python binding signature differences."""

        try:
            result = cuda_filter.apply(gpu_source)
            normalized_result = BlurEvaluator._normalize_cuda_result(result)
            if normalized_result is not None:
                return normalized_result
        except TypeError:
            pass
        gpu_destination = cv2.cuda.GpuMat()
        try:
            cuda_filter.apply(gpu_source, gpu_destination)
        except Exception as exc:
            # TODO: validate CUDA filter.apply(...) argument ordering on the target Metashape OpenCV build.
            raise PipelineError("OpenCV CUDA filter apply() failed: {0}".format(exc)) from exc
        return gpu_destination

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
        self._report = YoloBackendReport()
        self._report_cached = False
        self._active_device = "cpu"
        self._resolved_model_path: Optional[Path] = None

    def detect_backend_support(self) -> Dict[str, Any]:
        """Inspect YOLO / PyTorch CUDA availability and cache the result."""

        if self._report_cached:
            return self._report.to_dict()

        report = YoloBackendReport(
            ultralytics_available=YOLO is not None,
            torch_available=torch is not None,
            torch_version=getattr(torch, "__version__", "") if torch is not None else "",
            requested_mode=self.config.yolo_device_mode,
            selected_device=self._active_device,
            active_device=self._active_device,
            prefer_cuda=self.config.prefer_yolo_cuda,
            allow_fallback=self.config.yolo_allow_fallback,
            device_index=self.config.yolo_device_index,
            local_model_path=str(self.config.find_local_mask_model_path() or ""),
        )
        if YOLO is None:
            report.warnings.append("Ultralytics YOLO is not available in this Python runtime.")
        if torch is None:
            report.warnings.append("PyTorch is not available in this Python runtime.")
            self._report = report
            self._report_cached = True
            return report.to_dict()

        cuda_api = getattr(torch, "cuda", None)
        if cuda_api is None or not hasattr(cuda_api, "is_available"):
            report.warnings.append("torch.cuda is not available in this runtime.")
            self._report = report
            self._report_cached = True
            return report.to_dict()

        try:
            report.cuda_available = bool(cuda_api.is_available())
        except Exception as exc:
            report.warnings.append("Failed to query torch.cuda.is_available(): {0}".format(exc))
            report.cuda_available = False
        try:
            report.cuda_device_count = int(cuda_api.device_count())
        except Exception as exc:
            report.warnings.append("Failed to query torch.cuda.device_count(): {0}".format(exc))
            report.cuda_device_count = 0

        if report.cuda_available and report.cuda_device_count > 0:
            for device_index in range(report.cuda_device_count):
                try:
                    report.device_names.append(str(cuda_api.get_device_name(device_index)))
                except Exception:
                    report.device_names.append("cuda:{0}".format(device_index))
        elif self.config.yolo_device_mode == "cuda" or self.config.prefer_yolo_cuda:
            report.warnings.append("PyTorch CUDA device is not available; YOLO will use CPU when fallback is allowed.")

        self._report = report
        self._report_cached = True
        return report.to_dict()

    def resolve_device(self) -> str:
        """Choose the effective device for YOLO inference."""

        self.detect_backend_support()
        if self._model is not None and self._active_device:
            return self._active_device
        explicit_device = str(self.config.mask_device).strip() if self.config.mask_device else ""
        if explicit_device:
            self._report.notes.append("Using explicit mask_device override from config.")
            self._report.selected_device = explicit_device
            self._active_device = explicit_device
            return explicit_device

        requested = self.config.yolo_device_mode
        self._report.requested_mode = requested
        self._report.prefer_cuda = self.config.prefer_yolo_cuda
        self._report.allow_fallback = self.config.yolo_allow_fallback
        self._report.device_index = self.config.yolo_device_index

        if requested == "cpu":
            self._report.selected_device = "cpu"
            self._active_device = "cpu"
            return "cpu"

        if requested == "auto" and not self.config.prefer_yolo_cuda:
            self._report.notes.append("prefer_yolo_cuda=False forced CPU selection while yolo_device_mode=auto.")
            self._report.selected_device = "cpu"
            self._active_device = "cpu"
            return "cpu"

        cuda_ready = self._report.cuda_available and self._report.cuda_device_count > self.config.yolo_device_index
        if requested == "cuda":
            if cuda_ready:
                device = "cuda:{0}".format(self.config.yolo_device_index)
                self._set_active_device(device)
                return device
            reason = self._cuda_unavailable_reason()
            if self.config.yolo_allow_fallback:
                self.record_fallback(reason, context="device_selection", requested_device="cuda")
                self._set_active_device("cpu")
                return "cpu"
            raise PipelineError(reason)

        if requested == "auto":
            if cuda_ready:
                device = "cuda:{0}".format(self.config.yolo_device_index)
                self._set_active_device(device)
                return device
            reason = self._cuda_unavailable_reason()
            self.record_fallback(reason, context="device_selection", requested_device="auto")
            self._set_active_device("cpu")
            return "cpu"

        raise PipelineError("Unsupported yolo_device_mode: {0}".format(requested))

    def load_model(self) -> Any:
        """Load the configured YOLO segmentation model once per run."""

        self._require_yolo_runtime()
        if self._model is None:
            resolved_model_path = self.config.resolve_mask_model_path()
            selected_device = self.resolve_device()
            self._resolved_model_path = resolved_model_path
            self._report.local_model_path = str(resolved_model_path)
            LOGGER.info("Loading YOLO model from local path: %s", resolved_model_path)
            LOGGER.info("YOLO inference device selection: %s", selected_device)
            try:
                self._model = YOLO(str(resolved_model_path))
            except Exception as exc:
                raise PipelineError(
                    "Failed to load YOLO model from local file '{0}'.".format(resolved_model_path)
                ) from exc
            self._report.model_loaded = True
            self._report.active_device = self._active_device
            self.save_backend_report()
        return self._model

    def infer_mask(self, image_path: Path) -> Dict[str, Any]:
        """Run YOLO segmentation and return a binary mask summary."""

        self._require_cv_runtime()
        image = _read_image_with_unicode_path(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise PipelineError("Failed to read image for mask generation: {0}".format(image_path))

        model = self.load_model()
        device = self.resolve_device()
        predict_kwargs: Dict[str, Any] = {
            "source": str(image_path),
            "verbose": False,
            "conf": self.config.mask_confidence_threshold,
            "iou": self.config.mask_iou_threshold,
            "device": device,
        }

        try:
            results = model.predict(**predict_kwargs)
        except Exception as exc:
            if self._is_cuda_device(device) and self.config.yolo_allow_fallback:
                reason = "YOLO CUDA inference failed for {0}; fallback to CPU: {1}".format(image_path.name, exc)
                self.fallback_to_cpu(reason, context="predict", image_path=image_path)
                predict_kwargs["device"] = "cpu"
                try:
                    results = model.predict(**predict_kwargs)
                except Exception as fallback_exc:
                    raise PipelineError("YOLO inference failed for {0}.".format(image_path.name)) from fallback_exc
            else:
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
        mask_to_save = self._normalize_mask_for_save(mask)
        _write_image_with_unicode_path(out_path, mask_to_save)

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
            "mask_polarity": self.config.mask_polarity,
        }

    def run(self) -> PhaseResult:
        """Generate binary mask PNGs for each selected front/back pair."""

        self.detect_backend_support()
        self.save_backend_report()
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
                "mask_polarity",
            ),
        )
        self.save_backend_report()
        return PhaseResult(
            phase="generate_masks",
            status="ok",
            message="Generated binary mask PNGs for selected front/back frame pairs.",
            details={
                "pair_count": len(pairs),
                "image_count": len(rows),
                "masked_image_count": masked_images,
                "mask_summary_csv": self.config.mask_summary_log_path,
                "yolo_device": self._active_device,
                "yolo_backend_report": self.config.yolo_backend_report_path,
            },
        )

    def build_backend_report(self, probe_runtime: bool = True) -> Dict[str, Any]:
        """Return a serializable YOLO backend report with current fallback state."""

        if probe_runtime:
            self.detect_backend_support()
        elif not self._report_cached:
            _append_unique_message(
                self._report.notes,
                "YOLO device probe is deferred until explicit status refresh or pipeline execution.",
            )
            self._report.requested_mode = self.config.yolo_device_mode
            self._report.prefer_cuda = self.config.prefer_yolo_cuda
            self._report.allow_fallback = self.config.yolo_allow_fallback
            self._report.device_index = self.config.yolo_device_index
            self._report.local_model_path = str(self.config.find_local_mask_model_path() or "")
        self._report.selected_device = self._report.selected_device or self._active_device
        self._report.active_device = self._active_device
        if self._resolved_model_path is not None:
            self._report.local_model_path = str(self._resolved_model_path)
        payload = self._report.to_dict()
        payload["backend_report_path"] = str(self.config.yolo_backend_report_path)
        payload["save_backend_report"] = self.config.save_backend_report
        return payload

    def save_backend_report(self) -> None:
        if self.config.save_backend_report:
            self.logs.write_json(self.config.yolo_backend_report_path, self.build_backend_report())

    def record_fallback(
        self,
        reason: str,
        context: str,
        requested_device: Optional[str] = None,
        image_path: Optional[Path] = None,
    ) -> None:
        clean_reason = str(reason).strip() or "Unknown YOLO fallback reason."
        if clean_reason not in self._report.fallback_reasons:
            self._report.fallback_reasons.append(clean_reason)
        event = {
            "context": context,
            "reason": clean_reason,
            "requested_device": requested_device or self.config.yolo_device_mode,
            "image_path": str(image_path) if image_path else "",
        }
        self._report.fallback_events.append(event)
        LOGGER.warning("YOLO backend fallback [%s]: %s", context, clean_reason)
        self.logs.append_line(
            self.config.cuda_fallback_log_path,
            "yolo\t{0}\t{1}\t{2}\t{3}".format(
                context,
                event["requested_device"],
                event["image_path"] or "<unknown>",
                clean_reason,
            ),
        )

    def fallback_to_cpu(self, reason: str, context: str, image_path: Optional[Path] = None) -> None:
        self.record_fallback(reason, context=context, image_path=image_path)
        self._set_active_device("cpu")
        self.save_backend_report()

    def cleanup(self) -> None:
        """Release model references and cached device state on GUI shutdown."""

        self._model = None
        self._resolved_model_path = None
        self._active_device = "cpu"
        self._report = YoloBackendReport()
        self._report_cached = False
        cuda_api = getattr(torch, "cuda", None) if torch is not None else None
        if cuda_api is not None and hasattr(cuda_api, "empty_cache"):
            try:
                cuda_api.empty_cache()
            except Exception:
                pass

    def _set_active_device(self, device: str) -> None:
        self._active_device = device
        self._report.selected_device = device
        self._report.active_device = device
        if device.startswith("cuda:"):
            try:
                device_index = int(device.split(":", 1)[1])
            except ValueError:
                device_index = 0
            if len(self._report.device_names) > device_index:
                self._report.device_name = self._report.device_names[device_index]
        else:
            self._report.device_name = "cpu"

    @staticmethod
    def _is_cuda_device(device: str) -> bool:
        return str(device).startswith("cuda")

    def _cuda_unavailable_reason(self) -> str:
        if YOLO is None:
            return "Ultralytics YOLO is not available in this Python runtime."
        if torch is None:
            return "PyTorch is not available in this Python runtime."
        if not self._report.cuda_available or self._report.cuda_device_count <= 0:
            return "CUDA is not available for YOLO inference."
        if self._report.cuda_device_count <= self.config.yolo_device_index:
            return "yolo_device_index={0} is out of range for {1} detected CUDA device(s).".format(
                self.config.yolo_device_index,
                self._report.cuda_device_count,
            )
        return "YOLO CUDA device selection failed."

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
        mask_array = MaskGenerator._normalize_binary_mask(mask_array > 0.5)
        if mask_array.shape != shape:
            mask_array = cv2.resize(mask_array, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_array = MaskGenerator._normalize_binary_mask(mask_array)
        return mask_array

    @staticmethod
    def _mask_result(mask: Any, detection_count: int, target_classes: Sequence[str]) -> Dict[str, Any]:
        normalized_mask = MaskGenerator._normalize_binary_mask(mask)
        unique_classes = sorted(set(class_name for class_name in target_classes if class_name))
        return {
            "mask": normalized_mask,
            "detection_count": detection_count,
            "target_classes": unique_classes,
            "masked_pixels": int(np.count_nonzero(normalized_mask)),
        }

    @staticmethod
    def _normalize_binary_mask(mask: Any) -> Any:
        return np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)

    def _normalize_mask_for_save(self, mask: Any) -> Any:
        normalized_mask = self._normalize_binary_mask(mask)
        if self.config.mask_polarity == "target_black":
            return np.where(normalized_mask > 0, 0, 255).astype(np.uint8)
        return normalized_mask

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
        """Set all sensors to EquisolidFisheye as required by the spec."""

        for sensor in getattr(chunk, "sensors", []):
            sensor.type = Metashape.Sensor.Type.EquisolidFisheye

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
            # TODO: re-check camera.mask assignment and target_black PNG polarity on the current Metashape build with a small sample.
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

        save_metashape_document_instance(self.config, doc)

    def run(self) -> PhaseResult:
        """Create or reuse a chunk, import paired images, and apply per-camera masks."""

        self._require_metashape()
        doc = get_or_create_metashape_document(self.config, create=True)
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

    def disable_low_quality_cameras(
        self,
        chunk: Any,
        threshold: float = 0.5,
    ) -> Tuple[int, Dict[str, Dict[str, Any]]]:
        """Disable station pairs only when all measured sides fall below threshold."""

        decisions = self._quality_decisions_by_camera(chunk, threshold)
        disabled = 0
        for camera in getattr(chunk, "cameras", []):
            camera_label = getattr(camera, "label", "")
            decision = decisions.get(camera_label)
            if decision is None or not decision.get("disable_pair"):
                continue
            if not getattr(camera, "enabled", True):
                continue
            camera.enabled = False
            disabled += 1
        return disabled, decisions

    def export_quality_log(
        self,
        chunk: Any,
        csv_path: Path,
        quality_decisions: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        """Export current Metashape quality metadata."""

        rows: List[Dict[str, Any]] = []
        for camera in getattr(chunk, "cameras", []):
            camera_label = getattr(camera, "label", "")
            decision = quality_decisions.get(camera_label, {}) if quality_decisions else {}
            rows.append(
                {
                    "camera_label": camera_label,
                    "frame_id": self._frame_id_for_camera(camera),
                    "camera_side": self._side_for_camera(camera),
                    "enabled": getattr(camera, "enabled", True),
                    "aligned": self.camera_is_aligned(camera),
                    "image_quality": metashape_metadata_value(getattr(camera, "meta", None), "Image/Quality", ""),
                    "quality_keep_rule": decision.get("quality_keep_rule", ""),
                    "disabled_reason": decision.get("disabled_reason", ""),
                }
            )
        self.logs.write_csv(
            csv_path,
            rows,
            headers=(
                "camera_label",
                "frame_id",
                "camera_side",
                "enabled",
                "aligned",
                "image_quality",
                "quality_keep_rule",
                "disabled_reason",
            ),
        )

    def match_photos(self, chunk: Any, config: PipelineConfig, reset_matches: bool = False) -> None:
        """Run feature matching with current argument names from the checklist."""

        chunk.matchPhotos(
            downscale=config.match_downscale,
            generic_preselection=config.generic_preselection,
            reference_preselection=config.reference_preselection,
            filter_mask=True,
            mask_tiepoints=False,
            filter_stationary_points=config.filter_stationary_points,
            keep_keypoints=config.keep_keypoints,
            keypoint_limit=config.keypoint_limit,
            tiepoint_limit=config.tiepoint_limit,
            reset_matches=reset_matches,
        )

    def align_cameras(self, chunk: Any, reset_alignment: bool = True) -> None:
        """Run camera alignment with optional reset of current alignment."""

        chunk.alignCameras(reset_alignment=reset_alignment)

    @staticmethod
    def sensor_offset_summary(chunk: Any) -> Dict[str, Any]:
        """Return a compact summary of multi-camera slave-sensor offset estimation."""

        return summarize_multicamera_sensor_offsets(chunk)

    def realign_after_cleanup(self, chunk: Any, config: PipelineConfig) -> None:
        """Rerun alignment after overlap cleanup."""

        if sum(1 for camera in getattr(chunk, "cameras", []) if getattr(camera, "enabled", True)) < 2:
            return
        self.match_photos(chunk, config, reset_matches=True)
        self.align_cameras(chunk, reset_alignment=True)

    def run(self) -> PhaseResult:
        """Analyze quality and align the active chunk when cameras are available."""

        self._require_metashape()
        document = get_or_create_metashape_document(self.config, create=False)
        chunk = getattr(document, "chunk", None) if document is not None else None
        if chunk is None:
            return PhaseResult("align", "skipped", "No active chunk is available.", {})
        if not getattr(chunk, "cameras", []):
            return PhaseResult("align", "skipped", "Active chunk has no cameras to align.", {})

        self.analyze_image_quality(chunk)
        disabled_count, quality_decisions = self.disable_low_quality_cameras(
            chunk, threshold=self.config.metashape_image_quality_threshold
        )
        self.export_quality_log(chunk, self.config.metashape_quality_log_path, quality_decisions)
        enabled_camera_count = sum(1 for camera in getattr(chunk, "cameras", []) if getattr(camera, "enabled", True))
        if enabled_camera_count < 2:
            raise PipelineError("Fewer than two enabled cameras remain after Image/Quality filtering.")
        self.match_photos(chunk, self.config, reset_matches=True)
        self.align_cameras(chunk, reset_alignment=True)
        self.export_quality_log(chunk, self.config.metashape_quality_log_path, quality_decisions)
        sensor_offset_summary = self.sensor_offset_summary(chunk)
        if sensor_offset_summary["missing_slave_rotation_count"] > 0:
            LOGGER.warning(
                "Slave sensor rotation was not estimated for %d sensors: %s",
                sensor_offset_summary["missing_slave_rotation_count"],
                sensor_offset_summary["missing_slave_rotation"],
            )
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
                "sensor_offset_summary": sensor_offset_summary,
            },
        )

    def _quality_decisions_by_camera(
        self,
        chunk: Any,
        threshold: float,
    ) -> Dict[str, Dict[str, Any]]:
        stations: Dict[int, Dict[str, Any]] = {}
        decisions: Dict[str, Dict[str, Any]] = {}
        for camera in getattr(chunk, "cameras", []):
            label = getattr(camera, "label", "")
            side = camera_side_from_label(label)
            if side is None:
                continue
            frame_id = extract_frame_id_from_label(label)
            station = stations.setdefault(frame_id, {})
            station[side] = camera

        for station in stations.values():
            cameras = list(station.values())
            measured_qualities = [self.camera_quality(camera) for camera in cameras]
            valid_qualities = [quality for quality in measured_qualities if quality is not None]
            disable_pair = False
            quality_keep_rule = ""
            disabled_reason = ""

            if len(cameras) < 2:
                # TODO: validate how incomplete front/back stations should be handled on the current build.
                quality_keep_rule = "keep_incomplete_station"
            elif not valid_qualities:
                quality_keep_rule = "keep_pair_no_quality"
            elif len(valid_qualities) != len(cameras):
                quality_keep_rule = "keep_pair_missing_quality"
            elif any(quality >= threshold for quality in valid_qualities):
                quality_keep_rule = "keep_pair_either_side_ok"
            else:
                disable_pair = True
                quality_keep_rule = "disable_pair_both_below_threshold"
                disabled_reason = "both_measured_sides_below_image_quality_threshold"

            for camera in cameras:
                decisions[getattr(camera, "label", "")] = {
                    "disable_pair": disable_pair,
                    "quality_keep_rule": quality_keep_rule,
                    "disabled_reason": disabled_reason,
                }
        return decisions

    @staticmethod
    def camera_quality(camera: Any) -> Optional[float]:
        value = metashape_metadata_value(getattr(camera, "meta", None), "Image/Quality")
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
        document = get_or_create_metashape_document(self.config, create=False)
        chunk = getattr(document, "chunk", None) if document is not None else None
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


class PipelineController:
    """Shared orchestration layer for menu callbacks and GUI actions."""

    def __init__(
        self,
        config: PipelineConfig,
        logs: Optional[LogWriter] = None,
        progress_callback: Optional[Callable[[str, str, int, int], None]] = None,
        initialize_logging: bool = True,
    ) -> None:
        self.config = config
        self.logs = logs or LogWriter()
        self.extractor = FFmpegExtractor(self.config, self.logs)
        self.blur_evaluator = BlurEvaluator(self.config, self.logs)
        self.mask_generator = MaskGenerator(self.config, self.logs)
        self.gpu_status = GpuStatusAggregator(self.config, self.logs)
        self.importer = MetashapeImporter(self.config)
        self.aligner = MetashapeAligner(self.config, self.logs)
        self.overlap_reducer = OverlapReducer(self.config, self.logs)
        self.progress_callback = progress_callback
        self._current_step = ""
        self._logging_configured = False
        if initialize_logging:
            self.initialize_logging()

    def initialize_logging(self) -> None:
        """Configure file and stream logging when runtime work is about to begin."""

        if self._logging_configured:
            return
        configure_logging(self.config)
        self._logging_configured = True

    def run_full_pipeline(self) -> PhaseResult:
        """Execute the full pipeline using the existing phase components."""

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

        overall_status = "error" if any(result.status == "error" for result in phase_results) else "ok"
        summary_payload = self.build_log_summary({"phases": [result.to_dict() for result in phase_results]})
        summary_message = "Phase 1 through Phase 3 completed through alignment, overlap reduction, and log export."
        if overall_status == "error":
            failed_phase = next((result.phase for result in phase_results if result.status == "error"), "unknown")
            summary_message = "Phase 1 through Phase 3 stopped after an error in '{0}'.".format(failed_phase)
        summary = PhaseResult(
            phase="run_full_pipeline",
            status=overall_status,
            message=summary_message,
            details={"phases": summary_payload["phases"]},
        )
        self.logs.write_summary(self.config.summary_log_path, summary_payload)
        return summary

    def run_extract_streams(self) -> PhaseResult:
        return self._run_phase(self._run_extract_streams_impl, "extract_streams", require_input=True)

    def run_select_frames(self) -> PhaseResult:
        return self._run_phase(self._run_select_frames_impl, "select_frames")

    def run_generate_masks(self) -> PhaseResult:
        return self._run_phase(self._run_generate_masks_impl, "generate_masks")

    def run_import_to_metashape(self) -> PhaseResult:
        return self._run_phase(self._run_import_to_metashape_impl, "import_to_metashape")

    def run_align(self) -> PhaseResult:
        return self._run_phase(self._run_align_impl, "align")

    def run_reduce_overlap(self) -> PhaseResult:
        return self._run_phase(self._run_reduce_overlap_impl, "reduce_overlap")

    def run_export_logs(self) -> PhaseResult:
        return self._run_phase(self._run_export_logs_impl, "export_logs")

    def build_log_summary(
        self,
        extra: Optional[Mapping[str, Any]] = None,
        probe_backends: bool = True,
    ) -> Dict[str, Any]:
        """Build a compact summary of config, logs, and the active chunk."""

        required_logs = {
            "ffprobe.json": self.config.ffprobe_log_path.exists(),
            "frame_quality.csv": self.config.frame_quality_log_path.exists(),
            "opencv_backend_report.json": self.config.opencv_backend_report_path.exists()
            if self.config.save_backend_report
            else False,
            "yolo_backend_report.json": self.config.yolo_backend_report_path.exists()
            if self.config.save_backend_report
            else False,
            "metashape_gpu_report.json": self.config.metashape_gpu_report_path.exists()
            if self.config.save_backend_report
            else False,
            "gpu_summary_report.json": self.config.gpu_summary_report_path.exists()
            if self.config.save_backend_report
            else False,
            "mask_summary.csv": self.config.mask_summary_log_path.exists(),
            "metashape_quality.csv": self.config.metashape_quality_log_path.exists(),
            "overlap_reduction.csv": self.config.overlap_reduction_log_path.exists(),
            "pipeline_summary.json": True,
            "error.log": self.config.error_log_path.exists(),
            "cuda_fallback.log": self.config.cuda_fallback_log_path.exists(),
        }
        gpu_reports = self.gpu_status.collect_all(
            self.blur_evaluator.backend_manager,
            self.mask_generator,
            save=self.config.save_backend_report if probe_backends else False,
            probe_runtime=probe_backends,
        )
        summary: Dict[str, Any] = {
            "config": self.config.to_dict(),
            "required_logs": required_logs,
            "existing_logs": sorted(path.name for path in self.config.log_dir.glob("*") if path.is_file()),
            "opencv_backend": gpu_reports["opencv"],
            "yolo_backend": gpu_reports["yolo"],
            "metashape_gpu": gpu_reports["metashape"],
            "gpu_summary": gpu_reports["summary"],
        }
        document = get_or_create_metashape_document(self.config, create=False) if Metashape is not None else None
        if document is not None and getattr(document, "chunk", None) is not None:
            chunk = document.chunk
            summary["active_chunk"] = {
                "label": getattr(chunk, "label", ""),
                "camera_count": len(getattr(chunk, "cameras", [])),
                "enabled_camera_count": len(
                    [camera for camera in getattr(chunk, "cameras", []) if getattr(camera, "enabled", True)]
                ),
                "sensor_offset_summary": summarize_multicamera_sensor_offsets(chunk),
            }
        if extra:
            summary.update(extra)
        return summary

    def cleanup(self) -> None:
        """Release GUI-owned backend state and close managed log handlers."""

        self.blur_evaluator.backend_manager.cleanup()
        self.mask_generator.cleanup()
        self._logging_configured = False

    def _run_extract_streams_impl(self) -> PhaseResult:
        input_path = self.config.require_input_video()
        self._set_step("extract_streams", "probe_streams", 1, 5)
        payload = self.extractor.probe_streams(input_path)
        selection = self.extractor.stream_selection_from_probe(payload)
        front_stream_index = self.extractor.selected_stream_index_from_probe(payload, "front")
        back_stream_index = self.extractor.selected_stream_index_from_probe(payload, "back")

        self._set_step("extract_streams", "extract_front_stream", 2, 5)
        self.extractor.extract_front_stream(
            input_path, self.config.extracted_front_dir, front_stream_index
        )

        self._set_step("extract_streams", "extract_back_stream", 3, 5)
        self.extractor.extract_back_stream(
            input_path, self.config.extracted_back_dir, back_stream_index
        )

        self._set_step("extract_streams", "verify_frame_counts", 4, 5)
        front_count, back_count = self.extractor.verify_frame_counts(
            self.config.extracted_front_dir, self.config.extracted_back_dir
        )

        self._set_step("extract_streams", "complete", 5, 5)
        return PhaseResult(
            phase="extract_streams",
            status="ok",
            message=self.extractor.build_selection_message(selection, front_count, back_count),
            details={
                "front_count": front_count,
                "back_count": back_count,
                "video_stream_count": selection["video_stream_count"],
                "usable_video_stream_count": selection["usable_video_stream_count"],
                "selected_front_stream_index": front_stream_index,
                "selected_back_stream_index": back_stream_index,
                "ignored_stream_indices": selection["ignored_stream_indices"],
            },
        )

    def _run_select_frames_impl(self) -> PhaseResult:
        self._set_step("select_frames", "select_pairs", 1, 1)
        return self.blur_evaluator.select_pairs(
            self.config.extracted_front_dir,
            self.config.extracted_back_dir,
            self.config.selected_front_dir,
            self.config.selected_back_dir,
        )

    def _run_generate_masks_impl(self) -> PhaseResult:
        self._set_step("generate_masks", "process_directory_front_back", 1, 1)
        return self.mask_generator.run()

    def _run_import_to_metashape_impl(self) -> PhaseResult:
        MetashapeImporter._require_metashape()
        doc = get_or_create_metashape_document(self.config, create=True)

        self._set_step("import_to_metashape", "create_or_get_chunk", 1, 8)
        chunk = self.importer.create_or_get_chunk(doc, name=self.config.chunk_name)
        pairs = collect_frame_pairs(self.config.selected_front_dir, self.config.selected_back_dir)

        self._set_step("import_to_metashape", "build_filename_sequence", 2, 8)
        filenames = self.importer.build_filename_sequence(self.config.selected_front_dir, self.config.selected_back_dir)

        self._set_step("import_to_metashape", "build_filegroups", 3, 8)
        filegroups = self.importer.build_filegroups(len(pairs))
        self.importer.validate_import_plan(filenames, filegroups)

        expected_labels = self.importer.expected_camera_labels(
            self.config.selected_front_dir, self.config.selected_back_dir
        )
        import_state = self.importer.detect_existing_import_state(chunk, expected_labels)
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
            self._set_step("import_to_metashape", "import_multiplane_images", 4, 8)
            self.importer.import_multiplane_images(chunk, filenames, filegroups)
            imported_now = True

        self._set_step("import_to_metashape", "set_sensor_types", 5, 8)
        self.importer.set_sensor_types(chunk)

        self._set_step("import_to_metashape", "apply_rig_reference", 6, 8)
        self.importer.apply_rig_reference(chunk, self.config)

        self._set_step("import_to_metashape", "apply_masks_from_disk", 7, 8)
        applied_masks, missing_labels = self.importer.apply_masks_from_disk(
            chunk, self.config.mask_front_dir, self.config.mask_back_dir
        )
        if missing_labels:
            raise PipelineError(
                "Mask PNGs are missing for {0} camera(s). Run Generate Masks first.".format(len(missing_labels))
            )

        self._set_step("import_to_metashape", "save_document", 8, 8)
        self.importer.save_document(doc)
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
                "sample_pairs": self.importer.build_small_sample(
                    self.config.selected_front_dir,
                    self.config.selected_back_dir,
                    sample_pairs=min(4, len(pairs)),
                ),
            },
        )

    def _run_align_impl(self) -> PhaseResult:
        MetashapeAligner._require_metashape()
        document = get_or_create_metashape_document(self.config, create=False)
        chunk = getattr(document, "chunk", None) if document is not None else None
        if chunk is None:
            return PhaseResult("align", "skipped", "No active chunk is available.", {})
        if not getattr(chunk, "cameras", []):
            return PhaseResult("align", "skipped", "Active chunk has no cameras to align.", {})

        self._set_step("align", "analyze_image_quality", 1, 6)
        self.aligner.analyze_image_quality(chunk)

        self._set_step("align", "disable_low_quality_cameras", 2, 6)
        disabled_count, quality_decisions = self.aligner.disable_low_quality_cameras(
            chunk, threshold=self.config.metashape_image_quality_threshold
        )

        self._set_step("align", "export_quality_log_before_align", 3, 6)
        self.aligner.export_quality_log(chunk, self.config.metashape_quality_log_path, quality_decisions)

        enabled_camera_count = sum(1 for camera in getattr(chunk, "cameras", []) if getattr(camera, "enabled", True))
        if enabled_camera_count < 2:
            raise PipelineError("Fewer than two enabled cameras remain after Image/Quality filtering.")

        self._set_step("align", "match_photos", 4, 6)
        self.aligner.match_photos(chunk, self.config, reset_matches=True)

        self._set_step("align", "align_cameras", 5, 6)
        self.aligner.align_cameras(chunk)

        self._set_step("align", "export_quality_log_and_save", 6, 6)
        self.aligner.export_quality_log(chunk, self.config.metashape_quality_log_path, quality_decisions)
        aligned_camera_count = self.aligner.aligned_camera_count(chunk)
        sensor_offset_summary = self.aligner.sensor_offset_summary(chunk)
        if sensor_offset_summary["missing_slave_rotation_count"] > 0:
            LOGGER.warning(
                "Slave sensor rotation was not estimated for %d sensors: %s",
                sensor_offset_summary["missing_slave_rotation_count"],
                sensor_offset_summary["missing_slave_rotation"],
            )
        if aligned_camera_count < 2:
            raise PipelineError(
                "Alignment produced too few aligned cameras: {0}. Check masks, quality thresholds, and overlap.".format(
                    aligned_camera_count
                )
            )
        save_metashape_document(self.config)
        return PhaseResult(
            phase="align",
            status="ok",
            message="Ran image quality analysis, disabled low-quality cameras, and aligned the active chunk.",
            details={
                "camera_count": len(getattr(chunk, "cameras", [])),
                "enabled_camera_count": enabled_camera_count,
                "disabled_low_quality_count": disabled_count,
                "aligned_camera_count": aligned_camera_count,
                "quality_csv": self.config.metashape_quality_log_path,
                "sensor_offset_summary": sensor_offset_summary,
            },
        )

    def _run_reduce_overlap_impl(self) -> PhaseResult:
        OverlapReducer._require_metashape()
        document = get_or_create_metashape_document(self.config, create=False)
        chunk = getattr(document, "chunk", None) if document is not None else None
        if chunk is None:
            return PhaseResult("reduce_overlap", "skipped", "No active chunk is available.", {})

        self._set_step("reduce_overlap", "disable_redundant_cameras", 1, 4)
        rows = self.overlap_reducer.disable_redundant_cameras(chunk, self.config)
        disabled_station_count = len({row["disabled_frame_id"] for row in rows}) if rows else 0
        details: Dict[str, Any] = {
            "disabled_camera_count": len(rows),
            "disabled_station_count": disabled_station_count,
            "overlap_csv": self.config.overlap_reduction_log_path,
        }

        realigned = False
        if rows and self.config.realign_after_overlap_reduction and len(self.overlap_reducer.get_enabled_cameras(chunk)) >= 2:
            self._set_step("reduce_overlap", "realign_after_cleanup", 2, 4)
            self.aligner.realign_after_cleanup(chunk, self.config)
            realigned = True
            details["aligned_camera_count_after_realign"] = self.aligner.aligned_camera_count(chunk)

        if self.config.use_builtin_reduce_overlap:
            self._set_step("reduce_overlap", "run_reduce_overlap_builtin", 3, 4)
            details.update(self.overlap_reducer.run_reduce_overlap_builtin(chunk, overlap=self.config.overlap_target))

        self._set_step("reduce_overlap", "export_overlap_log_and_save", 4, 4)
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

    def _run_export_logs_impl(self) -> PhaseResult:
        self._set_step("export_logs", "write_summary", 1, 1)
        self.blur_evaluator.backend_manager.save_backend_report()
        self.mask_generator.save_backend_report()
        summary = self.build_log_summary()
        self.logs.write_summary(self.config.summary_log_path, summary)
        return PhaseResult(
            phase="export_logs",
            status="ok",
            message="Exported the current pipeline log summary.",
            details={"summary_path": self.config.summary_log_path},
        )

    def _set_step(self, phase_name: str, step_name: str, index: int, total: int) -> None:
        self._current_step = step_name
        LOGGER.info("%s/%s (%s/%s)", phase_name, step_name, index, total)
        if self.progress_callback is not None:
            self.progress_callback(phase_name, step_name, index, total)

    def _run_phase(self, callback: Callable[[], PhaseResult], phase_name: str, require_input: bool = False) -> PhaseResult:
        """Wrap each phase with common validation, progress tracking, and error logging."""

        self._current_step = ""
        self.initialize_logging()
        self.config.ensure_directories()
        try:
            self.config.validate(require_input=require_input)
            result = callback()
            LOGGER.info("%s: %s", result.phase, result.message)
            return result
        except Exception as exc:  # pragma: no cover - exercised mainly inside Metashape.
            LOGGER.exception("%s failed at step '%s': %s", phase_name, self._current_step or phase_name, exc)
            return PhaseResult(
                phase=phase_name,
                status="error",
                message=str(exc),
                details={
                    "exception_type": exc.__class__.__name__,
                    "failed_step": self._current_step or phase_name,
                },
            )
        finally:
            if self.config.save_backend_report:
                try:
                    self.gpu_status.collect_all(self.blur_evaluator.backend_manager, self.mask_generator, save=True)
                except Exception as gpu_exc:
                    LOGGER.warning("Failed to write GPU summary reports: %s", gpu_exc)


class DualFisheyePipeline:
    """Backward-compatible wrapper around the shared pipeline controller."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        progress_callback: Optional[Callable[[str, str, int, int], None]] = None,
        initialize_logging: bool = True,
    ) -> None:
        self.config = config or PipelineConfig()
        self.controller = PipelineController(
            self.config,
            progress_callback=progress_callback,
            initialize_logging=initialize_logging,
        )
        self.logs = self.controller.logs
        self.extractor = self.controller.extractor
        self.blur_evaluator = self.controller.blur_evaluator
        self.mask_generator = self.controller.mask_generator
        self.gpu_status = self.controller.gpu_status
        self.importer = self.controller.importer
        self.aligner = self.controller.aligner
        self.overlap_reducer = self.controller.overlap_reducer

    def run_full_pipeline(self) -> PhaseResult:
        return self.controller.run_full_pipeline()

    def run_extract_streams(self) -> PhaseResult:
        return self.controller.run_extract_streams()

    def run_select_frames(self) -> PhaseResult:
        return self.controller.run_select_frames()

    def run_generate_masks(self) -> PhaseResult:
        return self.controller.run_generate_masks()

    def run_import_to_metashape(self) -> PhaseResult:
        return self.controller.run_import_to_metashape()

    def run_align(self) -> PhaseResult:
        return self.controller.run_align()

    def run_reduce_overlap(self) -> PhaseResult:
        return self.controller.run_reduce_overlap()

    def run_export_logs(self) -> PhaseResult:
        return self.controller.run_export_logs()

    def build_log_summary(
        self,
        extra: Optional[Mapping[str, Any]] = None,
        probe_backends: bool = True,
    ) -> Dict[str, Any]:
        return self.controller.build_log_summary(extra, probe_backends=probe_backends)

    def cleanup(self) -> None:
        self.controller.cleanup()


if QtWidgets is not None:

    class DualFisheyeMainDialog(QtWidgets.QDialog):
        """Qt dialog that drives the existing pipeline components from Metashape."""

        def __init__(self, config: Optional[PipelineConfig] = None, parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self.persistence = ConfigPersistence()
            self.config = config or PipelineConfig()
            self.pipeline = self._build_pipeline(self.config)
            self._action_buttons: List[Any] = []
            self._widgets: Dict[str, Any] = {}
            self._summary_labels: Dict[str, Any] = {}
            self._cleaned_up = False
            self._gui_log_handler = GuiLogHandler(self._append_log_entry)
            LOGGER.addHandler(self._gui_log_handler)
            self._build_ui()
            self._populate_widgets_from_config(self.config)
            loaded_previous = self._load_last_used_config_if_available()
            if not loaded_previous:
                self._refresh_input_osv_field_state()
            self.refresh_summary(probe_backends=False)

        def closeEvent(self, event: Any) -> None:
            self.cleanup()
            _release_gui_dialog(self)
            super().closeEvent(event)

        def cleanup(self) -> None:
            """Release GUI-owned references so Metashape shutdown sees less live state."""

            if self._cleaned_up:
                return
            self._cleaned_up = True
            self._save_last_used_config(silent=True)
            if self._gui_log_handler is not None:
                try:
                    LOGGER.removeHandler(self._gui_log_handler)
                except ValueError:
                    pass
                try:
                    self._gui_log_handler.close()
                except Exception:
                    pass
                self._gui_log_handler = None
            if self.pipeline is not None:
                self.pipeline.cleanup()

        def _build_ui(self) -> None:
            self.setWindowTitle("デュアル魚眼パイプライン")
            self.setMinimumSize(560, 480)

            root_layout = QtWidgets.QVBoxLayout(self)
            root_layout.setContentsMargins(10, 10, 10, 10)
            root_layout.setSpacing(8)

            config_button_row = QtWidgets.QHBoxLayout()
            save_button = QtWidgets.QPushButton("設定保存")
            save_button.clicked.connect(self.save_config)
            load_button = QtWidgets.QPushButton("設定読込")
            load_button.clicked.connect(self.load_config)
            reset_button = QtWidgets.QPushButton("初期値に戻す")
            reset_button.clicked.connect(self.reset_to_default)
            config_button_row.addWidget(save_button)
            config_button_row.addWidget(load_button)
            config_button_row.addWidget(reset_button)
            config_button_row.addStretch(1)
            root_layout.addLayout(config_button_row)

            self._action_buttons.extend([save_button, load_button, reset_button])

            self.status_label = QtWidgets.QLabel("準備完了")
            self.status_label.setWordWrap(True)
            self.status_label.setStyleSheet("padding: 6px; border: 1px solid #bcbcbc;")
            root_layout.addWidget(self.status_label)

            self.progress_label = QtWidgets.QLabel("待機中")
            self.progress_label.setWordWrap(True)
            root_layout.addWidget(self.progress_label)

            self.progress_bar = QtWidgets.QProgressBar()
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            root_layout.addWidget(self.progress_bar)

            self.tabs = QtWidgets.QTabWidget()
            self.tabs.addTab(self._build_basic_tab(), "基本設定")
            self.tabs.addTab(self._build_preprocess_tab(), "前処理")
            self.tabs.addTab(self._build_mask_import_tab(), "マスク / 読込")
            self.tabs.addTab(self._build_align_tab(), "アライメント / 間引き")
            self.tabs.addTab(self._build_logs_tab(), "ログ / 状態")
            root_layout.addWidget(self.tabs, 1)
            self._configure_dialog_geometry()

        def _build_basic_tab(self) -> Any:
            content = self._create_tab_container()
            layout = QtWidgets.QVBoxLayout(content)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(8)

            path_group, path_layout = self._create_group_box("入力 / 作業フォルダ")
            path_form = self._create_form_layout()
            path_form.addRow(
                "入力OSV",
                self._build_path_field(
                    "input_mp4",
                    directory=False,
                    dialog_title="OSVを選択",
                    file_filter="OSV Files (*.osv);;All Files (*)",
                    placeholder=_INPUT_OSV_PLACEHOLDER,
                ),
            )
            path_form.addRow("作業フォルダ", self._build_path_field("work_root", directory=True))
            path_layout.addLayout(path_form)
            layout.addWidget(path_group)

            stream_group, stream_layout = self._create_group_box("ストリーム設定")
            stream_form = self._create_form_layout()
            stream_form.addRow("前方ストリーム番号", self._register_widget("front_stream_index", self._spin_box(0, 99)))
            stream_form.addRow("後方ストリーム番号", self._register_widget("back_stream_index", self._spin_box(0, 99)))
            stream_layout.addLayout(stream_form)
            layout.addWidget(stream_group)

            run_button = QtWidgets.QPushButton("フル実行")
            run_button.clicked.connect(lambda: self._run_action("run_full_pipeline", self.pipeline.run_full_pipeline))
            self._action_buttons.append(run_button)
            layout.addWidget(self._create_action_group("実行", (run_button,), columns=1))
            layout.addStretch(1)
            return self._wrap_scrollable_tab(content)

        def _build_preprocess_tab(self) -> Any:
            content = self._create_tab_container()
            layout = QtWidgets.QVBoxLayout(content)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(8)

            extraction_group, extraction_layout = self._create_group_box("抽出 / ブレ判定")
            extraction_form = self._create_form_layout()
            extraction_form.addRow(
                "何フレームごとに抽出",
                self._register_widget("extract_every_n_frames", self._spin_box(1, 10000)),
            )
            extraction_form.addRow(
                "前方ブレ閾値",
                self._register_widget("blur_threshold_front", self._double_spin_box(0.0, 100000.0, 4)),
            )
            extraction_form.addRow(
                "後方ブレ閾値",
                self._register_widget("blur_threshold_back", self._double_spin_box(0.0, 100000.0, 4)),
            )
            extraction_form.addRow("FFTブレ閾値", self._register_widget("fft_blur_threshold", QtWidgets.QLineEdit()))
            extraction_form.addRow("JPEG品質", self._register_widget("jpeg_quality", self._spin_box(1, 31)))
            extraction_layout.addLayout(extraction_form)
            layout.addWidget(extraction_group)

            opencv_group, opencv_layout = self._create_group_box("OpenCV / CUDA")
            opencv_form = self._create_form_layout()
            opencv_form.addRow(
                "OpenCV実行方式",
                self._register_widget("opencv_backend", self._combo_box(("auto", "cpu", "cuda"))),
            )
            opencv_form.addRow(
                "CUDAを優先",
                self._register_widget("prefer_cuda", QtWidgets.QCheckBox("有効にする")),
            )
            opencv_form.addRow(
                "CUDAデバイス番号",
                self._register_widget("cuda_device_index", self._spin_box(0, 31)),
            )
            opencv_form.addRow(
                "CPUフォールバックを許可",
                self._register_widget("cuda_allow_fallback", QtWidgets.QCheckBox("許可する")),
            )
            opencv_form.addRow(
                "CUDAデバイス情報を記録",
                self._register_widget("cuda_log_device_info", QtWidgets.QCheckBox("記録する")),
            )
            opencv_form.addRow(
                "CUDAガウシアン事前ぼかし",
                self._register_widget(
                    "cuda_use_gaussian_preblur",
                    QtWidgets.QCheckBox("有効にする"),
                ),
            )
            opencv_form.addRow(
                "CUDAベンチマークモード",
                self._register_widget("cuda_benchmark_mode", QtWidgets.QCheckBox("記録する")),
            )
            opencv_form.addRow(
                "backend report を保存",
                self._register_widget("save_backend_report", QtWidgets.QCheckBox("保存する")),
            )
            opencv_layout.addLayout(opencv_form)
            layout.addWidget(opencv_group)

            extract_button = QtWidgets.QPushButton("ストリーム抽出")
            extract_button.clicked.connect(lambda: self._run_action("extract_streams", self.pipeline.run_extract_streams))
            select_button = QtWidgets.QPushButton("フレーム選別")
            select_button.clicked.connect(lambda: self._run_action("select_frames", self.pipeline.run_select_frames))
            layout.addWidget(self._create_action_group("前処理実行", (extract_button, select_button), columns=2))
            self._action_buttons.extend([extract_button, select_button])
            layout.addStretch(1)
            return self._wrap_scrollable_tab(content)

        def _build_mask_import_tab(self) -> Any:
            content = self._create_tab_container()
            layout = QtWidgets.QVBoxLayout(content)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(8)

            mask_group, mask_layout = self._create_group_box("モデル / マスク")
            mask_form = self._create_form_layout()
            mask_form.addRow("YOLOモデルパス", self._build_path_field("mask_model_path", directory=False))
            mask_form.addRow("マスク対象クラス", self._register_widget("mask_classes", QtWidgets.QLineEdit()))
            mask_form.addRow("マスク膨張ピクセル", self._register_widget("mask_dilate_px", self._spin_box(0, 1024)))
            mask_form.addRow(
                "マスク極性",
                self._register_widget("mask_polarity", self._combo_box(("target_black",))),
            )
            mask_layout.addLayout(mask_form)
            layout.addWidget(mask_group)

            yolo_group, yolo_layout = self._create_group_box("YOLO / デバイス")
            yolo_form = self._create_form_layout()
            yolo_form.addRow(
                "YOLO実行方式",
                self._register_widget("yolo_device_mode", self._combo_box(("auto", "cpu", "cuda"))),
            )
            yolo_form.addRow(
                "YOLOでCUDAを優先",
                self._register_widget("prefer_yolo_cuda", QtWidgets.QCheckBox("有効にする")),
            )
            yolo_form.addRow(
                "YOLO CPUフォールバックを許可",
                self._register_widget("yolo_allow_fallback", QtWidgets.QCheckBox("許可する")),
            )
            yolo_form.addRow(
                "YOLOデバイス番号",
                self._register_widget("yolo_device_index", self._spin_box(0, 31)),
            )
            yolo_layout.addLayout(yolo_form)
            layout.addWidget(yolo_group)

            mask_button = QtWidgets.QPushButton("マスク生成")
            mask_button.clicked.connect(lambda: self._run_action("generate_masks", self.pipeline.run_generate_masks))
            import_button = QtWidgets.QPushButton("Metashapeへ読込")
            import_button.clicked.connect(
                lambda: self._run_action("import_to_metashape", self.pipeline.run_import_to_metashape)
            )
            layout.addWidget(self._create_action_group("マスク / 読込 実行", (mask_button, import_button), columns=2))
            self._action_buttons.extend([mask_button, import_button])
            layout.addStretch(1)
            return self._wrap_scrollable_tab(content)

        def _build_align_tab(self) -> Any:
            content = self._create_tab_container()
            layout = QtWidgets.QVBoxLayout(content)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(8)

            align_group, align_layout = self._create_group_box("アライメント / 間引き")
            align_form = self._create_form_layout()
            align_form.addRow(
                "画質閾値",
                self._register_widget(
                    "metashape_image_quality_threshold",
                    self._double_spin_box(0.0, 1.0, 3, single_step=0.05),
                ),
            )
            align_form.addRow("キーポイント上限", self._register_widget("keypoint_limit", self._spin_box(1, 10000000)))
            align_form.addRow("タイポイント上限", self._register_widget("tiepoint_limit", self._spin_box(1, 10000000)))
            align_form.addRow(
                "カメラ距離閾値",
                self._register_widget("camera_distance_threshold", self._double_spin_box(0.0, 1000.0, 4)),
            )
            align_form.addRow(
                "カメラ角度閾値（度）",
                self._register_widget("camera_angle_threshold_deg", self._double_spin_box(0.0, 180.0, 4)),
            )
            align_form.addRow("重複目標枚数", self._register_widget("overlap_target", self._spin_box(1, 99)))
            align_layout.addLayout(align_form)
            layout.addWidget(align_group)

            rig_group, rig_layout = self._create_group_box("rig参照")
            rig_form = self._create_form_layout()
            rig_form.addRow(
                "rig参照を有効化",
                self._register_widget("enable_rig_reference", QtWidgets.QCheckBox("有効にする")),
            )
            rig_location = self._register_widget("rig_relative_location", QtWidgets.QLineEdit())
            rig_location.setPlaceholderText("0.0, 0.0, 0.0")
            rig_form.addRow(
                "rig相対位置",
                rig_location,
            )
            rig_rotation = self._register_widget("rig_relative_rotation", QtWidgets.QLineEdit())
            rig_rotation.setPlaceholderText("0.0, 0.0, 0.0")
            rig_form.addRow(
                "rig相対回転",
                rig_rotation,
            )
            rig_layout.addLayout(rig_form)
            layout.addWidget(rig_group)

            align_button = QtWidgets.QPushButton("アライメント")
            align_button.clicked.connect(lambda: self._run_action("align", self.pipeline.run_align))
            reduce_button = QtWidgets.QPushButton("冗長画像削減")
            reduce_button.clicked.connect(lambda: self._run_action("reduce_overlap", self.pipeline.run_reduce_overlap))
            export_button = QtWidgets.QPushButton("ログ出力")
            export_button.clicked.connect(lambda: self._run_action("export_logs", self.pipeline.run_export_logs))
            layout.addWidget(
                self._create_action_group(
                    "アライメント関連実行",
                    (align_button, reduce_button, export_button),
                    columns=2,
                )
            )
            self._action_buttons.extend([align_button, reduce_button, export_button])
            layout.addStretch(1)
            return self._wrap_scrollable_tab(content)

        def _build_logs_tab(self) -> Any:
            content = self._create_tab_container()
            layout = QtWidgets.QVBoxLayout(content)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(8)

            summary_group = QtWidgets.QGroupBox("集計 / GPU状態")
            summary_layout = QtWidgets.QVBoxLayout(summary_group)
            summary_form = self._create_form_layout()
            for key, label in (
                ("selected_pairs", "採用ペア数"),
                ("discarded_pairs", "除外ペア数"),
                ("mask_rows", "マスク行数"),
                ("masked_images", "マスクあり画像数"),
                ("quality_rows", "画質ログ行数"),
                ("overlap_rows", "間引きログ行数"),
                ("enabled_cameras", "有効カメラ数"),
                ("aligned_cameras", "アライメント済みカメラ数"),
                ("opencv_status", "OpenCV状態"),
                ("yolo_status", "YOLO状態"),
                ("metashape_gpu_status", "Metashape GPU状態"),
                ("gpu_fallback", "GPUフォールバック"),
                ("device_indices", "使用デバイス番号"),
                ("backend_report_paths", "backend report 保存先"),
            ):
                value_label = QtWidgets.QLabel("-")
                value_label.setWordWrap(True)
                self._summary_labels[key] = value_label
                summary_form.addRow(label, value_label)
            summary_layout.addLayout(summary_form)
            layout.addWidget(summary_group)

            refresh_button = QtWidgets.QPushButton("状態更新")
            refresh_button.clicked.connect(lambda: self.refresh_summary(probe_backends=True))
            self._action_buttons.append(refresh_button)
            layout.addWidget(refresh_button)

            detail_group, detail_layout = self._create_group_box("詳細")
            self.summary_text = QtWidgets.QPlainTextEdit()
            self.summary_text.setReadOnly(True)
            self.summary_text.setMinimumHeight(150)
            detail_layout.addWidget(self.summary_text, 1)

            self.log_view = QtWidgets.QTextEdit()
            self.log_view.setReadOnly(True)
            self.log_view.setMinimumHeight(220)
            detail_layout.addWidget(self.log_view, 2)
            layout.addWidget(detail_group, 1)
            return self._wrap_scrollable_tab(content)

        def _build_path_field(
            self,
            name: str,
            directory: bool,
            dialog_title: Optional[str] = None,
            file_filter: str = "All Files (*)",
            placeholder: str = "",
        ) -> Any:
            container = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            line_edit = QtWidgets.QLineEdit()
            line_edit.setMinimumWidth(0)
            if placeholder:
                line_edit.setPlaceholderText(placeholder)
            line_edit.textChanged.connect(
                lambda value, widget=line_edit: self._update_path_widget_tooltip(widget, value)
            )
            line_edit.editingFinished.connect(
                lambda name=name, widget=line_edit: self._sync_path_field_value(name, widget.text())
            )
            browse_button = QtWidgets.QPushButton("参照")
            browse_button.clicked.connect(lambda: self._browse_path(name, directory, dialog_title, file_filter))
            row.addWidget(line_edit, 1)
            row.addWidget(browse_button)
            self._update_path_widget_tooltip(line_edit, "")
            self._widgets[name] = line_edit
            self._action_buttons.append(browse_button)
            return container

        @staticmethod
        def _qt_enum(owner: Any, *paths: str) -> Any:
            for path in paths:
                value = owner
                missing = False
                for part in path.split("."):
                    value = getattr(value, part, None)
                    if value is None:
                        missing = True
                        break
                if not missing:
                    return value
            raise AttributeError("Qt enum path not available: {0}".format(", ".join(paths)))

        def _create_tab_container(self) -> Any:
            return QtWidgets.QWidget()

        def _create_group_box(self, title: str) -> Tuple[Any, Any]:
            group = QtWidgets.QGroupBox(title)
            layout = QtWidgets.QVBoxLayout(group)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(8)
            return group, layout

        def _create_form_layout(self) -> Any:
            form = QtWidgets.QFormLayout()
            try:
                form.setFieldGrowthPolicy(
                    self._qt_enum(
                        QtWidgets.QFormLayout,
                        "AllNonFixedFieldsGrow",
                        "FieldGrowthPolicy.AllNonFixedFieldsGrow",
                    )
                )
            except AttributeError:
                pass
            try:
                form.setRowWrapPolicy(
                    self._qt_enum(
                        QtWidgets.QFormLayout,
                        "WrapLongRows",
                        "RowWrapPolicy.WrapLongRows",
                    )
                )
            except AttributeError:
                pass
            return form

        def _create_action_group(self, title: str, buttons: Sequence[Any], columns: int = 2) -> Any:
            group, layout = self._create_group_box(title)
            grid = QtWidgets.QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(8)
            for index, button in enumerate(buttons):
                grid.addWidget(button, index // columns, index % columns)
            layout.addLayout(grid)
            return group

        def _wrap_scrollable_tab(self, content: Any) -> Any:
            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(content)
            return scroll_area

        def _configure_dialog_geometry(self) -> None:
            geometry = self._available_screen_geometry()
            if geometry is None:
                self.resize(820, 720)
                return

            max_width = min(geometry.width(), max(360, int(geometry.width() * 0.88)))
            max_height = min(geometry.height(), max(320, int(geometry.height() * 0.88)))
            min_width = min(max_width, 560)
            min_height = min(max_height, 480)
            self.setMinimumSize(min_width, min_height)
            self.setMaximumSize(max_width, max_height)
            self.adjustSize()
            width = min(max(self.sizeHint().width(), min_width), max_width)
            height = min(max(self.sizeHint().height(), min_height), max_height)
            self.resize(width, height)
            self.move(
                geometry.x() + max(0, int((geometry.width() - width) / 2)),
                geometry.y() + max(0, int((geometry.height() - height) / 2)),
            )

        @staticmethod
        def _available_screen_geometry() -> Optional[Any]:
            app = QtWidgets.QApplication.instance()
            if app is None:
                return None
            screen = None
            if hasattr(app, "primaryScreen"):
                screen = app.primaryScreen()
            if screen is not None and hasattr(screen, "availableGeometry"):
                return screen.availableGeometry()
            desktop = app.desktop() if hasattr(app, "desktop") else None
            if desktop is not None and hasattr(desktop, "availableGeometry"):
                return desktop.availableGeometry()
            return None

        @staticmethod
        def _update_path_widget_tooltip(widget: Any, value: str, detail: Optional[str] = None) -> None:
            messages: List[str] = []
            text = value.strip()
            if text:
                messages.append(text)
            if detail:
                messages.append(detail)
            widget.setToolTip("\n".join(messages))

        def _register_widget(self, name: str, widget: Any) -> Any:
            self._widgets[name] = widget
            return widget

        def _build_pipeline(self, config: PipelineConfig) -> DualFisheyePipeline:
            """Create a GUI-side pipeline without forcing runtime validation on dialog open."""

            return DualFisheyePipeline(
                config,
                progress_callback=self._on_progress,
                initialize_logging=False,
            )

        @staticmethod
        def _combo_box(items: Sequence[str]) -> Any:
            widget = QtWidgets.QComboBox()
            widget.addItems(list(items))
            return widget

        @staticmethod
        def _spin_box(minimum: int, maximum: int) -> Any:
            widget = QtWidgets.QSpinBox()
            widget.setRange(minimum, maximum)
            return widget

        @staticmethod
        def _double_spin_box(
            minimum: float,
            maximum: float,
            decimals: int,
            single_step: float = 0.1,
        ) -> Any:
            widget = QtWidgets.QDoubleSpinBox()
            widget.setRange(minimum, maximum)
            widget.setDecimals(decimals)
            widget.setSingleStep(single_step)
            return widget

        def _browse_path(
            self,
            name: str,
            directory: bool,
            dialog_title: Optional[str] = None,
            file_filter: str = "All Files (*)",
        ) -> None:
            current_path = self._widgets[name].text().strip()
            resolved_current = normalize_input_video_path(current_path) if current_path else None
            if resolved_current is not None and not directory and resolved_current.suffix:
                start_dir = str(resolved_current.parent)
            elif resolved_current is not None:
                start_dir = str(resolved_current)
            else:
                start_dir = str(self.config.project_root)
            if directory:
                selected_path = QtWidgets.QFileDialog.getExistingDirectory(
                    self,
                    dialog_title or "フォルダを選択",
                    start_dir,
                )
            else:
                selected_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self,
                    dialog_title or "ファイルを選択",
                    start_dir,
                    file_filter,
                )
            if selected_path:
                self._widgets[name].setText(selected_path)
                self._sync_path_field_value(name, selected_path, announce=True)

        def _populate_widgets_from_config(self, config: PipelineConfig) -> None:
            self._widgets["input_mp4"].setText(_path_text(config.input_mp4))
            self._widgets["work_root"].setText(str(config.work_root))
            self._widgets["mask_model_path"].setText(str(config.mask_model_path))
            self._widgets["front_stream_index"].setValue(config.front_stream_index)
            self._widgets["back_stream_index"].setValue(config.back_stream_index)
            self._widgets["extract_every_n_frames"].setValue(config.extract_every_n_frames)
            self._widgets["blur_threshold_front"].setValue(config.blur_threshold_front)
            self._widgets["blur_threshold_back"].setValue(config.blur_threshold_back)
            self._widgets["fft_blur_threshold"].setText(
                "" if config.fft_blur_threshold is None else str(config.fft_blur_threshold)
            )
            self._widgets["jpeg_quality"].setValue(config.jpeg_quality)
            self._widgets["opencv_backend"].setCurrentText(config.opencv_backend)
            self._widgets["prefer_cuda"].setChecked(config.prefer_cuda)
            self._widgets["cuda_device_index"].setValue(config.cuda_device_index)
            self._widgets["cuda_allow_fallback"].setChecked(config.cuda_allow_fallback)
            self._widgets["cuda_log_device_info"].setChecked(config.cuda_log_device_info)
            self._widgets["cuda_use_gaussian_preblur"].setChecked(config.cuda_use_gaussian_preblur)
            self._widgets["cuda_benchmark_mode"].setChecked(config.cuda_benchmark_mode)
            self._widgets["yolo_device_mode"].setCurrentText(config.yolo_device_mode)
            self._widgets["prefer_yolo_cuda"].setChecked(config.prefer_yolo_cuda)
            self._widgets["yolo_allow_fallback"].setChecked(config.yolo_allow_fallback)
            self._widgets["yolo_device_index"].setValue(config.yolo_device_index)
            self._widgets["save_backend_report"].setChecked(config.save_backend_report)
            self._widgets["mask_classes"].setText(", ".join(config.mask_classes))
            self._widgets["mask_dilate_px"].setValue(config.mask_dilate_px)
            self._widgets["mask_polarity"].setCurrentText(config.mask_polarity)
            self._widgets["metashape_image_quality_threshold"].setValue(config.metashape_image_quality_threshold)
            self._widgets["keypoint_limit"].setValue(config.keypoint_limit)
            self._widgets["tiepoint_limit"].setValue(config.tiepoint_limit)
            self._widgets["camera_distance_threshold"].setValue(config.camera_distance_threshold)
            self._widgets["camera_angle_threshold_deg"].setValue(config.camera_angle_threshold_deg)
            self._widgets["overlap_target"].setValue(config.overlap_target)
            self._widgets["enable_rig_reference"].setChecked(config.enable_rig_reference)
            self._widgets["rig_relative_location"].setText(
                ", ".join(str(value) for value in config.rig_relative_location)
            )
            self._widgets["rig_relative_rotation"].setText(
                ", ".join(str(value) for value in config.rig_relative_rotation)
            )
            self._refresh_input_osv_field_state()
            self._refresh_mask_model_field_state()

        def _sync_path_field_value(self, name: str, value: str, announce: bool = False) -> None:
            updated = PipelineConfig.from_mapping(self.config.to_dict())
            updated.update_from_mapping({name: value})
            self.config = updated
            self.pipeline = self._build_pipeline(self.config)
            if name == "input_mp4":
                self._refresh_input_osv_field_state(
                    status_prefix="入力OSVを更新しました。" if announce else None
                )
            elif name == "mask_model_path":
                self._refresh_mask_model_field_state(
                    status_prefix="YOLOモデルパスを更新しました。" if announce else None
                )
            elif name == "work_root" and announce:
                self._set_status("info", "作業フォルダを変更しました。保存または実行すると設定パスが反映されます。")
            if name in ("input_mp4", "work_root"):
                self.refresh_summary(probe_backends=False)

        def _sync_config_from_widgets(self) -> None:
            config = PipelineConfig.from_mapping(self.config.to_dict())
            config.update_from_mapping(
                {
                    "input_mp4": self._widgets["input_mp4"].text().strip(),
                    "work_root": self._widgets["work_root"].text().strip(),
                    "front_stream_index": self._widgets["front_stream_index"].value(),
                    "back_stream_index": self._widgets["back_stream_index"].value(),
                    "extract_every_n_frames": self._widgets["extract_every_n_frames"].value(),
                    "blur_threshold_front": self._widgets["blur_threshold_front"].value(),
                    "blur_threshold_back": self._widgets["blur_threshold_back"].value(),
                    "fft_blur_threshold": self._parse_optional_float(self._widgets["fft_blur_threshold"].text()),
                    "jpeg_quality": self._widgets["jpeg_quality"].value(),
                    "opencv_backend": self._widgets["opencv_backend"].currentText(),
                    "prefer_cuda": self._widgets["prefer_cuda"].isChecked(),
                    "cuda_device_index": self._widgets["cuda_device_index"].value(),
                    "cuda_allow_fallback": self._widgets["cuda_allow_fallback"].isChecked(),
                    "cuda_log_device_info": self._widgets["cuda_log_device_info"].isChecked(),
                    "cuda_use_gaussian_preblur": self._widgets["cuda_use_gaussian_preblur"].isChecked(),
                    "cuda_benchmark_mode": self._widgets["cuda_benchmark_mode"].isChecked(),
                    "yolo_device_mode": self._widgets["yolo_device_mode"].currentText(),
                    "prefer_yolo_cuda": self._widgets["prefer_yolo_cuda"].isChecked(),
                    "yolo_allow_fallback": self._widgets["yolo_allow_fallback"].isChecked(),
                    "yolo_device_index": self._widgets["yolo_device_index"].value(),
                    "save_backend_report": self._widgets["save_backend_report"].isChecked(),
                    "mask_model_path": self._widgets["mask_model_path"].text().strip(),
                    "mask_classes": self._widgets["mask_classes"].text().strip(),
                    "mask_dilate_px": self._widgets["mask_dilate_px"].value(),
                    "mask_polarity": self._widgets["mask_polarity"].currentText(),
                    "metashape_image_quality_threshold": self._widgets[
                        "metashape_image_quality_threshold"
                    ].value(),
                    "keypoint_limit": self._widgets["keypoint_limit"].value(),
                    "tiepoint_limit": self._widgets["tiepoint_limit"].value(),
                    "camera_distance_threshold": self._widgets["camera_distance_threshold"].value(),
                    "camera_angle_threshold_deg": self._widgets["camera_angle_threshold_deg"].value(),
                    "overlap_target": self._widgets["overlap_target"].value(),
                    "enable_rig_reference": self._widgets["enable_rig_reference"].isChecked(),
                    "rig_relative_location": self._parse_vector(self._widgets["rig_relative_location"].text()),
                    "rig_relative_rotation": self._parse_vector(self._widgets["rig_relative_rotation"].text()),
                }
            )
            self.config = config
            self.pipeline = self._build_pipeline(self.config)
            self._refresh_input_osv_field_state()
            self._refresh_mask_model_field_state()

        def _refresh_input_osv_field_state(self, status_prefix: Optional[str] = None) -> None:
            input_widget = self._widgets.get("input_mp4")
            if input_widget is None:
                return
            issue = self.config.input_video_validation_error(require_exists=True)
            if issue is None:
                input_widget.setStyleSheet("")
                self._update_path_widget_tooltip(input_widget, input_widget.text(), "ffprobe / ffmpeg に渡せる OSV コンテナです。")
                if status_prefix:
                    self._set_status("ok", status_prefix)
                return

            input_widget.setStyleSheet("border: 1px solid #c98900; background: #fff8e1;")
            self._update_path_widget_tooltip(input_widget, input_widget.text(), str(issue))
            if status_prefix:
                self._set_status("warning", "{0} {1}".format(status_prefix, issue))
            elif not input_widget.text().strip():
                self._set_status("warning", str(issue))

        def _refresh_mask_model_field_state(self, status_prefix: Optional[str] = None) -> None:
            model_widget = self._widgets.get("mask_model_path")
            if model_widget is None:
                return

            issue = self.config.mask_model_validation_error()
            if issue is None:
                resolved_path = self.config.find_local_mask_model_path()
                model_widget.setStyleSheet("")
                self._update_path_widget_tooltip(
                    model_widget,
                    model_widget.text(),
                    "ローカル YOLO モデルを利用できます: {0}".format(
                        resolved_path if resolved_path is not None else ""
                    ),
                )
                if status_prefix:
                    self._set_status("ok", status_prefix)
                return

            model_widget.setStyleSheet("border: 1px solid #c98900; background: #fff8e1;")
            self._update_path_widget_tooltip(model_widget, model_widget.text(), str(issue))
            if status_prefix:
                self._set_status("warning", "{0} {1}".format(status_prefix, issue))

        def _config_issue_messages(self) -> List[str]:
            messages: List[str] = []
            input_issue = self.config.input_video_validation_error(require_exists=True)
            model_issue = self.config.mask_model_validation_error()
            if input_issue is not None:
                messages.append(str(input_issue))
            if model_issue is not None:
                messages.append(str(model_issue))
            return messages

        def _save_last_used_config(self, silent: bool = False) -> None:
            try:
                self._sync_config_from_widgets()
                target_path = self.persistence.save(self.config)
                if not silent:
                    self._set_status("ok", "設定を保存しました: {0}".format(target_path))
            except Exception as exc:
                if not silent:
                    self._set_status("error", "設定の保存に失敗しました: {0}".format(exc))

        def _load_last_used_config_if_available(self) -> bool:
            candidate_paths: List[Path] = []
            for path in (
                _default_last_used_config_path(self.config.project_root),
                self.config.last_used_config_path,
            ):
                if path not in candidate_paths:
                    candidate_paths.append(path)

            for candidate_path in candidate_paths:
                if not candidate_path.exists():
                    continue
                try:
                    loaded = self.persistence.load(candidate_path)
                except Exception as exc:
                    self._append_log_entry(logging.WARNING, "Failed to load last used config: {0}".format(exc))
                    continue
                self.config = loaded
                self.pipeline = self._build_pipeline(self.config)
                self._populate_widgets_from_config(self.config)
                issues = self._config_issue_messages()
                status = "ok" if not issues else "warning"
                message = "前回の設定を読み込みました: {0}".format(candidate_path)
                if issues:
                    message = "{0}. {1}".format(message, " ".join(issues))
                self._set_status(status, message)
                return True
            return False

        def save_config(self) -> None:
            self._save_last_used_config(silent=False)

        def load_config(self) -> None:
            current_dir = str(self.config.config_dir if self.config.config_dir.exists() else self.config.project_root)
            selected_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "設定JSONを読み込む",
                current_dir,
                "JSON Files (*.json);;All Files (*)",
            )
            if not selected_path:
                return
            try:
                loaded = self.persistence.load(Path(selected_path))
                self.config = loaded
                self.pipeline = self._build_pipeline(self.config)
                self._populate_widgets_from_config(self.config)
                self.refresh_summary(probe_backends=False)
                issues = self._config_issue_messages()
                status = "ok" if not issues else "warning"
                message = "設定を読み込みました: {0}".format(selected_path)
                if issues:
                    message = "{0}. {1}".format(message, " ".join(issues))
                self._set_status(status, message)
            except Exception as exc:
                self._set_status("error", "設定の読込に失敗しました: {0}".format(exc))

        def reset_to_default(self) -> None:
            self.config = PipelineConfig()
            self.pipeline = self._build_pipeline(self.config)
            self._populate_widgets_from_config(self.config)
            self.refresh_summary(probe_backends=False)
            self._refresh_input_osv_field_state("GUIの設定値を初期値に戻しました。")

        def _run_action(self, action_name: str, runner: Callable[[], PhaseResult]) -> None:
            self._set_running(True)
            try:
                self._sync_config_from_widgets()
                self._validate_preprocess_backend_action(action_name)
                self.persistence.save(self.config)
                self._set_status("info", "{0}を実行中...".format(JapaneseUiText.action_label(action_name)))
                result = runner()
                failed_step = result.details.get("failed_step", "") if result.details else ""
                if result.status == "error":
                    message = result.message
                    if failed_step:
                        message = "{0} (failed step: {1})".format(message, failed_step)
                    self._set_status("error", message)
                elif result.status == "skipped":
                    self._set_status("warning", result.message)
                else:
                    self._set_status("ok", result.message)
                self.refresh_summary(probe_backends=True)
            except Exception as exc:
                LOGGER.exception("GUI action '%s' failed: %s", action_name, exc)
                self._set_status("error", str(exc))
            finally:
                self._set_running(False)

        def _set_running(self, is_running: bool) -> None:
            for button in self._action_buttons:
                button.setEnabled(not is_running)
            if is_running:
                self.progress_label.setText("実行中...")
            else:
                self.progress_label.setText("待機中")
                self.progress_bar.setRange(0, 1)
                self.progress_bar.setValue(0)
            self._process_events()

        def _on_progress(self, phase_name: str, step_name: str, index: int, total: int) -> None:
            self.progress_label.setText(JapaneseUiText.progress_label(phase_name, step_name))
            self.progress_bar.setRange(0, max(1, total))
            self.progress_bar.setValue(index)
            self._process_events()

        def _append_log_entry(self, levelno: int, message: str) -> None:
            color = "#222222"
            if levelno >= logging.ERROR:
                color = "#b00020"
            elif levelno >= logging.WARNING:
                color = "#9c6500"
            elif levelno >= logging.INFO:
                color = "#0a4b78"
            localized_message = JapaneseUiText.translate(message)
            self.log_view.append('<span style="color:{0};">{1}</span>'.format(color, html.escape(localized_message)))
            self._process_events()

        def _set_status(self, status: str, message: str) -> None:
            palette = {
                "ok": ("#e7f6e7", "#2b6f2b"),
                "info": ("#e8f1fb", "#0a4b78"),
                "warning": ("#fff4df", "#9c6500"),
                "error": ("#fdecea", "#b00020"),
            }
            background, foreground = palette.get(status, ("#f0f0f0", "#333333"))
            self.status_label.setText(JapaneseUiText.translate(message))
            self.status_label.setStyleSheet(
                "padding: 6px; border: 1px solid {0}; background: {1}; color: {0};".format(
                    foreground, background
                )
            )
            self._process_events()

        def refresh_summary(self, probe_backends: bool = False) -> None:
            selected_pairs = 0
            discarded_pairs = 0
            if self.config.frame_quality_log_path.exists():
                with self.config.frame_quality_log_path.open("r", encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        if str(row.get("keep_pair", "")).strip() in ("1", "True", "true"):
                            selected_pairs += 1
                        else:
                            discarded_pairs += 1

            mask_rows = 0
            masked_images = 0
            if self.config.mask_summary_log_path.exists():
                with self.config.mask_summary_log_path.open("r", encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        mask_rows += 1
                        try:
                            if int(str(row.get("masked_pixels", "0")).strip() or "0") > 0:
                                masked_images += 1
                        except ValueError:
                            continue

            quality_rows = self._count_csv_rows(self.config.metashape_quality_log_path)
            overlap_rows = self._count_csv_rows(self.config.overlap_reduction_log_path)
            enabled_cameras = 0
            aligned_cameras = 0
            document = get_or_create_metashape_document(self.config, create=False) if Metashape is not None else None
            if document is not None and getattr(document, "chunk", None) is not None:
                chunk = document.chunk
                enabled_cameras = len(
                    [camera for camera in getattr(chunk, "cameras", []) if getattr(camera, "enabled", True)]
                )
                aligned_cameras = self.pipeline.aligner.aligned_camera_count(chunk)

            backend_manager = self.pipeline.blur_evaluator.backend_manager
            gpu_reports = self.pipeline.gpu_status.collect_all(
                backend_manager,
                self.pipeline.mask_generator,
                save=False,
                probe_runtime=probe_backends,
            )
            backend_report = self._merge_saved_report(self.config.opencv_backend_report_path, gpu_reports["opencv"])
            yolo_report = self._merge_saved_report(self.config.yolo_backend_report_path, gpu_reports["yolo"])
            metashape_report = self._merge_saved_report(
                self.config.metashape_gpu_report_path,
                gpu_reports["metashape"],
            )
            gpu_summary = self.pipeline.gpu_status.build_summary(backend_report, yolo_report, metashape_report)
            if not probe_backends and not backend_manager._backend_ensured:
                gpu_summary["opencv_status"] = self._preview_opencv_backend(backend_report)

            summary_values = {
                "selected_pairs": selected_pairs,
                "discarded_pairs": discarded_pairs,
                "mask_rows": mask_rows,
                "masked_images": masked_images,
                "quality_rows": quality_rows,
                "overlap_rows": overlap_rows,
                "enabled_cameras": enabled_cameras,
                "aligned_cameras": aligned_cameras,
                "opencv_status": gpu_summary["opencv_status"],
                "yolo_status": gpu_summary["yolo_status"],
                "metashape_gpu_status": gpu_summary["metashape_gpu_status"],
                "gpu_fallback": gpu_summary["gpu_fallback"],
                "device_indices": gpu_summary["device_indices"],
                "backend_report_paths": "\n".join(
                    (
                        str(self.config.opencv_backend_report_path),
                        str(self.config.yolo_backend_report_path),
                        str(self.config.metashape_gpu_report_path),
                        str(self.config.gpu_summary_report_path),
                    )
                ),
            }
            for key, value in summary_values.items():
                text = str(value)
                self._summary_labels[key].setText(text)
                self._summary_labels[key].setToolTip(text)

            summary_payload = self.pipeline.build_log_summary(
                extra={"gui_summary": summary_values},
                probe_backends=probe_backends,
            )
            self.summary_text.setPlainText(json.dumps(summary_payload, indent=2, ensure_ascii=False))
            self._process_events()

        def _validate_preprocess_backend_action(self, action_name: str) -> None:
            manager = self.pipeline.blur_evaluator.backend_manager
            if action_name in ("run_full_pipeline", "select_frames"):
                manager.detect_cuda_support()
                if self.config.opencv_backend == "cuda" and not self.config.cuda_allow_fallback:
                    manager.select_backend(self.config)
            if action_name in ("run_full_pipeline", "generate_masks"):
                generator = self.pipeline.mask_generator
                generator.detect_backend_support()
                if self.config.yolo_device_mode == "cuda" and not self.config.yolo_allow_fallback:
                    generator.resolve_device()

        def _preview_opencv_backend(self, backend_report: Mapping[str, Any]) -> str:
            device_count = int(backend_report.get("cuda_device_count", 0) or 0)
            cuda_ready = bool(backend_report.get("cuda_api_available")) and device_count > self.config.cuda_device_index
            if self.config.opencv_backend == "cpu":
                return "OpenCV: CPU 実行"
            if self.config.opencv_backend == "cuda":
                if cuda_ready:
                    return "OpenCV: CUDA 使用中"
                if self.config.cuda_allow_fallback:
                    return "OpenCV: CPU フォールバック"
                return "OpenCV: CUDA 利用不可"
            if self.config.prefer_cuda and cuda_ready:
                return "OpenCV: CUDA 使用中"
            return "OpenCV: CPU 実行"

        def _merge_saved_report(self, path: Path, current_report: Mapping[str, Any]) -> Dict[str, Any]:
            merged_report = dict(current_report)
            if not path.exists():
                return merged_report
            try:
                with path.open("r", encoding="utf-8") as handle:
                    saved_report = json.load(handle)
                if isinstance(saved_report, Mapping):
                    merged_report.update(saved_report)
            except Exception as exc:
                self._append_log_entry(logging.WARNING, "backend report の読込に失敗しました: {0}".format(exc))
            return merged_report

        @staticmethod
        def _count_csv_rows(path: Path) -> int:
            if not path.exists():
                return 0
            with path.open("r", encoding="utf-8", newline="") as handle:
                return sum(1 for _ in csv.DictReader(handle))

        @staticmethod
        def _parse_optional_float(value: str) -> Optional[float]:
            text = value.strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError as exc:
                raise PipelineError("FFTブレ閾値は空欄または数値で入力してください。") from exc

        @staticmethod
        def _parse_vector(value: str) -> Tuple[float, float, float]:
            if not value.strip():
                return 0.0, 0.0, 0.0
            parts = [item.strip() for item in value.split(",") if item.strip()]
            if len(parts) != 3:
                raise PipelineError("rigベクトルはカンマ区切りの3要素で入力してください。")
            try:
                return float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError as exc:
                raise PipelineError("rigベクトルには数値を入力してください。") from exc

        @staticmethod
        def _process_events() -> None:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.processEvents()


else:

    class DualFisheyeMainDialog(object):
        """Fallback placeholder when Qt bindings are unavailable."""

        def __init__(self, config: Optional[PipelineConfig] = None, parent: Optional[Any] = None) -> None:
            del config
            del parent
            raise PipelineError(
                "この Python 実行環境では Qt バインディングが利用できません。"
                "既存メニューを使用し、現在の Metashape Qt バインディングを確認してください。"
            )


def _show_result(result: PhaseResult) -> None:
    """Display a compact phase summary."""

    message = "[{0}] {1}".format(result.status.upper(), JapaneseUiText.translate(result.message))
    LOGGER.info(message)
    if Metashape is not None:
        Metashape.app.messageBox(message)
    else:
        print(message)


def get_pipeline(
    config: Optional[PipelineConfig] = None,
    progress_callback: Optional[Callable[[str, str, int, int], None]] = None,
) -> DualFisheyePipeline:
    """Create a fresh pipeline instance for each menu invocation."""

    return DualFisheyePipeline(config=config, progress_callback=progress_callback)


def _release_gui_dialog(dialog: Optional[Any] = None) -> None:
    """Clear the global GUI reference once the dialog has been closed or discarded."""

    global _GUI_DIALOG
    if dialog is None or _GUI_DIALOG is dialog:
        _GUI_DIALOG = None


def cleanup_gui_dialog() -> None:
    """Close and release the GUI dialog if it exists."""

    global _GUI_DIALOG
    dialog = _GUI_DIALOG
    _GUI_DIALOG = None
    if dialog is None:
        return
    if hasattr(dialog, "cleanup"):
        try:
            dialog.cleanup()
        except Exception as exc:
            LOGGER.warning("GUI cleanup raised an exception: %s", exc)
    if hasattr(dialog, "close"):
        try:
            dialog.close()
        except Exception:
            pass
    if hasattr(dialog, "deleteLater"):
        try:
            dialog.deleteLater()
        except Exception:
            pass


def _menu_items() -> Tuple[Tuple[str, Callable[[], None]], ...]:
    """Return the registered menu callbacks without constructing a pipeline."""

    return (
        ("00 GUIを開く", menu_open_gui),
        ("01 フルパイプライン実行", menu_run_full_pipeline),
        ("02 ストリーム抽出", menu_extract_streams),
        ("03 フレーム選別", menu_select_frames),
        ("04 マスク生成", menu_generate_masks),
        ("05 Metashapeへ読込", menu_import_to_metashape),
        ("06 アライメント", menu_align),
        ("07 冗長画像削減", menu_reduce_overlap),
        ("08 ログ出力", menu_export_logs),
    )


def _full_menu_label(suffix: str) -> str:
    return "{0}/{1}".format(_MENU_ROOT, suffix)


def register_application_shutdown() -> None:
    """Connect plugin cleanup to the Qt application shutdown when available."""

    global _APP_SHUTDOWN_CONNECTED
    if _APP_SHUTDOWN_CONNECTED or QtWidgets is None:
        return
    app = QtWidgets.QApplication.instance()
    if app is None or not hasattr(app, "aboutToQuit"):
        return
    try:
        app.aboutToQuit.connect(shutdown_plugin)
    except Exception:
        return
    _APP_SHUTDOWN_CONNECTED = True


def unregister_application_shutdown() -> None:
    """Disconnect the Qt shutdown hook when possible."""

    global _APP_SHUTDOWN_CONNECTED
    if not _APP_SHUTDOWN_CONNECTED or QtWidgets is None:
        return
    app = QtWidgets.QApplication.instance()
    if app is not None and hasattr(app, "aboutToQuit"):
        try:
            app.aboutToQuit.disconnect(shutdown_plugin)
        except Exception:
            pass
    _APP_SHUTDOWN_CONNECTED = False


def menu_open_gui() -> None:
    """Menu callback for the Qt-based GUI launcher."""

    global _GUI_DIALOG
    register_application_shutdown()
    if _GUI_DIALOG is not None and hasattr(_GUI_DIALOG, "isVisible") and _GUI_DIALOG.isVisible():
        _GUI_DIALOG.show()
        if hasattr(_GUI_DIALOG, "raise_"):
            _GUI_DIALOG.raise_()
        if hasattr(_GUI_DIALOG, "activateWindow"):
            _GUI_DIALOG.activateWindow()
        return
    try:
        _GUI_DIALOG = DualFisheyeMainDialog()
    except Exception as exc:
        LOGGER.exception("open_gui failed: %s", exc)
        _show_result(PhaseResult("open_gui", "error", str(exc), {"exception_type": exc.__class__.__name__}))
        return
    _GUI_DIALOG.show()
    if hasattr(_GUI_DIALOG, "raise_"):
        _GUI_DIALOG.raise_()
    if hasattr(_GUI_DIALOG, "activateWindow"):
        _GUI_DIALOG.activateWindow()


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
    app = getattr(Metashape, "app", None)
    if app is None:
        return
    remove_menu_item = getattr(app, "removeMenuItem", None)
    for suffix, callback in _menu_items():
        label = _full_menu_label(suffix)
        if callable(remove_menu_item):
            try:
                remove_menu_item(label)
            except Exception as exc:
                LOGGER.warning("Failed to remove existing menu item '%s': %s", label, exc)
        app.addMenuItem(label, callback)
    _MENU_REGISTERED = True
    register_application_shutdown()
    LOGGER.info("Registered Dual Fisheye menu items.")


def unregister_menu_items() -> None:
    """Remove the registered menu tree when the current runtime supports it."""

    global _MENU_REGISTERED
    if Metashape is None:
        _MENU_REGISTERED = False
        return
    app = getattr(Metashape, "app", None)
    remove_menu_item = getattr(app, "removeMenuItem", None) if app is not None else None
    if callable(remove_menu_item):
        for suffix, _callback in _menu_items():
            label = _full_menu_label(suffix)
            try:
                remove_menu_item(label)
            except Exception as exc:
                LOGGER.warning("Failed to remove menu item '%s': %s", label, exc)
    _MENU_REGISTERED = False


def initialize_plugin() -> None:
    """Explicit plugin entry point for Metashape script execution."""

    register_menu_items()


def shutdown_plugin() -> None:
    """Release GUI and logger state that can otherwise survive until application exit."""

    global _HEADLESS_DOCUMENT
    global _HEADLESS_DOCUMENT_PATH
    cleanup_gui_dialog()
    _HEADLESS_DOCUMENT = None
    _HEADLESS_DOCUMENT_PATH = None
    shutdown_logging(include_gui_handlers=True)
    unregister_application_shutdown()


if __name__ == "__main__":
    initialize_plugin()
