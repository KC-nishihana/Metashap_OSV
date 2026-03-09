import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "metashape_dual_fisheye_pipeline.py"


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("metashape_dual_fisheye_pipeline", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_config(module, tmp_path, **overrides):
    config = module.PipelineConfig(
        project_root=tmp_path,
        input_mp4=tmp_path / "input" / "source.mp4",
        work_root=tmp_path / "work",
        project_path=tmp_path / "project" / "dual_fisheye_project.psx",
    )
    config.update_from_mapping(overrides)
    config.ensure_directories()
    return config


class FakeCudaNamespace:
    def __init__(
        self,
        device_count=0,
        has_laplacian=True,
        has_gaussian=True,
        has_mean_stddev=True,
    ):
        self._device_count = device_count
        self._active_device = None
        if has_laplacian:
            self.createLaplacianFilter = lambda *args, **kwargs: object()
        if has_gaussian:
            self.createGaussianFilter = lambda *args, **kwargs: object()
        if has_mean_stddev:
            self.meanStdDev = lambda *args, **kwargs: ([0.0], [1.0])
        self.GpuMat = lambda: object()

    def getCudaEnabledDeviceCount(self):
        return self._device_count

    def setDevice(self, device_index):
        self._active_device = device_index

    def getDevice(self):
        return self._active_device


def test_detect_cuda_support_handles_cpu_only_runtime(monkeypatch, tmp_path):
    module = load_pipeline_module()
    monkeypatch.setattr(module, "cv2", SimpleNamespace(__version__="4.13.0"), raising=False)
    monkeypatch.setattr(module, "np", object(), raising=False)
    config = make_config(module, tmp_path)

    manager = module.OpenCVBackendManager(config, module.LogWriter())
    report = manager.detect_cuda_support()

    assert report["cv2_available"] is True
    assert report["cuda_namespace_available"] is False
    assert report["cuda_device_count"] == 0


def test_select_backend_prefers_cuda_for_auto_when_available(monkeypatch, tmp_path):
    module = load_pipeline_module()
    fake_cuda = FakeCudaNamespace(device_count=1)
    fake_cv2 = SimpleNamespace(__version__="4.13.0", cuda=fake_cuda)
    monkeypatch.setattr(module, "cv2", fake_cv2, raising=False)
    monkeypatch.setattr(module, "np", object(), raising=False)
    config = make_config(module, tmp_path, opencv_backend="auto", prefer_cuda=True)

    manager = module.OpenCVBackendManager(config, module.LogWriter())

    assert manager.select_backend(config) == "cuda"


def test_select_backend_returns_cpu_when_auto_and_prefer_cuda_disabled(monkeypatch, tmp_path):
    module = load_pipeline_module()
    fake_cuda = FakeCudaNamespace(device_count=2)
    fake_cv2 = SimpleNamespace(__version__="4.13.0", cuda=fake_cuda)
    monkeypatch.setattr(module, "cv2", fake_cv2, raising=False)
    monkeypatch.setattr(module, "np", object(), raising=False)
    config = make_config(module, tmp_path, opencv_backend="auto", prefer_cuda=False)

    manager = module.OpenCVBackendManager(config, module.LogWriter())

    assert manager.select_backend(config) == "cpu"


def test_select_backend_raises_for_strict_cuda_without_device(monkeypatch, tmp_path):
    module = load_pipeline_module()
    fake_cuda = FakeCudaNamespace(device_count=0)
    fake_cv2 = SimpleNamespace(__version__="4.13.0", cuda=fake_cuda)
    monkeypatch.setattr(module, "cv2", fake_cv2, raising=False)
    monkeypatch.setattr(module, "np", object(), raising=False)
    config = make_config(
        module,
        tmp_path,
        opencv_backend="cuda",
        cuda_allow_fallback=False,
    )

    manager = module.OpenCVBackendManager(config, module.LogWriter())

    with pytest.raises(module.PipelineError):
        manager.select_backend(config)


def test_laplacian_score_falls_back_to_cpu_after_cuda_failure(monkeypatch, tmp_path):
    module = load_pipeline_module()
    fake_cuda = FakeCudaNamespace(device_count=1)
    fake_cv2 = SimpleNamespace(__version__="4.13.0", cuda=fake_cuda)
    monkeypatch.setattr(module, "cv2", fake_cv2, raising=False)
    monkeypatch.setattr(module, "np", object(), raising=False)
    config = make_config(
        module,
        tmp_path,
        opencv_backend="cuda",
        cuda_allow_fallback=True,
        save_backend_report=True,
    )
    evaluator = module.BlurEvaluator(config, module.LogWriter())
    evaluator._current_image_path = tmp_path / "work" / "selected" / "images" / "front" / "F_000001.jpg"

    monkeypatch.setattr(
        evaluator,
        "laplacian_score_cuda",
        lambda image: (_ for _ in ()).throw(RuntimeError("forced cuda failure")),
    )
    monkeypatch.setattr(evaluator, "laplacian_score_cpu", lambda image: 12.5)

    score = evaluator.laplacian_score(object())

    assert score == 12.5
    assert evaluator.backend_manager.active_backend == "cpu"
    assert evaluator.backend_manager.fallback_detected is True
    assert evaluator._last_score_metadata["backend"] == "cpu"
    assert evaluator._last_score_metadata["fallback"] is True
    assert config.cuda_fallback_log_path.exists()
    assert config.opencv_backend_report_path.exists()
