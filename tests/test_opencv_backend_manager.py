import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "metashape_dual_fisheye_pipeline.py"


def load_pipeline_module(metashape_module=None):
    spec = importlib.util.spec_from_file_location("metashape_dual_fisheye_pipeline", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    previous_cv2 = sys.modules.get("cv2")
    previous_metashape = sys.modules.get("Metashape")
    sys.modules["cv2"] = SimpleNamespace(__version__="stub")
    if metashape_module is not None:
        sys.modules["Metashape"] = metashape_module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous_cv2
        if metashape_module is not None:
            if previous_metashape is None:
                sys.modules.pop("Metashape", None)
            else:
                sys.modules["Metashape"] = previous_metashape
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


class FakeMetashapeApp:
    def __init__(self):
        self.added = []
        self.removed = []
        self.document = SimpleNamespace(path="", chunk=None)

    def addMenuItem(self, label, callback):
        self.added.append((label, callback))

    def removeMenuItem(self, label):
        self.removed.append(label)


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


def test_import_does_not_register_menus_automatically():
    fake_app = FakeMetashapeApp()
    fake_metashape = SimpleNamespace(app=fake_app)

    load_pipeline_module(metashape_module=fake_metashape)

    assert fake_app.added == []
    assert fake_app.removed == []


def test_register_menu_items_is_idempotent():
    fake_app = FakeMetashapeApp()
    fake_metashape = SimpleNamespace(app=fake_app)
    module = load_pipeline_module(metashape_module=fake_metashape)

    module.register_menu_items()
    module.register_menu_items()

    labels = [label for label, _callback in fake_app.added]
    assert len(labels) == 9
    assert len(set(labels)) == 9
    assert fake_app.removed == labels


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
    log_text = config.cuda_fallback_log_path.read_text(encoding="utf-8")
    assert "BlurEvaluator.laplacian_score_cuda" in log_text
    assert str(evaluator._current_image_path) in log_text
    assert "forced cuda failure" in log_text


def test_apply_cuda_filter_accepts_direct_gpumat_return(monkeypatch, tmp_path):
    module = load_pipeline_module()

    class FakeGpuMat:
        def __init__(self, data=None):
            self.data = data

        def upload(self, data):
            self.data = data

        def download(self):
            return self.data

    class FakeFilter:
        def __init__(self, result):
            self.result = result

        def apply(self, source):
            return self.result

    fake_cv2 = SimpleNamespace(cuda=SimpleNamespace(GpuMat=FakeGpuMat))
    monkeypatch.setattr(module, "cv2", fake_cv2, raising=False)

    gpu_result = FakeGpuMat(data="laplacian")
    returned = module.BlurEvaluator._apply_cuda_filter(FakeFilter(gpu_result), FakeGpuMat(data="source"))

    assert returned is gpu_result


def test_laplacian_score_cuda_converts_to_float32_before_filter(monkeypatch, tmp_path):
    module = load_pipeline_module()

    class FakeGpuMat:
        def __init__(self, data=None):
            self.data = data
            self.convert_calls = []

        def upload(self, data):
            self.data = data

        def download(self):
            return self.data

        def convertTo(self, cv_type):
            self.convert_calls.append(cv_type)
            return FakeGpuMat(self.data.astype(np.float32, copy=False))

    class FakeFilter:
        def __init__(self, result):
            self.result = result

        def apply(self, source):
            return self.result

    class FakeCudaNamespace:
        def __init__(self):
            self.last_uploaded = None
            self.laplacian_args = None

        def GpuMat(self):
            gpu_mat = FakeGpuMat()
            self.last_uploaded = gpu_mat
            return gpu_mat

        def createLaplacianFilter(self, src_type, dst_type, ksize):
            self.laplacian_args = (src_type, dst_type, ksize)
            return FakeFilter(FakeGpuMat(np.ones((4, 4), dtype=np.float32)))

    fake_cuda = FakeCudaNamespace()
    fake_cv2 = SimpleNamespace(
        cuda=fake_cuda,
        CV_8UC1=1,
        CV_32FC1=2,
        CV_8U=3,
        CV_32F=4,
    )
    monkeypatch.setattr(module, "cv2", fake_cv2, raising=False)
    monkeypatch.setattr(module, "np", np, raising=False)

    config = make_config(module, tmp_path, opencv_backend="cuda", cuda_allow_fallback=False)
    evaluator = module.BlurEvaluator(config, module.LogWriter())
    monkeypatch.setattr(evaluator, "compute_center70_mask", lambda image: np.ones(image.shape[:2], dtype=np.uint8))

    score = evaluator.laplacian_score_cuda(np.zeros((4, 4), dtype=np.uint8))

    assert score == 0.0
    assert fake_cuda.laplacian_args == (fake_cv2.CV_32FC1, fake_cv2.CV_32FC1, 3)
    assert fake_cuda.last_uploaded is not None
    assert fake_cuda.last_uploaded.convert_calls == [fake_cv2.CV_32FC1]


def test_load_model_resolves_bare_filename_from_project_root(monkeypatch, tmp_path):
    module = load_pipeline_module()
    model_path = tmp_path / "yolo26x-seg.pt"
    model_path.write_bytes(b"weights")
    captured = {}

    def fake_yolo(path):
        captured["path"] = path
        return {"path": path}

    monkeypatch.setattr(module, "YOLO", fake_yolo, raising=False)
    config = make_config(module, tmp_path, mask_model_path="yolo26x-seg.pt")
    config.project_root = tmp_path
    generator = module.MaskGenerator(config, module.LogWriter())

    model = generator.load_model()

    assert model["path"] == str(model_path.resolve(strict=False))
    assert captured["path"] == str(model_path.resolve(strict=False))


def test_load_model_raises_clear_error_when_local_model_missing(monkeypatch, tmp_path):
    module = load_pipeline_module()
    yolo_called = {"value": False}

    def fake_yolo(path):
        yolo_called["value"] = True
        return {"path": path}

    monkeypatch.setattr(module, "YOLO", fake_yolo, raising=False)
    config = make_config(module, tmp_path, mask_model_path="missing-model.pt")
    config.project_root = tmp_path
    generator = module.MaskGenerator(config, module.LogWriter())

    with pytest.raises(module.PipelineError, match="Please select a local \\.pt file in GUI"):
        generator.load_model()

    assert yolo_called["value"] is False


def test_build_log_summary_can_skip_runtime_gpu_probes(monkeypatch, tmp_path):
    module = load_pipeline_module()
    config = make_config(module, tmp_path, save_backend_report=False)
    controller = module.PipelineController(config, initialize_logging=False)

    monkeypatch.setattr(
        controller.blur_evaluator.backend_manager,
        "detect_cuda_support",
        lambda: (_ for _ in ()).throw(AssertionError("OpenCV probe should be deferred")),
    )
    monkeypatch.setattr(
        controller.mask_generator,
        "detect_backend_support",
        lambda: (_ for _ in ()).throw(AssertionError("YOLO probe should be deferred")),
    )

    summary = controller.build_log_summary(probe_backends=False)

    assert summary["opencv_backend"]["notes"] == [
        "OpenCV CUDA probe is deferred until explicit status refresh or pipeline execution."
    ]
    assert summary["yolo_backend"]["notes"] == [
        "YOLO device probe is deferred until explicit status refresh or pipeline execution."
    ]
    assert summary["metashape_gpu"]["status"] == "deferred"
