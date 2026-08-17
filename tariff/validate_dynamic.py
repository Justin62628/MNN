"""Validate dynamic feature feeds and one Debug/OpenCL MNN fusion run.

The fusion ONNX graph intentionally retains the ``CustomSoftsplat`` operator;
ONNXRuntime does not provide that custom kernel, so this check compares the
standard ONNX feature graph separately and uses MNN for the complete fusion
path.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


MNN_ROOT = Path(__file__).resolve().parents[1]
RIFE_ROOT = Path(r"D:\60-fps-Project\Projects\RIFE GUI")
sys.path.insert(0, str(MNN_ROOT / "tariff"))
sys.path.insert(0, str(RIFE_ROOT))

import tariff  # noqa: E402
from VFI_uc.TariffDynamic import build_dynamic_cache  # noqa: E402


EXPORT_ROOT = RIFE_ROOT / "DevUtils" / "Torch2ONNX"
IMAGE_ROOT = RIFE_ROOT / "test_material" / "images"
TIMESTEP_VALUES = (0.25, 0.5, 0.75)


def _save(path: Path, array: np.ndarray) -> None:
    image = np.clip(array[0].transpose(1, 2, 0)[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), image)


def run_variant(model_type: str, size: tuple[int, int]) -> tariff.TariffProcessor:
    export_dir = EXPORT_ROOT / ("TariffPwr_export" if model_type == "pwr" else "TariffNb202_export")
    model_dir = (
        RIFE_ROOT / "models" / "vfi" / "mnn_tariff_dynamic" / "models"
        / ("Tariff_neu2_pwr_mnn" if model_type == "pwr" else "Tariff_neu2_nb202_mnn")
    )
    img0 = tariff._load_frame(IMAGE_ROOT / "001.png", size)
    img1 = tariff._load_frame(IMAGE_ROOT / "002.png", size)
    timestep_hw = size[::-1] if model_type == "pwr" else (size[1] // 2, size[0] // 2)
    timestep = np.full((1, 1, *timestep_hw), 0.5, dtype=np.float32)

    feature_session = ort.InferenceSession(str(export_dir / "feature_dynamic.onnx"), providers=["CPUExecutionProvider"])
    feature_feed = {"img0": img0, "img1": img1}
    feature_feed.update({name: value.numpy() for name, value in build_dynamic_cache(1, size[1], size[0]).items()})
    feature_values = feature_session.run(None, feature_feed)
    feature_names = [item.name for item in feature_session.get_outputs()]
    feature_map = dict(zip(feature_names, feature_values))

    feature_finite = all(np.isfinite(value).all() for value in feature_values)
    feature_shapes = ", ".join(f"{name}={value.shape}" for name, value in feature_map.items())
    print(f"{model_type}: ONNX feature outputs finite={feature_finite}; {feature_shapes}")

    processor = tariff.TariffProcessor(
        model_dir / "feature_dynamic.mnn",
        model_dir / "fusion_dynamic.mnn",
        model_type=model_type,
    )
    mnn_output = np.asarray(processor.process(img0, img1, timestep)[0])
    print(
        f"{model_type}: MNN shape={mnn_output.shape} range=({mnn_output.min():.6f},{mnn_output.max():.6f}) "
        f"finite={np.isfinite(mnn_output).all()}"
    )
    output_dir = IMAGE_ROOT / "out_dynamic"
    _save(output_dir / f"{model_type}_mnn_{size[0]}x{size[1]}.png", mnn_output)
    return processor


def run_batch_consistency(
    model_type: str,
    size: tuple[int, int],
    processor: tariff.TariffProcessor,
) -> None:
    """Check x4 batch fusion against repeated fusion-only runs.

    x4 in the GUI requests ``[0.25, 0.5, 0.75]`` after one feature pass.  The
    comparison deliberately keeps that feature cache fixed; re-running the
    OpenCL feature graph for every T=1 reference would test a different path
    and can hide the fusion/softsplat state bug this check is meant to catch.
    """

    width, height = size
    rng = np.random.default_rng(20260817)
    img0 = rng.random((1, 3, height, width), dtype=np.float32)
    img1 = rng.random((1, 3, height, width), dtype=np.float32)
    timestep_height = height if model_type == "pwr" else height // 2
    timestep_width = width if model_type == "pwr" else width // 2
    timesteps = np.stack(
        [
            np.full((1, timestep_height, timestep_width), value, dtype=np.float32)
            for value in TIMESTEP_VALUES
        ],
        axis=1,
    )
    processor._processor.prepare_feature(img0, img1)
    batched = processor._processor.process_cached(timesteps)
    singles = [
        processor._processor.process_cached(timesteps[:, index:index + 1])[0]
        for index in range(3)
    ]
    errors = [float(np.max(np.abs(batched[index] - singles[index]))) for index in range(3)]
    print(f"{model_type}: x4 batch-vs-single max_abs={errors} size={width}x{height}")
    if max(errors) > 1e-3:
        raise AssertionError(f"{model_type}: x4 batch mismatch exceeds 1e-3: {errors}")


if __name__ == "__main__":
    model_type = os.environ.get("TARIFF_VALIDATE_MODEL", "pwr").lower()
    width = int(os.environ.get("TARIFF_VALIDATE_WIDTH", "512"))
    height = int(os.environ.get("TARIFF_VALIDATE_HEIGHT", "384"))
    if width % 32 or height % 32:
        raise ValueError("validate_dynamic.py requires a 32-aligned test size; use GUI padding tests for arbitrary sizes")
    processor = run_variant(model_type, (width, height))
    run_batch_consistency(model_type, (width, height), processor)
