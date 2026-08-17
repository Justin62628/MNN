"""Python entry point for the dynamic-shape Tariff MNN pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


_MNN_ROOT = Path(__file__).resolve().parents[1]
_BUILD_CANDIDATES = [
    os.environ.get("TARIFF_MNN_BUILD", ""),
    str(_MNN_ROOT / "build-debug-ninja"),
    str(_MNN_ROOT / "build-debug" / "Debug"),
    str(_MNN_ROOT / "build-debug"),
    str(_MNN_ROOT / "build" / "Debug"),
    str(_MNN_ROOT / "build" / "Release"),
]
for _build_dir in _BUILD_CANDIDATES:
    while _build_dir and _build_dir in sys.path:
        sys.path.remove(_build_dir)
for _build_dir in _BUILD_CANDIDATES:
    if _build_dir and Path(_build_dir).is_dir():
        sys.path.insert(0, _build_dir)
        break

# The GUI adapter intentionally has a separate OpenCL cache namespace because
# its x2 path starts with native process(), while tariff.py uses the explicit
# prepare_feature/process_cached route.  MNN tuning/op cache entries are not
# safe to mix across those execution contracts.
os.environ.setdefault("TARIFF_OPENCL_CACHE_NAMESPACE", "tariff_py")

import tariff_mnn  # noqa: E402


class TariffProcessor:
    def __init__(
        self,
        feat_model_path: str,
        fusion_model_path: str,
        model_type: str = "nb202",
        platform_size: int = 1,
        platform_id: int = 0,
        device_id: int = 0,
    ):
        model_type = model_type.lower()
        if model_type not in {"nb202", "pwr"}:
            raise ValueError(f"model_type must be 'nb202' or 'pwr', got {model_type!r}")
        self._model_type = model_type
        self._feat_model_path = str(feat_model_path)
        self._fusion_model_path = str(fusion_model_path)
        self._platform_size = platform_size
        self._platform_id = platform_id
        self._device_id = device_id
        self._processor = tariff_mnn.TariffProcessor(
            self._feat_model_path,
            self._fusion_model_path,
            model_type,
            platform_size,
            platform_id,
            device_id,
        )
        self._native_shape = None

    def process(self, img0: np.ndarray, img1: np.ndarray, timesteps: np.ndarray) -> list:
        """Run all timesteps for BCHW input frames.

        Timesteps are ``[B,T,H/2,W/2]`` for NB202 and ``[B,T,H,W]`` for PWR.
        The native model is evaluated on the smallest 64-aligned frame shape
        using edge padding, then every result is cropped back to the caller's
        exact H/W.  This keeps the exported graph shape-dynamic while making
        the public Python entry point accept non-aligned video sizes.
        """

        img0 = np.ascontiguousarray(img0, dtype=np.float32)
        img1 = np.ascontiguousarray(img1, dtype=np.float32)
        timesteps = np.ascontiguousarray(timesteps, dtype=np.float32)
        if img0.ndim != 4 or img1.shape != img0.shape:
            raise ValueError("img0 and img1 must be identical 4D BCHW arrays")
        if img0.shape[1] != 3:
            raise ValueError(f"Tariff expects C=3, got {img0.shape[1]}")
        _, _, height, width = img0.shape
        expected_hw = (height // 2, width // 2) if self._model_type == "nb202" else (height, width)
        if timesteps.ndim != 4 or timesteps.shape[0] != img0.shape[0] or timesteps.shape[2:] != expected_hw:
            raise ValueError(
                f"timesteps must have shape [B,T,{expected_hw[0]},{expected_hw[1]}] for {self._model_type}"
            )
        padded_height = ((height + 63) // 64) * 64
        padded_width = ((width + 63) // 64) * 64
        native_shape = (padded_height, padded_width)
        if self._native_shape is not None and self._native_shape != native_shape:
            # Recreate both sessions when the padded spatial shape changes.
            # MNN/OpenCL can retain execution-local state across a dynamic
            # resize; keeping one native object per padded shape avoids
            # carrying a previous resolution into the next graph encoding.
            old_processor = self._processor
            self._processor = None
            del old_processor
            self._processor = tariff_mnn.TariffProcessor(
                self._feat_model_path,
                self._fusion_model_path,
                self._model_type,
                self._platform_size,
                self._platform_id,
                self._device_id,
            )
        self._native_shape = native_shape
        if padded_height == height and padded_width == width:
            self._processor.prepare_feature(img0, img1)
            return self._processor.process_cached(timesteps)

        def pad_spatial(array: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
            pad_h = target_height - array.shape[-2]
            pad_w = target_width - array.shape[-1]
            return np.pad(array, ((0, 0),) * (array.ndim - 2) + ((0, pad_h), (0, pad_w)), mode="edge")

        padded_img0 = pad_spatial(img0, padded_height, padded_width)
        padded_img1 = pad_spatial(img1, padded_height, padded_width)
        timestep_height = padded_height // 2 if self._model_type == "nb202" else padded_height
        timestep_width = padded_width // 2 if self._model_type == "nb202" else padded_width
        padded_timesteps = pad_spatial(timesteps, timestep_height, timestep_width)
        self._processor.prepare_feature(padded_img0, padded_img1)
        outputs = self._processor.process_cached(padded_timesteps)
        return [np.ascontiguousarray(output[..., :height, :width]) for output in outputs]


def _load_frame(path: Path, size_wh: tuple[int, int]) -> np.ndarray:
    import cv2

    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(path)
    width, height = size_wh
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0


def _make_timesteps(model_type: str, batch: int, n: int, height: int, width: int) -> np.ndarray:
    if model_type == "nb202":
        height //= 2
        width //= 2
    return np.stack(
        [np.full((batch, height, width), i / (n + 1), dtype=np.float32) for i in range(1, n + 1)],
        axis=1,
    )


if __name__ == "__main__":
    import cv2

    model_type = os.environ.get("TARIFF_MODEL_TYPE", "pwr").lower()
    model_dir = Path(
        os.environ.get(
            "TARIFF_MODEL_DIR",
            str(
                Path(r"D:\60-fps-Project\Projects\RIFE GUI\DevUtils\Torch2ONNX")
                / ("TariffPwr_export" if model_type == "pwr" else "TariffNb202_export")
            ),
        )
    )
    processor = TariffProcessor(
        model_dir / "feature_dynamic.mnn",
        model_dir / "fusion_dynamic.mnn",
        model_type=model_type,
    )

    image_root = Path(
        os.environ.get("TARIFF_IMAGE_ROOT", r"D:\60-fps-Project\Projects\RIFE GUI\test_material\images")
    )
    output_root = image_root / "out_dynamic"
    output_root.mkdir(parents=True, exist_ok=True)
    resolutions = [(960, 576), (512, 384), (511, 383), (480, 270)]
    n = 3
    for width, height in resolutions:
        img0 = _load_frame(image_root / "001.png", (width, height))
        img1 = _load_frame(image_root / "002.png", (width, height))
        timesteps = _make_timesteps(model_type, img0.shape[0], n, height, width)
        outputs = processor.process(img0, img1, timesteps)
        for index, output in enumerate(outputs):
            output = np.asarray(output)
            print(
                f"{model_type} {width}x{height} t={index}: shape={output.shape} "
                f"min={output.min():.6f} max={output.max():.6f} "
                f"finite={np.isfinite(output).all()}"
            )
            image = np.clip(output[0].transpose(1, 2, 0)[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(output_root / f"{model_type}_{width}x{height}_{index}.png"), image)
