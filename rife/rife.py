"""
RIFE v4.22 MNN inference. Single model: input [img0, img1, timestep, grid] -> output interpolated frame.
API: process(img0, img1, timestep) with HWC uint8 RGB in/out (normalization done in C++ via ImageProcess).
"""
import sys
import os

# Add MNN build output so rife_mnn can be imported (e.g. build/Debug or build/Release)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_mnn_root = os.path.normpath(os.path.join(_script_dir, ".."))
for _sub in ("build/Debug", "build/Release", "build"):
    _path = os.path.join(_mnn_root, _sub)
    if os.path.isdir(_path):
        if _path not in sys.path:
            sys.path.insert(0, _path)
        break

import numpy as np
import rife_mnn


class RIFEProcessor:
    def __init__(self, model_path: str):
        """
        :param model_path: Path to RIFE .mnn model (e.g. rife4.22.mnn).
        """
        self._processor = rife_mnn.RIFEProcessor(model_path)

    def process(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        timestep: float = 0.5,
    ) -> np.ndarray:
        """
        Run one interpolation step.
        :param img0: HWC uint8 RGB (h, w, 3).
        :param img1: HWC uint8 RGB (h, w, 3), same shape as img0.
        :param timestep: Interpolation time in [0, 1] (0.5 = midpoint).
        :return: HWC uint8 RGB (h, w, 3).
        """
        assert img0.ndim == 3 and img0.shape[2] == 3 and img0.dtype == np.uint8
        assert img1.shape == img0.shape and img1.dtype == np.uint8
        return self._processor.process(img0, img1, timestep)


if __name__ == "__main__":
    import cv2

    # Example paths; adjust to your MNN model and images.
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "60-fps-Project", "Projects", "RIFE GUI",
        "models", "vfi", "mnn_rife", "models", "rife4.22.mnn",
    )
    model_path = os.path.normpath(model_path)
    if not os.path.isfile(model_path):
        model_path = "D:/60-fps-Project/Projects/RIFE GUI/models/vfi/mnn_rife/models/rife4.22.mnn"
    if not os.path.isfile(model_path):
        print("Set model_path and image paths below.")
        sys.exit(1)

    image_root = "D:/60-fps-Project/Projects/RIFE GUI/test_material/images/"
    out_path = os.path.join(image_root, "out", "rife_mnn.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    size = (960, 544)
    img0 = cv2.resize(cv2.imread(os.path.join(image_root, "001.png")), size)
    img1 = cv2.resize(cv2.imread(os.path.join(image_root, "002.png")), size)
    # HWC uint8 RGB (cv2 gives BGR)

    processor = RIFEProcessor(model_path)
    out = processor.process(img0, img1, timestep=0.5)
    cv2.imwrite(out_path, out)
    print("Wrote", out_path)
