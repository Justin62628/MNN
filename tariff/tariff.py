import os
import numpy as np
import sys

sys.path.append("D:/Program/VSsource/comm_repos/MNN/build/Debug")
# sys.path.append("D:/Program/VSsource/comm_repos/MNN/build/Release")
import tariff_mnn


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
        """
        Initialize the Tariff pipeline.
        :param feat_model_path: Feature extraction model path (.mnn)
        :param fusion_model_path: Fusion model path (.mnn)
        :param model_type: "nb202" or "pwr"
        :param platform_size: Number of GPU cards (OpenCL)
        :param platform_id: Which GPU card to use (OpenCL)
        :param device_id: Which device on the card (OpenCL)
        """
        # tariff_mnn.print_opencl_devices()
        self._processor = tariff_mnn.TariffProcessor(
            feat_model_path,
            fusion_model_path,
            model_type,
            platform_size,
            platform_id,
            device_id,
        )
        self._model_type = model_type

    def process(self, img0: np.ndarray, img1: np.ndarray, timesteps: np.ndarray) -> list:
        """
        Execute full pipeline. NB202 and PWR share same interface: img0, img1, timesteps.
        :param img0: BCHW input image 1
        :param img1: BCHW input image 2
        :param timesteps: BTHW time step sequence (T typically 1)
        :return: List of BCHW output arrays
        """
        assert img0.ndim == 4 and img0.dtype == np.float32
        assert img1.shape == img0.shape
        return self._processor.process(img0, img1, timesteps)


if __name__ == "__main__":
    import cv2
    import torch

    # --- NB202 example ---
    # processor = TariffProcessor(
    #     "D:/60-fps-Project/Projects/RIFE GUI/models/vfi/mnn_tariff/models/Tariff_neu2_nb202_mnn/feature_1080.mnn",
    #     "D:/60-fps-Project/Projects/RIFE GUI/models/vfi/mnn_tariff/models/Tariff_neu2_nb202_mnn/fusion_1080.mnn",
    #     model_type="nb202")
    # size = (1920, 1088)

    # processor = TariffProcessor(
    #     "D:/60-fps-Project/Projects/RIFE GUI/models/vfi/mnn_tariff/models/Tariff_neu2_nb202_mnn/feature_540.mnn",
    #     "D:/60-fps-Project/Projects/RIFE GUI/models/vfi/mnn_tariff/models/Tariff_neu2_nb202_mnn/fusion_540.mnn",
    #     model_type="nb202")
    # size = (960, 576)


    # --- PWR example ---
    # processor = TariffProcessor(
    #     "D:/60-fps-Project/Projects/RIFE GUI/models/vfi/mnn_tariff/models/Tariff_neu2_pwr_mnn/feature_1080.mnn",
    #     "D:/60-fps-Project/Projects/RIFE GUI/models/vfi/mnn_tariff/models/Tariff_neu2_pwr_mnn/fusion_1080.mnn",
    #     model_type="pwr")
    # size = (1920, 1088)

    processor = TariffProcessor(
        "D:/60-fps-Project/Projects/RIFE GUI/models/vfi/mnn_tariff/models/Tariff_neu2_pwr_mnn/feature_540.mnn",
        "D:/60-fps-Project/Projects/RIFE GUI/models/vfi/mnn_tariff/models/Tariff_neu2_pwr_mnn/fusion_540.mnn",
        model_type="pwr")
    size = (960, 576)

    image_root = "D:/60-fps-Project/Projects/RIFE GUI/test_material/images/"
    output_root = os.path.join(image_root, 'out/')
    os.makedirs(output_root, exist_ok=True)

    img0 = cv2.resize(cv2.imread(os.path.join(image_root, "001.png")), size)
    img1 = cv2.resize(cv2.imread(os.path.join(image_root, "002.png")), size)
    img0, img1 = map(lambda x: torch.from_numpy(x)[None, ...].permute(0, 3, 1, 2).mul(1/255.).float(), (img0, img1))

    n = 3
    timesteps = torch.concat([torch.ones(1, 1, size[1], size[0]) * i / (n + 1) for i in range(1, n + 1)], dim=1)

    outputs = processor.process(img0.cpu().numpy(), img1.cpu().numpy(), timesteps.cpu().numpy())

    for i, output in enumerate(outputs):
        print(f"Output shape: {output.shape}")
        cv2.imwrite(os.path.join(output_root, f"mnn_{i}.png"), (output[0].transpose(1, 2, 0) * 255).astype(np.uint8))
