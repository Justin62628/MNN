import numpy as np
import sys
import numpy as np
# tariff.py
sys.path.append("D:/Program/VSsource/comm_repos/MNN/build/")
import tariff_mnn

class TariffProcessor:
    def __init__(self, feat_model_path: str, fusion_model_path: str):
        """
        初始化处理管道
        :param feat_model_path: 特征提取模型路径
        :param fusion_model_path: 融合模型路径
        """
        self._processor = tariff_mnn.TariffProcessor(feat_model_path, fusion_model_path)
    
    def process(self, img0: np.ndarray, img1: np.ndarray, coords: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
        """
        执行完整处理流程
        :param img0: BCHW格式输入图像1
        :param img1: BCHW格式输入图像2 
        :param coords: BCHW格式坐标信息
        :param timesteps: BTCHW格式时间步序列（实际T=1）
        :return: B x 3*T x H x W 格式输出
        """
        # 验证输入形状
        assert img0.ndim == 4 and img0.dtype == np.float32
        assert img1.shape == img0.shape
        
        print(os.getpid())
        # 执行推理
        result = self._processor.process(img0, img1, coords, timesteps)
        
        # 后处理（如果需要）
        return result

if __name__ == "__main__":
    # 示例用法
    import cv2
    import torch
    import torch.nn.functional as F
    import os
    # processor = TariffProcessor("D:/60-fps-Project/Projects/RIFE GUI/feature_s.mnn", "D:/60-fps-Project/Projects/RIFE GUI/fusion_s.mnn")
    processor = TariffProcessor("D:/60-fps-Project/Projects/RIFE GUI/models/vfi/mnn_tariff/models/Tariff_neu2_nb202_mnn/feature_540.mnn", 
                                "D:/60-fps-Project/Projects/RIFE GUI/models/vfi/mnn_tariff/models/Tariff_neu2_nb202_mnn/fusion_540.mnn")
    size = (960, 576)
    # size = (1920, 1088)
    
    image_root = "D:/60-fps-Project/Projects/RIFE GUI/test_material/images/"
    output_root = os.path.join(image_root, 'out/')
    img0 = cv2.resize(cv2.imread(os.path.join(image_root, "turbo0.png")), size)
    img0 = cv2.resize(cv2.imread(os.path.join(image_root, "00000000.jpg")), size)
    img1 = cv2.resize(cv2.imread(os.path.join(image_root, "00000002.jpg")), size)
    img0 = cv2.resize(cv2.imread(os.path.join(image_root, "001.png")), size)
    img1 = cv2.resize(cv2.imread(os.path.join(image_root, "002.png")), size)
    img0, img1 = map(lambda x: torch.from_numpy(x)[None, ...].permute(0, 3, 1, 2).mul(1/255.).float(), (img0, img1))
    def preprocess(x):
        return F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)

    img0f, img1f = preprocess(img0), preprocess(img1)

    def coords_grid(b, h, w, device=torch.device("cuda"), dtype: torch.dtype=torch.float32):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

        stacks = [x, y]

        grid = torch.stack(stacks, dim=0)  # [2, H, W] or [3, H, W]

        grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

        grid = grid.to(device, dtype=dtype)
        return grid
    
    coords = coords_grid(1, img0f.size(2), img0f.size(3), dtype=img0f.dtype, device=img0f.device)  # NOTE: MNN?

    n = 3

    timesteps = torch.concat([torch.ones_like(img0f[:, :1]) * i / (n + 1) for i in range(1, n+1)], dim=1)

    # 执行推理
    outputs = processor.process(img0.cpu().numpy(), img1.cpu().numpy(), coords.cpu().numpy(), timesteps.cpu().numpy())
    for i, output in enumerate(outputs):
        print(f"Output shape: {output.shape}")
        cv2.imwrite(os.path.join(output_root, f"mnn_{i}.png"), (output[0].transpose(1, 2, 0) * 255).astype(np.uint8))