import numpy as np
import sys
sys.path.append("D:/Program/VSsource/comm_repos/MNN/build/")
import scdet_mnn

_SceneDetector = scdet_mnn.SceneDetector

class SceneDetector:
    def __init__(self, model_path: str):
        """
        初始化场景检测器。
        :param model_path: MNN 格式模型文件路径
        """
        self._detector = _SceneDetector(model_path)

    def detect(self, img0: np.ndarray, img1: np.ndarray, img2: np.ndarray, img3: np.ndarray) -> tuple[float, float]:
        """
        执行场景检测推理。
        :param img0: HWC uint8 RGB 图像
        :param img1: HWC uint8 RGB 图像
        :return: (confidence0, confidence1)
        """
        assert img0.ndim == 3 and img0.dtype == np.uint8 and img0.shape[2] == 3
        assert img1.ndim == 3 and img1.dtype == np.uint8 and img1.shape[2] == 3
        assert img2.ndim == 3 and img2.dtype == np.uint8 and img2.shape[2] == 3
        assert img3.ndim == 3 and img3.dtype == np.uint8 and img3.shape[2] == 3
        result = [1/(1+np.exp(-i)) for i in self._detector.detect(img0, img1, img2, img3)]
        return result


if __name__ == "__main__":
    import cv2
    import time
    # 初始化
    sd = SceneDetector("D:/60-fps-Project/Projects/RIFE GUI/models/scdet/scdet.mnn")
    # 读取两张 RGB 图像
    img0 = cv2.resize(cv2.cvtColor(cv2.imread("demo/exec/test_input/input.png"), cv2.COLOR_BGR2RGB), (448, 252))
    img1 = cv2.resize(cv2.cvtColor(cv2.imread("demo/exec/test_input/output.png"), cv2.COLOR_BGR2RGB), (448, 252))
    # 推理

    for i in range(1000):
        start = time.time()
        conf0, conf1 = sd.detect(img0, img0, img0, img1)
        print(f"Confidence {i}: {conf0}, {conf1}, time: {time.time() - start}")

        start = time.time()
        conf0, conf1 = sd.detect(img0, img0, img1, img1)
        print(f"Confidence {i}: {conf0}, {conf1}, time: {time.time() - start}")

        start = time.time()
        conf0, conf1 = sd.detect(img0, img1, img1, img1)
        print(f"Confidence {i}: {conf0}, {conf1}, time: {time.time() - start}")

        start = time.time()
        conf0, conf1 = sd.detect(img1, img1, img1, img0)
        print(f"Confidence {i}: {conf0}, {conf1}, time: {time.time() - start}")