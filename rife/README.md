# Step by Step
- follow https://mnn-docs.readthedocs.io/en/latest/contribute/op.html
```bash
# administrative, Developer PowerShell for VS 2022
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
cd schema
.\generate.ps1
```

```bash
# copy zconf.h
# cnpy
cmake -B build -DZLIB_INCLUDE_DIR="..\zlib" -DZLIB_LIBRARY="..\zlib\build\Debug"

cd build
cmake -G "Ninja" -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_BUILD_CONVERTER=ON  -DMNN_WIN_RUNTIME_MT=ON .. -DMNN_BUILD_DEMO=ON  -DMNN_OPENCL=ON  -DMNN_USE_THREAD_POOL=OFF -DMNN_VULKAN=ON
```

## RIFE

ONNX is exported from RIFE GUI (`VFI_uc/RIFE/v4_21_onnx/RIFE_HDv4.py`). Single model: input `input` (1, 9, H, W) = [img0(3), img1(3), timestep(1), horizontal(1), vertical(1)], output `out` (1, 3, H, W) in [0, 1].

### Convert ONNX → MNN

```bash
.\build\Debug\MNNConvert.exe -f ONNX --modelFile "D:\60-fps-Project\Projects\RIFE GUI\rife4.22.onnx" --MNNModel "D:\60-fps-Project\Projects\RIFE GUI\models\vfi\mnn_rife\models\rife4.22.mnn"  --allowCustomOp --optimizeLevel 0 --batch 1 --keepInputFormat
```

### Build inference (C++ pybind11 module)

From MNN root, configure with `tariff` (and thus pybind11) then build. The rife folder is included by root `CMakeLists.txt` and produces `rife_mnn` (e.g. `build/Debug/rife_mnn.pyd`).

```bash
cd build
ninja rife_mnn
```

### Python usage

Ensure `build/Debug` (or `build/Release`) is on `PYTHONPATH`, then:

```python
import sys
sys.path.append("D:/Program/VSsource/comm_repos/MNN/build/Debug")  # or Release
import numpy as np
import rife_mnn

proc = rife_mnn.RIFEProcessor("path/to/rife4.22.mnn")
# img0, img1: (1, 3, H, W) float32 in [0, 1]
out = proc.process(img0, img1, timestep=0.5)  # (1, 3, H, W) float32 [0, 1]
```

Or run the wrapper script (edit paths inside if needed):

```bash
python rife/rife.py
```
