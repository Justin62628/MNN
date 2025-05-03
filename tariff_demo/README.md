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
cmake -G "Ninja" -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_BUILD_CONVERTER=ON  -DMNN_WIN_RUNTIME_MT=ON .. -DMNN_BUILD_DEMO=ON  -DMNN_OPENCL=ON  -DMNN_USE_THREAD_POOL=OFF
```

```bash
.\build\MNNConvert.exe -f ONNX --modelFile "D:\60-fps-Project\Projects\RIFE GUI\softsplat_test.onnx" --MNNModel  'd:\60-fps-Project\Projects\RIFE GUI\softsplat_test.mnn'

.\build\MNNConvert.exe -f ONNX --modelFile "D:\60-fps-Project\Projects\RIFE GUI\fusion.onnx" --MNNModel "D:\60-fps-Project\Projects\RIFE GUI\models\vfi\mnn_tariff\models\Tariff_neu2_nb202_mnn\fusion_540.mnn"  --allowCustomOp  --saveStaticModel --optimizeLevel 0 --batch 1 --keepInputFormat

.\build\MNNConvert.exe -f ONNX --modelFile "D:\60-fps-Project\Projects\RIFE GUI\feature.onnx" --MNNModel  'd:\60-fps-Project\Projects\RIFE GUI\feature_s.mnn'
```

## softsplat
```bash
cd D:\Program\VSsource\comm_repos\MNN\source\backend\opencl\execution\cl
python .\opencl_codegen.py .

cd root
python tools\script\register.py .
```