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

## Tariff NB202 (resolution-tagged ONNX)

Export ONNX from `DevUtils/Torch2ONNX/TariffNb202ToMnn.py` (outputs `feature_540.onnx`, `fusion_540.onnx`, `feature_1080.onnx`, `fusion_1080.onnx` in `TariffNb202_export/`).

```bash
# NB202 540p
.\build\MNNConvert.exe -f ONNX --modelFile "D:\60-fps-Project\Projects\RIFE GUI\DevUtils\Torch2ONNX\TariffNb202_export\fusion_540.onnx" --MNNModel "D:\60-fps-Project\Projects\RIFE GUI\models\vfi\mnn_tariff\models\Tariff_neu2_nb202_mnn\fusion_540.mnn"  --allowCustomOp  --saveStaticModel --optimizeLevel 0 --batch 1 --keepInputFormat

.\build\MNNConvert.exe -f ONNX --modelFile "D:\60-fps-Project\Projects\RIFE GUI\DevUtils\Torch2ONNX\TariffNb202_export\feature_540.onnx" --MNNModel "D:\60-fps-Project\Projects\RIFE GUI\models\vfi\mnn_tariff\models\Tariff_neu2_nb202_mnn\feature_540.mnn"  --allowCustomOp  --saveStaticModel --optimizeLevel 0 --batch 1 --keepInputFormat

# NB202 1080p
.\build\MNNConvert.exe -f ONNX --modelFile "D:\60-fps-Project\Projects\RIFE GUI\DevUtils\Torch2ONNX\TariffNb202_export\fusion_1080.onnx" --MNNModel "D:\60-fps-Project\Projects\RIFE GUI\models\vfi\mnn_tariff\models\Tariff_neu2_nb202_mnn\fusion_1080.mnn"  --allowCustomOp  --saveStaticModel --optimizeLevel 0 --batch 1 --keepInputFormat

.\build\MNNConvert.exe -f ONNX --modelFile "D:\60-fps-Project\Projects\RIFE GUI\DevUtils\Torch2ONNX\TariffNb202_export\feature_1080.onnx" --MNNModel "D:\60-fps-Project\Projects\RIFE GUI\models\vfi\mnn_tariff\models\Tariff_neu2_nb202_mnn\feature_1080.mnn"  --allowCustomOp  --saveStaticModel --optimizeLevel 0 --batch 1 --keepInputFormat
```

## Tariff PWR (resolution-tagged ONNX)

Export ONNX from `DevUtils/Torch2ONNX/TariffPwrToMnn.py` (outputs `feature_540.onnx`, `fusion_540.onnx`, `feature_1080.onnx`, `fusion_1080.onnx` in `TariffPwr_export/`).

```bash
# PWR 540p
.\build\MNNConvert.exe -f ONNX --modelFile "D:\60-fps-Project\Projects\RIFE GUI\DevUtils\Torch2ONNX\TariffPwr_export\fusion_540.onnx" --MNNModel "D:\60-fps-Project\Projects\RIFE GUI\models\vfi\mnn_tariff\models\Tariff_neu2_pwr_mnn\fusion_540.mnn"  --allowCustomOp  --saveStaticModel --optimizeLevel 0 --batch 1 --keepInputFormat

.\build\MNNConvert.exe -f ONNX --modelFile "D:\60-fps-Project\Projects\RIFE GUI\DevUtils\Torch2ONNX\TariffPwr_export\feature_540.onnx" --MNNModel "D:\60-fps-Project\Projects\RIFE GUI\models\vfi\mnn_tariff\models\Tariff_neu2_pwr_mnn\feature_540.mnn"  --allowCustomOp  --saveStaticModel --optimizeLevel 0 --batch 1 --keepInputFormat

# PWR 1080p
.\build\MNNConvert.exe -f ONNX --modelFile "D:\60-fps-Project\Projects\RIFE GUI\DevUtils\Torch2ONNX\TariffPwr_export\fusion_1080.onnx" --MNNModel "D:\60-fps-Project\Projects\RIFE GUI\models\vfi\mnn_tariff\models\Tariff_neu2_pwr_mnn\fusion_1080.mnn"  --allowCustomOp  --saveStaticModel --optimizeLevel 0 --batch 1 --keepInputFormat

.\build\MNNConvert.exe -f ONNX --modelFile "D:\60-fps-Project\Projects\RIFE GUI\DevUtils\Torch2ONNX\TariffPwr_export\feature_1080.onnx" --MNNModel "D:\60-fps-Project\Projects\RIFE GUI\models\vfi\mnn_tariff\models\Tariff_neu2_pwr_mnn\feature_1080.mnn"  --allowCustomOp  --saveStaticModel --optimizeLevel 0 --batch 1 --keepInputFormat
```

## softsplat
```bash
cd D:\Program\VSsource\comm_repos\MNN\source\backend\opencl\execution\cl
python .\opencl_codegen.py .

cd root
python tools\script\register.py .
```

```bash
.\build\MNNConvert.exe -f ONNX --modelFile "D:\60-fps-Project\Projects\RIFE GUI\softsplat_test.onnx" --MNNModel  'd:\60-fps-Project\Projects\RIFE GUI\softsplat_test.mnn'

```