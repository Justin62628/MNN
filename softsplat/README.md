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
```