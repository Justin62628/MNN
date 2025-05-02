# Step by Step
```bash
# start x64 Native Tools Command Prompt for VS 2022
# enter some desired venv
"D:/60-fps-Project/Projects/RIFE GUI/venv/Scripts/activate"
# open vscode
Code
# modify scdet_demo/modify_onnx.py and convert sudo's scdet onnx to 2-input model
python scdet_demo/modify_onnx.py
# convert to MNN model
MNNConvert -f ONNX --modelFile scdet/scdet.onnx --MNNModel scdet/scdet.mnn --keepInputFormat=0 --bizCode biz
```

- modify `.vscode/settings.json` path
- click `Configure` in `CMake` - `Project Status`
- `Ctrl+Shift+P`, `CMake: Set/Launch Target`, select `scdet_demo.cpp`,
- Click `Build` in `CMake` - `Project Status`
- `Ctrl+Shift+P`, `CMake: Debug`, should output 0.99, since that two completely different images have 99% possiblity of being a scene.
- `Ctrl+Shift+P`, `CMake: Select Variant` - `Release`
- Modify `scdet/scdet.py` path
- `python scdet/scdet.py`
- Modify `scdet/scdet_py.cpp` if needed.

```bash
./build/MNNConvert.exe -f ONNX --modelFile "E:/Library/Downloads/model.onnx" --MNNModel  "D:/60-fps-Project/Projects/RIFE GUI/models/scdet/scdet.mnn"  --allowCustomOp  --saveStaticModel --debug --optimizeLevel 3 --info
```
