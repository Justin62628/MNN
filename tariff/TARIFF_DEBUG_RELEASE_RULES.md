# Tariff dynamic-shape / MNN rules

这份规则记录本次调试中已经验证过的约束，后续导出、转换、部署和回归都按此执行。

## Python、源码边界和构建

1. RIFE GUI 目录下的 Python、ONNX 导出、MNN 转换、GUI 烟测统一使用：
   `D:\60-fps-Project\Projects\RIFE GUI\venv\Scripts\python.exe`。
2. 网络结构、原始 softsplat 和原始推理逻辑保持独立；动态 grid/cache 与 ONNX 适配放在 `VFI_uc\TariffDynamic`，Nuitka 要收集的 GUI MNN 适配器放在 `VFI\MNN\inference_tariff_mnn.py`。
3. Release 构建使用 VS 2022 x64 Developer Prompt、Ninja、`MNN_BUILD_SHARED_LIBS=OFF`、`MNN_BUILD_CONVERTER=ON`、`MNN_OPENCL=ON`、`MNN_WIN_RUNTIME_MT=ON`。
4. Windows 上 Converter 的静态构建需要显式指定仓库内 zlib：
   `-DZLIB_INCLUDE_DIR=D:/Program/VSsource/comm_repos/MNN/3rd_party/zlib`
   和 `-DZLIB_LIBRARY=.../3rd_party/zlib/build/Release/zlibstatic.lib`。

## 导出和转换

5. ONNX 不生成 grid cache。`coords`、backwarp grid、feature grid、delta、radius embedding 和初始 context 全部作为 `nn.Module` 输入；推理端按当前 H/W 构造同名输入。
6. 导出样例 H/W 取 64 的倍数，只用于让导出器看到完整算子路径；输入和输出的 H/W 轴仍必须声明 dynamic，不能把样例尺寸写回网络常量。
7. NB202 的 timestep 是 `[B,T,H//2,W//2]`；PWR 的 timestep 是 `[B,T,H,W]`。不能用同一个 ones tensor 规则覆盖两个变体。
8. MNN 转换固定使用 `--allowCustomOp --optimizeLevel 0 --batch 1 --keepInputFormat`。当前图的部署契约是 batch=1、空间分辨率动态；不要把它误称为任意 batch。
9. 每次转换后先检查 ONNX/MNN 的 input/output 名称、数量、rank、动态轴和 finite 统计，再运行联合推理；只看单独的 feature 或 fusion 结果不够。
10. 反向 flow 分支不要导出 `flip(0)`。MNN 对该形式的 batch 反转曾生成错误的 `flow10`；在独立网络包装层使用 `chunk(2,0)` 加 `cat((second, first),0)`。
11. PWR 的 linear softsplat 分母为零时必须用 `where(denom == 0, 1, denom)`，否则 PyTorch 导出 debug 结果会出现 NaN。

## MNN/OpenCL 运行时

12. feature 和 fusion 必须共享同一套 MNN runtime/session 语义；不要用旧的全局 cache、旧的 interpreter 输出或不同 backend 的悬空 tensor 交叉传递。
13. OpenCL feature 输出在传给 fusion 前保持 device tensor；需要 host 数据时必须先同步，再使用 `copyToHostTensor`，并保留 mapped-host fallback 的 shape/stride 检查。
14. OpenCL softsplat 使用独立的固定点 `int32` accumulation buffer（当前 scale 为 8192），再执行单独的 convert kernel；主 splat 和 convert 使用固定安全 workgroup，不要直接套用通用 tuner。
15. softsplat 回归至少覆盖 C=4、33、65、97、多个 timestep，并逐项与 CPU 结果比较。单测通过、fusion 单独通过，并不等于 feat+fusion 联合 plugin 通过。
16. OpenCL 的多 timestep 请求必须把 `[0.25, 0.5, 0.75]` 的批量结果与同一 feature cache 下的三个 fusion-only `T=1` 结果逐项比较；`CustomSoftsplat` 的固定点 accumulation buffer 必须在每次 `onExecute` 前清零，不能只在 `onEncode/resizeSession` 时清零。
17. 日志中的 `CL ERROR CODE : -52, info:run2d` 只能作为调优告警观察；只有进程退出成功、输出 shape 正确、finite、非全零并通过图像检查时才算成功。

## GUI 适配和分辨率

18. GUI 适配器接收 OpenCV HWC/BGR frame，转换为 BCHW float32/255；输出再转回调用方原始 H/W 和 uint8。不要在适配器里偷偷 resize 到 540/1080 档位。
19. native wrapper 对任意空间 H/W 做 edge pad 到 64 的倍数（feature/grid 本身虽只要求 32，但 IFNet 多级 concat 要求 64），NB202 的 timestep 按对应半分辨率 pad，返回前 crop 到调用方 H/W；因此必须同时测试对齐和非对齐尺寸。`tariff.py` 在对齐后的 native `(H,W)` 变化时重建 native processor；GUI adapter 则锁定一个 padded shape 并对分辨率切换显式报错，避免动态 graph 的跨 shape execution 状态残留。
20. GUI 部署目录使用独立包 `models\vfi\mnn_tariff_dynamic`，其 pyd、`feature_dynamic.mnn`、`fusion_dynamic.mnn` 与 `VFI\MNN\inference_tariff_mnn.py` 成套更新；旧模型文件保留作回滚基线。
21. 接入测试必须经过 `VfiPools.vfi_get_module`，确认 `MNN_Tariff` 分支加载的是 `VFI.MNN.inference_tariff_mnn`，不能只直接实例化 core。
22. GUI/Nuitka 的 x3/x4 请求必须保留 native 的“feature 一次 + fusion 多个 timestep”执行路径；不得用逐 timestep 重跑 feature 的 fallback 掩盖 OpenCL 状态问题。native `tariff.py` 和 GUI adapter 的回归都必须固定 feature cache 后比较 batch-vs-single fusion。
23. 修改任意 `.cl` 后必须重新运行 `opencl_codegen.py` 更新对应的 `*_mnn_cl.cpp`，再重编 MNN；只改嵌入源或只改 `.cl` 都会留下不一致的 kernel 签名。
24. 一个 custom op 有多个 kernel 时，所有 kernel 必须接收相同的 build macro 集合；固定点 convert kernel 也必须传入 `USE_FIXED_POINT_ACCUM`，否则它会把 `int32` accumulation 当成 `float` 读取，结果会接近全零或出现 NaN。
25. scatter 的 FP32 CAS 累加受 work-item 执行顺序影响，在 NB202 非 4 对齐 channel/非对齐分辨率会产生可见漂移；需要使用整数固定点 atomic add，把非确定性缩减到可控量化误差。
26. grid/cache host 数据必须按当前 H/W 重建并在每次 feature run 重新拷贝；同分辨率的 feature session 仍可能保留 OpenCL correlation/grid 执行状态，因此在每个新帧对边界重建 feature session，但绝不能在 timestep 循环中重复计算 feature。
27. 单个 OLS 进程只初始化一个模型变体（PWR 或 NB202）并固定一个 padded 分辨率；不要把 PWR→NB202、对象串行或进程内分辨率切换当作有效回归场景。需要测多个变体/分辨率时使用独立进程。
28. `tariff.py` 的公共入口统一采用 `prepare_feature` + `process_cached`；直接 native `process` 仅保留为单步底层接口，不能作为 x3/x4 的性能或一致性基线。
29. OpenCL kernel 的跨进程持久化必须通过 tariff 专用的 `Interpreter::setCacheFile` / `updateCacheFile`，不要复用旧的全局 `feat.cache`。feature 与 fusion 共享一个 `RuntimeInfo`，由 feature interpreter 作为唯一 cache owner，避免同一进程把同一份 binary 重复加载两次。
30. 持久化 cache 包含 OpenCL program binary、autotuning LWS、GEMM 参数和 pre-param；首次 shape resize 后必须更新 cache，才能保存动态分辨率路径的 kernel/tuning。它不包含进程内的 `cl_context`、Session、显存分配和 Tensor 内容；这些仍需每次进程启动/shape 初始化。
31. cache 文件名必须隔离执行 namespace、模型组合、OpenCL platform/device 和自定义 kernel schema；GUI 与 `tariff.py` 不共用 tuning/op cache，因为两者的 x2/x4 session 序列不同。修改 `.cl`、embedded source 或 build macro 时必须递增 schema，避免旧 binary 被错误复用。可用 `TARIFF_OPENCL_CACHE_FILE` 指定绝对路径，默认写在 feature MNN 所在目录。

## 最小回归矩阵

每次改动至少执行：

```powershell
$py = 'D:\60-fps-Project\Projects\RIFE GUI\venv\Scripts\python.exe'
$env:PYTHONPATH = 'D:\60-fps-Project\Projects\RIFE GUI'
& $py 'D:\60-fps-Project\Projects\RIFE GUI\VFI_uc\TariffDynamic\validate_gui_adapter.py'
```

该脚本一次只验证一个模型和一个分辨率；通过分别设置 `TARIFF_GUI_MODEL`、`TARIFF_GUI_WIDTH`、`TARIFF_GUI_HEIGHT` 覆盖 NB202/PWR、`960x576`、`511x383` 和其它非对齐尺寸，检查输出 shape、dtype、finite、非全零、x2/x4 中点一致性，并写出可视化 PNG。随后再执行 ONNXRuntime feature、CPU/OpenCL softsplat 对照和 `tariff.py` 的多分辨率回归。
