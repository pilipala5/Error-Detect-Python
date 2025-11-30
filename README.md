# Lpvs 粗差检测（Python 版）

本目录是对 `Error_Detect` 中 Lpvs（METHOD_ID=1）算法的纯 Python 改写，只关注 `Error_Detect-Python/data` 下的输入数据，并同时输出 **1-2 图、2-3 图及合并** 的粗差点列表。筛选标准在原 `OUTPUT_FORM` 基础上优化为 **4-100σ** 范围。

## 环境准备

```bash
# 安装依赖
pip install numpy
```

## 运行

```bash
# 生成粗差检测结果
python main.py
```

运行后会在 `Error_Detect-Python/data/lpvs_result.txt` 生成检测结果，同时在终端提示输出路径。

## 数据与文件说明

- `data/PRECAI.DAT`：相机内方位元素（仅使用第 2 行的 f、x0、y0）。
- `data/PRECKI.DAT`：地面控制点文件，读取但算法不直接使用。
- `data/PREPHIy10.DAT`：像点观测，按出现顺序包含三张影像；程序默认组合为 1-2 图、2-3 图。
- `main.py`：入口脚本，调度两对影像并生成结果。
- `error_detection.py`：Lpvs 实现，保留原 C++ 逻辑、仅输出 4-100σ 粗差点。

## 核心流程（与原 C++ 对应）

1. 读取内方位与像点，查找公共点。
2. 以 Bx=1 初始化，迭代解算相对定向（phi、omega、kappa、u、v），内层收敛阈值为 `|Δ| < 3e-5`。
3. 依据 Lpvs 重新选权，权值低于 0.001 的点记为粗差候选；外层权迭代变化小于 0.001 或循环超过 30 次时停止。
4. 在最终姿态下计算像空间 Y 向残差，换算为 σ（除以 m0=2.8e-3 mm），并筛选 4-100σ 之间的粗差点。
5. 分别输出 1-2、2-3 结果及合并表，写入 `data/lpvs_result.txt`。
