# CavityFlowNumpyOnly — 基于 NumPy FFT 的不可压缩 Navier-Stokes 与 Cahn-Hilliard 求解器

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-fft%20solver-green)](https://numpy.org/)

一套**仅依赖 NumPy / CuPy / PyTorch（无第三方 CFD 库）**的不可压缩 Navier-Stokes 与 Cahn-Hilliard 方程有限差分数值求解器。支持 2D / 3D 方腔驱动流、圆柱绕流、Cahn-Hilliard 相场模拟，并提供基于制造解的精度阶验证脚本。

---

## 目录

- [核心特性](#核心特性)
- [项目文件结构](#项目文件结构)
- [依赖安装](#依赖安装)
- [快速开始](#快速开始)
  - [1. 二维方腔驱动流 (2D Lid-Driven Cavity Flow)](#1-二维方腔驱动流-2d-lid-driven-cavity-flow)
  - [2. 三维方腔驱动流 (3D Lid-Driven Cavity Flow)](#2-三维方腔驱动流-3d-lid-driven-cavity-flow)
  - [3. 圆柱绕流 (Cylinder Flow)](#3-圆柱绕流-cylinder-flow)
  - [4. Cahn-Hilliard 二维相场模拟](#4-cahn-hilliard-二维相场模拟)
  - [5. Cahn-Hilliard 三维相场模拟](#5-cahn-hilliard-三维相场模拟)
  - [6. 精度验证脚本](#6-精度验证脚本)
- [数值方法](#数值方法)
- [核心算法详解](#核心算法详解)
- [结果展示](#结果展示)
- [参数说明](#参数说明)
- [引用参考](#引用参考)

---

## 核心特性

- 🚀 **纯 NumPy 实现** — 线性方程求解全部替换为 NumPy FFT / DCT / DST 谱方法，无需组装大型稀疏矩阵
- 🎮 **GPU 加速** — 提供 CuPy 版本，利用 GPU 批处理同时模拟多组雷诺数
- 📐 **高精度格式** — 支持 Euler 显式（1 阶）、Crank-Nicolson（2 阶）和 RK3（3 阶）时间推进
- 🧪 **制造解验证** — 内置 Taylor 涡等解析解，自动计算误差收敛阶
- 🖼️ **内置可视化** — 直接输出流线图、涡旋检测、等值面渲染

---

## 项目文件结构

```
CavityFlowNumpyOnly/
├── 方腔驱动流 (Lid-Driven Cavity Flow)
│   ├── Euler_Forward.py              # 1 阶 Euler 前向格式 (CPU, NumPy)
│   ├── stable_cavity_flow.py         # 2 阶 Crank-Nicolson + 3 阶 RK3 (CPU)
│   ├── Stable_CavityFlow_2or3order.py # 整合版: CN + RK3 双格式 (CPU)
│   ├── gpuCavityFlow.py              # GPU 批处理版 (CuPy), 多 Re 同时计算
│   └── cavity3d.py                   # 3D 方腔流 (PyTorch)
│
├── 圆柱绕流 (Cylinder Flow)
│   ├── cpuCylinder.py                # 2D 圆柱绕流 (CPU, NumPy)
│   └── maskCylinder.py              # GPU 批处理版 (CuPy) + Q-判据涡识别
│
├── Cahn-Hilliard 相场
│   ├── cpuCahnHilliard.py            # 2D Cahn-Hilliard (CPU, scipy DCT)
│   ├── gpuCahnHilliard.py            # 2D Cahn-Hilliard (GPU, CuPy DCT)
│   ├── cpuCahnHilliard3d.py          # 3D Cahn-Hilliard (CPU, Gauss-Seidel)
│   └── gpuCahnHilliard3d.py          # 3D Cahn-Hilliard (GPU, CuPy)
│
├── 精度验证 (Verification)
│   ├── firstOrderCheck.py            # NS 求解器制造解验证 (CNAB / RK3)
│   ├── secondOrderCheck.py           # 非线性迭代 + SAV 格式验证
│   ├── testFunc.py                   # Taylor 涡旋 + 多组制造解
│   └── helmholtzSolver.py            # Helmholtz 稀疏求解器 + FFT 谱求解器
│
├── 辅助
│   └── neumman_3d.py                 # 3D Neumann Poisson Kronecker 谱求解器
│
└── 输出样例
    ├── 79.png / cylinder.mp4          # 圆柱绕流结果
    ├── cavity3d3.png                  # 3D 方腔流结果
    ├── bubble_merge.gif / .mp4        # Cahn-Hilliard 2D 气泡融合
    └── cahn_hilliard_3d.jpg           # Cahn-Hilliard 3D 等值面
```

---

## 依赖安装

### CPU 版本（基础）

```bash
pip install numpy scipy matplotlib tqdm opencv-python scikit-image
```

### GPU 加速（可选）

```bash
# CuPy (NVIDIA GPU) — 用于 2D 腔流、圆柱绕流、Cahn-Hilliard 2D/3D
pip install cupy-cuda12x

# PyTorch (任意 GPU) — 用于 3D 方腔流
pip install torch torchvision
```

| 文件 | 所需依赖 |
|------|---------|
| `Euler_Forward.py`, `stable_cavity_flow.py`, `Stable_CavityFlow_2or3order.py` | NumPy, Matplotlib |
| `gpuCavityFlow.py`, `maskCylinder.py`, `gpuCahnHilliard.py`, `gpuCahnHilliard3d.py` | CuPy, NumPy, Matplotlib |
| `cavity3d.py`, `neumman_3d.py` | PyTorch, NumPy |
| `cpuCylinder.py` | NumPy, Matplotlib, OpenCV |
| `cpuCahnHilliard.py` | NumPy, SciPy, Matplotlib |
| `cpuCahnHilliard3d.py` | NumPy, SciPy, scikit-image, Matplotlib |
| `firstOrderCheck.py` | NumPy, PyTorch (可选), Matplotlib |
| `secondOrderCheck.py`, `helmholtzSolver.py` | NumPy, SciPy, Pandas |

---

## 快速开始

### 1. 二维方腔驱动流 (2D Lid-Driven Cavity Flow)

方腔驱动流是 CFD 经典基准算例：正方形腔体顶盖以恒定速度 $U_{top} = 1$ 驱动内部流体。

#### CPU — 1 阶 Euler 格式

```bash
python Euler_Forward.py
```

修改参数：编辑文件末尾 `__main__` 中的 `Re`, `Nx`, `Ny`, `dt`, `nt`。

#### CPU — 2 阶 Crank-Nicolson / 3 阶 RK3

```bash
python stable_cavity_flow.py
```

核心参数（在 `__main__` 中修改）：

```python
Re = 1000        # 雷诺数 (建议 100, 400, 1000, 3200, 5000, 10000)
Nx, Ny = 80, 80  # 网格分辨率
dt = 0.001        # 时间步长
nt = 20000        # 总时间步数
recordRate = 250  # 每隔多少步输出流线图
```

**切换时间格式**：在第 368-369 行，注释/取消注释以切换 CNAB 或 RK3：

```python
u, v, p = updateVelocityField_CNAB_bctop(...)   # Crank-Nicolson (2 阶)
# u, v, p = updateVelocityField_RK3_bctop(...)  # RK3 (3 阶)
```

#### GPU — CuPy 多雷诺数并行批处理

```bash
python gpuCavityFlow.py
```

可同时模拟多个雷诺数（如 100, 1000, 10000）：

```python
Re = cp.expand_dims(cp.round(cp.logspace(2, 4, 3)).reshape(-1, 1), axis=2)
# 生成 [100, 1000, 10000] 三个雷诺数
Nx, Ny = 256, 256   # GPU 可使用更高分辨率
nt = 200000
dt = 0.0001
```

结果自动保存到 `cavityFlow_inference/<Re>/T_*.png`。

---

### 2. 三维方腔驱动流 (3D Lid-Driven Cavity Flow)

使用 PyTorch + Kronecker 积谱方法求解 3D 不可压缩 Navier-Stokes 方程。

```bash
python cavity3d.py
```

关键参数：

```python
N = 128           # 网格分辨率 (N³)
Re = 100          # 雷诺数
dt = 0.0002       # 时间步长
inner_epoch = 20  # 每时间步内迭代次数
timeSteps = 200   # 总时间步
```

输出：`resTemp.npz` — 包含 `U, V, W` 三个速度分量，可用 ParaView 或 MATLAB 进行三维可视化。

> 详细 3D 可视化教程见 [知乎专栏](https://zhuanlan.zhihu.com/p/852110231)

---

### 3. 圆柱绕流 (Cylinder Flow)

模拟流体绕过圆柱的经典钝体绕流问题，支持涡旋识别（Q-判据 + $\lambda_2$ 判据）。

#### CPU 版本

```bash
python cpuCylinder.py
```

核心参数：

```python
Re = 1000         # 雷诺数
Lx, Ly = 5.0, 1.0 # 流域尺寸 (长 × 高)
Nx, Ny = 400, 80  # 网格分辨率
dt = 0.001
nt = 20000
# 圆柱位置和半径在 obstacle_mask 中设置:
# (xce[j] - 0.75)**2 + (yce[i] - 0.5)**2 < 0.2**2
```

内部使用障碍物掩码（`obstacle_mask`）标记圆柱区域，自动处理无滑移边界条件。

#### GPU 版本（带涡旋检测）

```bash
python maskCylinder.py
```

GPU 版本额外包含：
- **Q-判据**涡旋识别（红色散点标注涡心）
- 涡心坐标和 $\lambda_2$ 值输出到 `maskedCylinder.txt`
- 多雷诺数并行批处理

结果保存到 `CylinderLoader/` 目录。

---

### 4. Cahn-Hilliard 二维相场模拟

使用谱方法（DCT）求解 Cahn-Hilliard 方程，模拟二元混合物的相分离（spinodal decomposition）。

#### CPU 版本

```bash
python cpuCahnHilliard.py
```

核心参数：

```python
N = 512           # 单维网格数 (总自由度 (N+1)²)
T = 2500          # 时间步数
ep = 0.01         # 界面宽度参数 ε
mu = 1            # μ = dt / dx²
seed = 0          # 随机种子
k = 1             # 初始条件: 1=随机, 2=余弦波
saveInterval = 10 # 帧保存间隔
```

结果自动保存到 `CahnHilliard_frames/` 目录。

#### GPU 版本（推荐用于高分辨率）

```bash
python gpuCahnHilliard.py
```

GPU 版本支持三种初始条件：

| k | 初始条件 | 物理意义 |
|---|---------|---------|
| 1 | 随机扰动 | 旋节线分解 (spinodal decomposition) |
| 2 | 余弦波 | 规则相分离图案 |
| 3 | 双气泡 | 两气泡融合 (Ostwald ripening) |

```python
N = 512
T = 200000
ep = 0.01
mu = 1
k = 3             # 双气泡融合
saveInterval = 1000
```

双气泡融合效果见 `bubble_merge.gif`。

---

### 5. Cahn-Hilliard 三维相场模拟

使用 Gauss-Seidel 迭代求解 3D Cahn-Hilliard 方程组，并用 Marching Cubes 算法提取等值面进行三维可视化。

#### CPU 版本

```bash
python cpuCahnHilliard3d.py
```

关键参数：

```python
h = 1/64          # 网格间距 (N=64)
m = 2.5           # 界面参数
eps = h * m / (2 * sqrt(2) * arctanh(0.9))  # 界面宽度
dt = 0.1 * h      # 时间步长
# 外循环 2000 步, 内循环 (Gauss-Seidel) 50 次
```

输出保存到 `3dCahnHilliard_CUBE/` 目录，每个等值面截图约在每 10 个外迭代输出一次。

#### GPU 版本（推荐）

```bash
python gpuCahnHilliard3d.py
```

GPU 版本使用 CuPy 加速 Gauss-Seidel 迭代：

```python
h = 1/96          # 更高分辨率
eps = 0.01
dt = 0.1 * h
# 外循环 6600 步, 每 66 步输出一次等值面
```

---

### 6. 精度验证脚本

采用**制造解方法（Method of Manufactured Solutions, MMS）**验证求解器的数值精度阶。

#### 6.1 一阶/二阶精度验证 (`firstOrderCheck.py`)

验证 CNAB (2 阶) 和 RK3 (3 阶) 格式的收敛阶：

```bash
python firstOrderCheck.py
```

核心函数 `accuracy_validate()`：

```python
N = 512
tEnd = 1.0
dt = 2.0e-3
Re = 10.0
timeScheme = "CNAB"  # 或 "RK3"
```

该脚本：
1. 使用制造解（已知解析解 $u_{sol}, v_{sol}, p_{sol}$）和对应体积力 $f_u, f_v$
2. 以不同步长 $\Delta t$ 和 $\Delta t/2$ 运行求解器
3. 输出空间逐点的 $L_2$ 误差和最大误差
4. 通过 Richardson 外推计算收敛阶：

$$p = \log_2\left(\frac{\|e(\Delta t)\|}{\|e(\Delta t/2)\|}\right)$$

误差记录到 `firstOrder_tau*.txt`。

#### 6.2 非线性迭代与 SAV 格式验证 (`secondOrderCheck.py`)

支持两种高阶时间推进验证：

```bash
python secondOrderCheck.py
```

| 格式 | 函数 | 特点 |
|------|------|------|
| Nonlinear CNAB | `nonlinearMain()` | 每步内非线性迭代至收敛 |
| SAV (Scalar Auxiliary Variable) | `savMain()` | 能量稳定, 无条件梯度有界 |

```python
a = 2 * np.pi
Re = 10.0
N = 512
dt = 2.0e-3
T = 1.0
```

误差自动记录为 CSV 文件，包含 `timePoint, mseU, mseV, maxU, maxV` 等字段。

#### 6.3 辅助模块

- **`testFunc.py`** — `TaylorVortex` 类提供多组制造解（Taylor 涡、多项式解等），支持 `query(t, mode)` 查询任意时刻的 $u, v, p, f_u, f_v$
- **`helmholtzSolver.py`** — 提供稀疏 LU 分解和 FFT 两种 Helmholtz 方程求解器
- **`neumman_3d.py`** — `KroneckerBMM` 类：3D Neumann 边界条件 Poisson 方程的 Kronecker 积谱求解器

---

## 数值方法

### 空间离散

- **交错网格 (MAC Grid)**: $u$ 位于 $x$ 方向面心，$v$ 位于 $y$ 方向面心，$p$ 位于体心
- **对流项**: 二阶中心差分，动量守恒形式
- **扩散项**: 二阶中心差分

### 时间推进 — 投影法 (Projection Method)

采用分步投影法 (fractional-step / Chorin's projection)：

1. **预测步**: 忽略压力梯度，求解中间速度 $u^*, v^*$
   - Euler 显式（1 阶）
   - Crank-Nicolson + Adams-Bashforth (CNAB, 2 阶)
   - 低存储 RK3 (3 阶)
2. **压力 Poisson 方程**: 使用 DCT-II/DCT-III 谱方法直接求解
3. **校正步**: 利用压力梯度修正速度场，满足散度为零

$$
\begin{cases}
\frac{u^* - u^n}{\Delta t} = -\frac{3}{2}N(u^n) + \frac{1}{2}N(u^{n-1}) + \frac{1}{2Re}\nabla^2(u^* + u^n) \\
\nabla^2 p^{n+1} = \frac{1}{\Delta t}\nabla \cdot u^* \\
u^{n+1} = u^* - \Delta t \nabla p^{n+1}
\end{cases}
$$

### 压力 Poisson 方程求解

二维（Neumann 边界条件）：

```python
solvePoissonEquation_2dDCT(b, h)
```

使用 DCT-II / DCT-III 的 FFT 实现，复杂度 $O(N^2 \log N)$，无需迭代。

三维：使用 Kronecker 积分解耦 + 谱方法 (`KroneckerBMM`)。

### Cahn-Hilliard 求解

采用谱方法（DCT-I）求解 Cahn-Hilliard 方程（Neumann 边界条件）：

$$\frac{\partial c}{\partial t} = \nabla^2(c^3 - c - \varepsilon^2 \nabla^2 c)$$

在谱空间中解耦得到显式递推公式，避免非线性迭代。

---

## 结果展示

| 问题 | 效果 |
|------|------|
| 2D 方腔流 Re=1000 | 一次涡 + 二次角涡 |
| 3D 方腔流 | 三维涡结构 (ParaView 可视化) |
| 圆柱绕流 Re=1000 | 卡门涡街 |
| Cahn-Hilliard 2D | 旋节线分解 / 气泡融合 |
| Cahn-Hilliard 3D | 双连续结构等值面 |

---

## 参数说明

### 通用 NS 求解器参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `Re` | 雷诺数 $Re = UL/\nu$ | 100, 400, 1000, 10000 |
| `Nx, Ny` | $x, y$ 方向网格数 | 64 ~ 512 |
| `dt` | 时间步长 | 0.0001 ~ 0.01 |
| `nt` | 总时间步数 | 1000 ~ 200000 |
| `bctop` | 顶盖驱动速度 | 通常为 1.0 |
| `Lx, Ly` | 流域尺寸 | 通常为 1×1 (腔流), 5×1 (绕流) |

### Cahn-Hilliard 参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `ep` ($\varepsilon$) | 界面宽度 | 0.01 ~ 0.02 |
| `mu` | $\mu = \Delta t / \Delta x^2$ | 1.0 |
| `k` | 初始条件选择 | 1 (随机), 2 (余弦), 3 (气泡) |
| `N` | 单维网格数 | 128 ~ 512 |

### CFL 条件

为保证数值稳定性，建议满足：

$$\text{CFL} = \max\left(\frac{|u|}{\Delta x}, \frac{|v|}{\Delta y}\right) \cdot \Delta t \lesssim 0.5$$

对于显式处理的对流项尤其重要。GPU 版本中的 `dt` 通常取更小值（如 0.0001）以匹配更高分辨率。

---

## 引用参考

- **原始 MATLAB 实现**: [2D-Lid-Driven-Cavity-Flow-Incompressible-Navier-Stokes-Solver](https://github.com/mathworks/2D-Lid-Driven-Cavity-Flow-Incompressible-Navier-Stokes-Solver.git)
- **Cahn-Hilliard 数值方法**: "Numerical Methods for the Cahn-Hilliard Equation" by Matthew Geleta
- **投影法**: Chorin, A. J. (1968). Numerical solution of the Navier-Stokes equations.
- **3D 可视化教程**: [知乎专栏](https://zhuanlan.zhihu.com/p/852110231)

---

## License

MIT License — 详见仓库 LICENSE 文件。
