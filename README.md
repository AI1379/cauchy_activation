# Cauchy Activation & Residual Mixer

Python >= 3.12 的机器学习项目，对比 **Cauchy 残差混合** 与标准残差在曲线拟合和 PDE 求解中的表现。

## 项目结构

```
cauchy_activation/
├── src/
│   └── cauchy_res_mixer/           # 核心模块
│       ├── __init__.py
│       ├── model.py                 # CauchyResidualMLP 网络定义
│       └── train_utils.py           # 训练工具函数
├── notebooks/
│   └── cauchy_res_mixer/
│       ├── train_test_cauchy_res_mixer.ipynb        # 实验 1：曲线拟合
│       └── train_test_cauchy_res_mixer_pde.ipynb    # 实验 2：PDE PINN 求解
├── pyproject.toml                   # 项目配置与依赖
└── README.md                        # 本文件
```

## 安装

本项目使用 **uv** 包管理工具。如未安装，请参考 [uv 文档](https://docs.astral.sh/uv/)。

### 1. 克隆或准备项目

```bash
cd /path/to/cauchy_activation
```

### 2. 安装依赖

项目提供三种 PyTorch 安装选项：

#### 选项 A：CPU 版本

```bash
uv sync --extra cpu
```

#### 选项 B：CUDA 12.8 版本

```bash
uv sync --extra cu128
```

#### 选项 C：CUDA 13.0 版本

```bash
uv sync --extra cu130
```

同时安装开发依赖（包含 Jupyter）：

```bash
uv sync --extra cpu --with dev
# 或其他 PyTorch 选项
```

### 3. 激活虚拟环境（可选）

使用 uv 时通常无需手动激活，但如果需要手动使用：

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

## 运行 Notebooks

### 方式 1：VS Code（推荐）

1. 用 VS Code 打开项目根目录
2. 打开任一 notebook 文件：
   - `notebooks/cauchy_res_mixer/train_test_cauchy_res_mixer.ipynb`
   - `notebooks/cauchy_res_mixer/train_test_cauchy_res_mixer_pde.ipynb`
3. 选择正确的 Python 内核（应自动指向虚拟环境）
4. 逐个单元执行或全选运行

### 方式 2：Jupyter 命令行

```bash
# 启动 Jupyter Lab
jupyter lab

# 或启动 Jupyter Notebook
jupyter notebook
```

然后在浏览器中导航到对应 notebook 文件打开。

## Notebook 说明

### 实验 1：曲线拟合 `train_test_cauchy_res_mixer.ipynb`

**目标函数**：
$$z(x) = \sin(x)\cos(y) + 0.1\sin(5x)\cos(5y) + \sin(x^2 + y^2)$$

**测试对象**：

- 4 种网络架构：`(残差方式) × (激活函数)`
  - 标准残差 + Cauchy 激活
  - Cauchy 残差 + Cauchy 激活
  - 标准残差 + ReLU
  - Cauchy 残差 + ReLU
- 网络深度：30 层
- 隐层维度：128

**输出**：

- 训练/验证损失曲线对比
- 各模型在稠密网格上的 MSE 误差统计

---

### 实验 2：PDE PINN 求解 `train_test_cauchy_res_mixer_pde.ipynb`

**目标 PDE**（Poisson 方程）：
$$-\Delta u(x,y) = 2\pi^2\sin(\pi x)\sin(\pi y), \quad (x,y) \in [-1,1]^2$$

**边界条件**：
$$u(x,y) = 0, \quad (x,y) \in \partial[-1,1]^2$$

**解析解**：
$$u^*(x,y) = \sin(\pi x)\sin(\pi y)$$

**损失函数** (PINN 风格)：
$$L = L_{pde} + w_{bc} \cdot L_{bc}$$

其中：

- $L_{pde}$ 为 PDE 残差的 MSE（含自动微分二阶导）
- $L_{bc}$ 为边界条件的 MSE

**测试对象**：

- 同样 4 种网络架构
- 网络深度：30 层
- 隐层维度：128

**输出**：

- PINN 总损失、PDE 损失、边界损失曲线
- 各模型与解析解的 MSE 误差对比
- 最优模型与解析解的场图及误差热力图

---

## 依赖概览

关键依赖：

| 包 | 用途 |
|------|------|
| `torch` | 深度学习框架 |
| `numpy` | 科学计算 |
| `matplotlib` | 图表绘制 |
| `tqdm` | 进度条 |
| `scikit-learn` | 数据分割工具 |
| `jupyter` | Notebook 执行器 |

## 常见问题

### Q: Jupyter kernel 找不到？

**A**: 确保已用 `uv sync --with dev` 安装开发依赖，VS Code 中手动选择正确的 Python 解释器。

### Q: CUDA 相关错误？

**A**: 检查 CUDA 版本是否匹配（cu128 需 CUDA 12.8，cu130 需 CUDA 13.0），或选择 CPU 版本。

### Q: 导入错误 `ModuleNotFoundError: cauchy_res_mixer`？

**A**: 确保在项目根目录运行，且 `src/` 已添加到 Python 路径（Jupyter 会自动处理）。

### Q: Notebook 运行速度慢？

**A**:

- 检查是否在 GPU 上运行（输出会显示 `device`）
- 降低 `epochs` 或 `n_interior`、`n_boundary_each` 参数快速测试

## 修改与扩展

### 调整超参数

在 Notebook 中查找以下行并修改：

**曲线拟合**：

```python
epochs = 300
lr = 1e-5
```

**PDE 求解**：

```python
epochs = 1200
lr = 1e-3
w_bc = 20.0
```

### 更改采样点数 (PDE Only)

```python
n_interior = 4096
n_boundary_each = 1024
```

### 自定义 PDE

在 Notebook cell 3 中修改：

- `u_exact_numpy()` - 解析解
- `rhs_torch()` - 右端项
- `sample_boundary()` - 边界采样

## 许可证

未指定（请补充）
