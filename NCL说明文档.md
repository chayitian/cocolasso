# NCL (Nonconvex Lasso) 说明文档

## 概述

NCL (Nonconvex Lasso) 是基于 Datta & Zou (2017) 论文提出的测量误差模型方法，用于高维误差变量回归问题。本实现提供了完整的 NCL 算法及与 CoCoLasso 对比的模拟实验框架。

## 项目结构

```
cocolasso/
├── src/
│   ├── __init__.py          # 包入口，导出所有公共 API
│   ├── cocolasso.py         # CoCoLasso 核心实现
│   └── ncl.py               # NCL 核心实现
├── test/
│   ├── cocolasso_simulation.py  # CoCoLasso 模拟实验
│   └── ncl_simulation.py       # NCL 模拟实验
└── README.md
```

## 依赖

- Python >= 3.7
- NumPy >= 1.15.0
- Pandas

## 快速开始

### 导入模块

```python
from src import ncl_method, coco, cov_autoregressive
```

### 基本使用

```python
import numpy as np
from src import ncl_method, simulate_data

# 生成测试数据
data = simulate_data(n=200, p=200, tau=0.3, noise="additive", seed=42)
Z, y, X, beta = data["Z"], data["y"], data["X"], data["beta"]

# 运行 NCL
result = ncl_method(
    Z=Z, y=y, n=200, p=200, tau=0.3,
    noise="additive", K=5, step=100, n_R=100
)

beta_hat = result["beta"]
```

## API 文档

### `ncl_method` - NCL 主函数

```python
ncl_method(Z, y, n, p, tau, noise="additive", K=5, step=100, n_R=100, seed=42)
```

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `Z` | ndarray (n, p) | 观测到的带误差设计矩阵 |
| `y` | ndarray (n,) | 响应变量 |
| `n` | int | 样本数 |
| `p` | int | 特征数 |
| `tau` | float | 误差标准差 |
| `noise` | str | 误差类型：`"additive"` 或 `"multiplicative"` |
| `K` | int | 交叉验证折数 |
| `step` | int | lambda 路径长度 |
| `n_R` | int | L1 约束半径 R 的搜索点数 |
| `seed` | int | 随机种子 |

**返回值：**

```python
{
    "beta": ndarray,      # 估计系数
    "lambda": float,      # 最优 lambda
    "R": float            # 最优 L1 约束半径
}
```

### `ncl_coordinate_descent` - NCL 坐标下降

```python
ncl_coordinate_descent(Gamma, rho, lambda_val, R, p, beta_start=None, max_iter=1000, opt_tol=1e-5, zero_threshold=1e-6)
```

求解带 L1 约束的非凸优化问题：

$$\min_{\beta} \frac{1}{2}\beta^T\Gamma\beta - \rho^T\beta + \lambda\|\beta\|_1 \quad \text{s.t.} \quad \|\beta\|_1 \leq R$$

### `naive_lasso_cv` - 朴素 Lasso 交叉验证

用于初始化 NCL 的 R 参数。

### `compute_corrected_covariance_additive` - 加性误差修正协方差

```python
compute_corrected_covariance_additive(Z, y, n, tau)
```

计算：
- $\tilde{\Sigma} = \frac{1}{n}Z^TZ - \tau^2 I$
- $\tilde{\rho} = \frac{1}{n}Z^Ty$

## 模拟实验

### 实验设置

基于 Datta & Zou (2017) 的模拟设置：

| 参数 | 值 |
|------|-----|
| 样本数 n | 100 |
| 特征数 p | 250 |
| 真实系数 β* | (3, 1.5, 0, 0, 2, 0, ..., 0) |
| 噪声 σ | 3.0 |
| 蒙特卡洛重复 | 100 次 |
| Bootstrap 次数 | 500 次 |

### 协方差结构

- **AR (自回归)**: $\Sigma_{ij} = \rho^{|i-j|}$, $\rho=0.5$
- **CS (复合对称)**: $\Sigma_{ij} = \rho$ (非对角线), $\Sigma_{ii} = 1$, $\rho=0.5$

### 误差类型与 tau 值

| 误差类型 | tau 值 |
|---------|--------|
| 加性测量误差 | 0.75, 1.0, 1.25 |
| 乘性测量误差 | 0.25, 0.5, 0.75 |

### 评价指标

| 指标 | 说明 |
|------|------|
| **C** | 正确识别的非零系数数量（最大值 = 3） |
| **IC** | 错误识别的零系数数量（理想值 = 0） |
| **SE** | 平方误差 $\sum(\beta_j - \hat{\beta}_j)^2$ |
| **PE** | 预测误差 $(\beta - \hat{\beta})^T\Sigma(\beta - \hat{\beta})$ |

### 运行实验

```bash
# 完整实验 (100 次蒙特卡洛)
python test/ncl_simulation.py --mode full

# 快速测试 (3 次蒙特卡洛)
python test/ncl_simulation.py --mode quick

# 自定义参数
python test/ncl_simulation.py --mode full --n_mc 50 --n_bootstrap 200
```

### 输出结果

结果保存为 `test/ncl_simulation_results.csv`，格式如下：

| Method | CovType | ErrorType | Tau | C_median | C_se | IC_median | IC_se | SE_median | SE_se | PE_median | PE_se |
|--------|---------|-----------|-----|----------|------|-----------|-------|-----------|-------|-----------|-------|
| NCL | AR | additive | 0.75 | ... | ... | ... | ... | ... | ... | ... | ... |

## 与 CoCoLasso 对比

### 方法差异

| 特性 | NCL | CoCoLasso |
|------|-----|-----------|
| 优化目标 | 非凸 + L1 约束 | 凸修正 + L1 惩罚 |
| 参数选择 | 网格搜索 (lambda, R) | 交叉验证 (lambda) |
| 计算复杂度 | 较高（双层网格搜索） | 较低（单层路径算法） |
| 理论保证 | 需要已知 tau | 需要已知 tau |

### 对比实验

```python
from src import ncl_method, coco

# 相同数据下对比两种方法
ncl_result = ncl_method(Z, y, n, p, tau, noise="additive")
coco_result = coco(Z, y, n, p, tau=tau, noise="additive", block=False)

# 比较系数估计
print("NCL beta:", ncl_result["beta"][:5])
print("CoCoLasso beta:", coco_result["beta_opt"][:5])
```

## 注意事项

1. **tau 参数**：NCL 需要预先知道测量误差的标准差 tau
2. **计算时间**：NCL 的双层网格搜索（lambda × R）计算量较大
3. **numpy 版本**：需要 numpy >= 1.15.0（使用 `np.quantile`）
4. **Python 版本**：需要 Python >= 3.7（使用 `sys.stdout.reconfigure`）

## 参考文献

Datta, A., & Zou, H. (2017). CoCoLasso for High-dimensional Error-in-variables Regression. *The Annals of Statistics*, 45(6), 2421-2448.
