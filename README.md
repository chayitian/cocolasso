# BDcocolasso

参考 BDCoCoLasso 作者的 R 语言包实现 Python 的 BDcocolasso 并进行复现，之后拓展到 AdaCoCoLasso（自适应CoCoLasso）并进行复现。

BDCoCoLasso 作者的 R 语言包地址：https://github.com/celiaescribe/BDcocolasso.git


## 目录

1. [算法背景](#1-算法背景)
2. [问题设定](#2-问题设定)
3. [核心公式推导](#3-核心公式推导)
4. [算法流程](#4-算法流程)
5. [代码架构](#5-代码架构)
6. [核心函数详解](#6-核心函数详解)
7. [使用示例](#7-使用示例)
8. [参数说明](#8-参数说明)
9. [模拟实验](#9-模拟实验)
10. [参考文献](#10-参考文献)

---

## 1. 算法背景

**CoCoLasso**（Convex Corrected Lasso，凸修正Lasso）是一种高维误差变量回归算法，用于处理协变量中存在 **加性误差（additive error）** 或 **缺失数据（missing data）** 的情况。该算法由 Datta & Zou (2017) 提出，核心思想是对被噪声污染的协方差矩阵进行 **正半定投影（PSD projection）** 修正，然后基于修正后的协方差矩阵进行 Lasso 回归。

**BD-CoCoLasso**（Block-Descent CoCoLasso，块下降CoCoLasso）是 CoCoLasso 的扩展版本，适用于只有部分特征被污染的场景。它利用块坐标下降（block coordinate descent）策略，将变量分为未污染块和污染块分别优化，从而显著降低高维场景下的计算开销。

**三块 BD-CoCoLasso** 是进一步的推广，同时处理加性误差和缺失数据混合存在的场景，将变量分为三个块：未污染变量、加性误差变量、缺失数据变量。

---

## 2. 问题设定

### 2.1 线性模型

考虑标准线性回归模型：

$$y = X\beta^* + \varepsilon$$

其中 $y \in \mathbb{R}^n$ 为响应向量，$X \in \mathbb{R}^{n \times p}$ 为真实设计矩阵，$\beta^* \in \mathbb{R}^p$ 为真实系数，$\varepsilon \in \mathbb{R}^n$ 为噪声项。

### 2.2 误差变量设定

我们无法观测到真实的 $X$，只能观测到被污染的矩阵 $Z$：

**加性误差设定**：

$$Z = X + W$$

其中 $W \in \mathbb{R}^{n \times p}$ 为加性误差矩阵，各元素独立同分布 $W_{ij} \sim \mathcal{N}(0, \tau^2)$。

**缺失数据设定**：

$$Z_{ij} = \begin{cases} X_{ij}, & \text{以概率 } 1 - \delta \\ \text{NA}, & \text{以概率 } \delta \end{cases}$$

其中 $\delta$ 为缺失概率。

### 2.3 Lasso 回归

标准 Lasso 估计量为：

$$\hat{\beta} = \arg\min_{\beta} \frac{1}{2n} \|y - X\beta\|_2^2 + \lambda \|\beta\|_1$$

等价地，可以写成**协方差形式**：

$$\hat{\beta} = \arg\min_{\beta} \frac{1}{2} \beta^\top \Sigma \beta - \rho^\top \beta + \lambda \|\beta\|_1$$

其中 $\Sigma = \frac{1}{n} X^\top X$，$\rho = \frac{1}{n} X^\top y$。

---

## 3. 核心公式推导

### 3.1 污染协方差矩阵的修正

由于我们只能观测到 $Z$ 而非 $X$，直接使用 $\hat{\Sigma} = \frac{1}{n} Z^\top Z$ 和 $\hat{\rho} = \frac{1}{n} Z^\top y$ 作为 $\Sigma$ 和 $\rho$ 的估计会导致偏差。

#### 加性误差修正

在加性误差设定下：

$$\mathbb{E}[\hat{\Sigma}] = \mathbb{E}\left[\frac{1}{n} Z^\top Z\right] = \Sigma + \tau^2 I_p$$

因此修正估计为：

$$\tilde{\Sigma} = \hat{\Sigma} - \tau^2 I_p = \frac{1}{n} Z^\top Z - \tau^2 I_p$$

$$\tilde{\rho} = \hat{\rho} = \frac{1}{n} Z^\top y$$

#### 缺失数据修正

在缺失数据设定下，先将缺失值替换为0（中心化后），定义观测比例矩阵 $R$：

$$R_{jk} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}[Z_{ij} \neq \text{NA}] \cdot \mathbf{1}[Z_{ik} \neq \text{NA}]$$

修正估计为：

$$\tilde{\Sigma}_{jk} = \frac{\hat{\Sigma}_{jk}}{R_{jk}}, \quad \tilde{\rho}_j = \frac{\hat{\rho}_j}{R_{jj}}$$

### 3.2 正半定投影

修正后的 $\tilde{\Sigma}$ **不一定是正半定矩阵**，而 Lasso 的协方差形式要求协方差矩阵正半定以保证凸性。因此需要对 $\tilde{\Sigma}$ 进行投影：

$$\Sigma^{\text{PSD}} = \Pi_{\text{PSD}}(\tilde{\Sigma}) = \arg\min_{R \succeq 0} \|R - \tilde{\Sigma}\|$$

本实现提供两种投影方法：

#### ADMM 投影（最大范数）

使用交替方向乘子法（ADMM），求解：

$$\min_{R, S} \|R - \tilde{\Sigma}\|_{\max} \quad \text{s.t.} \quad R \succeq 0, \quad R - S = \tilde{\Sigma}, \quad \|S\|_1 \leq \mu/2$$

ADMM 迭代步骤：

1. **R 步**（特征值投影）：

$$R^{(k+1)} = V \cdot \text{diag}(\max(D, \epsilon)) \cdot V^\top$$

其中 $W = \tilde{\Sigma} + S^{(k)} + \mu L^{(k)}$ 的特征值分解为 $W = V D V^\top$。

2. **S 步**（L1 球投影）：

$$S^{(k+1)}_{\text{lower}} = M_{\text{lower}} - \text{L1Proj}(M_{\text{lower}}, \mu/2)$$

其中 $M = R^{(k+1)} - \tilde{\Sigma} - \mu L^{(k)}$。

3. **L 步**（对偶变量更新）：

$$L^{(k+1)} = L^{(k)} - \frac{R^{(k+1)} - S^{(k+1)} - \tilde{\Sigma}}{\mu}$$

每20次迭代将 $\mu$ 减半以加速收敛。

#### HM-Lasso 投影（Frobenius 范数）

基于 HM-Lasso 算法（High-dimensional Matrix Lasso），求解：

$$\min_{A \succeq 0} \|A - \tilde{\Sigma}\|_F$$

迭代步骤：

1. **A 步**（特征值投影）：

$$A^{(k+1)} = V \cdot \text{diag}(\max(D, \epsilon)) \cdot V^\top$$

2. **B 步**（Frobenius 范数下闭式解）：

$$B^{(k+1)} = \frac{A^{(k+1)} - \tilde{\Sigma} - \mu L^{(k)}}{\mu W \circ W + \mathbf{1}}$$

3. **L 步**：

$$L^{(k+1)} = L^{(k)} - \frac{A^{(k+1)} - B^{(k+1)} - \tilde{\Sigma}}{\mu}$$

### 3.3 协方差形式 Lasso（坐标下降法）

给定修正后的正半定协方差矩阵 $\Sigma^{\text{PSD}}$ 和修正后的 $\rho$，求解：

$$\hat{\beta} = \arg\min_{\beta} \frac{1}{2} \beta^\top \Sigma^{\text{PSD}} \beta - \rho^\top \beta + \lambda \|\beta\|_1$$

使用**坐标下降法**迭代更新每个 $\beta_j$：

$$S_j = \sum_{k \neq j} \Sigma^{\text{PSD}}_{jk} \beta_k - \rho_j$$

- 若 $S_j > \lambda$：$\beta_j = \frac{\lambda - S_j}{\Sigma^{\text{PSD}}_{jj}}$
- 若 $S_j < -\lambda$：$\beta_j = \frac{-\lambda - S_j}{\Sigma^{\text{PSD}}_{jj}}$
- 若 $|S_j| \leq \lambda$：$\beta_j = 0$

#### SCAD 惩罚

SCAD（Smoothly Clipped Absolute Deviations）惩罚使用自适应权重 $w_j$：

$$w_j = \begin{cases} 1, & |\beta_j| \leq \lambda \\ \frac{a\lambda - |\beta_j|}{\lambda(a-1)}, & \lambda < |\beta_j| \leq a\lambda \\ 0, & |\beta_j| > a\lambda \end{cases}$$

其中 $a = 3.7$（默认值）。有效惩罚参数为 $\lambda_j^{\text{eff}} = w_j \cdot \lambda$。

### 3.4 BD-CoCoLasso（块坐标下降）

当设计矩阵 $Z$ 的前 $p_1$ 列为未污染变量、后 $p_2$ 列为污染变量时（$p = p_1 + p_2$），将 $\beta$ 分为 $\beta_1$ 和 $\beta_2$。

**关键思想**：未污染块使用标准 Lasso，污染块使用 CoCoLasso 修正。

目标函数：

$$Q(\beta_1, \beta_2) = \frac{1}{2} \begin{pmatrix} \beta_1 \\ \beta_2 \end{pmatrix}^\top \begin{pmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22}^{\text{PSD}} \end{pmatrix} \begin{pmatrix} \beta_1 \\ \beta_2 \end{pmatrix} - \begin{pmatrix} \rho_1 \\ \rho_2 \end{pmatrix}^\top \begin{pmatrix} \beta_1 \\ \beta_2 \end{pmatrix} + \lambda (\|\beta_1\|_1 + \|\beta_2\|_1)$$

其中 $\Sigma_{11} = \frac{1}{n} X_1^\top X_1$（未污染），$\Sigma_{22}^{\text{PSD}}$ 为污染块的修正正半定协方差矩阵。

**块坐标下降迭代**：

1. **更新 $\beta_1$**（固定 $\beta_2$）：

$$\beta_1^{(k+1)} = \arg\min_{\beta_1} \frac{1}{2} \beta_1^\top \Sigma_{11} \beta_1 + \beta_1^\top \Sigma_{12} \beta_2^{(k)} - \rho_1^\top \beta_1 + \lambda \|\beta_1\|_1$$

2. **更新 $\beta_2$**（固定 $\beta_1$）：

$$\beta_2^{(k+1)} = \arg\min_{\beta_2} \frac{1}{2} \beta_2^\top \Sigma_{22}^{\text{PSD}} \beta_2 + \beta_2^\top \Sigma_{21} \beta_1^{(k+1)} - \rho_2^\top \beta_2 + \lambda \|\beta_2\|_1$$

### 3.5 三块 BD-CoCoLasso

将变量分为三个块：

- $\beta_1$（$p_1$ 维）：未污染变量
- $\beta_2$（$p_2$ 维）：加性误差变量
- $\beta_3$（$p_3$ 维）：缺失数据变量

目标函数：

$$Q(\beta_1, \beta_2, \beta_3) = \frac{1}{2} \beta^\top \Sigma^{\text{block}} \beta - \rho^\top \beta + \lambda \|\beta\|_1$$

其中：

$$\Sigma^{\text{block}} = \begin{pmatrix} \Sigma_{11} & \Sigma_{12} & \Sigma_{13} \\ \Sigma_{21} & \Sigma_{22}^{\text{PSD,add}} & \Sigma_{23} \\ \Sigma_{31} & \Sigma_{32} & \Sigma_{33}^{\text{PSD,mis}} \end{pmatrix}$$

$\Sigma_{22}^{\text{PSD,add}}$ 为加性误差块的修正正半定矩阵，$\Sigma_{33}^{\text{PSD,mis}}$ 为缺失数据块的修正正半定矩阵。

### 3.6 交叉验证

使用 K 折交叉验证选择最优正则化参数 $\lambda$：

1. 将数据随机分为 K 折
2. 对每个 $\lambda$ 值和每折 $k$：
   - 在训练集上计算修正协方差矩阵并拟合模型
   - 在测试集上计算预测误差：

$$\text{CV-Error}(\lambda) = \frac{1}{K} \sum_{k=1}^{K} \left( \hat{\beta}_\lambda^{(-k)\top} \Sigma_{\text{test}}^{(k)} \hat{\beta}_\lambda^{(-k)} - 2 \rho_{\text{test}}^{(k)\top} \hat{\beta}_\lambda^{(-k)} \right)$$

3. 选择 $\lambda_{\text{opt}} = \arg\min_\lambda \text{CV-Error}(\lambda)$
4. 一倍标准差规则：$\lambda_{\text{sd}} = \max\{\lambda : \text{CV-Error}(\lambda) \leq \text{CV-Error}(\lambda_{\text{opt}}) + \text{SD}(\lambda_{\text{opt}})\}$

### 3.7 L1 球投影

ADMM 算法中需要将向量投影到 L1 球上。使用 Duchi et al. (2008) 的高效算法：

给定向量 $v$ 和半径 $b > 0$，投影 $w = \Pi_{\|\cdot\|_1 \leq b}(v)$ 的计算步骤：

1. 令 $u = \text{sort}(|v|)_{\text{降序}}$
2. 计算累积和 $sv_i = \sum_{j=1}^{i} u_j$
3. 找到 $\rho = \max\{j : u_j > \frac{sv_j - b}{j}\}$
4. 计算 $\theta = \max(0, \frac{sv_\rho - b}{\rho + 1})$
5. $w = \text{sign}(v) \cdot \max(|v| - \theta, 0)$

---

## 4. 算法流程

### 4.1 CoCoLasso 流程

```
输入: Z (n×p 污染矩阵), y (n维响应), 噪声类型, 参数
输出: 最优系数 β, 最优 λ

1. 数据预处理：中心化、标准化 Z 和 y
2. 计算修正协方差矩阵 Σ̃ 和修正相关向量 ρ̃
3. K折交叉验证：
   a. 对每折：
      - 计算训练集的修正协方差矩阵
      - PSD 投影（ADMM 或 HM）
      - 计算测试集的修正协方差矩阵
      - PSD 投影
4. 路径式坐标下降：
   对 λ 从 λ_max 到 λ_min：
     a. 对每折：用训练集拟合，在测试集计算误差
     b. 用全数据拟合当前 λ
     c. 记录误差和系数
     d. 早停检查
5. 选择最优 λ（λ_opt 和 λ_sd）
6. 返回结果
```

### 4.2 BD-CoCoLasso 流程

```
输入: Z (n×p), y (n维), p1, p2, 噪声类型, 参数
输出: 最优系数 β = (β₁, β₂), 最优 λ

1. 数据预处理
2. 将 Z 分为 X₁ (未污染) 和 Z₂ (污染)
3. K折交叉验证：
   a. 对每折：
      - 计算训练集的 Σ₁₁, Σ₂₂^PSD, Σ₁₂, ρ₁, ρ₂
      - 计算测试集的对应量
4. 块坐标下降：
   对 λ 从 λ_max 到 λ_min：
     a. 对每折：
        - 块坐标下降求解 β₁, β₂
        - 在测试集计算误差
     b. 用全数据块坐标下降拟合
     c. 记录误差和系数
     d. 早停检查
5. 选择最优 λ
6. 返回结果
```

### 4.3 三块 BD-CoCoLasso 流程

与 BD-CoCoLasso 类似，但将 Z 分为三个块：$X_1$（未污染）、$Z_2$（加性误差）、$Z_3$（缺失数据），分别计算各自的修正协方差矩阵，并进行三块坐标下降。

---

## 5. 代码架构

```
src/
├── __init__.py              # 包入口，导出公共 API（coco, generalcoco 函数式接口）
├── cocolasso.py             # CoCoLasso 估计器模块
│   ├── CoCoLasso            # 标准 CoCoLasso 估计器（sklearn 风格）
│   ├── _pathwise_coordinate_descent()          # CoCoLasso 主算法
│   ├── _cv_covariance_matrices()               # 标准交叉验证
│   └── _lambda_max()                           # 计算最大 λ 值
├── bdcocolasso.py           # BDCoCoLasso 估计器模块
│   ├── BDCoCoLasso          # 二块下降 CoCoLasso 估计器（sklearn 风格）
│   ├── _blockwise_coordinate_descent()         # BD-CoCoLasso 主算法
│   ├── _cv_covariance_matrices_block()         # 两块交叉验证
│   ├── _lasso_covariance_block()               # 两块协方差 Lasso
│   └── _lambda_max_block()                     # 块下降的最大 λ 值
├── generalcocolasso.py      # GeneralCoCoLasso 估计器模块
│   ├── GeneralCoCoLasso     # 三块混合 CoCoLasso 估计器（sklearn 风格）
│   ├── _blockwise_coordinate_descent_general() # 三块 BD-CoCoLasso 主算法
│   ├── _cv_covariance_matrices_block_general() # 三块交叉验证
│   └── _lasso_covariance_block_general()       # 三块协方差 Lasso
└── _utils.py                # 共享工具函数（内部模块）
    ├── _l1_proj()                        # L1 球投影
    ├── _admm_proj()                      # ADMM 正半定投影
    ├── _hm_proj()                        # HM-Lasso 正半定投影
    ├── _scad_weight()                    # SCAD 自适应权重
    ├── _lasso_covariance()               # 标准协方差 Lasso
    ├── _compute_ratio_matrix()           # 缺失数据比例矩阵
    ├── _log_space()                      # 对数空间生成
    ├── _corrected_covariance_multiplicative() # 乘性误差校正
    └── _preprocess_data()                # 数据预处理
```

---

## 6. 核心函数详解

### 6.1 `_l1_proj(v, b)`

**功能**：将向量 $v$ 投影到半径为 $b$ 的 L1 球上。

**算法**：Duchi et al. (2008) 高效投影算法。

**公式**：

$$w = \text{sign}(v) \cdot \max(|v| - \theta, 0)$$

其中 $\theta = \max\left(0, \frac{sv_\rho - b}{\rho + 1}\right)$，$\rho = \max\{j : u_j > \frac{sv_j - b}{j}\}$。

**代码实现**：

```python
def _l1_proj(v: np.ndarray, b: float) -> np.ndarray:
    u = np.sort(np.abs(v))[::-1]
    sv = np.cumsum(u)
    rho = np.max(np.where(u > (sv - b) / np.arange(1, len(u) + 1)))
    theta = max(0, (sv[rho] - b) / (rho + 1))
    w = np.sign(v) * np.maximum(np.abs(v) - theta, 0)
    return w
```

### 6.2 `_admm_proj(mat, epsilon, mu, it_max, etol, etol_distance)`

**功能**：使用 ADMM 算法找到距离给定矩阵最近的正半定矩阵（最大范数意义下）。

**优化问题**：

$$\min_{R, S} \|R - \text{mat}\|_{\max} \quad \text{s.t.} \quad R \succeq 0, \quad R - S = \text{mat}, \quad \|S\|_1 \leq \mu/2$$

**迭代步骤**：

1. $R^{(k+1)} = V \cdot \text{diag}(\max(D, \epsilon)) \cdot V^\top$，其中 $W = \text{mat} + S^{(k)} + \mu L^{(k)} = VDV^\top$
2. $S^{(k+1)} = M - \text{L1Proj}(M_{\text{lower}}, \mu/2)$，其中 $M = R^{(k+1)} - \text{mat} - \mu L^{(k)}$
3. $L^{(k+1)} = L^{(k)} - (R^{(k+1)} - S^{(k+1)} - \text{mat}) / \mu$

**收敛条件**：$\|R^{(k+1)} - R^{(k)}\|_{\max} < \epsilon$ 且 $\|S^{(k+1)} - S^{(k)}\|_{\max} < \epsilon$ 且 $\|R^{(k+1)} - S^{(k+1)} - \text{mat}\|_{\max} < \epsilon$。

### 6.3 `_hm_proj(\hat{\Sigma}, R, a, iter_max, epsilon, mu, tolerance, norm)`

**功能**：使用 HM-Lasso 算法找到距离给定矩阵最近的正半定矩阵（Frobenius 范数意义下）。

**迭代步骤**：

1. **A 步**（特征值投影）：

$$A^{(k+1)} = V \cdot \text{diag}(\max(D, \epsilon)) \cdot V^\top$$

其中 $W = B^{(k)} + \hat{\Sigma} + \mu L^{(k)} = VDV^\top$

2. **B 步**（Frobenius 范数下闭式解）：

$$B^{(k+1)} = \frac{A^{(k+1)} - \hat{\Sigma} - \mu L^{(k)}}{\mu W \circ W + \mathbf{1}}$$

3. **L 步**（对偶变量更新）：

$$L^{(k+1)} = L^{(k)} - \frac{A^{(k+1)} - B^{(k+1)} - \hat{\Sigma}}{\mu}$$

### 6.4 `_lasso_covariance(n, p, lambda, control, XX, Xy, beta.start, penalty)`

**功能**：求解协方差形式的 Lasso 问题。

**优化问题**：

$$\hat{\beta} = \arg\min_{\beta} \frac{1}{2} \beta^\top \Sigma \beta - \rho^\top \beta + \lambda \|\beta\|_1$$

**坐标下降更新规则**：

对每个 $j = 1, \ldots, p$：

$$S_j = \sum_{k \neq j} \Sigma_{jk} \beta_k - \rho_j$$

- 若 $S_j > \lambda$：$\beta_j = \frac{\lambda - S_j}{\Sigma_{jj}}$
- 若 $S_j < -\lambda$：$\beta_j = \frac{-\lambda - S_j}{\Sigma_{jj}}$
- 若 $|S_j| \leq \lambda$：$\beta_j = 0$

### 6.5 `_lasso_covariance_block(n, p1, p2, X1, Z2, y, sigma1, sigma2, lambda_val, ...)`

**功能**：两块 BD-CoCoLasso 的子问题求解。

**优化问题**：

交替优化 $\beta_1$ 和 $\beta_2$：

$$\beta_1^{(k+1)} = \arg\min_{\beta_1} \frac{1}{2} \beta_1^\top \Sigma_{11} \beta_1 + \beta_1^\top \Sigma_{12} \beta_2^{(k)} - \rho_1^\top \beta_1 + \lambda \|\beta_1\|_1$$

$$\beta_2^{(k+1)} = \arg\min_{\beta_2} \frac{1}{2} \beta_2^\top \Sigma_{22}^{\text{PSD}} \beta_2 + \beta_2^\top \Sigma_{21} \beta_1^{(k+1)} - \rho_2^\top \beta_2 + \lambda \|\beta_2\|_1$$

### 6.6 `_lasso_covariance_block_general(n, p1, p2, p3, X1, Z2, Z3, y, sigma1, sigma2, sigma3, ...)`

**功能**：三块 BD-CoCoLasso 的子问题求解，同时处理加性误差和缺失数据。

**优化问题**：

交替优化 $\beta_1$、$\beta_2$、$\beta_3$，其中 $\Sigma_{22}^{\text{PSD}}$ 对应加性误差修正，$\Sigma_{33}^{\text{PSD}}$ 对应缺失数据修正。

### 6.7 `_pathwise_coordinate_descent(Z, y, n, p, ...)`

**功能**：CoCoLasso 主算法，沿 $\lambda$ 路径进行坐标下降。

**流程**：
1. 预处理数据
2. 计算 $\lambda_{\max}$ 和 $\lambda_{\min}$
3. K 折交叉验证计算各折的修正协方差矩阵
4. 沿 $\lambda$ 路径迭代：
   - 交叉验证计算误差
   - 全数据拟合
   - 早停检查
5. 选择 $\lambda_{\text{opt}}$ 和 $\lambda_{\text{sd}}$

### 6.8 `_blockwise_coordinate_descent(Z, y, n, p, p1, p2, ...)`

**功能**：BD-CoCoLasso 主算法。

与 `_pathwise_coordinate_descent` 类似，但使用块坐标下降策略，分别处理未污染和污染变量块。

### 6.9 `_blockwise_coordinate_descent_general(Z, y, n, p, p1, p2, p3, ...)`

**功能**：三块 BD-CoCoLasso 主算法，处理混合误差场景。

### 6.10 `coco(Z, y, n, p, p1, p2, ..., block, ...)`

**功能**：CoCoLasso / BD-CoCoLasso 统一入口。

- `block=False`：调用 `_pathwise_coordinate_descent`（标准 CoCoLasso）
- `block=True`：调用 `_blockwise_coordinate_descent`（BD-CoCoLasso）

### 6.11 `generalcoco(Z, y, n, p, p1, p2, p3, ...)`

**功能**：三块 BD-CoCoLasso 入口，调用 `_blockwise_coordinate_descent_general`。

### 6.12 `simulate_data(n, p, beta, noise, tau, missing_rate, seed)`

**功能**：生成模拟数据用于测试。

- 自回归协方差矩阵：$\Sigma_{ij} = \rho^{|i-j|}$
- 默认真实系数：$\beta = (3, 2, 0, 0, 1.5, 0, \ldots, 0)$

---

## 7. 使用示例

### 7.1 sklearn 风格接口（推荐）

```python
from src import CoCoLasso, BDCoCoLasso, GeneralCoCoLasso
import numpy as np

# ---- 标准 CoCoLasso ----
model = CoCoLasso(tau=0.75, noise="additive", cv_folds=4)
model.fit(Z, y)
print(model.coef_)           # 最优系数
print(model.lambda_opt_)     # 最优 lambda
y_pred = model.predict(Z_new)
score = model.score(Z_test, y_test)  # R²

# ---- BD-CoCoLasso（部分特征含误差）----
model = BDCoCoLasso(p1=50, p2=200, tau=0.75, noise="additive")
model.fit(Z, y)

# ---- 三块混合 CoCoLasso ----
model = GeneralCoCoLasso(p1=20, p2=100, p3=130, tau=0.75)
model.fit(Z, y)

# ---- 与 sklearn 工具兼容 ----
from sklearn.model_selection import GridSearchCV
param_grid = {"tau": [0.5, 0.75, 1.0]}
search = GridSearchCV(CoCoLasso(noise="additive"), param_grid, cv=3)
search.fit(Z, y)
```

### 7.2 函数式接口（向后兼容）

```python
from src import coco, generalcoco

# 标准 CoCoLasso
result = coco(Z=Z, y=y, n=n, p=p, step=100, K=4, mu=10,
              tau=0.3, noise="additive", block=False)
print(result["beta_opt"])
print(result["lambda_opt"])

# BD-CoCoLasso
result = coco(Z=Z, y=y, n=n, p=p, p1=50, p2=200, step=100, K=4,
              mu=10, tau=0.3, noise="additive", block=True)

# 三块 BD-CoCoLasso
result = generalcoco(Z=Z, y=y, n=n, p=p, p1=20, p2=100, p3=130,
                     step=100, K=4, mu=10, tau=0.3)
```

### 7.3 SCAD 惩罚

```python
model = CoCoLasso(tau=0.75, noise="additive", penalty="SCAD")
model.fit(Z, y)
```

---

## 8. 参数说明

### 8.1 CoCoLasso 估计器参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `alpha` | float or None | None | 正则化强度参数。None 时通过交叉验证自动选取 |
| `noise` | str | "additive" | 噪声类型：`"additive"`、`"missing"`、`"multiplicative"` |
| `tau` | float or None | None | 加性/乘性误差的标准差 |
| `penalty` | str | "lasso" | 惩罚类型：`"lasso"` 或 `"SCAD"` |
| `mode` | str | "ADMM" | PSD 投影模式：`"ADMM"` 或 `"HM"` |
| `mu` | float | 10 | ADMM 惩罚参数 |
| `max_iter` | int | 100 | lambda 路径上的最大搜索步数 |
| `cv_folds` | int | 4 | 交叉验证折数 |
| `tol` | float | 1e-5 | 收敛容差 |
| `etol` | float | 1e-4 | ADMM 收敛容差 |
| `early_stopping_max` | int | 10 | 早停最大连续递增次数 |
| `lambda_factor` | float or None | None | lambda_min / lambda_max 的比率 |
| `center_Z` | bool | True | 是否对 Z 中心化 |
| `scale_Z` | bool | True | 是否对 Z 标准化 |
| `center_y` | bool | True | 是否对 y 中心化 |
| `scale_y` | bool | True | 是否对 y 标准化 |

### 8.2 BDCoCoLasso 额外参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p1` | int | 0 | 无误差特征的列数 |
| `p2` | int | 0 | 含误差特征的列数 |

### 8.3 GeneralCoCoLasso 额外参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p1` | int | 0 | 无误差特征的列数 |
| `p2` | int | 0 | 含加性误差特征的列数 |
| `p3` | int | 0 | 含缺失数据特征的列数 |

### 8.4 估计器属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `coef_` | ndarray (p,) | 最优系数向量 |
| `intercept_` | float | 截距 |
| `coef_sd_` | ndarray (p,) | 1-std 准则下的系数向量 |
| `lambda_opt_` | float | 交叉验证选出的最优 lambda |
| `lambda_sd_` | float | 1-std 准则下的 lambda |
| `cv_results_` | dict | 交叉验证误差详情 |
| `coef_path_` | dict | 系数路径 |
| `n_iter_` | int | 实际迭代步数 |

### 8.5 `coco()` 函数式接口参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Z` | ndarray | - | 污染设计矩阵 (n×p) |
| `y` | ndarray | - | 响应向量 (n维) |
| `n` | int | - | 样本数 |
| `p` | int | - | 特征数 |
| `p1` | int | None | 未污染特征数（block=True时必需） |
| `p2` | int | None | 污染特征数（block=True时必需） |
| `center_Z` | bool | True | 是否中心化 Z |
| `scale_Z` | bool | True | 是否标准化 Z |
| `center_y` | bool | True | 是否中心化 y |
| `scale_y` | bool | True | 是否标准化 y |
| `lambda_factor` | float | None | λ_min/λ_max 比值 |
| `step` | int | 100 | λ 值的数量 |
| `K` | int | 4 | 交叉验证折数 |
| `mu` | float | 10 | ADMM 惩罚参数 |
| `tau` | float | None | 加性误差标准差 |
| `etol` | float | 1e-4 | ADMM 收敛容差 |
| `optTol` | float | 1e-5 | 优化收敛容差 |
| `earlyStopping_max` | int | 10 | 误差上升时的最大迭代次数 |
| `noise` | str | "additive" | 噪声类型："additive" 或 "missing" |
| `block` | bool | True | True=BD-CoCoLasso, False=CoCoLasso |
| `penalty` | str | "lasso" | 惩罚类型："lasso" 或 "SCAD" |
| `mode` | str | "ADMM" | PSD 投影模式："ADMM" 或 "HM" |

### 8.6 `generalcoco()` 函数式接口参数

与 `coco()` 类似，额外参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p1` | int | 0 | 未污染特征数 |
| `p2` | int | 0 | 加性误差特征数 |
| `p3` | int | 0 | 缺失数据特征数 |

### 8.7 函数式接口返回值

| 键 | 说明 |
|----|------|
| `lambda_opt` | 最优 λ（最小交叉验证误差） |
| `lambda_sd` | 一倍标准差规则对应的 λ |
| `beta_opt` | λ_opt 对应的系数 |
| `beta_sd` | λ_sd 对应的系数 |
| `data_error` | 各 λ 值的交叉验证误差数据 |
| `data_beta` | 各 λ 值的系数路径 |
| `early_stopping` | 早停时的迭代步数 |
| `mean_Z` | Z 的列均值 |
| `sd_Z` | Z 的列标准差 |
| `mean_y` | y 的均值 |
| `sd_y` | y 的标准差 |

### 8.8 重要注意事项

1. **加性误差设定**：必须提供 `tau` 参数（加性误差的标准差）。建议在加性误差设定下设置 `center_Z=False`，因为中心化可能引入偏差。
2. **缺失数据设定**：`tau=None`。建议设置 `center_Z=True`，先将缺失值排除后中心化，再将缺失位置0。
3. **BD-CoCoLasso**：Z 的前 p1 列必须是未污染变量，后 p2 列必须是污染变量。
4. **三块设定**：Z 的列顺序为：未污染 → 加性误差 → 缺失数据。
5. **缺失率过高**：缺失率超过 0.7 可能导致算法失败（特征值分解出现无穷大或 NaN）。
6. **计算开销**：BD-CoCoLasso 在高维场景下计算开销较大，可通过调整 `mu`、降低 `etol` 或 `optTol`、减小 `earlyStopping_max` 来加速。

---

## 9. 模拟实验

### 9.1 实验概述

基于 CoCoLasso 论文 (Datta & Zou, 2017) 的模拟设置，对比 **CoCoLasso** 与 **NCL**（Nonconvex Lasso, Loh & Wainwright 2012）两种方法在不同测量误差场景下的表现。

### 9.2 实验设置

| 参数 | 设定值 |
|------|--------|
| 样本量 $n$ | 100 |
| 特征维度 $p$ | 250 |
| 真实系数 $\beta^*$ | $(3, 1.5, 0, 0, 2, 0, \ldots, 0)$ |
| 模型噪声 $\sigma$ | 3 |
| 蒙特卡洛重复次数 | 100 |
| 交叉验证折数 | 5 |

**协方差结构**：

- 自回归（AR）：$\Sigma_{X,ij} = 0.5^{|i-j|}$
- 复合对称（CS）：$\Sigma_{X,ij} = 0.5 + I(i=j) \cdot 0.5$

**加性测量误差**：$Z = X + A$，$A$ 的行独立同分布于 $N(0, \tau^2 I)$，$\tau \in \{0.75, 1.0, 1.25\}$

**乘性测量误差**：$Z = X \odot M$，$\log(m_{ij}) \sim N(0, \tau^2)$，$\tau \in \{0.25, 0.5, 0.75\}$

### 9.3 乘性误差的协方差修正

对于 $Z = X \odot M$，其中 $\log(M_{ij}) \sim N(0, \tau^2)$：

$$E[M_{ij}] = e^{\tau^2/2}, \quad E[M_{ij}^2] = e^{2\tau^2}$$

修正协方差矩阵：

$$\tilde{\Sigma}_{jk} = \frac{\frac{1}{n}\sum_{i} Z_{ij} Z_{ik}}{e^{\tau^2}}, \quad j \neq k$$

$$\tilde{\Sigma}_{jj} = \frac{\frac{1}{n}\sum_{i} Z_{ij}^2}{e^{2\tau^2}}$$

修正交叉协方差向量：

$$\tilde{\rho}_j = \frac{\frac{1}{n}\sum_{i} Z_{ij} y_i}{e^{\tau^2/2}}$$

### 9.4 NCL 方法

NCL 求解如下非凸优化问题：

$$\hat{\beta} = \arg\min_{\beta} \frac{1}{2} \beta^\top \hat{\Gamma} \beta - \hat{\rho}^\top \beta + \lambda \|\beta\|_1 \quad \text{s.t.} \quad \|\beta\|_1 \leq R$$

其中 $\hat{\Gamma}$ 为修正后的协方差矩阵（**不做 PSD 投影**，因此问题可能非凸），$R$ 为 $\ell_1$ 约束参数。

**NCL 调参流程**：

1. 在 $(Z, y)$ 上拟合朴素 Lasso（5折CV），得到初始估计 $\hat{\beta}_{\text{init}}$
2. 令 $R_{\max} = \|\hat{\beta}_{\text{init}}\|_1$
3. 在 $[R_{\max}/500, \, 2R_{\max}]$ 区间内取 100 个等距 $R$ 值
4. 对每个 $(R, \lambda)$ 组合进行 5 折交叉验证，选择最优参数对

**求解算法**：投影坐标下降法——每轮坐标下降后，若 $\|\beta\|_1 > R$，则投影到 $\ell_1$ 球上。

### 9.5 评价指标

| 指标 | 公式 | 含义 |
|------|------|------|
| C（正确选择数） | $\sum_{j: \beta_j^* \neq 0} I(\hat{\beta}_j \neq 0)$ | 正确识别的非零系数数 |
| IC（错误选择数） | $\sum_{j: \beta_j^* = 0} I(\hat{\beta}_j \neq 0)$ | 错误选为非零的零系数数 |
| SE（平方误差） | $\|\beta^* - \hat{\beta}\|_2^2$ | 系数估计偏差 |
| PE（预测误差） | $(\beta^* - \hat{\beta})^\top \Sigma_X (\beta^* - \hat{\beta})$ | 预测精度 |

所有指标报告中位数，标准误通过 500 次 Bootstrap 自助法计算。

### 9.6 原文献模拟结果

以下为 Datta & Zou (2017) 原论文中报告的模拟结果，供参考对比。

#### 表 1：加性误差模拟研究汇总统计量

基于 100 次重复实验。报告的数值为中位数，标准误差（se）通过自助法计算。"CoCo" 代表 CoCoLasso。"NCL" 为 Loh 和 Wainwright [17] 的方法。AR 表示预测变量的自回归协方差，CS 表示复合对称协方差。

|       |         | τ = 0.75 |           | τ = 1.0 |           | τ = 1.25 |           |
| :---- | :------ | :------- | :-------- | :------ | :-------- | :------- | :-------- |
|       |         | CoCo     | NCL       | CoCo    | NCL       | CoCo     | NCL       |
| AR    | C       | 3 (0)    | 3 (0)     | 3 (0)   | 2 (0.07)  | 3 (0.07) | 2 (0.45)  |
|       | IC      | 11 (0.75)| 3 (0.69)  | 11 (1.09)| 1 (0.33)  | 10 (0.84)| 0 (0.17)  |
|       | PE      | 3.66 (0.19)| 4.13 (0.26)| 5.8 (0.26)| 6.91 (0.34)| 8.49 (0.5)| 10.92 (0.46)|
|       | SE      | 3.81 (0.19)| 3.76 (0.18)| 5.57 (0.2)| 6.07 (0.27)| 7.94 (0.24)| 8.36 (0.3) |
| CS    | C       | 2 (0.18) | 2 (0)     | 2 (0)   | 1.5 (0.48)| 2 (0.03) | 1 (0)     |
|       | IC      | 14 (0.64)| 11.5 (0.58)| 18 (0.71)| 7 (0.22)  | 21 (0.48)| 5 (0.28)  |
|       | PE      | 4.49 (0.22)| 4.57 (0.31)| 6.03 (0.22)| 6.91 (0.34)| 6.99 (0.25)| 10.47 (0.58)|
|       | SE      | 8.05 (0.33)| 8.03 (0.48)| 11.01 (0.4)| 10.31 (1.0)| 12.97 (0.34)| 15.06 (1.06)|

#### 表 2：乘性误差模拟研究汇总统计量

基于 100 次重复实验。报告的数值为中位数，标准误差（se）通过自助法计算。"CoCo" 代表 CoCoLasso。"NCL" 为 Loh 和 Wainwright [17] 的方法。AR 表示预测变量的自回归协方差，CS 表示复合对称协方差。

|       |         | τ = 0.25 |           | τ = 0.5 |           | τ = 0.75 |           |
| :---- | :------ | :------- | :-------- | :------ | :-------- | :------- | :-------- |
|       |         | CoCo     | NCL       | CoCo    | NCL       | CoCo     | NCL       |
| AR    | C       | 3 (0)    | 3 (0)    | 3 (0)     | 3 (0)    | 3 (0)     | 2 (0)     |
|       | IC      | 14 (1.41)| 12 (2.4) | 12 (0.87) | 6 (0.74) | 10 (0.81) | 1 (0.46)  |
|       | PE      | 2.02 (0.15)| 2.47 (0.18)| 3.25 (0.14)| 3.58 (0.25)| 7.32 (0.2)| 8.32 (0.29)|
|       | SE      | 1.95 (0.09)| 2.26 (0.14)| 2.93 (0.14)| 3.09 (0.18)| 6.19 (0.2)| 6.58 (0.26)|
| CS    | C       | 3 (0)    | 3 (0)    | 3 (0.18)  | 3 (0.18) | 2 (0)     | 1 (0.45)  |
|       | IC      | 15 (0.72)| 18 (1.49)| 13 (0.77) | 11 (0.7) | 16 (0.88) | 4 (0.36)  |
|       | PE      | 2.23 (0.16)| 2.37 (0.1)| 3.66 (0.15)| 3.82 (0.19)| 7.93 (0.3)| 9.31 (0.41)|
|       | SE      | 4.21 (0.27)| 4.32 (0.21)| 6.11 (0.27)| 5.75 (0.26)| 10.43 (0.25)| 9.34 (0.61)|

### 9.7 运行方式

```bash
# 完整实验（100次蒙特卡洛，所有场景）
python simulation.py --mode full

# 快速测试
python simulation.py --mode quick

# 仅运行 CoCoLasso
python simulation.py --mode coco_only

# 仅运行 NCL
python simulation.py --mode ncl_only

# 自定义参数
python simulation.py --mode full --n_mc 50 --n_bootstrap 300
```

结果保存为 `simulation_results.csv`，包含各场景下 C、IC、SE、PE 的中位数及 Bootstrap 标准误。

### 9.8 预期计算时间

| 方法 | 单次运行约耗时 | 100次×12场景 |
|------|---------------|-------------|
| CoCoLasso | ~30-60秒 | ~10-20小时 |
| NCL (n_R=100) | ~5-15分钟 | ~100-300小时 |

建议：先用 `--mode quick` 验证流程，再逐步运行完整实验。NCL 的 `n_R` 参数可适当减小以加速。

---

## 10. 参考文献

【1】Datta A, Zou H. CoCoLasso for High-dimensional Error-in-variables Regression[J]. Annals of Statistics, 2016, 45(6):2400-2426. [arXiv:1510.07123](https://arxiv.org/pdf/1510.07123.pdf)

【2】Yu Q. Adaptive CoCoLasso for High-Dimensional Measurement Error Models[J]. Entropy, 2025, 27(2).

【3】Loh P L, Wainwright M J. High-dimensional regression with noisy and missing data: Provable guarantees with nonconvexity[J]. Annals of Statistics, 2012, 40(3):1637-1664. [arXiv:1109.3714](https://arxiv.org/pdf/1109.3714.pdf)

【4】Duchi J, Shalev-Shwartz S, Singer Y, et al. Efficient projections onto the L1-ball for learning in high dimensions[C]. ICML, 2008.

【5】Boyd S, Parikh N, Chu E, et al. Distributed optimization and statistical learning via the alternating direction method of multipliers[J]. Foundations and Trends in Machine Learning, 2011, 3(1):1-122. [PDF](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)

【6】HM-Lasso: [arXiv:1811.00255](https://arxiv.org/pdf/1811.00255.pdf)
