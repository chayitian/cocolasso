# CoCoLasso 实现说明报告

## 1. 问题背景

### 1.1 测量误差下的高维回归

标准 Lasso 回归假设我们观测到无误差的协变量矩阵 X，求解：

$$\min_\beta \frac{1}{2n}\|y - X\beta\|^2 + \lambda\|\beta\|_1$$

然而在实际场景中，协变量往往含有测量误差。我们观测到的是含误差的矩阵 Z 而非真实矩阵 X。直接对 Z 使用 Lasso 会导致估计偏差。

### 1.2 三种噪声模型

CoCoLasso 支持三种测量误差类型：

| 噪声类型 | 观测模型 | 参数 |
|----------|----------|------|
| 加性噪声 | Z = X + U，其中 U_ij ~ N(0, τ²) | τ（误差标准差） |
| 缺失数据 | Z_ij = X_ij（以概率 1-π）或 NaN（以概率 π） | 无需额外参数 |
| 乘性噪声 | Z = X ⊙ exp(M)，其中 M_ij ~ N(0, τ²) | τ（对数误差标准差） |

---

## 2. 核心算法

### 2.1 算法总体流程

CoCoLasso 的核心思想是：**不直接用 Z 替代 X 进行 Lasso 回归，而是先对由 Z 计算得到的协方差矩阵 Σ̃ 和交叉协方差向量 ρ̃ 进行校正，再将校正后可能非半正定的 Σ̃ 投影到最近的半正定（PSD）矩阵，最后在协方差形式下求解 Lasso。**

算法流程如下：

```
输入: Z (n×p 含误差矩阵), y (n×1 响应向量), 噪声参数
输出: β_hat (p×1 估计系数)

1. 数据预处理：中心化、标准化 Z 和 y
2. 计算校正后的协方差 Σ̃ 和交叉协方差 ρ̃
3. 将 Σ̃ 投影到最近的 PSD 矩阵 Σ̂（ADMM 或 HM 算法）
4. 构造 lambda 路径：从 λ_max 到 λ_min 的对数等间距序列
5. 对路径上每个 λ：
   a. K 折交叉验证计算预测误差
   b. 在全量数据上求解 Lasso（协方差形式）
6. 选择最优 λ（最小误差或 1-std 准则）
7. 返回最优 λ 对应的系数 β_hat
```

### 2.2 协方差校正

这是 CoCoLasso 的关键步骤——根据噪声类型对样本协方差和交叉协方差进行解析校正。

#### 2.2.1 加性噪声校正

当 Z = X + U，U_ij ~ N(0, τ²) 独立同分布时：

$$\tilde{\Sigma} = \frac{1}{n}Z^TZ - \tau^2 I_p$$

$$\tilde{\rho} = \frac{1}{n}Z^Ty$$

**推导：** E[Z'Z/n] = E[(X+U)'(X+U)/n] = X'X/n + E[U'U]/n = Σ + τ²I，因此减去 τ²I 即可恢复 Σ。

注意 ρ 不需要校正，因为 E[Z'y/n] = E[(X+U)'y/n] = X'y/n = ρ（假设 U 与 y 独立）。

#### 2.2.2 缺失数据校正

当 Z 的某些元素为 NaN 时，定义观测比率矩阵 R：

$$R_{jk} = \frac{n_{jk}}{n}$$

其中 n_jk 是特征 j 和 k 同时被观测到的样本数。

$$\tilde{\Sigma} = \frac{1}{n}Z^TZ \oslash R$$

$$\tilde{\rho} = \frac{1}{n}Z^Ty \oslash \text{diag}(R)$$

其中 ⊘ 表示逐元素除法。NaN 在计算 Z'Z 时被替换为 0（预处理阶段完成）。

**推导：** 对于缺失数据，E[Z_j Z_k / n_jk] = X_j X_k（条件期望），因此除以观测比率可还原无偏估计。

#### 2.2.3 乘性噪声校正

当 Z = X ⊙ exp(M)，M_ij ~ N(0, τ²) 时：

$$\Gamma_{jk} = \begin{cases} \frac{1}{n}Z_j^TZ_k / e^{\tau^2} & j \neq k \\ \frac{1}{n}Z_j^TZ_j / e^{2\tau^2} & j = k \end{cases}$$

$$\tilde{\rho}_j = \frac{1}{n}Z_j^Ty / e^{\tau^2/2}$$

**推导：** E[exp(M_ij)·exp(M_ik)] = E[exp(M_ij)]·E[exp(M_ik)] = exp(τ²/2)·exp(τ²/2) = exp(τ²)（当 j≠k 时独立）。对角元素 E[exp(2M_ij)] = exp(2τ²)。对 ρ 的校正类似：E[Z_j'y/n] = E[X_j' ⊙ exp(M_j)' · y/n] = ρ_j · exp(τ²/2)。

### 2.3 PSD 投影

校正后的 Σ̃ 可能不是半正定矩阵（例如加性噪声校正中减去 τ²I 可能使某些特征值变负），因此需要投影到最近的 PSD 矩阵。项目实现了两种投影算法。

#### 2.3.1 ADMM 投影（最大范数）

ADMM 投影求解以下优化问题：

$$\min_{R, S} \|R - \tilde{\Sigma}\|_{\max} \quad \text{s.t.} \quad R \succeq 0, \quad R - S = \tilde{\Sigma}, \quad \|S\|_1 \leq \mu/2$$

其中 ||·||_max 是矩阵最大范数（逐元素绝对值的最大值），||·||_1 是矩阵 L1 范数（逐元素绝对值之和）。

**ADMM 迭代步骤：**

1. **R 更新（PSD 投影）：**
   $$W = \tilde{\Sigma} + S + \mu L$$
   对 W 做特征值分解，将负特征值截断为 ε（小正数）：
   $$R = V \cdot \text{diag}(\max(\lambda_i, \epsilon)) \cdot V^T$$

2. **S 更新（L1 投影）：**
   $$M = R - \tilde{\Sigma} - \mu L$$
   对 M 的下三角元素投影到半径为 μ/2 的 L1 球上（使用 Duchi et al. 2008 的高效算法），然后对称化。

3. **L 更新（对偶变量）：**
   $$L = L - (R - S - \tilde{\Sigma}) / \mu$$

4. **收敛判断：** 当 R、S 的变化量以及原始残差 R-S-Σ̃ 均小于容差时停止。

**关键点：** ADMM 投影使用的是**最大范数**（||·||_max）作为距离度量，而非 Frobenius 范数。这意味着投影结果是在逐元素最大偏差意义下最接近 Σ̃ 的 PSD 矩阵。

#### 2.3.2 HM 投影（Frobenius 范数）

HM（Higham Modified）投影求解以下优化问题：

$$\min_{A} \|A - \tilde{\Sigma}\|_F \quad \text{s.t.} \quad A \succeq 0$$

当有权重矩阵 R 时（缺失数据场景），使用加权 Frobenius 范数。

**HM 迭代步骤：**

1. **A 更新（PSD 投影）：**
   $$A = B + \tilde{\Sigma} + \mu L$$
   特征值分解后截断负特征值。

2. **B 更新（软阈值/加权收缩）：**
   - Frobenius 范数模式（norm="F"）：
     $$B = \frac{A - \tilde{\Sigma} - \mu L}{\mu W \odot W + \mathbf{1}}$$
   - 最大范数模式（norm="max"）：使用加权 L1 投影

3. **L 更新：**
   $$L = L - (A - B - \tilde{\Sigma}) / \mu$$

**默认配置下 HM 使用 Frobenius 范数**，与 ADMM 的最大范数形成互补。

#### 2.3.3 两种投影的对比

| 特性 | ADMM 投影 | HM 投影 |
|------|-----------|---------|
| 距离度量 | 最大范数 ||·||_max | Frobenius 范数 ||·||_F |
| L1 约束 | 有（||S||_1 ≤ μ/2） | 无 |
| 权重矩阵 | 不支持 | 支持 R 权重 |
| 适用场景 | 加性/乘性噪声 | 缺失数据（带 R 权重） |
| 收敛速度 | 较快 | 较慢但更稳定 |

### 2.4 协方差形式 Lasso 求解

投影得到 PSD 矩阵 Σ̂ 后，CoCoLasso 在协方差形式下求解 Lasso：

$$\min_\beta \frac{1}{2}\beta^T\hat{\Sigma}\beta - \tilde{\rho}^T\beta + \lambda\|\beta\|_1$$

这等价于标准 Lasso 的目标函数（将 X'X/n 替换为 Σ̂，X'y/n 替换为 ρ̃），但不需要原始数据矩阵，仅依赖充分的统计量。

#### 2.4.1 坐标下降求解器

对每个坐标 j，固定其他系数，求解一维优化：

$$\beta_j = \frac{S(\hat{\Sigma}_{j,\cdot}\beta_{-j} - \tilde{\rho}_j, \lambda)}{\hat{\Sigma}_{jj}}$$

其中 S 是软阈值算子：

$$S(z, \lambda) = \begin{cases} z - \lambda & z > \lambda \\ z + \lambda & z < -\lambda \\ 0 & |z| \leq \lambda \end{cases}$$

**SCAD 惩罚支持：** 坐标下降求解器还支持 SCAD（Smoothly Clipped Absolute Deviation）惩罚。SCAD 使用自适应权重 w_j：

$$w_j = \begin{cases} 1 & |\beta_j| \leq \lambda \\ \frac{a\lambda - |\beta_j|}{\lambda(a-1)} & \lambda < |\beta_j| \leq a\lambda \\ 0 & |\beta_j| > a\lambda \end{cases}$$

其中 a=3.7 为默认值。SCAD 的自适应权重使得大系数不被过度惩罚，具有 oracle 性质。

#### 2.4.2 sklearn 求解器

sklearn 求解器通过 Cholesky 分解将协方差形式转换为数据矩阵形式，再调用 sklearn 的 Lasso：

1. 对 Σ̂ 做 Cholesky 分解：Σ̂ = U^TU
2. 构造伪数据矩阵：W̃ = √p · U
3. 构造伪响应向量：Ỹ = U^{-T} · √p · ρ̃
4. 调用 sklearn Lasso：`Lasso(alpha=λ).fit(W̃, Ỹ)`

**优势：** 利用 sklearn 的高效 C 实现，加速约 6.4 倍。支持 warm_start（通过 `lasso.coef_` 初始化）。

**限制：** 仅支持 Lasso 惩罚，不支持 SCAD。

### 2.5 Lambda 路径与交叉验证

#### 2.5.1 Lambda 路径构造

从 λ_max（使所有系数为零的最小 lambda）到 λ_min = λ_factor × λ_max 的对数等间距序列：

$$\lambda_{\max} = \max_j |\tilde{\rho}_j|$$

$$\lambda_i = \lambda_{\max} \cdot \left(\frac{\lambda_{\min}}{\lambda_{\max}}\right)^{i/(K-1)}, \quad i = 0, 1, \ldots, K-1$$

其中 λ_factor 默认为 0.01（n < p）或 0.001（n ≥ p）。

#### 2.5.2 K 折交叉验证

对路径上每个 λ：

1. 将数据随机分为 K 折
2. 对每折 k：
   - 在训练集上计算校正后的 Σ 和 ρ，投影到 PSD
   - 在训练集的 (Σ_train, ρ_train) 上求解 Lasso 得到 β_λ
   - 在测试集上计算预测误差：err = β_λ^T Σ_test β_λ - 2ρ_test^T β_λ
3. 取 K 折误差的均值作为该 λ 的 CV 误差

#### 2.5.3 最优 Lambda 选择

提供两种选择准则：

1. **最小误差准则（lambda_opt）：** 选择 CV 误差最小的 λ
2. **1-std 准则（lambda_sd）：** 选择满足 CV 误差 ≤ min_error + std_error 且 λ ≥ λ_opt 的最小 λ。这倾向于选择更正则化（更稀疏）的模型。

#### 2.5.4 早停机制

路径搜索在以下条件之一满足时提前终止：

- CV 误差变化量小于 optTol（收敛）
- CV 误差连续 earlyStopping_max 次大于当前最优值（过拟合）
- 达到最大步数 max_iter

### 2.6 截距恢复

由于预处理阶段对 Z 和 y 进行了中心化和标准化，最终需要将系数还原到原始尺度：

$$\beta_j^{\text{original}} = \beta_j^{\text{standardized}} \cdot \frac{\text{sd}_y}{\text{sd}_{Z_j}}$$

$$\text{intercept} = \bar{y} - \bar{Z}^T \beta^{\text{original}}$$

---

## 3. 算法伪代码

```
算法: CoCoLasso 路径坐标下降
输入: Z, y, noise, tau, K, step, ...
输出: beta_opt, lambda_opt, beta_sd, lambda_sd

1.  预处理: Z_proc, y_proc = preprocess(Z, y)
2.  计算校正统计量:
    if noise == "additive":
        Sigma_tilde = Z_proc'Z_proc/n - tau^2 * I
        rho_tilde = Z_proc'y_proc/n
    elif noise == "missing":
        R = compute_ratio_matrix(Z_proc)
        Sigma_tilde = Z_proc'Z_proc/n / R
        rho_tilde = Z_proc'y_proc/n / diag(R)
    elif noise == "multiplicative":
        (Gamma, rho_tilde) = corrected_covariance_multiplicative(Z_proc, y_proc, tau)
        Sigma_tilde = Gamma

3.  PSD 投影:
    Sigma_hat = project_to_PSD(Sigma_tilde)  // ADMM 或 HM

4.  交叉验证准备:
    对 K 折，分别计算训练集和测试集的 Sigma_hat_train, rho_train, Sigma_hat_test, rho_test

5.  Lambda 路径搜索:
    lambda_max = max|rho_tilde|
    lambda_list = log_space(lambda_max, lambda_factor * lambda_max, step)
    beta_start = 0_p

    for i = 1 to step:
        lambda_i = lambda_list[i]

        // K 折 CV
        for k = 1 to K:
            beta_cv = lasso_solve(Sigma_hat_train[k], rho_train[k], lambda_i)
            err_k = beta_cv' * Sigma_hat_test[k] * beta_cv - 2 * rho_test[k]' * beta_cv

        cv_error = mean(err_1, ..., err_K)

        // 全量数据求解
        beta_i = lasso_solve(Sigma_hat, rho_tilde, lambda_i, beta_start)
        beta_start = beta_i  // warm start

        if cv_error <= best_error:
            best_error = cv_error
            lambda_opt = lambda_i
            beta_opt = beta_i

        // 早停检查
        if |cv_error - error_old| < optTol: break
        if cv_error 连续递增超过 earlyStopping_max 次: break

6.  1-std 准则:
    threshold = min_error + std_error_at_min
    lambda_sd = min{lambda: error(lambda) <= threshold 且 lambda >= lambda_opt}
    beta_sd = 对应的系数

7.  还原原始尺度:
    coef_original = beta_opt * sd_y / sd_Z
    intercept = mean_y - mean_Z' * coef_original

8.  返回 (beta_opt, lambda_opt, beta_sd, lambda_sd, intercept)
```

---

## 4. 关键实现细节

### 4.1 L1 球投影算法

ADMM 投影中的 S 更新步骤需要将向量投影到 L1 球上。实现采用 Duchi et al. (2008) 的高效算法，时间复杂度 O(p log p)：

1. 对 |v| 降序排列得到 u
2. 计算累积和 sv = cumsum(u)
3. 找到最大的 ρ 使得 u_ρ > (sv_ρ - b) / (ρ + 1)
4. 计算阈值 θ = max(0, (sv_ρ - b) / (ρ + 1))
5. 输出 w = sign(v) · max(|v| - θ, 0)

### 4.2 Cholesky 分解与 sklearn Lasso 的对接

sklearn Lasso 需要数据矩阵形式 (W, Y)，而非协方差形式 (Σ, ρ)。转换方法：

给定 Σ̂ = U^TU（Cholesky 分解），构造：

- W̃ = √p · U（p×p 矩阵，模拟 p 个样本 p 个特征）
- Ỹ = U^{-T} · √p · ρ̃（p 维向量）

验证：W̃^T Ỹ / p = √p · U^T · U^{-T} · √p · ρ̃ / p = p · ρ̃ / p = ρ̃ ✓

W̃^T W̃ / p = p · U^T U / p = U^T U = Σ̂ ✓

### 4.3 缺失数据的预处理

缺失数据场景下的预处理较为特殊：

1. 计算每列的非 NaN 均值，对非 NaN 元素中心化
2. 将 NaN 替换为 0（因为中心化后非 NaN 元素均值为 0，NaN 替换为 0 相当于用均值填充）
3. 计算观测比率矩阵 R
4. 标准化（除以非 NaN 标准差）

### 4.4 数值稳定性保护

- Cholesky 分解失败时返回零向量（而非崩溃）
- 系数中出现 NaN/Inf 时归零
- PSD 投影中特征值截断阈值为 ε=1e-6（而非 0），避免数值上的非正定
- 对角线元素过小时用 diag_eps=1e-10 替代，避免除零
