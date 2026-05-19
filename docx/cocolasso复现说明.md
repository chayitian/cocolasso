# CoCoLasso 算法说明与理论证明

本文档围绕标准 `CoCoLasso` 的实现与理论展开，对应 `src/cocolasso.py` 与 `src/_utils.py`。参考文献为 Datta and Zou (2017) 的 CoCoLasso 方法。

## 1. 模型与符号

令真实协变量为 $X_i \in \mathbb{R}^p$，响应为 $y_i \in \mathbb{R}$，样本独立同分布，$i=1,\ldots,n$。真实线性模型为

$$
y_i = X_i^\top \beta^* + \varepsilon_i,
$$

其中 $\beta^* \in \mathbb{R}^p$ 是真实参数，$\varepsilon_i$ 是均值为 0 的噪声，且 $\mathbb{E}(\varepsilon_i \mid X_i)=0$。

标准 Lasso 回归假设观测到无误差的协变量矩阵 $X$，求解

$$
\min_\beta \frac{1}{2n}\|y - X\beta\|^2 + \lambda\|\beta\|_1.
$$

然而在实际场景中，协变量往往含有测量误差，观测到的是含误差的矩阵 $Z$ 而非真实矩阵 $X$。直接对 $Z$ 使用 Lasso 会导致估计偏差。

### 1.1 三种噪声模型

CoCoLasso 支持三类测量误差：

**加性误差：**

$$
Z_i = X_i + W_i,
$$

其中 $W_i$ 与 $(X_i,\varepsilon_i)$ 独立，满足 $\mathbb{E}W_i=0$，$\operatorname{Cov}(W_i)=\tau^2 I_p$。

**缺失数据：**（代码已实现，未纳入复现实验）

$$
Z_{ij} = \begin{cases} X_{ij}, & \text{以概率 } 1 - \delta \\ \text{NA}, & \text{以概率 } \delta \end{cases}
$$

其中 $\delta$ 为缺失概率，缺失机制与 $(X_i,y_i)$ 独立。记观测比例

$$
R_{jk}=\mathbb{P}(Z_{ij}\text{ 被观测}, Z_{ik}\text{ 被观测})=1-2\delta+\delta^2.
$$

**乘性误差：**

$$
Z_{ij}=X_{ij}M_{ij}, \qquad \log M_{ij}\sim N(0,\tau^2).
$$

### 1.2 核心统计量

定义真实协方差矩阵和真实交叉矩：

$$
\Sigma_X=\mathbb{E}(X_iX_i^\top), \qquad \rho_X=\mathbb{E}(X_i y_i).
$$

由模型 $y_i=X_i^\top\beta^*+\varepsilon_i$ 和 $\mathbb{E}(X_i\varepsilon_i)=0$ 得到

$$
\rho_X=\mathbb{E}(X_iX_i^\top\beta^*)+\mathbb{E}(X_i\varepsilon_i)=\Sigma_X\beta^*.
$$

如果直接用污染变量 $Z_i$ 构造普通 Lasso，则 Gram 矩阵估计的是 $\mathbb{E}(Z_iZ_i^\top)$，一般不等于 $\Sigma_X$。CoCoLasso 的核心就是构造修正矩阵 $\tilde{\Sigma}$ 和修正向量 $\tilde{\rho}$，使它们满足

$$
\tilde{\Sigma} \approx \Sigma_X, \qquad \tilde{\rho} \approx \rho_X=\Sigma_X\beta^*.
$$

---

## 2. 核心算法

### 2.1 算法总体流程

CoCoLasso 的核心思想是：**不直接用 $Z$ 替代 $X$ 进行 Lasso 回归，而是先对由 $Z$ 计算得到的协方差矩阵和交叉协方差向量进行校正，再将校正后可能非半正定的矩阵投影到最近的半正定（PSD）矩阵，最后在协方差形式下求解 Lasso。**

```
输入: Z (n×p 含误差矩阵), y (n×1 响应向量), 噪声参数
输出: β_hat (p×1 估计系数)

1. 数据预处理：中心化、标准化 Z 和 y
2. 计算校正后的协方差 Γ̂ 和交叉协方差 γ̂
3. 将 Γ̂ 投影到最近的 PSD 矩阵 Γ̂_+（ADMM 或 HM 算法）
4. 构造 lambda 路径：从 λ_max 到 λ_min 的对数等间距序列
5. 对路径上每个 λ：
   a. K 折交叉验证计算预测误差
   b. 在全量数据上求解 Lasso（协方差形式）
6. 选择最优 λ（最小误差或 1-std 准则）
7. 返回最优 λ 对应的系数 β_hat
```

### 2.2 参数校验

函数 `_validate_common_options()` 检查：

1. `noise` 是否属于允许集合（`additive`、`missing`、`multiplicative`）。
2. `penalty` 是否为 `lasso` 或 `SCAD`。
3. `mode` 是否为 `ADMM` 或 `HM`。
4. `solver` 是否为 `coordinate_descent` 或 `sklearn`。
5. `solver="sklearn"` 时只允许 `penalty="lasso"`。
6. 加性误差或乘性误差要求提供 `tau`。

### 2.3 数据预处理

函数 `_preprocess_data()` 对输入矩阵 $Z$ 和响应 $y$ 做中心化与标准化。

设第 $j$ 列样本均值和标准差为

$$
\bar Z_j=\frac{1}{n}\sum_{i=1}^n Z_{ij}, \qquad s_j^2=\frac{1}{n-1}\sum_{i=1}^n (Z_{ij}-\bar Z_j)^2.
$$

若 `center_Z=True` 且 `scale_Z=True`，则内部使用

$$
Z_{ij}^{\ast}=\frac{Z_{ij}-\bar Z_j}{s_j}.
$$

响应变量类似，若 `center_y=True` 且 `scale_y=True`，内部使用

$$
y_i^{\ast}=\frac{y_i-\bar y}{s_y}.
$$

缺失数据场景中，代码先计算非缺失位置的列均值，再只对被观测位置中心化，并把缺失位置置为 0。该处理使得后续矩阵乘法可以直接运行，同时保留观测掩码用于构造比例矩阵。

### 2.4 协方差校正

这是 CoCoLasso 的关键步骤——根据噪声类型对样本协方差和交叉协方差进行解析校正。

#### 2.4.1 加性误差校正

原始尺度下 $Z_i=X_i+W_i$，因此

$$
\mathbb{E}(Z_iZ_i^\top)=\mathbb{E}(X_iX_i^\top)+\mathbb{E}(W_iW_i^\top)=\Sigma_X+\tau^2I_p.
$$

无标准化时的无偏修正为

$$
\tilde{\Sigma}=\frac{1}{n}Z^\top Z-\tau^2I_p.
$$

若代码内部使用标准化变量 $Z_{ij}^{\ast}=Z_{ij}/s_j$，则误差项也变为 $W_{ij}^{\ast}=W_{ij}/s_j$，于是 $\operatorname{Var}(W_{ij}^{\ast})=\tau^2/s_j^2$。因此当前实现使用

$$
\tilde{\Sigma}^{\ast}=\frac{1}{n}Z^{\ast\top}Z^{\ast}-\operatorname{diag}\left(\frac{\tau^2}{s_1^2},\ldots,\frac{\tau^2}{s_p^2}\right).
$$

对应的交叉矩为

$$
\tilde{\rho}^{\ast}=\frac{1}{n}Z^{\ast\top}y^{\ast}.
$$

因为 $W_i$ 与 $y_i$ 独立且均值为 0，所以 $\mathbb{E}(Z_iy_i)=\mathbb{E}(X_iy_i)+\mathbb{E}(W_iy_i)=\Sigma_X\beta^*$，交叉矩不需要减去 $\tau^2$ 项。

#### 2.4.2 缺失数据校正（代码已实现，未纳入复现实验）

在缺失数据场景中，若缺失机制与 $X_i,y_i$ 独立，则

$$
\mathbb{E}(Z_{ij}Z_{ik})=(1-\delta)^2\mathbb{E}(X_{ij}X_{ik})=R_{jk}\Sigma_{X,jk}.
$$

因此 $\Sigma_{X,jk}=\mathbb{E}(Z_{ij}Z_{ik})/R_{jk}$。代码用样本观测比例

$$
\widehat R_{jk}=\frac{1}{n}\sum_{i=1}^n 1\{Z_{ij}\text{ observed}, Z_{ik}\text{ observed}\}
$$

构造

$$
\tilde{\Sigma}_{jk}=\frac{n^{-1}\sum_i Z_{ij}Z_{ik}}{\widehat R_{jk}}.
$$

对交叉矩，若 $R_{jj}=\mathbb{P}(Z_{ij}\text{ 被观测})=1-\delta$，则

$$
\tilde{\rho}_j=\frac{n^{-1}\sum_i Z_{ij}y_i}{\widehat R_{jj}}.
$$

代码中 `_validate_ratio_matrix()` 要求比例矩阵所有元素为正且有限，否则修正会出现除零或无穷值。NaN 在计算 $Z^\top Z$ 时被替换为 0（预处理阶段完成）。

#### 2.4.3 乘性误差校正

乘性误差模型为 $Z_{ij}=X_{ij}M_{ij}$，$\log M_{ij}\sim N(0,\tau^2)$。正态矩母函数给出

$$
\mathbb{E}(M_{ij})=e^{\tau^2/2}, \qquad \mathbb{E}(M_{ij}^2)=e^{2\tau^2}.
$$

当 $j\ne k$ 且乘性误差相互独立时，

$$
\mathbb{E}(Z_{ij}Z_{ik})=\mathbb{E}(X_{ij}X_{ik})\mathbb{E}(M_{ij})\mathbb{E}(M_{ik})=\Sigma_{X,jk}e^{\tau^2}.
$$

因此非对角元素修正为

$$
\tilde{\Sigma}_{jk}=\frac{n^{-1}\sum_i Z_{ij}Z_{ik}}{e^{\tau^2}}, \qquad j\ne k.
$$

对角元素满足 $\mathbb{E}(Z_{ij}^2)=\mathbb{E}(X_{ij}^2)e^{2\tau^2}$，所以

$$
\tilde{\Sigma}_{jj}=\frac{n^{-1}\sum_i Z_{ij}^2}{e^{2\tau^2}}.
$$

交叉矩满足 $\mathbb{E}(Z_{ij}y_i)=\mathbb{E}(M_{ij})\mathbb{E}(X_{ij}y_i)=e^{\tau^2/2}\rho_{X,j}$，因此

$$
\tilde{\rho}_j=\frac{n^{-1}\sum_i Z_{ij}y_i}{e^{\tau^2/2}}.
$$

这些公式由 `_corrected_covariance_multiplicative()` 实现。

### 2.5 PSD 投影

修正矩阵 $\tilde{\Sigma}$ 在有限样本下可能不是正半定矩阵。若直接求解

$$
\frac{1}{2}\beta^\top\tilde{\Sigma}\beta-\tilde{\rho}^\top\beta+\lambda\|\beta\|_1,
$$

目标函数可能非凸。CoCoLasso 的关键步骤是把 $\tilde{\Sigma}$ 投影为正半定矩阵。代码支持两种方式。

#### 2.5.1 ADMM 投影（最大范数）

ADMM 投影求解以下优化问题：

$$
\Sigma^{\text{PSD}} = \arg\min_{\Gamma\succeq 0}\|\Gamma-\tilde{\Sigma}\|_{\max},
$$

其中 $\|\cdot\|_{\max}$ 是矩阵最大范数（逐元素绝对值的最大值）。实现中使用变量 $R,S,L$ 的 ADMM 迭代，把 PSD 约束和距离约束分离。每次迭代包括三步。

**第一步，R 更新（PSD 投影）：** 对矩阵

$$
W=\tilde{\Sigma}+S+\mu L
$$

做特征分解，并把特征值截断到非负：

$$
R \leftarrow Q\operatorname{diag}\{\max(d_j,\epsilon)\}Q^\top.
$$

**第二步，S 更新（L1 投影）：** 对下三角向量做 $\ell_1$ 球投影（使用 Duchi et al. 2008 的高效算法），从而控制最大范数距离，然后对称化。

**第三步，L 更新（对偶变量）：**

$$
L \leftarrow L-\frac{R-S-\tilde{\Sigma}}{\mu}.
$$

**收敛判断：** 当 $R$、$S$ 的变化量以及原始残差 $R-S-\tilde{\Sigma}$ 均小于容差时停止。

#### 2.5.2 HM-Lasso 投影（Frobenius 范数）

HM-Lasso 投影求解以下优化问题：

$$
\Sigma^{\text{PSD}} = \arg\min_{A\succeq 0}\|A-\tilde{\Sigma}\|_F,
$$

当有权重矩阵 $R$ 时（缺失数据场景），使用加权 Frobenius 范数。

**HM 迭代步骤：**

1. **A 更新（PSD 投影）：** 对 $A = B + \tilde{\Sigma} + \mu L$ 做特征值分解后截断负特征值。

2. **B 更新（软阈值/加权收缩）：**
   - Frobenius 范数模式（norm="F"）：
     $$
     B = \frac{A - \tilde{\Sigma} - \mu L}{\mu W \odot W + \mathbf{1}}
     $$
   - 最大范数模式（norm="max"）：使用加权 L1 投影。

3. **L 更新：**
   $$
   L = L - (A - B - \tilde{\Sigma}) / \mu
   $$

默认配置下 HM 使用 Frobenius 范数，与 ADMM 的最大范数形成互补。

#### 2.5.3 两种投影的对比

| 特性 | ADMM 投影 | HM-Lasso 投影 |
|------|-----------|---------|
| 距离度量 | 最大范数 $\|\cdot\|_{\max}$ | Frobenius 范数 $\|\cdot\|_F$ |
| L1 约束 | 有 | 无 |
| 权重矩阵 | 不使用 $R$ 权重 | 支持 $R$ 权重 |
| 代码入口 | `mode="ADMM"` | `mode="HM"` |
| 数值特点 | 倾向于控制逐元素最大偏差 | 倾向于控制 Frobenius 范数偏差 |

### 2.6 协方差形式 Lasso 求解

投影得到 PSD 矩阵 $\Sigma^{\text{PSD}}$ 后，CoCoLasso 在协方差形式下求解 Lasso：

$$
\widehat\beta_\lambda=\arg\min_{\beta\in\mathbb{R}^p}\left\{\frac{1}{2}\beta^\top\Sigma^{\text{PSD}}\beta-\tilde{\rho}^\top\beta+\lambda\|\beta\|_1\right\}.
$$

这等价于标准 Lasso 的目标函数（将 $X^\top X/n$ 替换为 $\Sigma^{\text{PSD}}$，$X^\top y/n$ 替换为 $\tilde{\rho}$），但不需要原始数据矩阵，仅依赖充分的统计量。

#### 2.6.1 坐标下降求解器

默认求解器 `_lasso_covariance()` 使用坐标下降。固定除第 $j$ 个坐标外的其他坐标，令

$$
s_j=\sum_{k=1}^p \Sigma^{\text{PSD}}_{jk}\beta_k.
$$

目标函数关于 $\beta_j$ 的一维部分为

$$
\frac{1}{2}\Sigma^{\text{PSD}}_{jj}\beta_j^2+\beta_j\left(\sum_{k\ne j}\Sigma^{\text{PSD}}_{jk}\beta_k-\tilde{\rho}_j\right)+\lambda|\beta_j|.
$$

记

$$
a_j=\Sigma^{\text{PSD}}_{jj}, \qquad b_j=\tilde{\rho}_j-\sum_{k\ne j}\Sigma^{\text{PSD}}_{jk}\beta_k.
$$

则一维问题为

$$
\min_t \frac{1}{2}a_jt^2-b_jt+\lambda|t|,
$$

软阈值解为

$$
t=\frac{S(b_j,\lambda)}{a_j},
$$

其中软阈值算子

$$
S(b,\lambda)=\operatorname{sign}(b)(|b|-\lambda)_+.
$$

代码中的更新式与该公式等价，只是用 $S_0=s_j-a_j\beta_j-\tilde{\rho}_j$ 表示负方向梯度。

**SCAD 惩罚支持：** 坐标下降求解器还支持 SCAD（Smoothly Clipped Absolute Deviation）惩罚。SCAD 使用自适应权重 $w_j$：

$$
w_j = \begin{cases} 1 & |\beta_j| \leq \lambda \\ \frac{a\lambda - |\beta_j|}{\lambda(a-1)} & \lambda < |\beta_j| \leq a\lambda \\ 0 & |\beta_j| > a\lambda \end{cases}
$$

其中 $a=3.7$ 为默认值。SCAD 的自适应权重使得大系数不被过度惩罚，具有 oracle 性质。

#### 2.6.2 sklearn 求解器

可选求解器 `_lasso_sklearn()` 使用 Cholesky 分解把协方差形式转换为伪数据形式，再调用 sklearn 的 Lasso。

若 $\Sigma^{\text{PSD}}=U^\top U$（Cholesky 分解），构造

$$
\widetilde W=\sqrt{p}U, \qquad \widetilde Y=U^{-\top}\sqrt{p}\tilde{\rho}.
$$

sklearn 的目标函数为

$$
\frac{1}{2p}\|\widetilde Y-\widetilde W\beta\|_2^2+\lambda\|\beta\|_1.
$$

展开平方项：

$$
\frac{1}{2p}\left(\beta^\top\widetilde W^\top\widetilde W\beta-2\widetilde Y^\top\widetilde W\beta+\widetilde Y^\top\widetilde Y\right)+\lambda\|\beta\|_1.
$$

代入 $\widetilde W=\sqrt p U$，得到

$$
\frac{1}{2}\beta^\top U^\top U\beta-\frac{1}{p}\widetilde Y^\top\sqrt p U\beta+\lambda\|\beta\|_1+\text{constant}.
$$

再由 $\widetilde Y=U^{-\top}\sqrt p\tilde{\rho}$ 得

$$
\frac{1}{p}\widetilde Y^\top\sqrt p U\beta=\frac{1}{p}(\sqrt p\tilde{\rho}^\top U^{-1})\sqrt p U\beta=\tilde{\rho}^\top\beta.
$$

所以 sklearn 伪数据目标与协方差形式目标只差一个常数项，两者等价。

**优势：** 利用 sklearn 的高效 C 实现，支持 warm_start（通过 `lasso.coef_` 初始化）。

**限制：** 仅支持 Lasso 惩罚，不支持 SCAD。

### 2.7 Lambda 路径与交叉验证

#### 2.7.1 Lambda 路径构造

若 `alpha=None`，代码先计算最大正则化参数：

$$
\lambda_{\max}=\|\tilde{\rho}\|_\infty=\max_j|\tilde{\rho}_j|.
$$

然后构造从 $\lambda_{\max}$ 到 $\lambda_{\min}=\text{lambda\_factor}\cdot\lambda_{\max}$ 的对数等间距序列：

$$
\lambda_i = \lambda_{\max} \cdot \left(\frac{\lambda_{\min}}{\lambda_{\max}}\right)^{i/(K-1)}, \quad i = 0, 1, \ldots, K-1,
$$

其中 $\text{lambda\_factor}$ 默认为 0.01（$n < p$）或 0.001（$n \geq p$）。

若 `alpha` 是数值，则代码只使用一个固定正则化参数 $\lambda=\alpha$。

#### 2.7.2 K 折交叉验证

对路径上每个 $\lambda$：

1. 将数据随机分为 $K$ 折
2. 对每折 $k$：
   - 仅用训练折估计中心化、标准化、缺失比例和加性误差尺度
   - 用训练折预处理参数转换测试折，避免全局预处理数据泄漏
   - 在训练集上计算校正后的 $\tilde{\Sigma}$ 和 $\tilde{\rho}$，投影到 PSD
   - 在训练集的 $(\Sigma^{\text{PSD}}_{\text{train}},\tilde{\rho}_{\text{train}})$ 上求解 Lasso 得到 $\widehat\beta_\lambda$
   - 在测试集上计算预测误差

每个 $\lambda$ 的验证误差为

$$
\operatorname{CV}(\lambda)=\frac{1}{K}\sum_{k=1}^K\left(\widehat\beta_{\lambda}^{(-k)\top}{\Sigma^{\text{PSD}}_{\text{test}}}^{(k)}\widehat\beta_{\lambda}^{(-k)}-2\tilde{\rho}_{\text{test}}^{(k)\top}\widehat\beta_{\lambda}^{(-k)}\right).
$$

#### 2.7.3 最优 Lambda 选择

提供两种选择准则：

1. **最小误差准则（lambda_opt）：** 选择 CV 误差最小的 $\lambda$
2. **1-std 准则（lambda_sd）：** 选择满足 CV 误差 $\leq \min\_\text{error} + \text{std\_error}$ 且 $\lambda \geq \lambda_{\text{opt}}$ 的最大 $\lambda$。这倾向于选择更强正则化（更稀疏）的模型。

#### 2.7.4 早停机制

路径搜索在以下条件之一满足时提前终止：

- CV 误差变化量小于 optTol（收敛）
- CV 误差连续 earlyStopping_max 次大于当前最优值（过拟合）
- 达到最大步数 max_iter

### 2.8 原始尺度恢复

内部估计量是在预处理尺度下得到的。若内部变量为

$$
Z_j^{\ast}=\frac{Z_j-\bar Z_j}{s_j}, \qquad y^{\ast}=\frac{y-\bar y}{s_y},
$$

则内部模型为 $y^{\ast}=Z^{\ast\top}\beta^{\ast}$。代回原始尺度：

$$
\frac{y-\bar y}{s_y}=\sum_{j=1}^p\frac{Z_j-\bar Z_j}{s_j}\beta_j^{\ast}.
$$

两边乘以 $s_y$：

$$
y-\bar y=\sum_{j=1}^p Z_j\frac{s_y}{s_j}\beta_j^{\ast}-\sum_{j=1}^p\bar Z_j\frac{s_y}{s_j}\beta_j^{\ast}.
$$

因此原始尺度系数为

$$
\beta_j=\frac{s_y}{s_j}\beta_j^{\ast},
$$

截距为

$$
\beta_0^{\text{intercept}}=\bar y-\bar Z^\top\beta.
$$

若用户关闭中心化或标准化，代码按实际预处理开关恢复对应尺度。当前代码中，函数式接口返回的 `beta_opt`、`beta_sd` 仍是算法内部预处理尺度上的系数；sklearn 风格估计器的 `coef_`、`coef_sd_` 和 `coef_path_` 已转换到原始特征尺度。若需要查看估计器内部的预处理尺度系数，可使用 `coef_scaled_`、`coef_sd_scaled_` 和 `coef_path_scaled_`。

---

## 3. 算法伪代码

```
算法: CoCoLasso 路径坐标下降
输入: Z, y, noise, tau, K, step, ...
输出: beta_opt, lambda_opt, beta_sd, lambda_sd

1.  预处理: Z_proc, y_proc = preprocess(Z, y)
2.  计算校正统计量:
    if noise == "additive":
        Σ̃ = Z_proc'Z_proc/n - τ²I
        ρ̃ = Z_proc'y_proc/n
    elif noise == "missing":
        R = compute_ratio_matrix(Z_proc)
        Σ̃ = Z_proc'Z_proc/n / R
        ρ̃ = Z_proc'y_proc/n / diag(R)
    elif noise == "multiplicative":
        (Σ̃, ρ̃) = corrected_covariance_multiplicative(Z_proc, y_proc, τ)

3.  PSD 投影:
    Σ^PSD = project_to_PSD(Σ̃)  // ADMM 或 HM

4.  交叉验证准备:
    对 K 折，分别计算训练集和测试集的 Σ^PSD_train, ρ_train, Σ^PSD_test, ρ_test

5.  Lambda 路径搜索:
    λ_max = max|ρ̃|
    λ_list = log_space(λ_max, λ_factor * λ_max, step)
    β_start = 0_p

    for i = 1 to step:
        λ_i = λ_list[i]

        // K 折 CV
        for k = 1 to K:
            β_cv = lasso_solve(Σ^PSD_train[k], ρ_train[k], λ_i)
            err_k = β_cv' * Σ^PSD_test[k] * β_cv - 2 * ρ_test[k]' * β_cv

        cv_error = mean(err_1, ..., err_K)

        // 全量数据求解
        β_i = lasso_solve(Σ^PSD, ρ̃, λ_i, β_start)
        β_start = β_i  // warm start

        if cv_error <= best_error:
            best_error = cv_error
            λ_opt = λ_i
            β_opt = β_i

        // 早停检查
        if |cv_error - error_old| < optTol: break
        if cv_error 连续递增超过 earlyStopping_max 次: break

6.  1-std 准则:
    threshold = min_error + std_error_at_min
    λ_sd = max{λ: error(λ) <= threshold 且 λ >= λ_opt}
    β_sd = 对应的系数

7.  还原原始尺度:
    coef_original = β_opt * s_y / s_Z
    intercept = ȳ - Z̄' * coef_original

8.  返回 (β_opt, λ_opt, β_sd, λ_sd, intercept)
```

---

## 4. 关键实现细节

### 4.1 L1 球投影算法

ADMM 投影中的 $S$ 更新步骤需要将向量投影到 L1 球上。实现采用 Duchi et al. (2008) 的高效算法，时间复杂度 $O(p\log p)$：

1. 对 $|v|$ 降序排列得到 $u$
2. 计算累积和 $sv = \operatorname{cumsum}(u)$
3. 找到最大的 $\rho$ 使得 $u_\rho > (sv_\rho - b) / (\rho + 1)$
4. 计算阈值 $\theta = \max(0, (sv_\rho - b) / (\rho + 1))$
5. 输出 $w = \operatorname{sign}(v) \cdot \max(|v| - \theta, 0)$

### 4.2 Cholesky 分解与 sklearn Lasso 的对接

sklearn Lasso 需要数据矩阵形式 $(W, Y)$，而非协方差形式 $(\Sigma^{\text{PSD}}, \tilde{\rho})$。转换方法：

给定 $\Sigma^{\text{PSD}} = U^\top U$（Cholesky 分解），构造：

- $\widetilde W = \sqrt{p} \cdot U$（$p \times p$ 矩阵，模拟 $p$ 个样本 $p$ 个特征）
- $\widetilde Y = U^{-\top} \cdot \sqrt{p} \cdot \tilde{\rho}$（$p$ 维向量）

验证：

$$
\widetilde W^\top \widetilde Y / p = \sqrt{p} \cdot U^\top \cdot U^{-\top} \cdot \sqrt{p} \cdot \tilde{\rho} / p = p \cdot \tilde{\rho} / p = \tilde{\rho} \quad \checkmark
$$

$$
\widetilde W^\top \widetilde W / p = p \cdot U^\top U / p = U^\top U = \Sigma^{\text{PSD}} \quad \checkmark
$$

### 4.3 缺失数据的预处理（代码已实现，未纳入复现实验）

缺失数据场景下的预处理较为特殊：

1. 计算每列的非 NaN 均值，对非 NaN 元素中心化
2. 将 NaN 替换为 0（因为中心化后非 NaN 元素均值为 0，NaN 替换为 0 相当于用均值填充）
3. 计算观测比率矩阵 $R$
4. 标准化（除以非 NaN 标准差）

### 4.4 数值稳定性保护

- Cholesky 分解失败时返回零向量（而非崩溃）
- 系数中出现 NaN/Inf 时归零
- PSD 投影中特征值截断阈值为 $\epsilon=10^{-6}$（而非 0），避免数值上的非正定
- 对角线元素过小时用 $\text{diag\_eps}=10^{-10}$ 替代，避免除零

### 4.5 当前 API 与运行入口

当前仓库未提供 `requirements.txt` 或 `pyproject.toml`。直接运行核心代码至少需要 `numpy`、`scipy`、`scikit-learn`；运行复现实验脚本还需要 `pandas`。

主要公开入口：

| 入口 | 说明 |
|------|------|
| `CoCoLasso` | 标准 CoCoLasso 估计器，支持 `additive`、`missing`、`multiplicative` |
| `coco()` | 函数式接口 |

重要参数与限制：

| 参数 | 当前默认值/限制 |
|------|----------------|
| `mu` | 默认 `1.0`，用于 ADMM/HM 投影中的惩罚参数 |
| `solver` | 默认 `"coordinate_descent"`；可选 `"sklearn"` |
| `penalty` | 默认 `"lasso"`；`"SCAD"` 仅由 `solver="coordinate_descent"` 支持 |
| `alpha` | 固定正则化强度；为 `None` 时沿 lambda 路径做交叉验证自动选择 |
| `mode` | `"ADMM"` 或 `"HM"`，控制 PSD 投影方式 |

复现实验脚本位于 `reproduce/cocolasso_simulation.py`：

```bash
python reproduce/cocolasso_simulation.py --mode quick
python reproduce/cocolasso_simulation.py --mode full --n_mc 50 --n_bootstrap 300
```

---

## 4.1 复现实验设计

本节描述 `reproduce/cocolasso_simulation.py` 中实现的复现实验，参照 Datta & Zou (2017) 的模拟设置。

### 4.1.1 实验设置

- **样本量** $n=100$，**特征数** $p=250$
- **真实参数** $\beta^*=(3, 1.5, 0, 0, 2, 0, \ldots, 0)$，非零位置为 $j=1,2,5$，稀疏度 $s=3$
- **模型噪声** $\sigma=3$
- **交叉验证** $K=5$ 折

### 4.1.2 协方差结构

| 类型 | 公式 | 参数 |
|------|------|------|
| AR（自回归） | $\Sigma_{X,jk}=\rho^{\|j-k\|}$ | $\rho=0.5$ |
| CS（复合对称） | $\Sigma_{X,jk}=\rho$（$j\ne k$），$\Sigma_{X,jj}=\rho+0.5$ | $\rho=0.5$ |

### 4.1.3 误差类型与参数

| 误差类型 | $\tau$ 取值 | 说明 |
|----------|-------------|------|
| 加性测量误差 | $0.75, 1.0, 1.25$ | $Z=X+W$，$W_{ij}\sim N(0,\tau^2)$ |
| 乘性测量误差 | $0.25, 0.5, 0.75$ | $Z=X\odot M$，$\log M_{ij}\sim N(0,\tau^2)$ |

> **注意：** 缺失数据场景虽在代码中实现，但未纳入当前复现实验。

### 4.1.4 评价指标

| 指标 | 含义 | 公式 |
|------|------|------|
| C | 正确选择数 | $\|\{j\in S:\hat\beta_j\ne0\}\|$ |
| IC | 错误选择数 | $\|\{j\notin S:\hat\beta_j\ne0\}\|$ |
| PE | 预测误差 | $(\beta^*-\hat\beta)^\top\Sigma_X(\beta^*-\hat\beta)$ |
| SE | 平方误差 | $\|\beta^*-\hat\beta\|_2^2$ |

其中零阈值为 $10^{-6}$。

### 4.1.5 蒙特卡洛与统计汇总

- 100 次蒙特卡洛重复，每次使用不同随机种子
- 报告各指标的中位数
- Bootstrap 标准误：500 次 bootstrap 重采样估计中位数的标准差

### 4.1.6 实验流程

1. 根据 `cov_type` 生成 $\Sigma_X$
2. 对每次蒙特卡洛重复：
   - 生成 $X\sim N(0,\Sigma_X)$，中心化并按列范数标准化
   - 生成 $y=X\beta^*+\varepsilon$，$\varepsilon\sim N(0,\sigma^2 I)$
   - 根据 `error_type` 和 $\tau$ 生成含误差观测 $Z$
   - 调用 `coco()` 求解，参数：`step=100, K=5, mu=1.0, penalty="lasso", mode="ADMM", solver="sklearn"`
   - 将标准化系数还原为原始尺度
   - 计算 C、IC、PE、SE
3. 汇总所有重复的中位数和 bootstrap 标准误
4. 输出至 `results/simulation_results.csv`

---

## 5. 相合性证明

本节证明标准 CoCoLasso 估计量的相合性。证明采用高维 Lasso 的基本不等式、修正矩阵的一致性和限制特征值条件。

### 5.1 估计量定义

令 $\Sigma^{\text{PSD}}$ 表示投影后的修正 Gram 矩阵，$\tilde{\rho}$ 表示修正交叉矩。估计量定义为

$$
\widehat\beta=\arg\min_{\beta}\left\{L_n(\beta)+\lambda_n\|\beta\|_1\right\},
$$

其中

$$
L_n(\beta)=\frac{1}{2}\beta^\top\Sigma^{\text{PSD}}\beta-\tilde{\rho}^\top\beta.
$$

记真实支持集为 $S=\{j:\beta_j^0\ne0\}$，$|S|=s$。记误差向量 $\Delta=\widehat\beta-\beta^*$。

### 5.2 条件

**条件 A1：** 样本独立同分布，$X_i$、误差变量和 $\varepsilon_i$ 具有足够的亚高斯或有限高阶矩尾部，且测量误差与 $(X_i,\varepsilon_i)$ 独立。

**条件 A2：** 修正矩阵和交叉矩满足最大范数集中不等式。存在 $a_n\to0$，通常 $a_n=\sqrt{\log p/n}$，使得以概率趋于 1，

$$
\|\tilde{\Sigma}-\Sigma_X\|_{\max}\le C_1a_n, \qquad \|\tilde{\rho}-\Sigma_X\beta^*\|_\infty\le C_2a_n.
$$

**条件 A3：** PSD 投影误差不改变上述阶数。即存在数值误差 $r_n$，满足

$$
\|\Sigma^{\text{PSD}}-\Sigma_X\|_{\max}\le C_3a_n+r_n,
$$

其中 $r_n=o(\lambda_n)$。若投影算法精确且总体矩阵 $\Sigma_X$ 正半定，该条件是 CoCoLasso 理论中投影步骤的基本要求。

**条件 A4：** 限制特征值条件成立。存在常数 $\kappa>0$，使得对所有满足锥约束 $\|v_{S^c}\|_1\le3\|v_S\|_1$ 的向量 $v$，有

$$
v^\top\Sigma^{\text{PSD}}v\ge\kappa\|v\|_2^2.
$$

**条件 A5：** 正则化参数满足 $\lambda_n\ge2\eta_n$，其中 $\eta_n=\|\tilde{\rho}-\Sigma^{\text{PSD}}\beta^*\|_\infty$。由 A2 和 A3 可得

$$
\eta_n\le \|\tilde{\rho}-\Sigma_X\beta^*\|_\infty+\|\Sigma^{\text{PSD}}-\Sigma_X\|_{\max}\|\beta^*\|_1=O_p(a_n+r_n).
$$

所以通常取 $\lambda_n\asymp\sqrt{\log p/n}$ 即可满足 A5。

### 5.3 基本不等式

由于 $\widehat\beta$ 最小化目标函数，必有

$$
L_n(\widehat\beta)+\lambda_n\|\widehat\beta\|_1\le L_n(\beta^*)+\lambda_n\|\beta^*\|_1.
$$

代入 $\widehat\beta=\beta^*+\Delta$，展开二次项：

$$
L_n(\beta^*+\Delta)=\frac{1}{2}(\beta^*+\Delta)^\top\Sigma^{\text{PSD}}(\beta^*+\Delta)-\tilde{\rho}^\top(\beta^*+\Delta).
$$

继续展开：

$$
L_n(\beta^*+\Delta)=\frac{1}{2}\beta^{*\top}\Sigma^{\text{PSD}}\beta^*+\Delta^\top\Sigma^{\text{PSD}}\beta^*+\frac{1}{2}\Delta^\top\Sigma^{\text{PSD}}\Delta-\tilde{\rho}^\top\beta^*-\tilde{\rho}^\top\Delta.
$$

而 $L_n(\beta^*)=\frac{1}{2}\beta^{*\top}\Sigma^{\text{PSD}}\beta^*-\tilde{\rho}^\top\beta^*$。两式相减得到

$$
L_n(\beta^*+\Delta)-L_n(\beta^*)=\frac{1}{2}\Delta^\top\Sigma^{\text{PSD}}\Delta-\Delta^\top(\tilde{\rho}-\Sigma^{\text{PSD}}\beta^*).
$$

代回基本不等式：

$$
\frac{1}{2}\Delta^\top\Sigma^{\text{PSD}}\Delta-\Delta^\top(\tilde{\rho}-\Sigma^{\text{PSD}}\beta^*)+\lambda_n\|\beta^*+\Delta\|_1\le\lambda_n\|\beta^*\|_1.
$$

移项：

$$
\frac{1}{2}\Delta^\top\Sigma^{\text{PSD}}\Delta\le\Delta^\top(\tilde{\rho}-\Sigma^{\text{PSD}}\beta^*)+\lambda_n(\|\beta^*\|_1-\|\beta^*+\Delta\|_1).
$$

### 5.4 随机误差项界

由 Hölder 不等式，

$$
\Delta^\top(\tilde{\rho}-\Sigma^{\text{PSD}}\beta^*)\le\|\Delta\|_1\|\tilde{\rho}-\Sigma^{\text{PSD}}\beta^*\|_\infty=\eta_n\|\Delta\|_1.
$$

由 A5，$\eta_n\le\lambda_n/2$，所以

$$
\Delta^\top(\tilde{\rho}-\Sigma^{\text{PSD}}\beta^*)\le\frac{\lambda_n}{2}\|\Delta\|_1.
$$

### 5.5 惩罚项分解

因为 $\beta^*_{S^c}=0$，有 $\|\beta^*\|_1=\|\beta^*_S\|_1$。同时

$$
\|\beta^*+\Delta\|_1=\|\beta^*_S+\Delta_S\|_1+\|\Delta_{S^c}\|_1.
$$

由三角不等式，$\|\beta^*_S+\Delta_S\|_1\ge\|\beta^*_S\|_1-\|\Delta_S\|_1$，因此

$$
\|\beta^*\|_1-\|\beta^*+\Delta\|_1\le\|\Delta_S\|_1-\|\Delta_{S^c}\|_1.
$$

代回基本不等式：

$$
\frac{1}{2}\Delta^\top\Sigma^{\text{PSD}}\Delta\le\frac{\lambda_n}{2}(\|\Delta_S\|_1+\|\Delta_{S^c}\|_1)+\lambda_n(\|\Delta_S\|_1-\|\Delta_{S^c}\|_1).
$$

合并同类项：

$$
\frac{1}{2}\Delta^\top\Sigma^{\text{PSD}}\Delta\le\frac{3\lambda_n}{2}\|\Delta_S\|_1-\frac{\lambda_n}{2}\|\Delta_{S^c}\|_1.
$$

左边非负，因此右边非负，得到 $\|\Delta_{S^c}\|_1\le3\|\Delta_S\|_1$，即锥约束。

### 5.6 L2 误差界

在锥约束下，由 A4，$\Delta^\top\Sigma^{\text{PSD}}\Delta\ge\kappa\|\Delta\|_2^2$。又由上一节的不等式去掉负项，

$$
\frac{1}{2}\Delta^\top\Sigma^{\text{PSD}}\Delta\le\frac{3\lambda_n}{2}\|\Delta_S\|_1.
$$

利用 Cauchy 不等式，$\|\Delta_S\|_1\le\sqrt{s}\|\Delta_S\|_2\le\sqrt{s}\|\Delta\|_2$，因此

$$
\frac{\kappa}{2}\|\Delta\|_2^2\le\frac{3\lambda_n}{2}\sqrt{s}\|\Delta\|_2.
$$

若 $\Delta\ne0$，两边除以 $\|\Delta\|_2$，得到

$$
\|\widehat\beta-\beta^*\|_2=\|\Delta\|_2\le\frac{3\sqrt{s}\lambda_n}{\kappa}.
$$

若 $\Delta=0$，该不等式显然成立。因此只要 $\sqrt{s}\lambda_n\to0$，就有 $\|\widehat\beta-\beta^*\|_2\xrightarrow{p}0$。

### 5.7 L1 误差界

由锥约束，$\|\Delta\|_1=\|\Delta_S\|_1+\|\Delta_{S^c}\|_1\le4\|\Delta_S\|_1$。再用 $\|\Delta_S\|_1\le\sqrt{s}\|\Delta\|_2$，得到

$$
\|\Delta\|_1\le4\sqrt{s}\|\Delta\|_2.
$$

代入 L2 界：

$$
\|\widehat\beta-\beta^*\|_1\le4\sqrt{s}\frac{3\sqrt{s}\lambda_n}{\kappa}=\frac{12s\lambda_n}{\kappa}.
$$

因此只要 $s\lambda_n\to0$，就有 $\|\widehat\beta-\beta^*\|_1\xrightarrow{p}0$。

### 5.8 预测误差界

预测误差可定义为 $\Delta^\top\Sigma_X\Delta$。若 $\lambda_{\max}(\Sigma_X)\le C_\Sigma$，则

$$
\Delta^\top\Sigma_X\Delta\le C_\Sigma\|\Delta\|_2^2.
$$

代入 L2 界：

$$
\Delta^\top\Sigma_X\Delta\le C_\Sigma\frac{9s\lambda_n^2}{\kappa^2}.
$$

因此若 $s\lambda_n^2\to0$，则预测误差收敛到 0。

### 5.9 结论

在 A1 到 A5 条件下，若 $\lambda_n\asymp\sqrt{\log p/n}$ 且 $\sqrt{s}\lambda_n\to0$，则 CoCoLasso 估计量满足

$$
\|\widehat\beta-\beta^*\|_2=O_p(\sqrt{s}\lambda_n).
$$

若进一步 $s\lambda_n\to0$，则

$$
\|\widehat\beta-\beta^*\|_1=O_p(s\lambda_n)=o_p(1).
$$

这就是相合性。

---

## 6. 渐进正态性证明

本节说明在低维或 oracle 条件下的渐进正态性。需要强调：高维 Lasso 估计量本身含有 $\ell_1$ 惩罚偏差，通常不能在不去偏、不 refit、不额外限制 $\lambda_n$ 的情况下直接得到普通渐进正态性。当前项目没有实现去偏估计器，因此这里证明的是当前修正二次目标在低维或惩罚足够小条件下的渐进正态性。

### 6.1 固定维条件

设 $p$ 固定，$n\to\infty$。假设：

1. $\Sigma_X$ 正定，最小特征值满足 $\lambda_{\min}(\Sigma_X)>c>0$。
2. $\tilde{\Sigma}\xrightarrow{p}\Sigma_X$，$\tilde{\rho}\xrightarrow{p}\Sigma_X\beta^*$。
3. 中心极限定理适用于修正矩：

$$
\frac{1}{\sqrt n}\sum_{i=1}^n \psi_i\xrightarrow{d}N(0,\Omega),
$$

其中 $\psi_i=g_i-G_i\beta^*$，$g_i$ 是单样本修正交叉矩贡献，$G_i$ 是单样本修正 Gram 矩阵贡献。

4. 惩罚参数满足 $\sqrt n\lambda_n\to0$。
5. PSD 投影在一阶渐近上不改变修正矩阵，即 $\sqrt n\|\Sigma^{\text{PSD}}-\tilde{\Sigma}\|=o_p(1)$。当 $p$ 固定且 $\lambda_{\min}(\Sigma_X)>c>0$ 时，$\tilde{\Sigma}$ 以概率趋于 1 已经正定，所以 PSD 投影最终不改变矩阵；此时第 5 条自然成立。

### 6.2 单样本影响函数：加性误差

先以加性误差为例推导。原始尺度下

$$
G_i=Z_iZ_i^\top-\tau^2I_p, \qquad g_i=Z_iy_i.
$$

于是 $\tilde{\Sigma}=\frac{1}{n}\sum_i G_i$，$\tilde{\rho}=\frac{1}{n}\sum_i g_i$。定义

$$
\psi_i=g_i-G_i\beta^*=Z_iy_i-(Z_iZ_i^\top-\tau^2I_p)\beta^*.
$$

代入 $y_i=X_i^\top\beta^*+\varepsilon_i$ 和 $Z_i=X_i+W_i$：

$$
\psi_i=(X_i+W_i)(X_i^\top\beta^*+\varepsilon_i)-\{(X_i+W_i)(X_i+W_i)^\top-\tau^2I_p\}\beta^*.
$$

展开第一项：

$$
(X_i+W_i)(X_i^\top\beta^*+\varepsilon_i)=(X_i+W_i)X_i^\top\beta^*+(X_i+W_i)\varepsilon_i.
$$

展开第二项：

$$
(X_i+W_i)(X_i+W_i)^\top\beta^*=(X_i+W_i)X_i^\top\beta^*+(X_i+W_i)W_i^\top\beta^*.
$$

两者相减：

$$
\psi_i=(X_i+W_i)\varepsilon_i-(X_i+W_i)W_i^\top\beta^*+\tau^2\beta^*.
$$

验证均值为 0。先看第一项：

$$
\mathbb{E}\{(X_i+W_i)\varepsilon_i\}=\mathbb{E}(X_i\varepsilon_i)+\mathbb{E}(W_i)\mathbb{E}(\varepsilon_i)=0.
$$

再看第二项：

$$
\mathbb{E}\{(X_i+W_i)W_i^\top\beta^*\}=\mathbb{E}(X_iW_i^\top)\beta^*+\mathbb{E}(W_iW_i^\top)\beta^*.
$$

由于 $X_i$ 与 $W_i$ 独立且 $\mathbb{E}W_i=0$，$\mathbb{E}(X_iW_i^\top)=0$。又 $\mathbb{E}(W_iW_i^\top)=\tau^2I_p$，所以

$$
\mathbb{E}\{(X_i+W_i)W_i^\top\beta^*\}=\tau^2\beta^*.
$$

因此 $\mathbb{E}\psi_i=0-\tau^2\beta^*+\tau^2\beta^*=0$。

若 $\mathbb{E}\|\psi_i\|^2<\infty$，多元中心极限定理给出

$$
\frac{1}{\sqrt n}\sum_{i=1}^n\psi_i\xrightarrow{d}N(0,\Omega),
$$

其中 $\Omega=\operatorname{Var}(\psi_i)$。

### 6.3 修正得分展开

估计量的一阶条件为

$$
\Sigma^{\text{PSD}}\widehat\beta-\tilde{\rho}+\lambda_n\widehat z=0,
$$

其中 $\widehat z\in\partial\|\widehat\beta\|_1$ 是次梯度。把 $\widehat\beta=\beta^*+\Delta$ 代入：

$$
\Sigma^{\text{PSD}}(\beta^*+\Delta)-\tilde{\rho}+\lambda_n\widehat z=0.
$$

整理：

$$
\Sigma^{\text{PSD}}\Delta=\tilde{\rho}-\Sigma^{\text{PSD}}\beta^*-\lambda_n\widehat z.
$$

两边乘以 $\sqrt n$：

$$
\sqrt n\Delta={\Sigma^{\text{PSD}}}^{-1}\sqrt n(\tilde{\rho}-\Sigma^{\text{PSD}}\beta^*)-\sqrt n\lambda_n{\Sigma^{\text{PSD}}}^{-1}\widehat z.
$$

由 $\sqrt n\lambda_n\to0$ 且 $\|\widehat z\|$ 有界，$\sqrt n\lambda_n{\Sigma^{\text{PSD}}}^{-1}\widehat z=o_p(1)$。

现在分解第一项中的矩阵误差：

$$
\tilde{\rho}-\Sigma^{\text{PSD}}\beta^*=(\tilde{\rho}-\tilde{\Sigma}\beta^*)+(\tilde{\Sigma}-\Sigma^{\text{PSD}})\beta^*.
$$

由 PSD 投影一阶可忽略条件，$\sqrt n(\tilde{\Sigma}-\Sigma^{\text{PSD}})\beta^*=o_p(1)$，因此

$$
\sqrt n(\tilde{\rho}-\Sigma^{\text{PSD}}\beta^*)=\sqrt n(\tilde{\rho}-\tilde{\Sigma}\beta^*)+o_p(1).
$$

又

$$
\sqrt n(\tilde{\rho}-\tilde{\Sigma}\beta^*)=\frac{1}{\sqrt n}\sum_{i=1}^n(g_i-G_i\beta^*)=\frac{1}{\sqrt n}\sum_{i=1}^n\psi_i.
$$

所以 $\sqrt n(\tilde{\rho}-\Sigma^{\text{PSD}}\beta^*)\xrightarrow{d}N(0,\Omega)$。

由于 ${\Sigma^{\text{PSD}}}^{-1}\xrightarrow{p}\Sigma_X^{-1}$，Slutsky 定理给出

$$
\sqrt n(\widehat\beta-\beta^*)\xrightarrow{d}N(0,\Sigma_X^{-1}\Omega\Sigma_X^{-1}).
$$

这就是固定维且惩罚足够小条件下的渐进正态性。

### 6.4 活动集 oracle 渐进正态性

高维场景下，如果能正确识别支持集 $S$，或者理论上讨论 oracle 估计量，则可在活动集上得到类似结论。

oracle 估计量定义为

$$
\widehat\beta_S^{\text{or}}=\arg\min_{b\in\mathbb{R}^s}\left\{\frac{1}{2}b^\top\Sigma^{\text{PSD}}_{SS}b-\tilde{\rho}_S^\top b\right\}, \qquad \widehat\beta_{S^c}^{\text{or}}=0.
$$

一阶条件为 $\Sigma^{\text{PSD}}_{SS}\widehat\beta_S^{\text{or}}=\tilde{\rho}_S$，所以

$$
\widehat\beta_S^{\text{or}}-\beta_S^*={\Sigma^{\text{PSD}}_{SS}}^{-1}(\tilde{\rho}_S-\Sigma^{\text{PSD}}_{SS}\beta_S^*).
$$

乘以 $\sqrt n$：

$$
\sqrt n(\widehat\beta_S^{\text{or}}-\beta_S^*)={\Sigma^{\text{PSD}}_{SS}}^{-1}\sqrt n(\tilde{\rho}_S-\Sigma^{\text{PSD}}_{SS}\beta_S^*).
$$

若 $s$ 固定，$\Sigma^{\text{PSD}}_{SS}\xrightarrow{p}\Sigma_{X,SS}$，并且 $\sqrt n(\tilde{\rho}_S-\Sigma^{\text{PSD}}_{SS}\beta_S^*)\xrightarrow{d}N(0,\Omega_{SS})$，则

$$
\sqrt n(\widehat\beta_S^{\text{or}}-\beta_S^*)\xrightarrow{d}N(0,\Sigma_{X,SS}^{-1}\Omega_{SS}\Sigma_{X,SS}^{-1}).
$$

若还满足 beta-min 条件 $\min_{j\in S}|\beta_j^0|\gg\lambda_n$ 以及适当的 irrepresentable 或 primal-dual witness 条件，Lasso 可以以概率趋于 1 选择正确支持集。此时活动集上的 Lasso 与 oracle 解渐近等价，进而获得活动集上的渐进正态性。但这类变量选择一致性条件比估计相合性更强，当前项目未显式检验这些条件。

### 6.5 缺失与乘性误差下的影响函数

缺失与乘性误差的证明结构相同。只需把单样本 $G_i$ 和 $g_i$ 换为相应的无偏修正贡献。

缺失数据中，若 $R_{jk}$ 已知，则

$$
G_{i,jk}=\frac{\mathbb{1}\{Z_{ij}\text{ 被观测}\}\cdot\mathbb{1}\{Z_{ik}\text{ 被观测}\}\cdot X_{ij}X_{ik}}{R_{jk}}, \qquad g_{i,j}=\frac{\mathbb{1}\{Z_{ij}\text{ 被观测}\}\cdot X_{ij}y_i}{R_{jj}}.
$$

由独立缺失机制，$\mathbb{E}G_{i,jk}=\Sigma_{X,jk}$，$\mathbb{E}g_{i,j}=\rho_{X,j}$。

乘性误差中，定义

$$
G_{i,jk}=\begin{cases}
Z_{ij}Z_{ik}/e^{\tau^2}, & j\ne k,\\
Z_{ij}^2/e^{2\tau^2}, & j=k,
\end{cases}
$$

以及 $g_{i,j}=Z_{ij}y_i/e^{\tau^2/2}$。由 log-normal 矩公式，$\mathbb{E}G_i=\Sigma_X$，$\mathbb{E}g_i=\Sigma_X\beta^*$。

于是仍可定义 $\psi_i=g_i-G_i\beta^*$，并在有限二阶矩和 CLT 条件下得到

$$
\frac{1}{\sqrt n}\sum_i\psi_i\xrightarrow{d}N(0,\Omega).
$$

后续一阶条件展开与加性误差完全相同。

---

## 7. 与代码的对应关系

| 代码函数 | 对应章节 |
|----------|----------|
| `_preprocess_data()` | 2.3 数据预处理 |
| `_additive_noise_variance()` | 2.4.1 加性误差校正（标准化后的 $\tau^2/s_j^2$ 修正） |
| `_ratio_matrix_from_mask()` 和 `_validate_ratio_matrix()` | 2.4.2 缺失数据校正（比例矩阵） |
| `_corrected_covariance_multiplicative()` | 2.4.3 乘性误差校正 |
| `_admm_proj()` | 2.5.1 ADMM 投影 |
| `_hm_proj()` | 2.5.2 HM-Lasso 投影 |
| `_lasso_covariance()` | 2.6.1 坐标下降求解器 |
| `_lasso_sklearn()` | 2.6.2 sklearn 求解器 |
| `_cv_covariance_matrices()` | 2.7 Lambda 路径与交叉验证 |
| `_restore_coefficients()`、`_restore_coefficient_path()`、`_restore_intercept()` | 2.8 原始尺度恢复 |

**复现实验脚本 `reproduce/cocolasso_simulation.py`：**

| 代码函数 | 对应章节 |
|----------|----------|
| `cov_autoregressive()`、`generate_cs_covariance()`、`generate_covariance()` | 4.1.2 协方差结构 |
| `generate_data()` | 4.1.6 实验流程（数据生成） |
| `add_additive_error()` | 4.1.3 加性测量误差 |
| `add_multiplicative_error()` | 4.1.3 乘性测量误差 |
| `compute_metrics()` | 4.1.4 评价指标（C、IC、PE、SE） |
| `bootstrap_se()` | 4.1.5 蒙特卡洛与统计汇总 |
| `run_single_experiment()` | 4.1.6 实验流程（单次运行） |
| `run_simulation()` | 4.1.6 实验流程（主循环） |

---

## 8. 结论与限制

第一，项目实现的标准 CoCoLasso 主线符合 CoCoLasso 的基本结构：修正协方差矩阵、PSD 投影、凸 Lasso 求解和交叉验证选择正则化参数。

第二，在修正矩阵集中、PSD 投影误差可控、限制特征值成立、$\lambda_n\asymp\sqrt{\log p/n}$ 且 $\sqrt{s}\lambda_n\to0$ 的条件下，估计量满足 L2 相合性；若 $s\lambda_n\to0$，还满足 L1 相合性。

第三，渐进正态性不是高维惩罚估计量自动具备的无条件结论。它需要固定维、oracle 活动集、惩罚项满足 $\sqrt n\lambda_n\to0$，或额外实现去偏估计。本项目当前没有实现去偏估计、活动集 refit 的标准误估计或置信区间输出，因此文档中的渐进正态性是算法对应理论目标在额外条件下的数学性质。

第四，若要把渐进正态性转化为可运行的推断模块，还需要实现 $\Omega$ 的一致估计、活动集或去偏校正、以及置信区间和假设检验接口。
