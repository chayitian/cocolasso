# CoCoLasso 算法说明与理论证明

本文档围绕标准 `CoCoLasso` 的实现与理论展开，对应 `src/cocolasso.py` 与 `src/_utils.py`。参考文献为 Datta and Zou (2017) 的 CoCoLasso 方法。`BDCoCoLasso`、`GeneralCoCoLasso`、`SCAD` 惩罚、`HM` 投影和 `sklearn` 伪数据求解器属于本仓库扩展或工程实现，不属于该 PDF 中标准 CoCoLasso 的理论陈述。

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
R_{jj}=\mathbb{P}(Z_{ij}\text{ 被观测})=1-\delta, \qquad R_{jk}=\mathbb{P}(Z_{ij}\text{ 与 }Z_{ik}\text{ 同时被观测})=(1-\delta)^2\ (j\ne k).
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

在缺失数据场景中，若缺失机制与 $X_i,y_i$ 独立，则对非对角元素有

$$
\mathbb{E}(Z_{ij}Z_{ik})=(1-\delta)^2\mathbb{E}(X_{ij}X_{ik})=R_{jk}\Sigma_{X,jk}.
$$

对角元素对应单个变量被观测，使用 $R_{jj}=1-\delta$。代码不假设所有特征缺失概率相同，而是直接用样本观测掩码构造 $\widehat R_{jk}$。

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

论文 Appendix A 中的 CoCoLasso 投影求解以下最大范数 PSD 投影问题：

$$
\Sigma^{\text{PSD}} = \arg\min_{\Gamma\succeq 0}\|\Gamma-\tilde{\Sigma}\|_{\max},
$$

其中 $\|\cdot\|_{\max}$ 是矩阵最大范数（逐元素绝对值的最大值）。论文通过引入辅助变量，把问题等价写成 $B=A-\tilde\Sigma$，然后对增广拉格朗日函数做 ADMM。代码中的变量名为 $R,S,L$，分别对应论文中的 PSD 变量、差异变量和拉格朗日乘子；其目标是数值近似该最大范数投影。

**第一步，R 更新（PSD 投影）：** 对矩阵

$$
W=\tilde{\Sigma}+S+\mu L
$$

做特征分解，并把特征值截断到非负：

$$
R \leftarrow Q\operatorname{diag}\{\max(d_j,\epsilon)\}Q^\top.
$$

**第二步，S 更新（L1 投影）：** 对下三角向量做 $\ell_1$ 球投影（使用 Duchi et al. 2008 的高效算法），再对称化。该步骤来自最大范数与 $\ell_1$ 球投影的对偶关系；代码使用下三角向量和半径 `mu/2` 的实现细节来避免重复计算对称矩阵元素。

**第三步，L 更新（对偶变量）：**

$$
L \leftarrow L-\frac{R-S-\tilde{\Sigma}}{\mu}.
$$

**收敛判断：** 当 $R$、$S$ 的变化量以及原始残差 $R-S-\tilde{\Sigma}$ 均小于容差时停止。

#### 2.5.2 HM-Lasso 投影（Frobenius 范数，扩展选项）

HM-Lasso 投影不是 Datta & Zou (2017) PDF 中的标准 CoCoLasso 投影，而是本仓库提供的工程扩展。它求解以下 Frobenius 范数意义下的 PSD 近似问题：

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

其中 $a=3.7$ 为默认值。SCAD 的自适应权重使得大系数不被过度惩罚；这是本仓库的非凸惩罚扩展，不属于 PDF 中标准 CoCoLasso 的凸 Lasso 理论。

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
\lambda_i = \lambda_{\max} \cdot \left(\frac{\lambda_{\min}}{\lambda_{\max}}\right)^{i/(\text{step}-1)}, \quad i = 0, 1, \ldots, \text{step}-1,
$$

其中 $\text{lambda\_factor}$ 默认为 0.01（$n < p$）或 0.001（$n \geq p$）。

若 `alpha` 是数值，则代码只使用一个固定正则化参数 $\lambda=\alpha$。

#### 2.7.2 K 折交叉验证

对路径上每个 $\lambda$：

1. 将数据随机分为 $K$ 折；若传入 `random_state`，折分由该局部随机种子控制
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

当前仓库提供了 `requirements.txt`。直接运行核心代码至少需要 `numpy`、`scipy`、`scikit-learn`；运行复现实验脚本还需要 `pandas`。

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
| `random_state` | 默认 `None`；控制交叉验证折分，`None` 时沿用全局 numpy 随机状态 |

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
- 每次蒙特卡洛中，数据生成仍使用该次实验的 `seed`；CV 折分显式使用 `random_state=seed + 20000`
- Bootstrap 标准误：500 次 bootstrap 重采样估计中位数的标准差；bootstrap 使用按场景和指标固定的局部随机种子

### 4.1.6 实验流程

1. 根据 `cov_type` 生成 $\Sigma_X$
2. 对每次蒙特卡洛重复：
   - 生成 $X\sim N(0,\Sigma_X)$，中心化并按列范数标准化
   - 生成 $y=X\beta^*+\varepsilon$，$\varepsilon\sim N(0,\sigma^2 I)$
   - 根据 `error_type` 和 $\tau$ 生成含误差观测 $Z$
   - 调用 `coco()` 求解，参数：`step=100, K=5, mu=1.0, penalty="lasso", mode="ADMM", solver="sklearn", random_state=seed+20000`
   - 将标准化系数还原为原始尺度
   - 计算 C、IC、PE、SE
3. 汇总所有重复的中位数和 bootstrap 标准误
4. 输出至 `results/simulation_results.csv`

---

## 5. 论文定理条件、误差界与证明对应

本节对应 Datta & Zou (2017) Theorem 1 和 Theorem 2。重点不是重新证明一个不同的 Lasso 结论，而是说明论文定理使用的条件、结论，以及这些条件如何进入 CoCoLasso 的基本不等式推导。

### 5.1 估计量定义

令真实 Gram 矩阵和真实交叉矩为

$$
\Sigma=\frac{1}{n}X^\top X, \qquad \rho=\frac{1}{n}X^\top y.
$$

令 $\widetilde\Sigma$ 表示由污染数据 $Z$ 得到的修正 Gram 矩阵，$\tilde\rho$ 表示修正交叉矩。论文定义最大范数 PSD 投影

$$
\widehat\Sigma=(\widetilde\Sigma)_+=\arg\min_{A\succeq0}\|A-\widetilde\Sigma\|_{\max}.
$$

标准 CoCoLasso 估计量为

$$
\widehat\beta=\arg\min_{\beta}\left\{\frac{1}{2}\beta^\top\widehat\Sigma\beta-\tilde\rho^\top\beta+\lambda\|\beta\|_1\right\}.
$$

记真实支持集为 $S=\{j:\beta_j^*\ne0\}$，$|S|=s$。不失一般性，论文把真实系数写为 $\beta^*=(\beta_S^{*\top},0)^\top$。下文记误差向量

$$
\Delta=\widehat\beta-\beta^*.
$$

### 5.2 Theorem 1 的条件与结论

**条件 A1：修正统计量的集中性。** 论文假设存在常数 $C,c$、误差尺度函数 $\zeta$ 和 $\varepsilon_0>0$，使得对任意 $\varepsilon\le\varepsilon_0$，逐元素有

$$
\Pr(|\widetilde\Sigma_{ij}-\Sigma_{ij}|\ge\varepsilon)\le C\exp(-cn\varepsilon^2\zeta^{-1}),
$$

$$
\Pr(|\tilde\rho_j-\rho_j|\ge\varepsilon)\le C\exp(-cns^{-2}\varepsilon^2\zeta^{-1}).
$$

其中 $\zeta$ 汇总响应噪声、测量误差强度和误差模型常数；加性误差和乘性误差在论文 Lemma 1 和 Lemma 2 中都满足该条件。

**条件 A2：PSD 投影误差不放大阶数。** 因为真实 $\Sigma$ 本身半正定，而 $\widehat\Sigma$ 是离 $\widetilde\Sigma$ 最近的半正定矩阵之一，所以

$$
\|\widehat\Sigma-\widetilde\Sigma\|_{\max}\le\|\Sigma-\widetilde\Sigma\|_{\max}.
$$

由三角不等式得到论文公式 (2.3)：

$$
\|\widehat\Sigma-\Sigma\|_{\max}\le2\|\widetilde\Sigma-\Sigma\|_{\max}.
$$

这一步是 CoCoLasso 理论的关键：投影保证凸性，同时保持最大范数收敛阶数。

**条件 A3：真实 Gram 矩阵满足限制特征值条件。** 论文使用真实 $\Sigma$ 上的兼容/限制特征值条件：

$$
0<\Lambda=\min_{v\ne0,\ \|v_{S^c}\|_1\le3\|v_S\|_1}\frac{v^\top\Sigma v}{\|v\|_2^2}.
$$

**条件 A4：正则化参数尺度。** Theorem 1 要求 $\lambda$ 足够大以控制随机误差项，又不能大到破坏信号：

$$
s\sqrt{\zeta\log p/n}\lesssim\lambda\le\min(\varepsilon_0,12\varepsilon_0\|\beta_S^*\|_\infty).
$$

在这些条件下，论文 Theorem 1 给出高概率误差界：存在常数 $C,c$，使得概率至少为 $1-C\exp(-c\log p)$ 时，

$$
\|\widehat\beta-\beta^*\|_2\le C\lambda\sqrt{s}/\Lambda, \qquad \|\widehat\beta-\beta^*\|_1\le C\lambda s/\Lambda,
$$

$$
\|X(\widehat\beta-\beta^*)\|_2/\sqrt n\le C\lambda\sqrt{s}/\sqrt\Lambda.
$$

因此 $s^2\zeta\log p/n\to0$ 保证 Theorem 1 中用于控制随机项的下界可以趋小；若同时选择的 $\lambda$ 还满足 $\lambda\sqrt{s}/\Lambda\to0$ 和 $\lambda s/\Lambda\to0$，上述误差界随样本量增长而收缩。实际应用中 $\lambda$ 由校准交叉验证选择；理论条件说明的是高概率误差控制所需的量级。

### 5.3 证明思路和高概率事件

证明从两个高概率事件开始。设 $D=\widehat\Sigma-\Sigma$，$B=\|\beta_S^*\|_\infty$。第一步要控制

$$
\|\tilde\rho-\widehat\Sigma\beta^*\|_\infty.
$$

按三角不等式分解：

$$
\|\tilde\rho-\widehat\Sigma\beta^*\|_\infty
\le \|\tilde\rho-\rho\|_\infty+\|\rho-\Sigma\beta^*\|_\infty+\|D\beta^*\|_\infty.
$$

三项分别由论文条件控制：第一项由 A1 的 $\tilde\rho$ 集中性控制；第二项等于 $\|X^\top w/n\|_\infty$，由响应噪声的亚高斯集中性控制；第三项满足 $\|D\beta^*\|_\infty\le sB\|D\|_{\max}$，再由 A2 和 A1 控制。由 $\lambda\gtrsim s\sqrt{\zeta\log p/n}$ 可得高概率事件

$$
\mathcal E_1=\{\|\tilde\rho-\widehat\Sigma\beta^*\|_\infty\le\lambda/2\}.
$$

第二步要保证投影误差不会破坏限制特征值。由 A2 可得高概率事件

$$
\mathcal E_2=\{16s\|D\|_{\max}\le\Lambda/4\}.
$$

下面的基本不等式和锥约束推导都在 $\mathcal E_1\cap\mathcal E_2$ 上进行；论文用 union bound 得到该事件的概率至少为 $1-C\exp(-c\log p)$。

### 5.4 基本不等式

为和代码说明中的记号保持一致，以下把 $\widehat\Sigma$ 记为 $\Sigma^{\text{PSD}}$，并写

$$
L_n(\beta)=\frac{1}{2}\beta^\top\Sigma^{\text{PSD}}\beta-\tilde\rho^\top\beta.
$$

由于 $\widehat\beta$ 最小化目标函数，必有

$$
L_n(\widehat\beta)+\lambda\|\widehat\beta\|_1\le L_n(\beta^*)+\lambda\|\beta^*\|_1.
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
\frac{1}{2}\Delta^\top\Sigma^{\text{PSD}}\Delta-\Delta^\top(\tilde{\rho}-\Sigma^{\text{PSD}}\beta^*)+\lambda\|\beta^*+\Delta\|_1\le\lambda\|\beta^*\|_1.
$$

移项：

$$
\frac{1}{2}\Delta^\top\Sigma^{\text{PSD}}\Delta\le\Delta^\top(\tilde{\rho}-\Sigma^{\text{PSD}}\beta^*)+\lambda(\|\beta^*\|_1-\|\beta^*+\Delta\|_1).
$$

### 5.5 随机误差项界

由 Hölder 不等式，

$$
\Delta^\top(\tilde{\rho}-\Sigma^{\text{PSD}}\beta^*)\le\|\Delta\|_1\|\tilde{\rho}-\Sigma^{\text{PSD}}\beta^*\|_\infty.
$$

在事件 $\mathcal E_1$ 上，$\|\tilde\rho-\Sigma^{\text{PSD}}\beta^*\|_\infty\le\lambda/2$，所以

$$
\Delta^\top(\tilde{\rho}-\Sigma^{\text{PSD}}\beta^*)\le\frac{\lambda}{2}\|\Delta\|_1.
$$

### 5.6 惩罚项分解与锥约束

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
\frac{1}{2}\Delta^\top\Sigma^{\text{PSD}}\Delta\le\frac{\lambda}{2}(\|\Delta_S\|_1+\|\Delta_{S^c}\|_1)+\lambda(\|\Delta_S\|_1-\|\Delta_{S^c}\|_1).
$$

合并同类项：

$$
\frac{1}{2}\Delta^\top\Sigma^{\text{PSD}}\Delta\le\frac{3\lambda}{2}\|\Delta_S\|_1-\frac{\lambda}{2}\|\Delta_{S^c}\|_1.
$$

左边非负，因此右边非负，得到 $\|\Delta_{S^c}\|_1\le3\|\Delta_S\|_1$，即锥约束。

### 5.7 L2 误差界

上一步只使用了 $\Sigma^{\text{PSD}}\succeq0$ 得到锥约束。为了得到 $L_2$ 界，需要回到真实 Gram 矩阵 $\Sigma$ 上使用限制特征值条件。由 $D=\Sigma^{\text{PSD}}-\Sigma$，有

$$
\Delta^\top\Sigma\Delta=\Delta^\top\Sigma^{\text{PSD}}\Delta-\Delta^\top D\Delta.
$$

在锥约束下，$\|\Delta\|_1\le4\|\Delta_S\|_1$，所以

$$
|\Delta^\top D\Delta|\le\|D\|_{\max}\|\Delta\|_1^2\le16\|D\|_{\max}\|\Delta_S\|_1^2.
$$

再由限制特征值条件，$\|\Delta_S\|_1^2\le s\|\Delta\|_2^2\le s(\Delta^\top\Sigma\Delta)/\Lambda$。在事件 $\mathcal E_2$ 上，$16s\|D\|_{\max}\le\Lambda/4$，因此

$$
|\Delta^\top D\Delta|\le\frac{1}{4}\Delta^\top\Sigma\Delta.
$$

由基本不等式去掉负项可得

$$
\frac{1}{2}\Delta^\top\Sigma^{\text{PSD}}\Delta\le\frac{3\lambda}{2}\|\Delta_S\|_1.
$$

结合 $\Delta^\top\Sigma^{\text{PSD}}\Delta\ge\Delta^\top\Sigma\Delta-|\Delta^\top D\Delta|\ge(3/4)\Delta^\top\Sigma\Delta$，得到

$$
\Delta^\top\Sigma\Delta\le C\lambda\|\Delta_S\|_1.
$$

再用 $\|\Delta_S\|_1\le\sqrt{s}\|\Delta\|_2$ 和 $\Delta^\top\Sigma\Delta\ge\Lambda\|\Delta\|_2^2$，得到

$$
\Lambda\|\Delta\|_2^2\le C\lambda\sqrt{s}\|\Delta\|_2.
$$

若 $\Delta\ne0$，两边除以 $\|\Delta\|_2$；若 $\Delta=0$，结论显然成立。因此

$$
\|\widehat\beta-\beta^*\|_2\le C\lambda\sqrt{s}/\Lambda.
$$

### 5.8 L1 误差界

由锥约束，$\|\Delta\|_1=\|\Delta_S\|_1+\|\Delta_{S^c}\|_1\le4\|\Delta_S\|_1$。再用 $\|\Delta_S\|_1\le\sqrt{s}\|\Delta\|_2$，得到

$$
\|\Delta\|_1\le4\sqrt{s}\|\Delta\|_2.
$$

代入 L2 界：

$$
\|\widehat\beta-\beta^*\|_1\le C\lambda s/\Lambda.
$$

因此只要 $s\lambda/\Lambda\to0$，就有 $\|\widehat\beta-\beta^*\|_1\xrightarrow{p}0$。

### 5.9 预测误差界

论文中的预测误差为

$$
\|X\Delta\|_2^2/n=\Delta^\top\Sigma\Delta.
$$

由上一节已经得到 $\Delta^\top\Sigma\Delta\le C\lambda\|\Delta_S\|_1$，再代入 $\|\Delta_S\|_1\le\sqrt{s}\|\Delta\|_2\le C\lambda s/\Lambda$，可得

$$
\Delta^\top\Sigma\Delta\le C\lambda^2s/\Lambda.
$$

取平方根即

$$
\|X(\widehat\beta-\beta^*)\|_2/\sqrt n\le C\lambda\sqrt{s}/\sqrt\Lambda.
$$

### 5.10 结论

在 Theorem 1 的条件下，若 $s^2\zeta\log p/n\to0$ 并选取满足论文要求且使误差界右端趋小的 $\lambda$，则 CoCoLasso 估计量满足

$$
\|\widehat\beta-\beta^*\|_2=O_p(\lambda\sqrt{s}/\Lambda).
$$

同时

$$
\|\widehat\beta-\beta^*\|_1=O_p(\lambda s/\Lambda), \qquad \|X(\widehat\beta-\beta^*)\|_2/\sqrt n=O_p(\lambda\sqrt{s}/\sqrt\Lambda).
$$

这就是论文 Theorem 1 的误差界逻辑。若这些右端项趋于 0，即得到相合性。

### 5.11 Theorem 2：符号一致性与支持恢复

Theorem 2 讨论的是变量选择一致性，条件比 Theorem 1 更强。它额外要求真实 Gram 矩阵在真实支持集 $S$ 上满足两个条件。

**条件 B1：不可表示条件（irrepresentable condition）。** 存在 $\gamma>0$，使得

$$
\|\Sigma_{S^c,S}\Sigma_{S,S}^{-1}\|_\infty\le1-\gamma.
$$

这个条件限制无关变量不能被相关变量过强表示，是干净数据 Lasso 符号一致性的经典条件。

**条件 B2：活动集 Gram 矩阵可逆。** 存在 $C_{\min}>0$，使得

$$
\lambda_{\min}(\Sigma_{S,S})=C_{\min}>0.
$$

在 A1 的集中性条件、B1、B2 以及适当的 $\lambda$ 和 $\varepsilon$ 选择下，论文 Theorem 2 得到以下结果：

1. CoCoLasso 目标存在唯一解，且其支持集包含在真实支持集内，即 $\operatorname{supp}(\widehat\beta)\subseteq S$。
2. 活动集上的最大范数误差满足 $\|\widehat\beta_S-\beta_S^*\|_\infty\le\kappa_\infty\lambda$，其中 $\kappa_\infty$ 由 $\|\Sigma_{S,S}^{-1}\|_\infty$ 和 $C_{\min}$ 控制。
3. 若 beta-min 条件 $\min_{j\in S}|\beta_j^*|\ge\kappa_\infty\lambda$ 成立，则 $\operatorname{sign}(\widehat\beta_S)=\operatorname{sign}(\beta_S^*)$。

证明逻辑使用 primal-dual witness 构造。第一步只在真实支持集 $S$ 上解受限问题，得到候选解 $\widehat\beta_S$，并令 $\widehat\beta_{S^c}=0$。第二步写出 KKT 条件：活动集上必须满足

$$
\widehat\Sigma_{S,S}\widehat\beta_S-\tilde\rho_S+\lambda u_S=0,
$$

非活动集上必须存在对偶变量 $u_{S^c}$ 使得

$$
\widehat\Sigma_{S^c,S}\widehat\beta_S-\tilde\rho_{S^c}+\lambda u_{S^c}=0, \qquad \|u_{S^c}\|_\infty<1.
$$

第三步用 B1 控制干净 Gram 矩阵下的对偶变量，用 A1 和 PSD 投影误差控制污染校正后的扰动，从而证明严格对偶可行性 $\|u_{S^c}\|_\infty<1$。严格对偶可行性意味着非活动变量系数为 0，且解唯一。第四步对活动集解显式展开：

$$
\widehat\beta_S-\beta_S^*=\widehat\Sigma_{S,S}^{-1}(\tilde\rho_S-\widehat\Sigma_{S,S}\beta_S^* - \lambda u_S),
$$

再用逆矩阵扰动界和集中性条件得到 $\ell_\infty$ 误差界。最后，若最小信号强度大于该误差界，估计值不会跨过 0，因此符号恢复正确。

代码没有显式检查 B1、B2 或 beta-min 条件；这些是理论保证条件，不是运行时校验条件。

---

## 6. 渐进正态性证明（额外推导，非 PDF 主结论）

本节不是 Datta & Zou (2017) PDF 的主定理内容，而是说明在低维或 oracle 条件下可以怎样得到渐进正态性。需要强调：高维 Lasso 估计量本身含有 $\ell_1$ 惩罚偏差，通常不能在不去偏、不 refit、不额外限制 $\lambda_n$ 的情况下直接得到普通渐进正态性。当前项目没有实现去偏估计器，因此这里证明的是当前修正二次目标在额外条件下的数学性质。

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
| `_hm_proj()` | 2.5.2 HM-Lasso 投影（扩展选项，非 PDF 标准投影） |
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

第二，在论文 Theorem 1 的集中性条件、PSD 投影误差控制、真实 Gram 矩阵限制特征值条件以及 $s^2\zeta\log p/n\to0$ 可使随机项下界趋小的缩放下，估计量满足 $\ell_2$、$\ell_1$ 和预测误差界；当这些误差界右端趋于 0 时得到相合性。

第三，渐进正态性不是高维惩罚估计量自动具备的无条件结论。它需要固定维、oracle 活动集、惩罚项满足 $\sqrt n\lambda_n\to0$，或额外实现去偏估计。本项目当前没有实现去偏估计、活动集 refit 的标准误估计或置信区间输出，因此文档中的渐进正态性是算法对应理论目标在额外条件下的数学性质。

第四，若要把渐进正态性转化为可运行的推断模块，还需要实现 $\Omega$ 的一致估计、活动集或去偏校正、以及置信区间和假设检验接口。
