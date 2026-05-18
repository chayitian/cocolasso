# CoCoLasso 实现说明与理论证明

本文档仅围绕本项目中的标准 `CoCoLasso` 实现展开，主要对应 `src/cocolasso.py` 与 `src/_utils.py`。`BDCoCoLasso` 与 `GeneralCoCoLasso` 是块结构扩展，使用相同的修正协方差、PSD 投影和协方差形式 Lasso 思路，但本文不把它们作为理论证明主体。

参考文献基础为 Datta and Zou (2017) 的 Convex Conditioned Lasso / CoCoLasso 思路，以及误差变量 Lasso 中常用的修正 Gram 矩阵理论。本文中的相合性证明针对本项目实现的凸校正 Lasso 目标函数；渐进正态性证明需要额外条件，尤其是固定维或低维活动集、惩罚项足够小、或使用已知支持集的 oracle 版本。当前项目代码没有实现去偏 Lasso 或渐近方差估计，因此渐进正态性应理解为算法目标对应估计量在相应理论条件下的性质，而不是当前 API 已直接输出置信区间。

## 1. 模型与符号

令真实协变量为 $X_i \in \mathbb{R}^p$，响应为 $y_i \in \mathbb{R}$，样本独立同分布，$i=1,\ldots,n$。

真实线性模型为

$$
y_i = X_i^\top \beta^0 + \varepsilon_i,
$$

其中 $\beta^0 \in \mathbb{R}^p$ 是真实参数，$\varepsilon_i$ 是均值为 0 的噪声，并且满足

$$
\mathbb{E}(\varepsilon_i \mid X_i)=0.
$$

本项目的标准 `CoCoLasso` 支持三类观测协变量 $Z_i$。

第一类是加性误差：

$$
Z_i = X_i + W_i,
$$

其中 $W_i$ 与 $(X_i,\varepsilon_i)$ 独立，满足

$$
\mathbb{E}W_i=0, \qquad \operatorname{Cov}(W_i)=\tau^2 I_p.
$$

第二类是缺失数据。令 $M_{ij}\in\{0,1\}$ 表示第 $j$ 个变量是否被观测，观测变量可写为

$$
Z_{ij}=M_{ij}X_{ij},
$$

其中项目实现先把缺失值按列均值中心化后置 0，再利用观测比例矩阵校正协方差。记

$$
R_{jk}=\mathbb{P}(M_{ij}=1,M_{ik}=1).
$$

第三类是乘性误差：

$$
Z_{ij}=X_{ij}U_{ij}, \qquad \log U_{ij}\sim N(0,\tau^2).
$$

定义真实协方差矩阵

$$
\Sigma_X=\mathbb{E}(X_iX_i^\top),
$$

并定义真实交叉矩

$$
\rho_X=\mathbb{E}(X_i y_i).
$$

由模型 $y_i=X_i^\top\beta^0+\varepsilon_i$ 和 $\mathbb{E}(X_i\varepsilon_i)=0$ 得到

$$
\rho_X=\mathbb{E}(X_iX_i^\top\beta^0)+\mathbb{E}(X_i\varepsilon_i)=\Sigma_X\beta^0.
$$

如果直接用污染变量 $Z_i$ 构造普通 Lasso，则 Gram 矩阵估计的是 $\mathbb{E}(Z_iZ_i^\top)$，一般不等于 $\Sigma_X$。CoCoLasso 的核心就是构造修正矩阵 $\widehat\Gamma$ 和修正向量 $\widehat\gamma$，使它们满足

$$
\widehat\Gamma \approx \Sigma_X, \qquad \widehat\gamma \approx \rho_X=\Sigma_X\beta^0.
$$

## 2. 项目中的实现流程

标准 `CoCoLasso.fit(Z, y)` 调用 `_pathwise_coordinate_descent()`。实现流程如下。

### 2.1 参数校验

函数 `_validate_common_options()` 检查：

1. `noise` 是否属于允许集合。
2. `penalty` 是否为 `lasso` 或 `SCAD`。
3. `mode` 是否为 `ADMM` 或 `HM`。
4. `solver` 是否为 `coordinate_descent` 或 `sklearn`。
5. `solver="sklearn"` 时只允许 `penalty="lasso"`。
6. 加性误差或乘性误差要求提供 `tau`。

这些检查保证后续修正协方差公式有明确含义。

### 2.2 数据预处理

函数 `_preprocess_data()` 对输入矩阵 $Z$ 和响应 $y$ 做中心化与标准化。

设第 $j$ 列样本均值和标准差为

$$
\bar Z_j=\frac{1}{n}\sum_{i=1}^n Z_{ij}, \qquad s_j^2=\frac{1}{n-1}\sum_{i=1}^n (Z_{ij}-\bar Z_j)^2.
$$

如果 `center_Z=True` 且 `scale_Z=True`，则内部使用

$$
Z_{ij}^{\ast}=\frac{Z_{ij}-\bar Z_j}{s_j}.
$$

响应变量类似，若 `center_y=True` 且 `scale_y=True`，内部使用

$$
y_i^{\ast}=\frac{y_i-\bar y}{s_y}.
$$

缺失数据场景中，代码先计算非缺失位置的列均值，再只对被观测位置中心化，并把缺失位置置为 0。该处理使得后续矩阵乘法可以直接运行，同时保留观测掩码用于构造比例矩阵。

### 2.3 加性误差修正

原始尺度下

$$
Z_i=X_i+W_i.
$$

因此

$$
\mathbb{E}(Z_iZ_i^\top)=\mathbb{E}(X_iX_i^\top)+\mathbb{E}(W_iW_i^\top)=\Sigma_X+\tau^2I_p.
$$

所以无标准化时的无偏修正为

$$
\widehat\Gamma=\frac{1}{n}Z^\top Z-\tau^2I_p.
$$

若代码内部使用标准化变量 $Z_{ij}^{\ast}=Z_{ij}/s_j$，则误差项也变为

$$
W_{ij}^{\ast}=\frac{W_{ij}}{s_j}.
$$

于是

$$
\operatorname{Var}(W_{ij}^{\ast})=\frac{\tau^2}{s_j^2}.
$$

因此当前实现使用

$$
\widehat\Gamma^{\ast}=\frac{1}{n}Z^{\ast\top}Z^\ast-\operatorname{diag}\left(\frac{\tau^2}{s_1^2},\ldots,\frac{\tau^2}{s_p^2}\right).
$$

对应的交叉矩为

$$
\widehat\gamma^{\ast}=\frac{1}{n}Z^{\ast\top}y^\ast.
$$

因为 $W_i$ 与 $y_i$ 独立且均值为 0，所以

$$
\mathbb{E}(Z_iy_i)=\mathbb{E}(X_iy_i)+\mathbb{E}(W_iy_i)=\Sigma_X\beta^0.
$$

这说明交叉矩不需要减去 $\tau^2$ 项。

### 2.4 缺失数据修正

在缺失数据场景中，记中心化后真实变量仍为 $X_{ij}$。令

$$
Z_{ij}=M_{ij}X_{ij}.
$$

若缺失机制与 $X_i,y_i$ 独立，则

$$
\mathbb{E}(Z_{ij}Z_{ik})=\mathbb{E}(M_{ij}M_{ik})\mathbb{E}(X_{ij}X_{ik})=R_{jk}\Sigma_{X,jk}.
$$

因此

$$
\Sigma_{X,jk}=\frac{\mathbb{E}(Z_{ij}Z_{ik})}{R_{jk}}.
$$

代码用样本观测比例

$$
\widehat R_{jk}=\frac{1}{n}\sum_{i=1}^n 1\{Z_{ij}\text{ observed}, Z_{ik}\text{ observed}\}
$$

构造

$$
\widehat\Gamma_{jk}=\frac{n^{-1}\sum_i Z_{ij}Z_{ik}}{\widehat R_{jk}}.
$$

对交叉矩，若

$$
R_{jj}=\mathbb{P}(M_{ij}=1),
$$

则

$$
\widehat\gamma_j=\frac{n^{-1}\sum_i Z_{ij}y_i}{\widehat R_{jj}}.
$$

代码中 `_validate_ratio_matrix()` 要求比例矩阵所有元素为正且有限，否则修正会出现除零或无穷值。

### 2.5 乘性误差修正

乘性误差模型为

$$
Z_{ij}=X_{ij}U_{ij}, \qquad \log U_{ij}\sim N(0,\tau^2).
$$

正态矩母函数给出

$$
\mathbb{E}(U_{ij})=e^{\tau^2/2}, \qquad \mathbb{E}(U_{ij}^2)=e^{2\tau^2}.
$$

当 $j\ne k$ 且乘性误差相互独立时，

$$
\mathbb{E}(Z_{ij}Z_{ik})=\mathbb{E}(X_{ij}X_{ik})\mathbb{E}(U_{ij})\mathbb{E}(U_{ik})=\Sigma_{X,jk}e^{\tau^2}.
$$

因此非对角元素修正为

$$
\widehat\Gamma_{jk}=\frac{n^{-1}\sum_i Z_{ij}Z_{ik}}{e^{\tau^2}}, \qquad j\ne k.
$$

对角元素满足

$$
\mathbb{E}(Z_{ij}^2)=\mathbb{E}(X_{ij}^2)e^{2\tau^2},
$$

所以

$$
\widehat\Gamma_{jj}=\frac{n^{-1}\sum_i Z_{ij}^2}{e^{2\tau^2}}.
$$

交叉矩满足

$$
\mathbb{E}(Z_{ij}y_i)=\mathbb{E}(U_{ij})\mathbb{E}(X_{ij}y_i)=e^{\tau^2/2}\rho_{X,j},
$$

因此

$$
\widehat\gamma_j=\frac{n^{-1}\sum_i Z_{ij}y_i}{e^{\tau^2/2}}.
$$

这些公式由 `_corrected_covariance_multiplicative()` 实现。

### 2.6 PSD 投影

修正矩阵 $\widehat\Gamma$ 在有限样本下可能不是正半定矩阵。若直接求解

$$
\frac{1}{2}\beta^\top\widehat\Gamma\beta-\widehat\gamma^\top\beta+\lambda\|\beta\|_1,
$$

目标函数可能非凸。CoCoLasso 的关键步骤是把 $\widehat\Gamma$ 投影为正半定矩阵。

代码支持两种方式。

第一种是 ADMM 投影 `_admm_proj()`。其目标可理解为在最大范数意义下寻找接近 $\widehat\Gamma$ 的正半定矩阵：

$$
\widehat\Gamma_+ = \arg\min_{\Gamma\succeq 0}\|\Gamma-\widehat\Gamma\|_{\max}.
$$

实现中使用变量 $R,S,L$ 的 ADMM 迭代，把 PSD 约束和距离约束分离。每次迭代包括三步。

第一步，对矩阵

$$
W=\widehat\Gamma+S+\mu L
$$

做特征分解，并把特征值截断到非负：

$$
R \leftarrow Q\operatorname{diag}\{\max(d_j,\epsilon)\}Q^\top.
$$

第二步，更新 $S$。代码对下三角向量做 $\ell_1$ 球投影，从而控制最大范数距离。

第三步，更新拉格朗日变量：

$$
L \leftarrow L-\frac{R-S-\widehat\Gamma}{\mu}.
$$

第二种是 HM 投影 `_hm_proj()`。该方法使用 Frobenius 范数或最大范数形式，核心仍然是交替进行 PSD 特征值截断和残差更新。

### 2.7 协方差形式 Lasso

完成修正与投影后，代码求解

$$
\widehat\beta_\lambda=\arg\min_{\beta\in\mathbb{R}^p}\left\{\frac{1}{2}\beta^\top\widehat\Gamma_+\beta-\widehat\gamma^\top\beta+\lambda\|\beta\|_1\right\}.
$$

默认求解器 `_lasso_covariance()` 使用坐标下降。固定除第 $j$ 个坐标外的其他坐标，令

$$
s_j=\sum_{k=1}^p \widehat\Gamma_{+,jk}\beta_k.
$$

目标函数关于 $\beta_j$ 的一维部分为

$$
\frac{1}{2}\widehat\Gamma_{+,jj}\beta_j^2+eta_j\left(\sum_{k\ne j}\widehat\Gamma_{+,jk}\beta_k-\widehat\gamma_j\right)+\lambda|\beta_j|.
$$

记

$$
a_j=\widehat\Gamma_{+,jj}, \qquad b_j=\widehat\gamma_j-\sum_{k\ne j}\widehat\Gamma_{+,jk}\beta_k.
$$

则一维问题为

$$
\min_t \frac{1}{2}a_jt^2-b_jt+\lambda|t|.
$$

软阈值解为

$$
t=\frac{S(b_j,\lambda)}{a_j},
$$

其中

$$
S(b,\lambda)=\operatorname{sign}(b)(|b|-\lambda)_+.
$$

代码中的更新式与该公式等价，只是用 $S0=s_j-a_j\beta_j-\widehat\gamma_j$ 表示负方向梯度。

可选求解器 `_lasso_sklearn()` 使用 Cholesky 分解把协方差形式转换为伪数据形式。若

$$
\widehat\Gamma_+=U^\top U,
$$

构造

$$
\widetilde W=\sqrt{p}U, \qquad \widetilde Y=U^{-\top}\sqrt{p}\widehat\gamma.
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

再由 $\widetilde Y=U^{-\top}\sqrt p\widehat\gamma$ 得

$$
\frac{1}{p}\widetilde Y^\top\sqrt p U\beta
=\frac{1}{p}(\sqrt p\widehat\gamma^\top U^{-1})\sqrt p U\beta
=\widehat\gamma^\top\beta.
$$

所以 sklearn 伪数据目标与协方差形式目标只差一个常数项。

### 2.8 Lambda 路径与交叉验证

若 `alpha=None`，代码先计算最大正则化参数。对于加性误差，

$$
\lambda_{\max}=\left\|\frac{1}{n}Z^\top y\right\|_\infty.
$$

缺失数据和乘性误差下，代码使用相应修正后的 $\widehat\gamma$ 计算 $\lambda_{\max}$。

然后构造对数网格

$$
\lambda_1>\lambda_2>\cdots>\lambda_m,
$$

其中

$$
\lambda_m=\text{lambda_factor}\cdot\lambda_1.
$$

若 `alpha` 是数值，则代码只使用一个固定正则化参数：

$$
\lambda=\alpha.
$$

交叉验证中，每一折只用训练折估计中心化、标准化、缺失比例和加性误差尺度，再把同一组训练折预处理参数应用到验证折。这样避免使用验证折信息估计预处理参数导致的数据泄漏。

每个 $\lambda$ 的验证误差为

$$
\operatorname{CV}(\lambda)=\frac{1}{K}\sum_{k=1}^K\left(\widehat\beta_{\lambda}^{(-k)\top}\widehat\Gamma_{+,\text{test}}^{(k)}\widehat\beta_{\lambda}^{(-k)}-2\widehat\gamma_{\text{test}}^{(k)\top}\widehat\beta_{\lambda}^{(-k)}\right).
$$

代码返回最小 CV 误差对应的 `lambda_opt`，并按 1-std 规则返回更强正则化的 `lambda_sd`。

### 2.9 原始尺度恢复

内部估计量是在预处理尺度下得到的。若内部变量为

$$
Z_j^\ast=\frac{Z_j-\bar Z_j}{s_j}, \qquad y^\ast=\frac{y-\bar y}{s_y},
$$

则内部模型为

$$
y^\ast=Z^{\ast\top}\beta^\ast.
$$

代回原始尺度：

$$
\frac{y-\bar y}{s_y}=\sum_{j=1}^p\frac{Z_j-\bar Z_j}{s_j}\beta_j^\ast.
$$

两边乘以 $s_y$：

$$
y-\bar y=\sum_{j=1}^p Z_j\frac{s_y}{s_j}\beta_j^\ast-\sum_{j=1}^p\bar Z_j\frac{s_y}{s_j}\beta_j^\ast.
$$

因此原始尺度系数为

$$
\beta_j=\frac{s_y}{s_j}\beta_j^\ast,
$$

截距为

$$
\beta_0^{\text{intercept}}=\bar y-\bar Z^\top\beta.
$$

若用户关闭中心化或标准化，代码按实际预处理开关恢复对应尺度。

## 3. 相合性证明

本节证明标准 CoCoLasso 估计量的相合性。证明采用高维 Lasso 的基本不等式、修正矩阵的一致性和限制特征值条件。

### 3.1 估计量定义

令 $\widehat\Gamma_+$ 表示投影后的修正 Gram 矩阵，$\widehat\gamma$ 表示修正交叉矩。估计量定义为

$$
\widehat\beta=\arg\min_{\beta}\left\{L_n(\beta)+\lambda_n\|\beta\|_1\right\},
$$

其中

$$
L_n(\beta)=\frac{1}{2}\beta^\top\widehat\Gamma_+\beta-\widehat\gamma^\top\beta.
$$

记真实支持集为

$$
S=\{j:\beta_j^0\ne0\}, \qquad |S|=s.
$$

记误差向量

$$
\Delta=\widehat\beta-\beta^0.
$$

### 3.2 条件

条件 A1：样本独立同分布，$X_i$、误差变量和 $\varepsilon_i$ 具有足够的亚高斯或有限高阶矩尾部，且测量误差与 $(X_i,\varepsilon_i)$ 独立。

条件 A2：修正矩阵和交叉矩满足最大范数集中不等式。存在 $a_n\to0$，通常

$$
a_n=\sqrt{\frac{\log p}{n}},
$$

使得以概率趋于 1，

$$
\|\widehat\Gamma-\Sigma_X\|_{\max}\le C_1a_n,
$$

且

$$
\|\widehat\gamma-\Sigma_X\beta^0\|_\infty\le C_2a_n.
$$

条件 A3：PSD 投影误差不改变上述阶数。即存在数值误差 $r_n$，满足

$$
\|\widehat\Gamma_+-\Sigma_X\|_{\max}\le C_3a_n+r_n,
$$

其中 $r_n=o(\lambda_n)$。若投影算法精确且总体矩阵 $\Sigma_X$ 正半定，该条件是 CoCoLasso 理论中投影步骤的基本要求。

条件 A4：限制特征值条件成立。存在常数 $\kappa>0$，使得对所有满足锥约束

$$
\|v_{S^c}\|_1\le3\|v_S\|_1
$$

的向量 $v$，有

$$
v^\top\widehat\Gamma_+v\ge\kappa\|v\|_2^2.
$$

条件 A5：正则化参数满足

$$
\lambda_n\ge2\eta_n,
$$

其中

$$
\eta_n=\|\widehat\gamma-\widehat\Gamma_+\beta^0\|_\infty.
$$

由 A2 和 A3 可得

$$
\eta_n\le \|\widehat\gamma-\Sigma_X\beta^0\|_\infty+\|\widehat\Gamma_+-\Sigma_X\|_{\max}\|\beta^0\|_1=O_p(a_n+r_n).
$$

所以通常取

$$
\lambda_n\asymp\sqrt{\frac{\log p}{n}}
$$

即可满足 A5。

### 3.3 基本不等式

由于 $\widehat\beta$ 最小化目标函数，必有

$$
L_n(\widehat\beta)+\lambda_n\|\widehat\beta\|_1\le L_n(\beta^0)+\lambda_n\|\beta^0\|_1.
$$

代入 $\widehat\beta=\beta^0+\Delta$。先展开二次项：

$$
L_n(\beta^0+\Delta)=\frac{1}{2}(\beta^0+\Delta)^\top\widehat\Gamma_+(\beta^0+\Delta)-\widehat\gamma^\top(\beta^0+\Delta).
$$

继续展开：

$$
L_n(\beta^0+\Delta)=\frac{1}{2}\beta^{0\top}\widehat\Gamma_+\beta^0+\Delta^\top\widehat\Gamma_+\beta^0+\frac{1}{2}\Delta^\top\widehat\Gamma_+\Delta-\widehat\gamma^\top\beta^0-\widehat\gamma^\top\Delta.
$$

而

$$
L_n(\beta^0)=\frac{1}{2}\beta^{0\top}\widehat\Gamma_+\beta^0-\widehat\gamma^\top\beta^0.
$$

两式相减得到

$$
L_n(\beta^0+\Delta)-L_n(\beta^0)=\frac{1}{2}\Delta^\top\widehat\Gamma_+\Delta-\Delta^\top(\widehat\gamma-\widehat\Gamma_+\beta^0).
$$

代回基本不等式：

$$
\frac{1}{2}\Delta^\top\widehat\Gamma_+\Delta-\Delta^\top(\widehat\gamma-\widehat\Gamma_+\beta^0)+\lambda_n\|\beta^0+\Delta\|_1\le\lambda_n\|\beta^0\|_1.
$$

移项：

$$
\frac{1}{2}\Delta^\top\widehat\Gamma_+\Delta\le\Delta^\top(\widehat\gamma-\widehat\Gamma_+\beta^0)+\lambda_n(\|\beta^0\|_1-\|\beta^0+\Delta\|_1).
$$

### 3.4 随机误差项界

由 Holder 不等式，

$$
\Delta^\top(\widehat\gamma-\widehat\Gamma_+\beta^0)\le\|\Delta\|_1\|\widehat\gamma-\widehat\Gamma_+\beta^0\|_\infty=\eta_n\|\Delta\|_1.
$$

由 A5，$\eta_n\le\lambda_n/2$，所以

$$
\Delta^\top(\widehat\gamma-\widehat\Gamma_+\beta^0)\le\frac{\lambda_n}{2}\|\Delta\|_1.
$$

### 3.5 惩罚项分解

因为 $\beta^0_{S^c}=0$，

$$
\|\beta^0\|_1=\|\beta^0_S\|_1.
$$

同时

$$
\|\beta^0+\Delta\|_1=\|\beta^0_S+\Delta_S\|_1+\|\Delta_{S^c}\|_1.
$$

由三角不等式，

$$
\|\beta^0_S+\Delta_S\|_1\ge\|\beta^0_S\|_1-\|\Delta_S\|_1.
$$

因此

$$
\|\beta^0\|_1-\|\beta^0+\Delta\|_1\le\|\Delta_S\|_1-\|\Delta_{S^c}\|_1.
$$

代回基本不等式：

$$
\frac{1}{2}\Delta^\top\widehat\Gamma_+\Delta\le\frac{\lambda_n}{2}(\|\Delta_S\|_1+\|\Delta_{S^c}\|_1)+\lambda_n(\|\Delta_S\|_1-\|\Delta_{S^c}\|_1).
$$

合并同类项：

$$
\frac{1}{2}\Delta^\top\widehat\Gamma_+\Delta\le\frac{3\lambda_n}{2}\|\Delta_S\|_1-\frac{\lambda_n}{2}\|\Delta_{S^c}\|_1.
$$

左边非负，因此右边非负，得到

$$
\frac{3\lambda_n}{2}\|\Delta_S\|_1-\frac{\lambda_n}{2}\|\Delta_{S^c}\|_1\ge0.
$$

两边乘以 $2/\lambda_n$：

$$
3\|\Delta_S\|_1-\|\Delta_{S^c}\|_1\ge0.
$$

所以

$$
\|\Delta_{S^c}\|_1\le3\|\Delta_S\|_1.
$$

这就是锥约束。

### 3.6 L2 误差界

在锥约束下，由 A4，

$$
\Delta^\top\widehat\Gamma_+\Delta\ge\kappa\|\Delta\|_2^2.
$$

又由上一节的不等式去掉负项，

$$
\frac{1}{2}\Delta^\top\widehat\Gamma_+\Delta\le\frac{3\lambda_n}{2}\|\Delta_S\|_1.
$$

利用 Cauchy 不等式，

$$
\|\Delta_S\|_1\le\sqrt{s}\|\Delta_S\|_2\le\sqrt{s}\|\Delta\|_2.
$$

因此

$$
\frac{\kappa}{2}\|\Delta\|_2^2\le\frac{3\lambda_n}{2}\sqrt{s}\|\Delta\|_2.
$$

若 $\Delta\ne0$，两边除以 $\|\Delta\|_2$，得到

$$
\|\widehat\beta-\beta^0\|_2=\|\Delta\|_2\le\frac{3\sqrt{s}\lambda_n}{\kappa}.
$$

若 $\Delta=0$，该不等式显然成立。

因此只要

$$
\sqrt{s}\lambda_n\to0,
$$

就有

$$
\|\widehat\beta-\beta^0\|_2\xrightarrow{p}0.
$$

### 3.7 L1 误差界

由锥约束，

$$
\|\Delta\|_1=\|\Delta_S\|_1+\|\Delta_{S^c}\|_1\le4\|\Delta_S\|_1.
$$

再用

$$
\|\Delta_S\|_1\le\sqrt{s}\|\Delta\|_2,
$$

得到

$$
\|\Delta\|_1\le4\sqrt{s}\|\Delta\|_2.
$$

代入 L2 界：

$$
\|\widehat\beta-\beta^0\|_1\le4\sqrt{s}\frac{3\sqrt{s}\lambda_n}{\kappa}=\frac{12s\lambda_n}{\kappa}.
$$

因此只要

$$
s\lambda_n\to0,
$$

就有

$$
\|\widehat\beta-\beta^0\|_1\xrightarrow{p}0.
$$

### 3.8 预测误差界

预测误差可定义为

$$
\Delta^\top\Sigma_X\Delta.
$$

若 $\lambda_{\max}(\Sigma_X)\le C_\Sigma$，则

$$
\Delta^\top\Sigma_X\Delta\le C_\Sigma\|\Delta\|_2^2.
$$

代入 L2 界：

$$
\Delta^\top\Sigma_X\Delta\le C_\Sigma\frac{9s\lambda_n^2}{\kappa^2}.
$$

因此若

$$
s\lambda_n^2\to0,
$$

则预测误差收敛到 0。

### 3.9 结论

在 A1 到 A5 条件下，若

$$
\lambda_n\asymp\sqrt{\frac{\log p}{n}}, \qquad \sqrt{s}\lambda_n\to0,
$$

则本项目 CoCoLasso 目标函数对应的估计量满足

$$
\|\widehat\beta-\beta^0\|_2=O_p(\sqrt{s}\lambda_n).
$$

若进一步 $s\lambda_n\to0$，则

$$
\|\widehat\beta-\beta^0\|_1=O_p(s\lambda_n)=o_p(1).
$$

这就是相合性。

## 4. 渐进正态性证明

本节说明在低维或 oracle 条件下的渐进正态性。需要强调：高维 Lasso 估计量本身含有 $\ell_1$ 惩罚偏差，通常不能在不去偏、不 refit、不额外限制 $\lambda_n$ 的情况下直接得到普通渐进正态性。当前项目没有实现去偏估计器，因此这里证明的是当前修正二次目标在低维或惩罚足够小条件下的渐进正态性。

### 4.1 固定维条件

设 $p$ 固定，$n\to\infty$。假设：

1. $\Sigma_X$ 正定，最小特征值满足 $\lambda_{\min}(\Sigma_X)>c>0$。
2. $\widehat\Gamma\xrightarrow{p}\Sigma_X$，$\widehat\gamma\xrightarrow{p}\Sigma_X\beta^0$。
3. 中心极限定理适用于修正矩：

$$
\frac{1}{\sqrt n}\sum_{i=1}^n \psi_i\xrightarrow{d}N(0,\Omega),
$$

其中

$$
\psi_i=g_i-G_i\beta^0,
$$

$g_i$ 是单样本修正交叉矩贡献，$G_i$ 是单样本修正 Gram 矩阵贡献。

4. 惩罚参数满足

$$
\sqrt n\lambda_n\to0.
$$

5. PSD 投影在一阶渐近上不改变修正矩阵，即

$$
\sqrt n\|\widehat\Gamma_+-\widehat\Gamma\|=o_p(1).
$$

当 $p$ 固定且 $\lambda_{\min}(\Sigma_X)>c>0$ 时，$\widehat\Gamma$ 以概率趋于 1 已经正定，所以 PSD 投影最终不改变矩阵；此时第 5 条自然成立。

### 4.2 单样本影响函数：加性误差

先以加性误差为例推导。原始尺度下

$$
G_i=Z_iZ_i^\top-\tau^2I_p,
$$

$$
g_i=Z_iy_i.
$$

于是

$$
\widehat\Gamma=\frac{1}{n}\sum_i G_i, \qquad \widehat\gamma=\frac{1}{n}\sum_i g_i.
$$

定义

$$
\psi_i=g_i-G_i\beta^0=Z_iy_i-(Z_iZ_i^\top-\tau^2I_p)\beta^0.
$$

代入 $y_i=X_i^\top\beta^0+\varepsilon_i$ 和 $Z_i=X_i+W_i$：

$$
\psi_i=(X_i+W_i)(X_i^\top\beta^0+\varepsilon_i)-\{(X_i+W_i)(X_i+W_i)^\top-\tau^2I_p\}\beta^0.
$$

展开第一项：

$$
(X_i+W_i)(X_i^\top\beta^0+\varepsilon_i)=(X_i+W_i)X_i^\top\beta^0+(X_i+W_i)\varepsilon_i.
$$

展开第二项：

$$
(X_i+W_i)(X_i+W_i)^\top\beta^0=(X_i+W_i)X_i^\top\beta^0+(X_i+W_i)W_i^\top\beta^0.
$$

两者相减：

$$
\psi_i=(X_i+W_i)\varepsilon_i-(X_i+W_i)W_i^\top\beta^0+\tau^2\beta^0.
$$

验证均值为 0。先看第一项：

$$
\mathbb{E}\{(X_i+W_i)\varepsilon_i\}=\mathbb{E}(X_i\varepsilon_i)+\mathbb{E}(W_i)\mathbb{E}(\varepsilon_i)=0.
$$

再看第二项：

$$
\mathbb{E}\{(X_i+W_i)W_i^\top\beta^0\}=\mathbb{E}(X_iW_i^\top)\beta^0+\mathbb{E}(W_iW_i^\top)\beta^0.
$$

由于 $X_i$ 与 $W_i$ 独立且 $\mathbb{E}W_i=0$，

$$
\mathbb{E}(X_iW_i^\top)=0.
$$

又

$$
\mathbb{E}(W_iW_i^\top)=\tau^2I_p.
$$

所以

$$
\mathbb{E}\{(X_i+W_i)W_i^\top\beta^0\}=\tau^2\beta^0.
$$

因此

$$
\mathbb{E}\psi_i=0-\tau^2\beta^0+\tau^2\beta^0=0.
$$

若 $\mathbb{E}\|\psi_i\|^2<\infty$，多元中心极限定理给出

$$
\frac{1}{\sqrt n}\sum_{i=1}^n\psi_i\xrightarrow{d}N(0,\Omega),
$$

其中

$$
\Omega=\operatorname{Var}(\psi_i).
$$

### 4.3 修正得分展开

估计量的一阶条件为

$$
\widehat\Gamma_+\widehat\beta-\widehat\gamma+\lambda_n\widehat z=0,
$$

其中 $\widehat z\in\partial\|\widehat\beta\|_1$ 是次梯度。

把 $\widehat\beta=\beta^0+\Delta$ 代入：

$$
\widehat\Gamma_+(\beta^0+\Delta)-\widehat\gamma+\lambda_n\widehat z=0.
$$

整理：

$$
\widehat\Gamma_+\Delta=\widehat\gamma-\widehat\Gamma_+\beta^0-\lambda_n\widehat z.
$$

两边乘以 $\sqrt n$：

$$
\sqrt n\Delta=\widehat\Gamma_+^{-1}\sqrt n(\widehat\gamma-\widehat\Gamma_+\beta^0)-\sqrt n\lambda_n\widehat\Gamma_+^{-1}\widehat z.
$$

由 $\sqrt n\lambda_n\to0$ 且 $\|\widehat z\|$ 有界，

$$
\sqrt n\lambda_n\widehat\Gamma_+^{-1}\widehat z=o_p(1).
$$

现在分解第一项中的矩阵误差：

$$
\widehat\gamma-\widehat\Gamma_+\beta^0=(\widehat\gamma-\widehat\Gamma\beta^0)+(\widehat\Gamma-\widehat\Gamma_+)\beta^0.
$$

由 PSD 投影一阶可忽略条件，

$$
\sqrt n(\widehat\Gamma-\widehat\Gamma_+)\beta^0=o_p(1).
$$

因此

$$
\sqrt n(\widehat\gamma-\widehat\Gamma_+\beta^0)=\sqrt n(\widehat\gamma-\widehat\Gamma\beta^0)+o_p(1).
$$

又

$$
\sqrt n(\widehat\gamma-\widehat\Gamma\beta^0)=\frac{1}{\sqrt n}\sum_{i=1}^n(g_i-G_i\beta^0)=\frac{1}{\sqrt n}\sum_{i=1}^n\psi_i.
$$

所以

$$
\sqrt n(\widehat\gamma-\widehat\Gamma_+\beta^0)\xrightarrow{d}N(0,\Omega).
$$

由于

$$
\widehat\Gamma_+^{-1}\xrightarrow{p}\Sigma_X^{-1},
$$

Slutsky 定理给出

$$
\sqrt n(\widehat\beta-\beta^0)\xrightarrow{d}N(0,\Sigma_X^{-1}\Omega\Sigma_X^{-1}).
$$

这就是固定维且惩罚足够小条件下的渐进正态性。

### 4.4 活动集 oracle 渐进正态性

高维场景下，如果能正确识别支持集 $S$，或者理论上讨论 oracle 估计量，则可在活动集上得到类似结论。

oracle 估计量定义为

$$
\widehat\beta_S^{\text{or}}=\arg\min_{b\in\mathbb{R}^s}\left\{\frac{1}{2}b^\top\widehat\Gamma_{+,SS}b-\widehat\gamma_S^\top b\right\}, \qquad \widehat\beta_{S^c}^{\text{or}}=0.
$$

一阶条件为

$$
\widehat\Gamma_{+,SS}\widehat\beta_S^{\text{or}}=\widehat\gamma_S.
$$

所以

$$
\widehat\beta_S^{\text{or}}-\beta_S^0=\widehat\Gamma_{+,SS}^{-1}(\widehat\gamma_S-\widehat\Gamma_{+,SS}\beta_S^0).
$$

乘以 $\sqrt n$：

$$
\sqrt n(\widehat\beta_S^{\text{or}}-\beta_S^0)=\widehat\Gamma_{+,SS}^{-1}\sqrt n(\widehat\gamma_S-\widehat\Gamma_{+,SS}\beta_S^0).
$$

若 $s$ 固定，$\widehat\Gamma_{+,SS}\xrightarrow{p}\Sigma_{X,SS}$，并且

$$
\sqrt n(\widehat\gamma_S-\widehat\Gamma_{+,SS}\beta_S^0)\xrightarrow{d}N(0,\Omega_{SS}),
$$

则

$$
\sqrt n(\widehat\beta_S^{\text{or}}-\beta_S^0)\xrightarrow{d}N(0,\Sigma_{X,SS}^{-1}\Omega_{SS}\Sigma_{X,SS}^{-1}).
$$

若还满足 beta-min 条件

$$
\min_{j\in S}|\beta_j^0|\gg\lambda_n
$$

以及适当的 irrepresentable 或 primal-dual witness 条件，Lasso 可以以概率趋于 1 选择正确支持集。此时活动集上的 Lasso 与 oracle 解渐近等价，进而获得活动集上的渐进正态性。但这类变量选择一致性条件比估计相合性更强，当前项目未显式检验这些条件。

### 4.5 缺失与乘性误差下的影响函数

缺失与乘性误差的证明结构相同。只需把单样本 $G_i$ 和 $g_i$ 换为相应的无偏修正贡献。

缺失数据中，若 $R_{jk}$ 已知，则

$$
G_{i,jk}=\frac{M_{ij}M_{ik}X_{ij}X_{ik}}{R_{jk}},
$$

$$
g_{i,j}=\frac{M_{ij}X_{ij}y_i}{R_{jj}}.
$$

由独立缺失机制，

$$
\mathbb{E}G_{i,jk}=\Sigma_{X,jk}, \qquad \mathbb{E}g_{i,j}=\rho_{X,j}.
$$

乘性误差中，定义

$$
G_{i,jk}=\begin{cases}
Z_{ij}Z_{ik}/e^{\tau^2}, & j\ne k,\\
Z_{ij}^2/e^{2\tau^2}, & j=k,
\end{cases}
$$

以及

$$
g_{i,j}=Z_{ij}y_i/e^{\tau^2/2}.
$$

由 log-normal 矩公式，

$$
\mathbb{E}G_i=\Sigma_X, \qquad \mathbb{E}g_i=\Sigma_X\beta^0.
$$

于是仍可定义

$$
\psi_i=g_i-G_i\beta^0,
$$

并在有限二阶矩和 CLT 条件下得到

$$
\frac{1}{\sqrt n}\sum_i\psi_i\xrightarrow{d}N(0,\Omega).
$$

后续一阶条件展开与加性误差完全相同。

## 5. 与当前项目实现的对应关系

`_preprocess_data()` 对应第 2.2 节。

`_additive_noise_variance()` 对应第 2.3 节标准化后的 $\tau^2/s_j^2$ 修正。

`_ratio_matrix_from_mask()` 和 `_validate_ratio_matrix()` 对应第 2.4 节缺失比例矩阵。

`_corrected_covariance_multiplicative()` 对应第 2.5 节乘性误差修正。

`_admm_proj()` 和 `_hm_proj()` 对应第 2.6 节 PSD 投影。

`_lasso_covariance()` 和 `_lasso_sklearn()` 对应第 2.7 节协方差形式 Lasso 求解。

`_cv_covariance_matrices()` 对应第 2.8 节 fold 内预处理与交叉验证误差。

`_restore_coefficients()`、`_restore_coefficient_path()` 和 `_restore_intercept()` 对应第 2.9 节原始尺度恢复。

## 6. 结论与限制

第一，项目实现的标准 `CoCoLasso` 主线符合 CoCoLasso 的基本结构：修正协方差矩阵、PSD 投影、凸 Lasso 求解和交叉验证选择正则化参数。

第二，在修正矩阵集中、PSD 投影误差可控、限制特征值成立、$\lambda_n\asymp\sqrt{\log p/n}$ 且 $\sqrt{s}\lambda_n\to0$ 的条件下，估计量满足 L2 相合性；若 $s\lambda_n\to0$，还满足 L1 相合性。

第三，渐进正态性不是高维惩罚估计量自动具备的无条件结论。它需要固定维、oracle 活动集、惩罚项满足 $\sqrt n\lambda_n\to0$，或额外实现去偏估计。本项目当前没有实现去偏估计、活动集 refit 的标准误估计或置信区间输出，因此文档中的渐进正态性是算法对应理论目标在额外条件下的数学性质。

第四，若要把渐进正态性转化为可运行的推断模块，还需要实现 $\Omega$ 的一致估计、活动集或去偏校正、以及置信区间和假设检验接口。
