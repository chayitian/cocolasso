"""
CoCoLasso（凸校正 Lasso）Python 实现

高维误差变量回归，支持：
- 协变量加性测量误差
- 协变量缺失数据
- 块下降 CoCoLasso（BD-CoCoLasso）
- 三块推广（混合加性误差 + 缺失数据）

参考文献: https://arxiv.org/pdf/1510.07123.pdf
"""

import numpy as np
from typing import Optional, List, Dict


def _l1_proj(v: np.ndarray, b: float) -> np.ndarray:
    """
    高效投影到半径为 b 的 L1 球上。

    参考文献: Duchi et al. (2008), Efficient Projections onto the L1-Ball
    for Learning in High Dimensions, ICML.

    参数
    ----------
    v : ndarray, 待投影向量
    b : float, L1 球半径（必须 > 0）

    返回
    ----------
    w : ndarray, 投影后的向量
    """
    assert b > 0, "半径 b 必须为正数"
    u = np.sort(np.abs(v))[::-1]
    sv = np.cumsum(u)
    rho = np.max(np.where(u > (sv - b) / np.arange(1, len(u) + 1)))
    theta = max(0, (sv[rho] - b) / (rho + 1))
    w = np.sign(v) * np.maximum(np.abs(v) - theta, 0)
    return w


def _admm_proj(
    mat: np.ndarray,
    epsilon: float = 1e-4,
    mu: float = 10,
    it_max: float = 1e3,
    etol: float = 1e-4,
    etol_distance: float = 1e-4,
) -> Dict:
    """
    ADMM 算法：寻找关于最大范数距离最近的半正定矩阵。

    求解: min_{R,S} ||R - mat||_max  s.t.  R >= 0, R - S = mat, ||S||_1 <= mu/2

    参数
    ----------
    mat : (p, p) ndarray, 待投影矩阵
    epsilon : float, PSD 空间的近似阈值
    mu : float, ADMM 惩罚参数
    it_max : int, 最大迭代次数
    etol : float, 原始/对偶残差收敛容差
    etol_distance : float, 距离收敛容差

    返回
    ----------
    dict, 包含:
        'mat': 投影后的 PSD 矩阵
        'df_ADMM': 收敛诊断信息字典
    """
    p = mat.shape[0]
    R = np.diag(np.diag(mat)).copy()
    S = np.zeros((p, p))
    L = np.zeros((p, p))

    itr = 0
    it_max = int(it_max)
    iterations, eps_R_list, eps_S_list, eps_primal_list, distance_list = [], [], [], [], []

    while itr < it_max:
        Rp = R.copy()
        Sp = S.copy()

        W = mat + S + mu * L
        W_eigvals, W_eigvecs = np.linalg.eigh(W)
        W_eigvals = np.maximum(W_eigvals, epsilon)
        R = W_eigvecs @ np.diag(W_eigvals) @ W_eigvecs.T

        M = R - mat - mu * L
        lower_idx = np.tril_indices(p)
        M_lower = M[lower_idx]
        S_lower = M_lower - _l1_proj(M_lower, mu / 2)
        S = np.zeros((p, p))
        S[lower_idx] = S_lower
        S = S + S.T - np.diag(np.diag(S))

        L = L - (R - S - mat) / mu

        iterations.append(itr)
        eps_R_list.append(np.max(np.abs(R - Rp)))
        eps_S_list.append(np.max(np.abs(S - Sp)))
        eps_primal_list.append(np.max(np.abs(R - S - mat)))
        distance_list.append(np.max(np.abs(R - mat)))

        if (
            (np.max(np.abs(R - Rp)) < etol)
            and (np.max(np.abs(S - Sp)) < etol)
            and (np.max(np.abs(R - S - mat)) < etol)
        ) or (abs(np.max(np.abs(Rp - mat)) - np.max(np.abs(R - mat))) < etol_distance):
            break

        itr += 1
        if itr % 20 == 0:
            mu = mu / 2

    return {
        "mat": R,
        "df_ADMM": {
            "iteration": iterations,
            "eps_R": eps_R_list,
            "eps_S": eps_S_list,
            "eps_primal": eps_primal_list,
            "distance": distance_list,
        },
    }


def _hm_proj(
    sigma_hat: np.ndarray,
    R: Optional[np.ndarray] = None,
    a: float = 1,
    iter_max: int = 1000,
    epsilon: float = 1e-4,
    mu: float = 10,
    tolerance: float = 1e-4,
    norm: str = "F",
) -> np.ndarray:
    """
    HM-lasso 算法：寻找关于 Frobenius 范数距离最近的半正定矩阵。

    参数
    ----------
    sigma_hat : (p, p) ndarray, 待投影协方差矩阵
    R : (p, p) ndarray or None, 权重矩阵
    a : float, hmlasso 取 1，cocolasso 取 0
    iter_max : int, 最大迭代次数
    epsilon : float, 停止准则
    mu : float, 惩罚参数
    tolerance : float, 收敛容差
    norm : str, 'F' 为 Frobenius 范数，'max' 为最大范数

    返回
    ----------
    Ak : (p, p) ndarray, 投影后的 PSD 矩阵
    """
    n = sigma_hat.shape[0]
    S_paired = sigma_hat.copy()

    if R is None:
        W = np.ones((n, n)) ** a
    else:
        W = R ** a

    Ak = S_paired.copy()
    Bk = np.zeros((n, n))
    Lk = np.zeros((n, n))

    for _ in range(iter_max):
        A = Bk + S_paired + mu * Lk
        A_eigvals, A_eigvecs = np.linalg.eigh(A)
        A_eigvals = np.maximum(A_eigvals, epsilon)
        Akp1 = A_eigvecs @ np.diag(A_eigvals) @ A_eigvecs.T

        if norm == "F":
            Bkp1 = (Akp1 - S_paired - mu * Lk) / (mu * W * W + np.ones((n, n)))
        else:
            C = Akp1 - S_paired - mu * Lk
            C_vec = C.flatten()
            W_vec = W.flatten()
            WC = np.abs(C_vec) * W_vec
            WC_sort_idx = np.argsort(WC)[::-1]
            W_sort = W_vec[WC_sort_idx]
            C_sort = C_vec[WC_sort_idx]
            l = 0
            frac = 0
            while l < len(C_vec) and W_sort[l] * abs(C_sort[l]) > frac:
                l += 1
                frac = (np.sum(np.abs(C_sort[:l])) - mu / 2) / np.sum(1.0 / W_sort[:l])
            d = frac
            b_vec = np.where(
                W_vec * np.abs(C_vec) > d,
                d * np.sign(C_vec) / W_vec,
                C_vec,
            )
            Bkp1 = b_vec.reshape(n, n)

        Lkp1 = Lk - (Akp1 - Bkp1 - S_paired) / mu

        if (
            max(
                np.max(np.abs(Akp1 - Ak)),
                np.max(np.abs(Bkp1 - Bk)),
                np.max(np.abs(Lkp1 - Lk)),
            )
            < tolerance
        ):
            Ak = Akp1
            break

        Ak = Akp1
        Bk = Bkp1
        Lk = Lkp1

    return Ak


def _scad_weight(beta_j: float, lambda_val: float, a: float = 3.7) -> float:
    """
    计算给定系数值的 SCAD 惩罚权重。

    SCAD 惩罚导数:
      - |beta| <= lambda:  w = 1
      - lambda < |beta| <= a*lambda:  w = (a*lambda - |beta|) / (lambda*(a-1))
      - |beta| > a*lambda:  w = 0
    """
    abs_beta = abs(beta_j)
    if abs_beta <= lambda_val:
        return 1.0
    elif abs_beta <= a * lambda_val:
        return (a * lambda_val - abs_beta) / (lambda_val * (a - 1))
    else:
        return 0.0


def _lasso_covariance(
    n: int,
    p: int,
    lambda_val: float,
    XX: np.ndarray,
    Xy: np.ndarray,
    beta_start: np.ndarray,
    penalty: str = "lasso",
    max_iter: int = 1000,
    opt_tol: float = 1e-5,
    zero_threshold: float = 1e-6,
) -> Dict:
    """
    协方差形式求解 Lasso:
        min_beta  (1/2) beta' XX beta - Xy' beta + lambda * ||beta||_1

    使用坐标下降法，可选 SCAD 自适应权重。

    参数
    ----------
    n : int, 样本数
    p : int, 特征数
    lambda_val : float, 惩罚参数
    XX : (p, p) ndarray, 协方差矩阵 (Sigma)
    Xy : (p,) or (p,1) ndarray, 交叉协方差向量 (rho)
    beta_start : (p,) ndarray, 初始 beta 值
    penalty : str, 'lasso' 或 'SCAD'
    max_iter : int, 最大迭代次数
    opt_tol : float, 收敛容差
    zero_threshold : float, 小系数归零阈值

    返回
    ----------
    dict, 包含:
        'coefficients': 最终 beta 向量
        'num_it': 实际迭代次数
    """
    beta = beta_start.copy().astype(float)
    Xy = Xy.ravel()

    s = XX @ beta
    lambda0 = lambda_val
    m = 1

    while m < max_iter:
        beta_old = beta.copy()
        for j in range(p):
            S0 = s[j] - XX[j, j] * beta_old[j] - Xy[j]

            if np.any(np.isnan(S0)):
                beta[j] = 0
                continue

            w_j = 1.0
            if penalty == "SCAD":
                w_j = _scad_weight(beta[j], lambda0)

            lambda_val = w_j * lambda0

            if S0 > lambda_val:
                beta[j] = (lambda_val - S0) / XX[j, j]
                s += XX[:, j] * (beta[j] - beta_old[j])
            elif S0 < -lambda_val:
                beta[j] = (-lambda_val - S0) / XX[j, j]
                s += XX[:, j] * (beta[j] - beta_old[j])
            else:
                beta[j] = 0

        if np.sum(np.abs(beta - beta_old)) < opt_tol:
            break
        m += 1

    beta[np.abs(beta) < zero_threshold] = 0
    return {"coefficients": beta, "num_it": m}


def _lasso_covariance_block(
    n: int,
    p1: int,
    p2: int,
    X1: np.ndarray,
    Z2: np.ndarray,
    y: np.ndarray,
    sigma1: np.ndarray,
    sigma2: np.ndarray,
    lambda_val: float,
    noise: str = "additive",
    ratio_matrix: Optional[np.ndarray] = None,
    beta1_start: Optional[np.ndarray] = None,
    beta2_start: Optional[np.ndarray] = None,
    penalty: str = "lasso",
    max_iter: int = 1000,
    opt_tol: float = 1e-5,
    zero_threshold: float = 1e-6,
) -> Dict:
    """
    二块 Lasso 协方差形式求解（BD-CoCoLasso）。

    交替块坐标下降：
      - 块 1（无误差）：标准 Lasso
      - 块 2（含误差）：CoCoLasso 校正 Lasso

    参数
    ----------
    n : int, 样本数
    p1 : int, 无误差特征数
    p2 : int, 含误差特征数
    X1 : (n, p1) ndarray, 无误差设计矩阵
    Z2 : (n, p2) ndarray, 含误差设计矩阵
    y : (n,) ndarray, 响应向量
    sigma1 : (p1, p1) ndarray, X1 的协方差矩阵
    sigma2 : (p2, p2) ndarray, Z2 的校正后 PSD 协方差矩阵
    lambda_val : float, 惩罚参数
    noise : str, 'additive' 或 'missing'
    ratio_matrix : (p2, p2) ndarray or None, 缺失数据的观测比率矩阵
    beta1_start : (p1,) ndarray, 初始 beta1
    beta2_start : (p2,) ndarray, 初始 beta2
    penalty : str, 'lasso' 或 'SCAD'
    max_iter : int, 最大迭代次数
    opt_tol : float, 收敛容差
    zero_threshold : float, 小系数归零阈值

    返回
    ----------
    dict, 包含:
        'coefficients_beta1': 最终 beta1
        'coefficients_beta2': 最终 beta2
        'num_it': 实际迭代次数
    """
    beta1 = beta1_start.copy() if beta1_start is not None else np.zeros(p1)
    beta2 = beta2_start.copy() if beta2_start is not None else np.zeros(p2)
    y = y.ravel()

    if noise == "additive":
        Z2_tilde = Z2
        rho1 = (1 / n) * X1.T @ y
        rho2 = (1 / n) * Z2.T @ y
    elif noise == "missing":
        Z2_tilde = Z2 / np.diag(ratio_matrix)[np.newaxis, :]
        rho1 = (1 / n) * X1.T @ y
        rho2 = (1 / n) * (Z2.T @ y) / np.diag(ratio_matrix)
    else:
        raise ValueError(f"Unknown noise type: {noise}")

    m = 1
    while m < max_iter:
        beta1_old = beta1.copy()
        beta2_old = beta2.copy()

        if noise == "additive":
            Xy1 = (1 / n) * X1.T @ (y - Z2 @ beta2)
        else:
            Z2_tilde = Z2 / np.diag(ratio_matrix)[np.newaxis, :]
            Xy1 = (1 / n) * X1.T @ (y - Z2_tilde @ beta2)

        beta1 = _lasso_covariance(
            n=n, p=p1, lambda_val=lambda_val, XX=sigma1, Xy=Xy1,
            beta_start=beta1_old, penalty=penalty, max_iter=max_iter,
            opt_tol=opt_tol, zero_threshold=zero_threshold,
        )["coefficients"]

        if noise == "additive":
            Xy2 = (1 / n) * Z2.T @ (y - X1 @ beta1)
        else:
            Xy2 = (1 / n) * (Z2.T @ (y - X1 @ beta1)) / np.diag(ratio_matrix)

        beta2 = _lasso_covariance(
            n=n, p=p2, lambda_val=lambda_val, XX=sigma2, Xy=Xy2,
            beta_start=beta2_old, penalty=penalty, max_iter=max_iter,
            opt_tol=opt_tol, zero_threshold=zero_threshold,
        )["coefficients"]

        if (np.sum(np.abs(beta1 - beta1_old)) < opt_tol and
                np.sum(np.abs(beta2 - beta2_old)) < opt_tol):
            break
        m += 1

    beta1[np.abs(beta1) < zero_threshold] = 0
    beta2[np.abs(beta2) < zero_threshold] = 0
    return {"coefficients_beta1": beta1, "coefficients_beta2": beta2, "num_it": m}


def _lasso_covariance_block_general(
    n: int,
    p1: int,
    p2: int,
    p3: int,
    X1: Optional[np.ndarray],
    Z2: np.ndarray,
    Z3: np.ndarray,
    y: np.ndarray,
    sigma1: Optional[np.ndarray],
    sigma2: np.ndarray,
    sigma3: np.ndarray,
    lambda_val: float,
    ratio_matrix: Optional[np.ndarray] = None,
    beta1_start: Optional[np.ndarray] = None,
    beta2_start: Optional[np.ndarray] = None,
    beta3_start: Optional[np.ndarray] = None,
    penalty: str = "lasso",
    max_iter: int = 1000,
    opt_tol: float = 1e-5,
    zero_threshold: float = 1e-6,
) -> Dict:
    """
    三块 Lasso 协方差形式求解（三块 BD-CoCoLasso）。

    三块交替坐标下降：
      - 块 1（无误差）：标准 Lasso
      - 块 2（加性误差）：CoCoLasso 校正 Lasso
      - 块 3（缺失数据）：CoCoLasso 校正 Lasso

    参数
    ----------
    n : int, 样本数
    p1 : int, 无误差特征数
    p2 : int, 含加性误差特征数
    p3 : int, 含缺失数据特征数
    X1 : (n, p1) ndarray or None, 无误差设计矩阵
    Z2 : (n, p2) ndarray, 含加性误差设计矩阵
    Z3 : (n, p3) ndarray, 含缺失数据设计矩阵
    y : (n,) ndarray, 响应向量
    sigma1 : (p1, p1) ndarray or None, X1 的协方差矩阵
    sigma2 : (p2, p2) ndarray, Z2 的校正后 PSD 协方差矩阵
    sigma3 : (p3, p3) ndarray, Z3 的校正后 PSD 协方差矩阵
    lambda_val : float, 惩罚参数
    ratio_matrix : (p3, p3) ndarray or None, 缺失数据的观测比率矩阵
    beta1_start, beta2_start, beta3_start : ndarray, 初始系数向量
    penalty : str, 'lasso' 或 'SCAD'
    max_iter : int, 最大迭代次数
    opt_tol : float, 收敛容差
    zero_threshold : float, 小系数归零阈值

    返回
    ----------
    dict, 包含:
        'coefficients_beta1': 最终 beta1
        'coefficients_beta2': 最终 beta2
        'coefficients_beta3': 最终 beta3
        'num_it': 实际迭代次数
    """
    y = y.ravel()
    Z3_tilde = Z3 / np.diag(ratio_matrix)[np.newaxis, :]

    if p1 > 0 and X1 is not None:
        beta1 = beta1_start.copy() if beta1_start is not None else np.zeros(p1)
        beta2 = beta2_start.copy() if beta2_start is not None else np.zeros(p2)
        beta3 = beta3_start.copy() if beta3_start is not None else np.zeros(p3)

        m = 1
        while m < max_iter:
            beta1_old = beta1.copy()
            beta2_old = beta2.copy()
            beta3_old = beta3.copy()

            Xy1 = (1 / n) * X1.T @ (y - Z2 @ beta2 - Z3_tilde @ beta3)
            beta1 = _lasso_covariance(
                n=n, p=p1, lambda_val=lambda_val, XX=sigma1, Xy=Xy1,
                beta_start=beta1_old, penalty=penalty, max_iter=max_iter,
                opt_tol=opt_tol, zero_threshold=zero_threshold,
            )["coefficients"]

            Xy2 = (1 / n) * Z2.T @ (y - X1 @ beta1 - Z3_tilde @ beta3)
            beta2 = _lasso_covariance(
                n=n, p=p2, lambda_val=lambda_val, XX=sigma2, Xy=Xy2,
                beta_start=beta2_old, penalty=penalty, max_iter=max_iter,
                opt_tol=opt_tol, zero_threshold=zero_threshold,
            )["coefficients"]

            Xy3 = (1 / n) * (Z3.T @ (y - X1 @ beta1 - Z2 @ beta2)) / np.diag(ratio_matrix)
            beta3 = _lasso_covariance(
                n=n, p=p3, lambda_val=lambda_val, XX=sigma3, Xy=Xy3,
                beta_start=beta3_old, penalty=penalty, max_iter=max_iter,
                opt_tol=opt_tol, zero_threshold=zero_threshold,
            )["coefficients"]

            if (np.sum(np.abs(beta1 - beta1_old)) < opt_tol and
                    np.sum(np.abs(beta2 - beta2_old)) < opt_tol and
                    np.sum(np.abs(beta3 - beta3_old)) < opt_tol):
                break
            m += 1

        beta1[np.abs(beta1) < zero_threshold] = 0
        beta2[np.abs(beta2) < zero_threshold] = 0
        beta3[np.abs(beta3) < zero_threshold] = 0
        return {"coefficients_beta1": beta1, "coefficients_beta2": beta2, "coefficients_beta3": beta3, "num_it": m}

    else:
        beta2 = beta2_start.copy() if beta2_start is not None else np.zeros(p2)
        beta3 = beta3_start.copy() if beta3_start is not None else np.zeros(p3)

        m = 1
        while m < max_iter:
            beta2_old = beta2.copy()
            beta3_old = beta3.copy()

            Xy2 = (1 / n) * Z2.T @ (y - Z3_tilde @ beta3)
            beta2 = _lasso_covariance(
                n=n, p=p2, lambda_val=lambda_val, XX=sigma2, Xy=Xy2,
                beta_start=beta2_old, penalty=penalty, max_iter=max_iter,
                opt_tol=opt_tol, zero_threshold=zero_threshold,
            )["coefficients"]

            Xy3 = (1 / n) * (Z3.T @ (y - Z2 @ beta2)) / np.diag(ratio_matrix)
            beta3 = _lasso_covariance(
                n=n, p=p3, lambda_val=lambda_val, XX=sigma3, Xy=Xy3,
                beta_start=beta3_old, penalty=penalty, max_iter=max_iter,
                opt_tol=opt_tol, zero_threshold=zero_threshold,
            )["coefficients"]

            if (np.sum(np.abs(beta2 - beta2_old)) < opt_tol and
                    np.sum(np.abs(beta3 - beta3_old)) < opt_tol):
                break
            m += 1

        beta2[np.abs(beta2) < zero_threshold] = 0
        beta3[np.abs(beta3) < zero_threshold] = 0
        return {"coefficients_beta2": beta2, "coefficients_beta3": beta3, "num_it": m}


def _compute_ratio_matrix(Z: np.ndarray, p: Optional[int] = None, offset: int = 0) -> np.ndarray:
    """
    计算缺失数据的观测比率矩阵。

    R_{jk} = n_{jk} / n，其中 n_{jk} 是特征 j 和 k 同时被观测到（非 NaN）的行数。

    参数
    ----------
    Z : (n, p2) ndarray, 含缺失值的含误差设计矩阵
    p : int or None, 特征数（使用 Z 的前 p 列）
    offset : int, Z 中的列偏移量

    返回
    ----------
    ratio_matrix : (p2, p2) ndarray
    """
    if p is None:
        p = Z.shape[1]
    n = Z.shape[0]
    ratio_matrix = np.zeros((p, p))
    Z_sub = Z[:, offset:offset + p] if offset > 0 else Z[:, :p]

    mask = ~np.isnan(Z_sub)
    for i in range(p):
        for j in range(i, p):
            n_ij = np.sum(mask[:, i] & mask[:, j])
            ratio_matrix[i, j] = n_ij
            ratio_matrix[j, i] = n_ij
    ratio_matrix = ratio_matrix / n
    return ratio_matrix


def _log_space(start: float, stop: float, num: int) -> np.ndarray:
    """生成对数等间距序列（类似 R 的 lseq）。"""
    return np.exp(np.linspace(np.log(start), np.log(stop), num))


def _corrected_covariance_multiplicative(mat: np.ndarray, y: np.ndarray,
                                          n: int, p: int,
                                          tau: float) -> tuple:
    """
    计算乘性误差的校正协方差和交叉协方差。

    对于 Z = X * M，其中 log(M_ij) ~ N(0, tau^2)：
      Gamma_jk = (1/n) Z_j'Z_k / exp(tau^2)       当 j != k
      Gamma_jj = (1/n) Z_j'Z_j / exp(2*tau^2)
      rho_j    = (1/n) Z_j'y / exp(tau^2/2)

    返回
    ----------
    (Gamma, rho_tilde) : 校正后的协方差矩阵和交叉协方差向量
    """
    exp_tau2 = np.exp(tau ** 2)
    exp_2tau2 = np.exp(2 * tau ** 2)
    exp_tau2_half = np.exp(tau ** 2 / 2)

    Sigma_raw = (1 / n) * mat.T @ mat
    Gamma = Sigma_raw / exp_tau2
    np.fill_diagonal(Gamma, np.diag(Sigma_raw) / exp_2tau2)
    rho_tilde = ((1 / n) * mat.T @ y.ravel()) / exp_tau2_half
    return Gamma, rho_tilde


def _lambda_max(Z: np.ndarray, y: np.ndarray, n: int,
                ratio_matrix: Optional[np.ndarray] = None,
                noise: str = "additive",
                tau: Optional[float] = None) -> float:
    """
    计算最大 lambda 值（使所有系数为零的最小 lambda）。

    加性噪声:       lambda_max = max|rho_tilde|，其中 rho_tilde = (1/n) Z'y
    缺失数据:       lambda_max = max|rho_tilde|，其中 rho_tilde = (1/n) Z'y / diag(R)
    乘性噪声:       lambda_max = max|rho_tilde|，其中 rho_tilde = (1/n) Z'y / exp(tau^2/2)
    """
    y = y.ravel()
    if noise == "additive":
        rho_tilde = (1 / n) * Z.T @ y
    elif noise == "missing":
        rho_tilde = (1 / n) * Z.T @ y / np.diag(ratio_matrix)
    elif noise == "multiplicative":
        exp_tau2_half = np.exp(tau ** 2 / 2)
        rho_tilde = ((1 / n) * Z.T @ y) / exp_tau2_half
    else:
        raise ValueError(f"Unknown noise type: {noise}")
    return float(np.max(np.abs(rho_tilde)))


def _lambda_max_block(Z: np.ndarray, y: np.ndarray, n: int, p1: int, p2: int,
                      ratio_matrix: Optional[np.ndarray] = None,
                      noise: str = "additive") -> float:
    """计算块下降设定下的 lambda_max。"""
    y = y.ravel()
    X1 = Z[:, :p1]
    Z2 = Z[:, p1:]
    if noise == "additive":
        rho_tilde = (1 / n) * Z.T @ y
        return float(np.max(np.abs(rho_tilde)))
    else:
        rho_tilde1 = (1 / n) * X1.T @ y
        rho_tilde2 = (1 / n) * Z2.T @ y / np.diag(ratio_matrix)
        return float(max(np.max(np.abs(rho_tilde1)), np.max(np.abs(rho_tilde2))))


def _cv_covariance_matrices(
    K: int,
    mat: np.ndarray,
    y: np.ndarray,
    p: int,
    mu: float = 10,
    tau: Optional[float] = None,
    ratio_matrix: Optional[np.ndarray] = None,
    etol: float = 1e-4,
    noise: str = "additive",
    mode: str = "ADMM",
) -> Dict:
    """
    为 CoCoLasso 交叉验证创建投影后的 PSD 协方差矩阵。

    对每一折 k：
      - 在训练集上计算校正后的协方差
      - 投影到最近的 PSD 矩阵
      - 在测试集上计算校正后的协方差

    参数
    ----------
    K : int, 交叉验证折数
    mat : (n, p) ndarray, 设计矩阵
    y : (n,) ndarray, 响应向量
    p : int, 特征数
    mu : float, ADMM 惩罚参数
    tau : float or None, 加性误差的标准差
    ratio_matrix : (p, p) ndarray or None, 观测比率矩阵
    etol : float, ADMM 容差
    noise : str, 'additive'、'missing' 或 'multiplicative'
    mode : str, 'ADMM' 或 'HM'

    返回
    ----------
    dict, 包含:
        'sigma_global', 'rho_global',
        'list_matrices_lasso', 'list_matrices_error',
        'list_rho_lasso', 'list_rho_error'
    """
    n = mat.shape[0]
    n_without_fold = n - n // K
    n_one_fold = n // K

    folds = np.random.permutation(np.repeat(np.arange(1, K + 1), n // K))

    def _project(cov_mat, R=None):
        if mode == "ADMM":
            return _admm_proj(cov_mat, mu=mu, etol=etol)["mat"]
        else:
            return _hm_proj(cov_mat, R=R, mu=mu, tolerance=etol)

    if noise == "additive":
        cov_modified = (1 / n) * mat.T @ mat - tau ** 2 * np.eye(p)
        sigma_global = _project(cov_modified)
        rho_global = (1 / n) * mat.T @ y

    elif noise == "missing":
        cov_modified = (1 / n) * mat.T @ mat / ratio_matrix
        sigma_global = _project(cov_modified, R=ratio_matrix)
        rho_global = (1 / n) * mat.T @ y / np.diag(ratio_matrix)

    elif noise == "multiplicative":
        Gamma_global, rho_global = _corrected_covariance_multiplicative(
            mat, y, n, p, tau)
        sigma_global = _project(Gamma_global)

    else:
        raise ValueError(f"Unknown noise type: {noise}")

    list_matrices_lasso = []
    list_matrices_error = []
    list_rho_lasso = []
    list_rho_error = []

    for i in range(1, K + 1):
        index = np.where(folds == i)[0]

        mat_train = np.delete(mat, index, axis=0)
        y_train = np.delete(y, index)

        mat_test = mat[index]
        y_test = y[index]

        if noise == "additive":
            cov_train = (1 / n_without_fold) * mat_train.T @ mat_train - tau ** 2 * np.eye(p)
            cov_test = (1 / n_one_fold) * mat_test.T @ mat_test - tau ** 2 * np.eye(p)
            rho_train = (1 / n_without_fold) * mat_train.T @ y_train
            rho_test = (1 / n_one_fold) * mat_test.T @ y_test
        elif noise == "missing":
            cov_train = (1 / n_without_fold) * mat_train.T @ mat_train / ratio_matrix
            cov_test = (1 / n_one_fold) * mat_test.T @ mat_test / ratio_matrix
            rho_train = (1 / n_without_fold) * mat_train.T @ y_train / np.diag(ratio_matrix)
            rho_test = (1 / n_one_fold) * mat_test.T @ y_test / np.diag(ratio_matrix)
        elif noise == "multiplicative":
            cov_train, rho_train = _corrected_covariance_multiplicative(
                mat_train, y_train, n_without_fold, p, tau)
            cov_test, rho_test = _corrected_covariance_multiplicative(
                mat_test, y_test, n_one_fold, p, tau)

        list_matrices_lasso.append(_project(cov_train, R=ratio_matrix if noise == "missing" else None))
        list_matrices_error.append(_project(cov_test, R=ratio_matrix if noise == "missing" else None))
        list_rho_lasso.append(rho_train)
        list_rho_error.append(rho_test)

    return {
        "sigma_global": sigma_global,
        "rho_global": rho_global,
        "list_matrices_lasso": list_matrices_lasso,
        "list_matrices_error": list_matrices_error,
        "list_rho_lasso": list_rho_lasso,
        "list_rho_error": list_rho_error,
    }


def _cv_covariance_matrices_block(
    K: int,
    mat: np.ndarray,
    y: np.ndarray,
    p: int,
    p1: int,
    p2: int,
    mu: float = 10,
    tau: Optional[float] = None,
    ratio_matrix: Optional[np.ndarray] = None,
    etol: float = 1e-4,
    noise: str = "additive",
    mode: str = "ADMM",
) -> Dict:
    """
    为 BD-CoCoLasso 交叉验证创建投影后的 PSD 协方差矩阵。

    参数
    ----------
    K : int, 交叉验证折数
    mat : (n, p) ndarray, 设计矩阵（前 p1 列无误差，后 p2 列含误差）
    y : (n,) ndarray, 响应向量
    p : int, 总特征数
    p1 : int, 无误差特征数
    p2 : int, 含误差特征数
    mu : float, ADMM 惩罚参数
    tau : float or None, 加性误差的标准差
    ratio_matrix : (p2, p2) ndarray or None, 观测比率矩阵
    etol : float, ADMM 容差
    noise : str, 'additive' 或 'missing'
    mode : str, 'ADMM' 或 'HM'

    返回
    ----------
    dict, 包含:
        'sigma_global_uncorrupted', 'sigma_global_corrupted',
        'list_PSD_lasso', 'list_PSD_error',
        'list_sigma_lasso', 'list_sigma_error',
        'folds'
    """
    n = mat.shape[0]
    n_without_fold = n - n // K
    n_one_fold = n // K
    start = p1

    folds = np.random.permutation(np.repeat(np.arange(1, K + 1), n // K))

    def _project(cov_mat, R=None):
        if mode == "ADMM":
            return _admm_proj(cov_mat, mu=mu, etol=etol)["mat"]
        else:
            return _hm_proj(cov_mat, R=R, mu=mu, tolerance=etol)

    mat_uncorrupted = mat[:, :p1]
    mat_corrupted = mat[:, start:p]

    if noise == "additive":
        cov_modified = (1 / n) * mat_corrupted.T @ mat_corrupted - tau ** 2 * np.eye(p2)
    elif noise == "missing":
        cov_modified = (1 / n) * mat_corrupted.T @ mat_corrupted / ratio_matrix
    else:
        raise ValueError(f"Unknown noise type: {noise}")

    sigma_global_corrupted = _project(cov_modified, R=ratio_matrix if noise == "missing" else None)
    sigma_global_uncorrupted = (1 / n) * mat_uncorrupted.T @ mat_uncorrupted

    list_PSD_lasso = []
    list_PSD_error = []
    list_sigma_lasso = []
    list_sigma_error = []

    for i in range(1, K + 1):
        index = np.where(folds == i)[0]

        mat_train_corrupted = np.delete(mat_corrupted, index, axis=0)
        mat_test_corrupted = mat_corrupted[index]

        if noise == "additive":
            cov_train = (1 / n_without_fold) * mat_train_corrupted.T @ mat_train_corrupted - tau ** 2 * np.eye(p2)
            cov_test = (1 / n_one_fold) * mat_test_corrupted.T @ mat_test_corrupted - tau ** 2 * np.eye(p2)
        elif noise == "missing":
            cov_train = (1 / n_without_fold) * mat_train_corrupted.T @ mat_train_corrupted / ratio_matrix
            cov_test = (1 / n_one_fold) * mat_test_corrupted.T @ mat_test_corrupted / ratio_matrix

        list_PSD_lasso.append(_project(cov_train, R=ratio_matrix if noise == "missing" else None))
        list_PSD_error.append(_project(cov_test, R=ratio_matrix if noise == "missing" else None))

        mat_train_uncorrupted = np.delete(mat_uncorrupted, index, axis=0)
        mat_test_uncorrupted = mat_uncorrupted[index]

        list_sigma_lasso.append((1 / n_without_fold) * mat_train_uncorrupted.T @ mat_train_uncorrupted)
        list_sigma_error.append((1 / n_one_fold) * mat_test_uncorrupted.T @ mat_test_uncorrupted)

    return {
        "sigma_global_uncorrupted": sigma_global_uncorrupted,
        "sigma_global_corrupted": sigma_global_corrupted,
        "list_PSD_lasso": list_PSD_lasso,
        "list_PSD_error": list_PSD_error,
        "list_sigma_lasso": list_sigma_lasso,
        "list_sigma_error": list_sigma_error,
        "folds": folds,
    }


def _cv_covariance_matrices_block_general(
    K: int,
    mat: np.ndarray,
    y: np.ndarray,
    p: int,
    p1: int,
    p2: int,
    p3: int,
    mu: float = 10,
    tau: Optional[float] = None,
    ratio_matrix: Optional[np.ndarray] = None,
    etol: float = 1e-4,
    mode: str = "ADMM",
) -> Dict:
    """
    为三块 BD-CoCoLasso 交叉验证创建投影后的 PSD 协方差矩阵。

    参数
    ----------
    K : int, 交叉验证折数
    mat : (n, p) ndarray, 设计矩阵
    y : (n,) ndarray, 响应向量
    p : int, 总特征数 (p1 + p2 + p3)
    p1 : int, 无误差特征数
    p2 : int, 含加性误差特征数
    p3 : int, 含缺失数据特征数
    mu : float, ADMM 惩罚参数
    tau : float or None, 加性误差的标准差
    ratio_matrix : (p3, p3) ndarray or None, 观测比率矩阵
    etol : float, ADMM 容差
    mode : str, 'ADMM' 或 'HM'

    返回
    ----------
    dict, 包含每折的投影矩阵
    """
    n = mat.shape[0]
    n_without_fold = n - n // K
    n_one_fold = n // K

    folds = np.random.permutation(np.repeat(np.arange(1, K + 1), n // K))

    def _project(cov_mat, R=None):
        if mode == "ADMM":
            return _admm_proj(cov_mat, mu=mu, etol=etol)["mat"]
        else:
            return _hm_proj(cov_mat, R=R, mu=mu, tolerance=etol)

    mat_uncorrupted = mat[:, :p1] if p1 > 0 else None
    mat_corrupted_additive = mat[:, p1:p1 + p2]
    mat_corrupted_missing = mat[:, p1 + p2:p]

    cov_modified_additive = (1 / n) * mat_corrupted_additive.T @ mat_corrupted_additive - tau ** 2 * np.eye(p2)
    cov_modified_missing = (1 / n) * mat_corrupted_missing.T @ mat_corrupted_missing / ratio_matrix

    sigma_global_corrupted_additive = _project(cov_modified_additive)
    sigma_global_corrupted_missing = _project(cov_modified_missing, R=ratio_matrix)

    sigma_global_uncorrupted = None
    if p1 > 0:
        sigma_global_uncorrupted = (1 / n) * mat_uncorrupted.T @ mat_uncorrupted

    list_PSD_lasso_additive = []
    list_PSD_error_additive = []
    list_PSD_lasso_missing = []
    list_PSD_error_missing = []
    list_sigma_lasso = []
    list_sigma_error = []

    for i in range(1, K + 1):
        index = np.where(folds == i)[0]

        mat_train_add = np.delete(mat_corrupted_additive, index, axis=0)
        mat_test_add = mat_corrupted_additive[index]
        cov_train_add = (1 / n_without_fold) * mat_train_add.T @ mat_train_add - tau ** 2 * np.eye(p2)
        cov_test_add = (1 / n_one_fold) * mat_test_add.T @ mat_test_add - tau ** 2 * np.eye(p2)
        list_PSD_lasso_additive.append(_project(cov_train_add))
        list_PSD_error_additive.append(_project(cov_test_add))

        mat_train_miss = np.delete(mat_corrupted_missing, index, axis=0)
        mat_test_miss = mat_corrupted_missing[index]
        cov_train_miss = (1 / n_without_fold) * mat_train_miss.T @ mat_train_miss / ratio_matrix
        cov_test_miss = (1 / n_one_fold) * mat_test_miss.T @ mat_test_miss / ratio_matrix
        list_PSD_lasso_missing.append(_project(cov_train_miss, R=ratio_matrix))
        list_PSD_error_missing.append(_project(cov_test_miss, R=ratio_matrix))

        if p1 > 0:
            mat_train_unc = np.delete(mat_uncorrupted, index, axis=0)
            mat_test_unc = mat_uncorrupted[index]
            list_sigma_lasso.append((1 / n_without_fold) * mat_train_unc.T @ mat_train_unc)
            list_sigma_error.append((1 / n_one_fold) * mat_test_unc.T @ mat_test_unc)

    result = {
        "sigma_global_corrupted_additive": sigma_global_corrupted_additive,
        "sigma_global_corrupted_missing": sigma_global_corrupted_missing,
        "list_PSD_lasso_additive": list_PSD_lasso_additive,
        "list_PSD_error_additive": list_PSD_error_additive,
        "list_PSD_lasso_missing": list_PSD_lasso_missing,
        "list_PSD_error_missing": list_PSD_error_missing,
        "folds": folds,
    }
    if p1 > 0:
        result["sigma_global_uncorrupted"] = sigma_global_uncorrupted
        result["list_sigma_lasso"] = list_sigma_lasso
        result["list_sigma_error"] = list_sigma_error
    return result


def _preprocess_data(
    Z: np.ndarray,
    y: np.ndarray,
    n: int,
    p: int,
    center_Z: bool = True,
    scale_Z: bool = True,
    center_y: bool = True,
    scale_y: bool = True,
    noise: str = "additive",
    p1: int = 0,
    p2: int = 0,
) -> Dict:
    """
    数据预处理：中心化和标准化 Z 和 y，处理缺失值。

    缺失数据：忽略 NaN 对列中心化，将 NaN 替换为 0，再标准化。
    加性误差：标准中心化和标准化。

    返回
    ----------
    dict, 包含:
        'Z', 'y', 'mean_Z', 'sd_Z', 'mean_y', 'sd_y', 'ratio_matrix'
    """
    Z = Z.copy().astype(float)
    y = y.copy().astype(float).ravel()

    mean_Z = np.nanmean(Z, axis=0)
    sd_Z = np.nanstd(Z, axis=0, ddof=1)

    ratio_matrix = None

    if noise == "missing":
        if p2 > 0:
            ratio_matrix = _compute_ratio_matrix(Z, p2, offset=p1)
        else:
            ratio_matrix = _compute_ratio_matrix(Z, p)

    if center_Z:
        if noise == "missing":
            if p2 > 0:
                for j in range(p1, p):
                    col_mask = ~np.isnan(Z[:, j])
                    Z[col_mask, j] -= mean_Z[j]
                Z[np.isnan(Z[:, p1:])] = 0
                if p1 > 0:
                    Z[:, :p1] = Z[:, :p1] - mean_Z[:p1]
            else:
                for j in range(p):
                    col_mask = ~np.isnan(Z[:, j])
                    Z[col_mask, j] -= mean_Z[j]
                Z[np.isnan(Z)] = 0
        else:
            Z = Z - mean_Z[np.newaxis, :]

        if scale_Z:
            sd_Z_safe = np.where(sd_Z != 0, sd_Z, 1.0)
            Z = Z / sd_Z_safe[np.newaxis, :]
    else:
        if scale_Z:
            sd_Z_safe = np.where(sd_Z != 0, sd_Z, 1.0)
            Z = Z / sd_Z_safe[np.newaxis, :]

    mean_y = np.mean(y)
    sd_y = np.std(y, ddof=1)

    if center_y:
        y = y - mean_y
    if scale_y:
        if sd_y != 0:
            y = y / sd_y

    return {
        "Z": Z,
        "y": y,
        "mean_Z": mean_Z,
        "sd_Z": sd_Z,
        "mean_y": mean_y,
        "sd_y": sd_y,
        "ratio_matrix": ratio_matrix,
    }


def _pathwise_coordinate_descent(
    Z: np.ndarray,
    y: np.ndarray,
    n: int,
    p: int,
    center_Z: bool = True,
    scale_Z: bool = True,
    center_y: bool = True,
    scale_y: bool = True,
    lambda_factor: Optional[float] = None,
    step: int = 100,
    K: int = 4,
    mu: float = 10,
    tau: Optional[float] = None,
    etol: float = 1e-4,
    optTol: float = 1e-5,
    earlyStopping_max: int = 10,
    noise: str = "additive",
    penalty: str = "lasso",
    mode: str = "ADMM",
) -> Dict:
    """
    CoCoLasso 的路径坐标下降算法。

    实现 CoCoLasso 算法，通过 K 折交叉验证选择最优正则化参数 lambda。

    参数
    ----------
    Z : (n, p) ndarray, 含误差设计矩阵
    y : (n,) ndarray, 响应向量
    n : int, 样本数
    p : int, 特征数
    center_Z : bool, 是否对 Z 中心化
    scale_Z : bool, 是否对 Z 标准化
    center_y : bool, 是否对 y 中心化
    scale_y : bool, 是否对 y 标准化
    lambda_factor : float, lambda_min/lambda_max 的比率
    step : int, lambda 路径上的搜索步数
    K : int, 交叉验证折数
    mu : float, ADMM 惩罚参数
    tau : float or None, 加性误差的标准差
    etol : float, ADMM 容差
    optTol : float, 路径下降的收敛容差
    earlyStopping_max : int, 误差连续递增的最大次数
    noise : str, 'additive'、'missing' 或 'multiplicative'
    penalty : str, 'lasso' 或 'SCAD'
    mode : str, 'ADMM' 或 'HM'

    返回
    ----------
    dict, 包含:
        'lambda_opt', 'lambda_sd', 'beta_opt', 'beta_sd',
        'data_error', 'data_beta', 'early_stopping',
        'mean_Z', 'sd_Z', 'mean_y', 'sd_y'
    """
    if lambda_factor is None:
        lambda_factor = 0.01 if n < p else 0.001

    preprocessed = _preprocess_data(Z, y, n, p, center_Z, scale_Z, center_y, scale_y, noise)
    Z_proc = preprocessed["Z"]
    y_proc = preprocessed["y"]
    ratio_matrix = preprocessed["ratio_matrix"]

    earlyStopping = step
    lam_max = _lambda_max(Z_proc, y_proc, n, ratio_matrix, noise, tau)
    lam_min = lambda_factor * lam_max
    lambda_list = _log_space(lam_max, lam_min, step)

    beta_start = np.zeros(p)
    best_lambda = lam_max
    beta_opt = beta_start.copy()
    best_error = 1000.0
    error_list = np.zeros((step, 4))
    error = 1000.0
    earlyStopping_high = 0
    matrix_beta = np.zeros((step, p))

    output = _cv_covariance_matrices(
        K=K, mat=Z_proc, y=y_proc, p=p, mu=mu, tau=tau,
        ratio_matrix=ratio_matrix, etol=etol, noise=noise, mode=mode,
    )
    list_matrices_lasso = output["list_matrices_lasso"]
    list_matrices_error = output["list_matrices_error"]
    list_rho_lasso = output["list_rho_lasso"]
    list_rho_error = output["list_rho_error"]
    ZZ = output["sigma_global"]
    Zy = output["rho_global"]

    for i in range(step):
        lambda_step = lambda_list[i]
        error_old = error

        cv_errors = []
        for k in range(K):
            sigma_train = list_matrices_lasso[k]
            rho_train = list_rho_lasso[k]
            coef_lambda = _lasso_covariance(
                n=n, p=p, lambda_val=lambda_step, XX=sigma_train, Xy=rho_train,
                beta_start=beta_start, penalty=penalty,
            )["coefficients"]

            sigma_test = list_matrices_error[k]
            rho_test = list_rho_error[k]
            err = coef_lambda @ sigma_test @ coef_lambda - 2 * rho_test @ coef_lambda
            cv_errors.append(err)

        error = np.mean(cv_errors)
        error_list[i, 0] = error
        error_list[i, 1] = np.quantile(cv_errors, 0.1)
        error_list[i, 2] = np.quantile(cv_errors, 0.9)
        error_list[i, 3] = np.std(cv_errors)

        coef_tot = _lasso_covariance(
            n=n, p=p, lambda_val=lambda_step, XX=ZZ, Xy=Zy,
            beta_start=beta_start, penalty=penalty,
        )["coefficients"]
        beta_start = coef_tot
        matrix_beta[i, :] = beta_start

        if error <= best_error:
            best_error = error
            best_lambda = lambda_step
            beta_opt = coef_tot.copy()

        if abs(error - error_old) < optTol:
            earlyStopping = i + 1
            break

        if i >= step // 2 and error >= error_list[0, 0]:
            earlyStopping = i + 1
            break

        if error > best_error:
            earlyStopping_high += 1
            if earlyStopping_high >= earlyStopping_max:
                earlyStopping = i + 1
                break

    df_error = {
        "lambda": lambda_list[:earlyStopping],
        "error": error_list[:earlyStopping, 0],
        "error_inf": error_list[:earlyStopping, 1],
        "error_sup": error_list[:earlyStopping, 2],
        "error_sd": error_list[:earlyStopping, 3],
    }

    step_min = int(np.argmin(df_error["error"]))
    sd_best = df_error["error_sd"][step_min]
    candidates = np.where(
        (df_error["error"] > best_error + sd_best) &
        (df_error["lambda"] > df_error["lambda"][step_min])
    )[0]
    step_sd = int(np.max(candidates)) if len(candidates) > 0 else step_min
    lambda_sd = df_error["lambda"][step_sd]
    beta_sd = matrix_beta[step_sd, :]

    return {
        "lambda_opt": best_lambda,
        "lambda_sd": lambda_sd,
        "beta_opt": beta_opt,
        "beta_sd": beta_sd,
        "data_error": df_error,
        "data_beta": {"lambda": lambda_list[:earlyStopping], "beta": matrix_beta[:earlyStopping, :]},
        "early_stopping": earlyStopping,
        "mean_Z": preprocessed["mean_Z"],
        "sd_Z": preprocessed["sd_Z"],
        "mean_y": preprocessed["mean_y"],
        "sd_y": preprocessed["sd_y"],
    }


def _blockwise_coordinate_descent(
    Z: np.ndarray,
    y: np.ndarray,
    n: int,
    p: int,
    p1: int,
    p2: int,
    center_Z: bool = True,
    scale_Z: bool = True,
    center_y: bool = True,
    scale_y: bool = True,
    lambda_factor: Optional[float] = None,
    step: int = 100,
    K: int = 4,
    mu: float = 10,
    tau: Optional[float] = None,
    etol: float = 1e-4,
    optTol: float = 1e-5,
    earlyStopping_max: int = 10,
    noise: str = "additive",
    penalty: str = "lasso",
    mode: str = "ADMM",
) -> Dict:
    """
    BD-CoCoLasso 的块坐标下降算法。

    实现块下降 CoCoLasso 算法，通过 K 折交叉验证选择最优正则化参数。
    设计矩阵 Z 的前 p1 列为无误差特征，后 p2 列为含误差特征。

    参数
    ----------
    Z : (n, p) ndarray, 设计矩阵（前 p1 列无误差，后 p2 列含误差）
    y : (n,) ndarray, 响应向量
    n : int, 样本数
    p : int, 总特征数 (p1 + p2)
    p1 : int, 无误差特征数
    p2 : int, 含误差特征数
    center_Z : bool, 是否对 Z 中心化
    scale_Z : bool, 是否对 Z 标准化
    center_y : bool, 是否对 y 中心化
    scale_y : bool, 是否对 y 标准化
    lambda_factor : float, lambda_min/lambda_max 的比率
    step : int, lambda 路径上的搜索步数
    K : int, 交叉验证折数
    mu : float, ADMM 惩罚参数
    tau : float or None, 加性误差的标准差
    etol : float, ADMM 容差
    optTol : float, 收敛容差
    earlyStopping_max : int, 误差连续递增的最大次数
    noise : str, 'additive' 或 'missing'
    penalty : str, 'lasso' 或 'SCAD'
    mode : str, 'ADMM' 或 'HM'

    返回
    ----------
    dict, 包含:
        'lambda_opt', 'lambda_sd', 'beta_opt', 'beta_sd',
        'data_error', 'data_beta', 'early_stopping',
        'mean_Z', 'sd_Z', 'mean_y', 'sd_y'
    """
    if lambda_factor is None:
        lambda_factor = 0.01 if n < p else 0.001

    preprocessed = _preprocess_data(
        Z, y, n, p, center_Z, scale_Z, center_y, scale_y, noise, p1, p2,
    )
    Z_proc = preprocessed["Z"]
    y_proc = preprocessed["y"]
    ratio_matrix = preprocessed["ratio_matrix"]

    earlyStopping = step
    lam_max = _lambda_max_block(Z_proc, y_proc, n, p1, p2, ratio_matrix, noise)
    lam_min = lambda_factor * lam_max
    lambda_list = _log_space(lam_max, lam_min, step)

    beta1_start = np.zeros(p1)
    beta2_start = np.zeros(p2)
    beta_start = np.concatenate([beta1_start, beta2_start])
    best_lambda = lam_max
    beta_opt = beta_start.copy()
    best_error = 10000.0
    error_list = np.zeros((step, 4))
    error = 0.0
    earlyStopping_high = 0
    matrix_beta = np.zeros((step, p))

    output = _cv_covariance_matrices_block(
        K=K, mat=Z_proc, y=y_proc, p=p, p1=p1, p2=p2, mu=mu, tau=tau,
        ratio_matrix=ratio_matrix, etol=etol, noise=noise, mode=mode,
    )
    list_PSD_lasso = output["list_PSD_lasso"]
    list_PSD_error = output["list_PSD_error"]
    list_sigma_lasso = output["list_sigma_lasso"]
    list_sigma_error = output["list_sigma_error"]
    sigma1 = output["sigma_global_uncorrupted"]
    sigma2 = output["sigma_global_corrupted"]
    folds = output["folds"]

    n_without_fold = n - n // K
    n_one_fold = n // K
    X1 = Z_proc[:, :p1]
    Z2 = Z_proc[:, p1:]

    for i in range(step):
        lambda_step = lambda_list[i]
        error_old = error

        cv_errors = []
        for k in range(K):
            index = np.where(folds == k + 1)[0]
            sigma_corrupted_train = list_PSD_lasso[k]
            sigma_uncorrupted_train = list_sigma_lasso[k]

            X1_cv_train = np.delete(X1, index, axis=0)
            Z2_cv_train = np.delete(Z2, index, axis=0)
            y_cv_train = np.delete(y_proc, index)

            out = _lasso_covariance_block(
                n=n_without_fold, p1=p1, p2=p2, X1=X1_cv_train, Z2=Z2_cv_train,
                y=y_cv_train, sigma1=sigma_uncorrupted_train, sigma2=sigma_corrupted_train,
                lambda_val=lambda_step, noise=noise, ratio_matrix=ratio_matrix,
                beta1_start=beta1_start, beta2_start=beta2_start, penalty=penalty,
            )
            beta1_lambda = out["coefficients_beta1"]
            beta2_lambda = out["coefficients_beta2"]

            sigma_corrupted_test = list_PSD_error[k]
            sigma_uncorrupted_test = list_sigma_error[k]
            X1_cv_test = X1[index]
            Z2_cv_test = Z2[index]
            y_cv_test = y_proc[index]

            rho_1 = (1 / n_one_fold) * X1_cv_test.T @ y_cv_test
            if noise == "additive":
                rho_2 = (1 / n_one_fold) * Z2_cv_test.T @ y_cv_test
                sigma3 = (1 / n_one_fold) * Z2_cv_test.T @ X1_cv_test @ beta1_lambda
            else:
                Z2_tilde = Z2_cv_test / np.diag(ratio_matrix)[np.newaxis, :]
                rho_2 = (1 / n_one_fold) * Z2_tilde.T @ y_cv_test
                sigma3 = (1 / n_one_fold) * Z2_tilde.T @ X1_cv_test @ beta1_lambda

            err = (beta1_lambda @ sigma_uncorrupted_test @ beta1_lambda +
                   beta2_lambda @ sigma_corrupted_test @ beta2_lambda -
                   2 * rho_1 @ beta1_lambda - 2 * rho_2 @ beta2_lambda +
                   2 * beta2_lambda @ sigma3)
            cv_errors.append(err)

        error = np.mean(cv_errors)
        error_list[i, 0] = error
        error_list[i, 1] = np.quantile(cv_errors, 0.1)
        error_list[i, 2] = np.quantile(cv_errors, 0.9)
        error_list[i, 3] = np.std(cv_errors)

        out = _lasso_covariance_block(
            n=n, p1=p1, p2=p2, X1=X1, Z2=Z2, y=y_proc,
            sigma1=sigma1, sigma2=sigma2, lambda_val=lambda_step,
            noise=noise, ratio_matrix=ratio_matrix,
            beta1_start=beta1_start, beta2_start=beta2_start, penalty=penalty,
        )
        beta1 = out["coefficients_beta1"]
        beta2 = out["coefficients_beta2"]
        beta = np.concatenate([beta1, beta2])

        beta1_start = beta1
        beta2_start = beta2
        matrix_beta[i, :] = beta

        if error <= best_error:
            best_error = error
            best_lambda = lambda_step
            beta_opt = beta.copy()

        if abs(error - error_old) < optTol:
            earlyStopping = i + 1
            break

        if i >= step // 2 and error >= error_list[0, 0]:
            earlyStopping = i + 1
            break

        if error > best_error:
            earlyStopping_high += 1
            if earlyStopping_high >= earlyStopping_max:
                earlyStopping = i + 1
                break

    df_error = {
        "lambda": lambda_list[:earlyStopping],
        "error": error_list[:earlyStopping, 0],
        "error_inf": error_list[:earlyStopping, 1],
        "error_sup": error_list[:earlyStopping, 2],
        "error_sd": error_list[:earlyStopping, 3],
    }

    step_min = int(np.argmin(df_error["error"]))
    sd_best = df_error["error_sd"][step_min]
    candidates = np.where(
        (df_error["error"] > best_error + sd_best) &
        (df_error["lambda"] > df_error["lambda"][step_min])
    )[0]
    step_sd = int(np.max(candidates)) if len(candidates) > 0 else step_min
    lambda_sd = df_error["lambda"][step_sd]
    beta_sd = matrix_beta[step_sd, :]

    return {
        "lambda_opt": best_lambda,
        "lambda_sd": lambda_sd,
        "beta_opt": beta_opt,
        "beta_sd": beta_sd,
        "data_error": df_error,
        "data_beta": {"lambda": lambda_list[:earlyStopping], "beta": matrix_beta[:earlyStopping, :]},
        "early_stopping": earlyStopping,
        "mean_Z": preprocessed["mean_Z"],
        "sd_Z": preprocessed["sd_Z"],
        "mean_y": preprocessed["mean_y"],
        "sd_y": preprocessed["sd_y"],
    }


def coco(
    Z: np.ndarray,
    y: np.ndarray,
    n: int,
    p: int,
    p1: Optional[int] = None,
    p2: Optional[int] = None,
    center_Z: bool = True,
    scale_Z: bool = True,
    center_y: bool = True,
    scale_y: bool = True,
    lambda_factor: Optional[float] = None,
    step: int = 100,
    K: int = 4,
    mu: float = 10,
    tau: Optional[float] = None,
    etol: float = 1e-4,
    optTol: float = 1e-5,
    earlyStopping_max: int = 10,
    noise: str = "additive",
    block: bool = True,
    penalty: str = "lasso",
    mode: str = "ADMM",
) -> Dict:
    """
    CoCoLasso / BD-CoCoLasso 的主入口函数。

    若 block=True，使用块坐标下降（BD-CoCoLasso）。
    若 block=False，使用路径坐标下降（CoCoLasso）。

    参数
    ----------
    Z : (n, p) ndarray, 含误差设计矩阵
    y : (n,) ndarray, 响应向量
    n : int, 样本数
    p : int, 特征数
    p1 : int or None, 无误差特征数（block=True 时必填）
    p2 : int or None, 含误差特征数（block=True 时必填）
    center_Z : bool, 是否对 Z 中心化
    scale_Z : bool, 是否对 Z 标准化
    center_y : bool, 是否对 y 中心化
    scale_y : bool, 是否对 y 标准化
    lambda_factor : float, lambda_min/lambda_max 的比率
    step : int, lambda 路径上的搜索步数
    K : int, 交叉验证折数
    mu : float, ADMM 惩罚参数
    tau : float or None, 加性误差的标准差
    etol : float, ADMM 容差
    optTol : float, 收敛容差
    earlyStopping_max : int, 误差连续递增的最大次数
    noise : str, 'additive'、'missing' 或 'multiplicative'
    block : bool, 使用 BD-CoCoLasso (True) 或 CoCoLasso (False)
    penalty : str, 'lasso' 或 'SCAD'
    mode : str, 'ADMM' 或 'HM'

    返回
    ----------
    dict, 包含:
        'lambda_opt', 'lambda_sd', 'beta_opt', 'beta_sd',
        'data_error', 'data_beta', 'early_stopping',
        'mean_Z', 'sd_Z', 'mean_y', 'sd_y'
    """
    if block:
        if p1 is None or p2 is None:
            raise ValueError("p1 and p2 must be provided when block=True")
        return _blockwise_coordinate_descent(
            Z=Z, y=y, n=n, p=p, p1=p1, p2=p2,
            center_Z=center_Z, scale_Z=scale_Z, center_y=center_y, scale_y=scale_y,
            lambda_factor=lambda_factor, step=step, K=K, mu=mu, tau=tau,
            etol=etol, optTol=optTol, earlyStopping_max=earlyStopping_max,
            noise=noise, penalty=penalty, mode=mode,
        )
    else:
        return _pathwise_coordinate_descent(
            Z=Z, y=y, n=n, p=p,
            center_Z=center_Z, scale_Z=scale_Z, center_y=center_y, scale_y=scale_y,
            lambda_factor=lambda_factor, step=step, K=K, mu=mu, tau=tau,
            etol=etol, optTol=optTol, earlyStopping_max=earlyStopping_max,
            noise=noise, penalty=penalty, mode=mode,
        )


def generalcoco(
    Z: np.ndarray,
    y: np.ndarray,
    n: int,
    p: int,
    p1: int = 0,
    p2: int = 0,
    p3: int = 0,
    center_Z: bool = True,
    scale_Z: bool = True,
    center_y: bool = True,
    scale_y: bool = True,
    lambda_factor: Optional[float] = None,
    step: int = 100,
    K: int = 4,
    mu: float = 10,
    tau: Optional[float] = None,
    etol: float = 1e-4,
    optTol: float = 1e-5,
    earlyStopping_max: int = 10,
    penalty: str = "lasso",
    mode: str = "ADMM",
) -> Dict:
    """
    三块 BD-CoCoLasso（混合加性误差 + 缺失数据）。

    blockwise_coordinate_descent_general 的封装函数。

    参数
    ----------
    Z : (n, p) ndarray, 设计矩阵
    y : (n,) ndarray, 响应向量
    n : int, 样本数
    p : int, 总特征数 (p1 + p2 + p3)
    p1 : int, 无误差特征数
    p2 : int, 含加性误差特征数
    p3 : int, 含缺失数据特征数
    center_Z : bool, 是否对 Z 中心化
    scale_Z : bool, 是否对 Z 标准化
    center_y : bool, 是否对 y 中心化
    scale_y : bool, 是否对 y 标准化
    lambda_factor : float, lambda_min/lambda_max 的比率
    step : int, lambda 路径上的搜索步数
    K : int, 交叉验证折数
    mu : float, ADMM 惩罚参数
    tau : float or None, 加性误差的标准差
    etol : float, ADMM 容差
    optTol : float, 收敛容差
    earlyStopping_max : int, 误差连续递增的最大次数
    penalty : str, 'lasso' 或 'SCAD'
    mode : str, 'ADMM' 或 'HM'

    返回
    ----------
    dict, 包含:
        'lambda_opt', 'lambda_sd', 'beta_opt', 'beta_sd',
        'data_error', 'data_beta', 'early_stopping',
        'mean_Z', 'sd_Z', 'mean_y', 'sd_y'
    """
    return _blockwise_coordinate_descent_general(
        Z=Z, y=y, n=n, p=p, p1=p1, p2=p2, p3=p3,
        center_Z=center_Z, scale_Z=scale_Z, center_y=center_y, scale_y=scale_y,
        lambda_factor=lambda_factor, step=step, K=K, mu=mu, tau=tau,
        etol=etol, optTol=optTol, earlyStopping_max=earlyStopping_max,
        penalty=penalty, mode=mode,
    )


def _blockwise_coordinate_descent_general(
    Z: np.ndarray,
    y: np.ndarray,
    n: int,
    p: int,
    p1: int,
    p2: int,
    p3: int,
    center_Z: bool = True,
    scale_Z: bool = True,
    center_y: bool = True,
    scale_y: bool = True,
    lambda_factor: Optional[float] = None,
    step: int = 100,
    K: int = 4,
    mu: float = 10,
    tau: Optional[float] = None,
    etol: float = 1e-4,
    optTol: float = 1e-5,
    earlyStopping_max: int = 10,
    penalty: str = "lasso",
    mode: str = "ADMM",
) -> Dict:
    """
    三块 BD-CoCoLasso（混合加性误差 + 缺失数据）。

    设计矩阵 Z 的列组织方式：
      - 第 0 ~ p1-1 列：无误差特征
      - 第 p1 ~ p1+p2-1 列：含加性误差的特征
      - 第 p1+p2 ~ p-1 列：含缺失数据的特征

    参数
    ----------
    Z : (n, p) ndarray, 设计矩阵
    y : (n,) ndarray, 响应向量
    n : int, 样本数
    p : int, 总特征数 (p1 + p2 + p3)
    p1 : int, 无误差特征数
    p2 : int, 含加性误差特征数
    p3 : int, 含缺失数据特征数
    center_Z : bool, 是否对 Z 中心化
    scale_Z : bool, 是否对 Z 标准化
    center_y : bool, 是否对 y 中心化
    scale_y : bool, 是否对 y 标准化
    lambda_factor : float, lambda_min/lambda_max 的比率
    step : int, lambda 路径上的搜索步数
    K : int, 交叉验证折数
    mu : float, ADMM 惩罚参数
    tau : float or None, 加性误差的标准差
    etol : float, ADMM 容差
    optTol : float, 收敛容差
    earlyStopping_max : int, 误差连续递增的最大次数
    penalty : str, 'lasso' 或 'SCAD'
    mode : str, 'ADMM' 或 'HM'

    返回
    ----------
    dict, 包含:
        'lambda_opt', 'lambda_sd', 'beta_opt', 'beta_sd',
        'data_error', 'data_beta', 'early_stopping',
        'mean_Z', 'sd_Z', 'mean_y', 'sd_y'
    """
    if lambda_factor is None:
        lambda_factor = 0.01 if n < p else 0.001

    preprocessed = _preprocess_data(
        Z, y, n, p, center_Z, scale_Z, center_y, scale_y,
        noise="missing", p1=p1 + p2, p2=p3,
    )
    Z_proc = preprocessed["Z"]
    y_proc = preprocessed["y"]
    ratio_matrix = preprocessed["ratio_matrix"]

    earlyStopping = step

    X1 = Z_proc[:, :p1] if p1 > 0 else None
    Z2 = Z_proc[:, p1:p1 + p2]
    Z3 = Z_proc[:, p1 + p2:p]
    rho_tilde1 = (1 / n) * Z_proc[:, :p1 + p2].T @ y_proc if p1 > 0 else np.array([])
    rho_tilde2 = (1 / n) * Z3.T @ y_proc / np.diag(ratio_matrix)
    lam_max = float(max(
        np.max(np.abs(rho_tilde1)) if len(rho_tilde1) > 0 else 0,
        np.max(np.abs(rho_tilde2)),
    ))
    lam_min = lambda_factor * lam_max
    lambda_list = _log_space(lam_max, lam_min, step)

    beta1_start = np.zeros(p1)
    beta2_start = np.zeros(p2)
    beta3_start = np.zeros(p3)
    beta_start = (np.concatenate([beta1_start, beta2_start, beta3_start])
                  if p1 > 0 else np.concatenate([beta2_start, beta3_start]))
    best_lambda = lam_max
    beta_opt = beta_start.copy()
    best_error = 10000.0
    error_list = np.zeros((step, 4))
    error = 0.0
    earlyStopping_high = 0
    matrix_beta = np.zeros((step, p))

    output = _cv_covariance_matrices_block_general(
        K=K, mat=Z_proc, y=y_proc, p=p, p1=p1, p2=p2, p3=p3, mu=mu,
        tau=tau, ratio_matrix=ratio_matrix, etol=etol, mode=mode,
    )
    list_PSD_lasso_additive = output["list_PSD_lasso_additive"]
    list_PSD_error_additive = output["list_PSD_error_additive"]
    list_PSD_lasso_missing = output["list_PSD_lasso_missing"]
    list_PSD_error_missing = output["list_PSD_error_missing"]
    sigma1 = output.get("sigma_global_uncorrupted")
    sigma2 = output["sigma_global_corrupted_additive"]
    sigma3 = output["sigma_global_corrupted_missing"]
    list_sigma_lasso = output.get("list_sigma_lasso")
    list_sigma_error = output.get("list_sigma_error")
    folds = output["folds"]

    n_without_fold = n - n // K
    n_one_fold = n // K

    for i in range(step):
        lambda_step = lambda_list[i]
        error_old = error

        cv_errors = []
        for k in range(K):
            index = np.where(folds == k + 1)[0]

            sigma_corrupted_train_additive = list_PSD_lasso_additive[k]
            sigma_corrupted_train_missing = list_PSD_lasso_missing[k]
            sigma_uncorrupted_train = list_sigma_lasso[k] if list_sigma_lasso is not None else None

            X1_cv_train = np.delete(X1, index, axis=0) if X1 is not None else None
            Z2_cv_train = np.delete(Z2, index, axis=0)
            Z3_cv_train = np.delete(Z3, index, axis=0)
            y_cv_train = np.delete(y_proc, index)

            out = _lasso_covariance_block_general(
                n=n_without_fold, p1=p1, p2=p2, p3=p3,
                X1=X1_cv_train, Z2=Z2_cv_train, Z3=Z3_cv_train, y=y_cv_train,
                sigma1=sigma_uncorrupted_train, sigma2=sigma_corrupted_train_additive,
                sigma3=sigma_corrupted_train_missing, lambda_val=lambda_step,
                ratio_matrix=ratio_matrix,
                beta1_start=beta1_start, beta2_start=beta2_start, beta3_start=beta3_start,
                penalty=penalty,
            )

            beta1_lambda = out.get("coefficients_beta1", np.zeros(p1))
            beta2_lambda = out["coefficients_beta2"]
            beta3_lambda = out["coefficients_beta3"]

            sigma_corrupted_test_additive = list_PSD_error_additive[k]
            sigma_corrupted_test_missing = list_PSD_error_missing[k]

            X1_cv_test = X1[index] if X1 is not None else None
            Z2_cv_test = Z2[index]
            Z3_cv_test = Z3[index]
            y_cv_test = y_proc[index]

            Z3_tilde_test = Z3_cv_test / np.diag(ratio_matrix)[np.newaxis, :]

            if p1 > 0 and X1_cv_test is not None:
                sigma_uncorrupted_test = list_sigma_error[k]
                rho_1 = (1 / n_one_fold) * X1_cv_test.T @ y_cv_test
                rho_2 = (1 / n_one_fold) * Z2_cv_test.T @ y_cv_test
                rho_3 = (1 / n_one_fold) * Z3_tilde_test.T @ y_cv_test

                sigma21 = (1 / n_one_fold) * Z2_cv_test.T @ X1_cv_test @ beta1_lambda
                sigma31 = (1 / n_one_fold) * Z3_tilde_test.T @ X1_cv_test @ beta1_lambda
                sigma32 = (1 / n_one_fold) * Z3_tilde_test.T @ Z2_cv_test @ beta2_lambda

                err = (beta1_lambda @ sigma_uncorrupted_test @ beta1_lambda +
                       beta2_lambda @ sigma_corrupted_test_additive @ beta2_lambda +
                       beta3_lambda @ sigma_corrupted_test_missing @ beta3_lambda -
                       2 * rho_1 @ beta1_lambda - 2 * rho_2 @ beta2_lambda -
                       2 * rho_3 @ beta3_lambda +
                       2 * beta2_lambda @ sigma21 +
                       2 * beta3_lambda @ sigma31 +
                       2 * beta3_lambda @ sigma32)
            else:
                rho_2 = (1 / n_one_fold) * Z2_cv_test.T @ y_cv_test
                rho_3 = (1 / n_one_fold) * Z3_tilde_test.T @ y_cv_test
                sigma32 = (1 / n_one_fold) * Z3_tilde_test.T @ Z2_cv_test @ beta2_lambda

                err = (beta2_lambda @ sigma_corrupted_test_additive @ beta2_lambda +
                       beta3_lambda @ sigma_corrupted_test_missing @ beta3_lambda -
                       2 * rho_2 @ beta2_lambda - 2 * rho_3 @ beta3_lambda +
                       2 * beta3_lambda @ sigma32)

            cv_errors.append(err)

        error = np.mean(cv_errors)
        error_list[i, 0] = error
        error_list[i, 1] = np.quantile(cv_errors, 0.1)
        error_list[i, 2] = np.quantile(cv_errors, 0.9)
        error_list[i, 3] = np.std(cv_errors)

        out = _lasso_covariance_block_general(
            n=n, p1=p1, p2=p2, p3=p3, X1=X1, Z2=Z2, Z3=Z3, y=y_proc,
            sigma1=sigma1, sigma2=sigma2, sigma3=sigma3, lambda_val=lambda_step,
            ratio_matrix=ratio_matrix,
            beta1_start=beta1_start, beta2_start=beta2_start, beta3_start=beta3_start,
            penalty=penalty,
        )

        if p1 > 0:
            beta1 = out["coefficients_beta1"]
        else:
            beta1 = np.zeros(p1)
        beta2 = out["coefficients_beta2"]
        beta3 = out["coefficients_beta3"]

        if p1 > 0:
            beta = np.concatenate([beta1, beta2, beta3])
        else:
            beta = np.concatenate([beta2, beta3])

        beta1_start = beta1 if p1 > 0 else np.zeros(p1)
        beta2_start = beta2
        beta3_start = beta3
        matrix_beta[i, :] = beta

        if error <= best_error:
            best_error = error
            best_lambda = lambda_step
            beta_opt = beta.copy()

        if abs(error - error_old) < optTol:
            earlyStopping = i + 1
            break

        if i >= step // 2 and error >= error_list[0, 0]:
            earlyStopping = i + 1
            break

        if error > best_error:
            earlyStopping_high += 1
            if earlyStopping_high >= earlyStopping_max:
                earlyStopping = i + 1
                break

    df_error = {
        "lambda": lambda_list[:earlyStopping],
        "error": error_list[:earlyStopping, 0],
        "error_inf": error_list[:earlyStopping, 1],
        "error_sup": error_list[:earlyStopping, 2],
        "error_sd": error_list[:earlyStopping, 3],
    }

    step_min = int(np.argmin(df_error["error"]))
    sd_best = df_error["error_sd"][step_min]
    candidates = np.where(
        (df_error["error"] > best_error + sd_best) &
        (df_error["lambda"] > df_error["lambda"][step_min])
    )[0]
    step_sd = int(np.max(candidates)) if len(candidates) > 0 else step_min
    lambda_sd = df_error["lambda"][step_sd]
    beta_sd = matrix_beta[step_sd, :]

    return {
        "lambda_opt": best_lambda,
        "lambda_sd": lambda_sd,
        "beta_opt": beta_opt,
        "beta_sd": beta_sd,
        "data_error": df_error,
        "data_beta": {"lambda": lambda_list[:earlyStopping], "beta": matrix_beta[:earlyStopping, :]},
        "early_stopping": earlyStopping,
        "mean_Z": preprocessed["mean_Z"],
        "sd_Z": preprocessed["sd_Z"],
        "mean_y": preprocessed["mean_y"],
        "sd_y": preprocessed["sd_y"],
    }