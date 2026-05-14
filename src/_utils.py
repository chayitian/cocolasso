"""
BDcocolasso 共享工具函数。

包含 PSD 投影、Lasso 求解器、数据预处理等底层算法，
供 cocolasso / bdcocolasso / generalcocolasso 模块共用。
"""

import numpy as np
from typing import Optional, Dict


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
    epsilon: float = 1e-6,
    mu: float = 1.0,
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
    epsilon: float = 1e-6,
    mu: float = 1.0,
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
    diag_eps = 1e-10

    while m < max_iter:
        beta_old = beta.copy()
        for j in range(p):
            S0 = s[j] - XX[j, j] * beta_old[j] - Xy[j]

            if not np.isfinite(S0):
                beta[j] = 0
                s = XX @ beta
                continue

            w_j = 1.0
            if penalty == "SCAD":
                w_j = _scad_weight(beta[j], lambda0)

            lambda_val = w_j * lambda0
            diag_j = XX[j, j] if XX[j, j] > diag_eps else diag_eps

            if S0 > lambda_val:
                beta[j] = (lambda_val - S0) / diag_j
            elif S0 < -lambda_val:
                beta[j] = (-lambda_val - S0) / diag_j
            else:
                beta[j] = 0

            if not np.isfinite(beta[j]):
                beta[j] = 0
                s = XX @ beta
                continue

            s += XX[:, j] * (beta[j] - beta_old[j])

        if not np.all(np.isfinite(beta)):
            beta = np.zeros(p)
            break

        if np.sum(np.abs(beta - beta_old)) < opt_tol:
            break
        m += 1

    beta[np.abs(beta) < zero_threshold] = 0
    return {"coefficients": beta, "num_it": m}


def _lasso_sklearn(
    n: int,
    p: int,
    lambda_val: float,
    XX: np.ndarray,
    Xy: np.ndarray,
    beta_start: Optional[np.ndarray] = None,
    max_iter: int = 10000,
    tol: float = 1e-6,
) -> Dict:
    """
    使用 Cholesky 分解 + sklearn Lasso 求解:
        min_beta  (1/2) beta' XX beta - Xy' beta + lambda * ||beta||_1

    将协方差形式转换为数据矩阵形式后，调用 sklearn 的坐标下降求解器。
    仅支持 penalty='lasso'，不支持 SCAD。

    参数
    ----------
    n : int, 原始样本数
    p : int, 特征数
    lambda_val : float, 惩罚参数
    XX : (p, p) ndarray, 协方差矩阵 (Sigma_tilde)
    Xy : (p,) or (p,1) ndarray, 交叉协方差向量 (rho_hat)
    beta_start : (p,) ndarray or None, 初始系数向量（用于 warm start）
    max_iter : int, sklearn Lasso 最大迭代次数
    tol : float, sklearn Lasso 收敛容差

    返回
    ----------
    dict, 包含:
        'coefficients': 最终 beta 向量
        'num_it': sklearn 实际迭代次数
    """
    from scipy.linalg import cholesky
    from sklearn.linear_model import Lasso

    if not (np.all(np.isfinite(XX)) and np.all(np.isfinite(Xy))):
        return {"coefficients": np.zeros(p), "num_it": 0}

    try:
        U = cholesky(XX, lower=False)
    except np.linalg.LinAlgError:
        return {"coefficients": np.zeros(p), "num_it": 0}

    W_tilde = np.sqrt(p) * U

    Y_tilde = np.linalg.solve(U.T, np.sqrt(p) * Xy.ravel())

    if not (np.all(np.isfinite(W_tilde)) and np.all(np.isfinite(Y_tilde))):
        return {"coefficients": np.zeros(p), "num_it": 0}

    alpha = lambda_val

    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter, tol=tol,
                  warm_start=True)
    if beta_start is not None:
        lasso.coef_ = beta_start.copy().astype(float)
    else:
        lasso.coef_ = np.zeros(p)

    lasso.fit(W_tilde, Y_tilde)

    coef = lasso.coef_
    if not np.all(np.isfinite(coef)):
        coef = np.zeros(p)

    return {"coefficients": coef, "num_it": int(lasso.n_iter_)}


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
                Z[:, p1:][np.isnan(Z[:, p1:])] = 0
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
