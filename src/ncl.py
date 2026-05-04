"""
NCL (Nonconvex Lasso) Implementation

基于 Datta & Zou (2017) 的 NCL 方法实现。
用于与 CoCoLasso 对比的测量误差模型方法。

本模块完全独立实现，不依赖 CoCoLasso。
"""

import numpy as np
from typing import Optional, Dict


def _l1_proj(v: np.ndarray, b: float) -> np.ndarray:
    """
    将向量投影到半径为 b 的 L1 球上。
    基于 Duchi et al. (2008) 的高效投影算法。

    Parameters
    ----------
    v : array, 待投影向量
    b : float, L1 球半径 (必须 > 0)

    Returns
    -------
    w : array, 投影后的向量
    """
    assert b > 0, "Radius b must be positive"
    u = np.sort(np.abs(v))[::-1]
    sv = np.cumsum(u)
    rho = np.max(np.where(u > (sv - b) / np.arange(1, len(u) + 1)))
    theta = max(0, (sv[rho] - b) / (rho + 1))
    w = np.sign(v) * np.maximum(np.abs(v) - theta, 0)
    return w


def _log_space(start: float, stop: float, num: int) -> np.ndarray:
    """生成对数等比序列（类似 R 的 lseq）。"""
    return np.exp(np.linspace(np.log(start), np.log(stop), num))


def _lasso_covariance(p: int, lambda_val: float,
                      Sigma: np.ndarray, rho: np.ndarray,
                      beta_start: np.ndarray = None,
                      max_iter: int = 1000,
                      opt_tol: float = 1e-5,
                      zero_threshold: float = 1e-6) -> np.ndarray:
    """
    协方差形式的 Lasso 坐标下降求解。

    min_beta (1/2) beta' Sigma beta - rho' beta + lambda * ||beta||_1

    Parameters
    ----------
    p : int, 特征数
    lambda_val : float, 惩罚参数
    Sigma : (p, p) array, 协方差矩阵
    rho : (p,) array, 交叉协方差向量
    beta_start : (p,) array, 初始值
    max_iter : int, 最大迭代次数
    opt_tol : float, 收敛容差
    zero_threshold : float, 零阈值

    Returns
    -------
    beta : (p,) array, 估计系数
    """
    if beta_start is None:
        beta = np.zeros(p)
    else:
        beta = beta_start.copy().astype(float)
    rho = rho.ravel()

    for m in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            if Sigma[j, j] <= 0:
                beta[j] = 0
                continue
            S_j = Sigma[j, :] @ beta - Sigma[j, j] * beta[j] - rho[j]
            if S_j > lambda_val:
                beta[j] = (lambda_val - S_j) / Sigma[j, j]
            elif S_j < -lambda_val:
                beta[j] = (-lambda_val - S_j) / Sigma[j, j]
            else:
                beta[j] = 0

        if np.sum(np.abs(beta - beta_old)) < opt_tol:
            break

    beta[np.abs(beta) < zero_threshold] = 0
    return beta


def _ensure_positive_semidefinite(mat: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    对矩阵进行正定性修正：将负特征值截断为 epsilon。

    当修正协方差矩阵 Sigma_tilde = (1/n)Z'Z - tau^2*I 不正定时，
    需要先投影到 PSD 锥上再进行优化。

    Parameters
    ----------
    mat : (p, p) array, 待修正矩阵
    epsilon : float, 最小特征值

    Returns
    -------
    mat_psd : (p, p) array, 修正后的正定矩阵
    """
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.maximum(eigvals, epsilon)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def compute_corrected_covariance_additive(Z: np.ndarray, y: np.ndarray,
                                           n: int, tau: float,
                                           ensure_psd: bool = True) -> tuple:
    """
    计算加性误差下的修正协方差矩阵。

    Sigma_tilde = (1/n) Z'Z - tau^2 * I
    rho_tilde = (1/n) Z'y

    Parameters
    ----------
    Z : (n, p) array, 观测设计矩阵
    y : (n,) array, 响应变量
    n : int, 样本数
    tau : float, 误差标准差
    ensure_psd : bool, 是否对修正后的协方差矩阵做正定性修正

    Returns
    -------
    (Sigma_tilde, rho_tilde) : 修正后的协方差矩阵和交叉协方差向量
    """
    Sigma_tilde = (1 / n) * Z.T @ Z - tau ** 2 * np.eye(Z.shape[1])
    rho_tilde = (1 / n) * Z.T @ y
    if ensure_psd:
        Sigma_tilde = _ensure_positive_semidefinite(Sigma_tilde)
    return Sigma_tilde, rho_tilde


def _corrected_covariance_multiplicative(Z: np.ndarray, y: np.ndarray,
                                          n: int, p: int,
                                          tau: float,
                                          ensure_psd: bool = True) -> tuple:
    """
    计算乘性误差下的修正协方差矩阵。

    对于 Z = X * M, 其中 log(M_ij) ~ N(0, tau^2):
      Gamma_jk = (1/n) Z_j'Z_k / exp(tau^2)       for j != k
      Gamma_jj = (1/n) Z_j'Z_j / exp(2*tau^2)
      rho_j    = (1/n) Z_j'y / exp(tau^2/2)

    Parameters
    ----------
    Z : (n, p) array, 观测设计矩阵
    y : (n,) array, 响应变量
    n : int, 样本数
    p : int, 特征数
    tau : float, 误差标准差
    ensure_psd : bool, 是否做正定性修正

    Returns
    -------
    (Gamma, rho_tilde) : 修正后的协方差矩阵和交叉协方差向量
    """
    exp_tau2 = np.exp(tau ** 2)
    exp_2tau2 = np.exp(2 * tau ** 2)
    exp_tau2_half = np.exp(tau ** 2 / 2)

    Sigma_raw = (1 / n) * Z.T @ Z
    Gamma = Sigma_raw / exp_tau2
    np.fill_diagonal(Gamma, np.diag(Sigma_raw) / exp_2tau2)
    rho_tilde = ((1 / n) * Z.T @ y.ravel()) / exp_tau2_half

    if ensure_psd:
        Gamma = _ensure_positive_semidefinite(Gamma)

    return Gamma, rho_tilde


def ncl_coordinate_descent(Gamma: np.ndarray, rho: np.ndarray,
                            lambda_val: float, R: float, p: int,
                            beta_start: np.ndarray = None,
                            max_iter: int = 1000,
                            opt_tol: float = 1e-5,
                            zero_threshold: float = 1e-6) -> np.ndarray:
    """
    NCL 坐标下降算法。

    求解: min_beta (1/2) beta' Gamma beta - rho' beta + lambda * ||beta||_1
          s.t. ||beta||_1 <= R

    Parameters
    ----------
    Gamma : (p, p) array, 修正后的协方差矩阵（应已做正定性修正）
    rho : (p,) array, 修正后的交叉协方差向量
    lambda_val : float, L1 惩罚参数
    R : float, L1 约束半径
    p : int, 特征数
    beta_start : (p,) array, 初始值
    max_iter : int, 最大迭代次数
    opt_tol : float, 收敛容差
    zero_threshold : float, 零阈值

    Returns
    -------
    beta : (p,) array, 估计系数
    """
    if beta_start is None:
        beta = np.zeros(p)
    else:
        beta = beta_start.copy()

    for m in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            if Gamma[j, j] <= 0:
                beta[j] = 0
                continue
            S_j = Gamma[j, :] @ beta - Gamma[j, j] * beta[j] - rho[j]
            if S_j > lambda_val:
                beta[j] = (lambda_val - S_j) / Gamma[j, j]
            elif S_j < -lambda_val:
                beta[j] = (-lambda_val - S_j) / Gamma[j, j]
            else:
                beta[j] = 0

        if np.sum(np.abs(beta)) > R:
            beta = _l1_proj(beta, R)

        if np.sum(np.abs(beta - beta_old)) < opt_tol:
            break

    beta[np.abs(beta) < zero_threshold] = 0
    return beta


def naive_lasso_cv(Z: np.ndarray, y: np.ndarray, n: int, p: int,
                   K: int = 5, step: int = 100,
                   seed: int = 42) -> np.ndarray:
    """
    朴素 Lasso 交叉验证，用于初始化 NCL 的 R 参数。

    Parameters
    ----------
    Z : (n, p) array, 观测设计矩阵
    y : (n,) array, 响应变量
    n : int, 样本数
    p : int, 特征数
    K : int, 交叉验证折数
    step : int, lambda 路径长度
    seed : int, 随机种子

    Returns
    -------
    best_beta : (p,) array, 朴素 Lasso 估计系数
    """
    Sigma_naive = (1 / n) * Z.T @ Z
    rho_naive = (1 / n) * Z.T @ y
    lam_max = np.max(np.abs(rho_naive))
    if lam_max < 1e-10:
        lam_max = 1.0
    lam_min = 0.001 * lam_max
    lambda_list = _log_space(lam_max, lam_min, step)

    rng = np.random.RandomState(seed)
    folds = rng.randint(1, K + 1, size=n)

    best_cv = np.inf
    best_beta = np.zeros(p)
    beta_start = np.zeros(p)

    for lam in lambda_list:
        cv_errors = []
        for k in range(1, K + 1):
            idx_test = np.where(folds == k)[0]
            idx_train = np.where(folds != k)[0]
            n_test = len(idx_test)
            n_train = len(idx_train)

            Sigma_train = (1 / n_train) * Z[idx_train].T @ Z[idx_train]
            rho_train = (1 / n_train) * Z[idx_train].T @ y[idx_train]
            Sigma_test = (1 / n_test) * Z[idx_test].T @ Z[idx_test]
            rho_test = (1 / n_test) * Z[idx_test].T @ y[idx_test]

            beta_cv = _lasso_covariance(
                p, lam, Sigma_train, rho_train,
                beta_start=beta_start
            )
            err = beta_cv @ Sigma_test @ beta_cv - 2 * rho_test @ beta_cv
            cv_errors.append(err)

        mean_cv = np.mean(cv_errors)

        beta_full = _lasso_covariance(
            p, lam, Sigma_naive, rho_naive,
            beta_start=beta_start
        )

        if mean_cv < best_cv:
            best_cv = mean_cv
            best_beta = beta_full.copy()

        beta_start = beta_full

    return best_beta


def ncl_method(Z: np.ndarray, y: np.ndarray, n: int, p: int,
               tau: float, noise: str = "additive",
               K: int = 5, step: int = 100, n_R: int = 100,
               seed: int = 42) -> dict:
    """
    NCL 方法主函数。

    Parameters
    ----------
    Z : (n, p) array, 观测到的带误差设计矩阵
    y : (n,) array, 响应变量
    n : int, 样本数
    p : int, 特征数
    tau : float, 误差标准差
    noise : str, 误差类型 ('additive' 或 'multiplicative')
    K : int, 交叉验证折数
    step : int, lambda 路径长度
    n_R : int, R 参数搜索点数
    seed : int, 随机种子

    Returns
    -------
    dict : {'beta': 估计系数, 'lambda_val': 最优lambda, 'R': 最优R}
    """
    beta_init = naive_lasso_cv(Z, y, n, p, K=K, step=step, seed=seed)
    R_max = np.sum(np.abs(beta_init))
    if R_max < 1e-10:
        R_max = 1.0

    R_values = np.linspace(R_max / 500, 2 * R_max, n_R)

    if noise == "additive":
        Gamma_full, rho_full = compute_corrected_covariance_additive(Z, y, n, tau)
    else:
        Gamma_full, rho_full = _corrected_covariance_multiplicative(Z, y, n, p, tau)

    lam_max = np.max(np.abs(rho_full))
    if lam_max < 1e-10:
        lam_max = 1.0
    lam_min = 0.001 * lam_max
    lambda_list = _log_space(lam_max, lam_min, step)

    rng = np.random.RandomState(seed)
    folds = rng.randint(1, K + 1, size=n)

    best_cv_error = np.inf
    best_lambda = lam_max
    best_R = R_values[0]
    best_beta = np.zeros(p)

    for ri, R in enumerate(R_values):
        beta_start_r = np.zeros(p)
        for li, lam in enumerate(lambda_list):
            cv_errors = []
            for k in range(1, K + 1):
                idx_test = np.where(folds == k)[0]
                idx_train = np.where(folds != k)[0]
                n_test = len(idx_test)
                n_train = len(idx_train)

                Z_train, Z_test = Z[idx_train], Z[idx_test]
                y_train, y_test = y[idx_train], y[idx_test]

                if noise == "additive":
                    Gamma_train, rho_train = compute_corrected_covariance_additive(
                        Z_train, y_train, n_train, tau)
                    Gamma_test, rho_test = compute_corrected_covariance_additive(
                        Z_test, y_test, n_test, tau)
                else:
                    Gamma_train, rho_train = _corrected_covariance_multiplicative(
                        Z_train, y_train, n_train, p, tau)
                    Gamma_test, rho_test = _corrected_covariance_multiplicative(
                        Z_test, y_test, n_test, p, tau)

                beta_cv = ncl_coordinate_descent(
                    Gamma_train, rho_train, lam, R, p,
                    beta_start=beta_start_r
                )
                err = beta_cv @ Gamma_test @ beta_cv - 2 * rho_test @ beta_cv
                cv_errors.append(err)

            mean_cv = np.mean(cv_errors)

            beta_full = ncl_coordinate_descent(
                Gamma_full, rho_full, lam, R, p,
                beta_start=beta_start_r
            )

            if mean_cv < best_cv_error:
                best_cv_error = mean_cv
                best_lambda = lam
                best_R = R
                best_beta = beta_full.copy()

            beta_start_r = beta_full

    return {"beta": best_beta, "lambda_val": best_lambda, "R": best_R}
