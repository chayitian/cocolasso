"""
NCL (Nonconvex Lasso) Implementation

基于 Datta & Zou (2017) 的 NCL 方法实现。
用于与 CoCoLasso 对比的测量误差模型方法。
"""

import numpy as np
from typing import Optional, Dict

try:
    from .cocolasso import (
        lasso_covariance, l1_proj, _log_space,
        _corrected_covariance_multiplicative
    )
except ImportError:
    from cocolasso import (
        lasso_covariance, l1_proj, _log_space,
        _corrected_covariance_multiplicative
    )


def compute_corrected_covariance_additive(Z: np.ndarray, y: np.ndarray,
                                           n: int, tau: float) -> tuple:
    """
    计算加性误差下的修正协方差矩阵。

    Sigma_tilde = (1/n) Z'Z - tau^2 * I
    rho_tilde = (1/n) Z'y
    """
    Sigma_tilde = (1 / n) * Z.T @ Z - tau ** 2 * np.eye(Z.shape[1])
    rho_tilde = (1 / n) * Z.T @ y
    return Sigma_tilde, rho_tilde


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
            beta = l1_proj(beta, R)

        if np.sum(np.abs(beta - beta_old)) < opt_tol:
            break

    beta[np.abs(beta) < zero_threshold] = 0
    return beta


def naive_lasso_cv(Z: np.ndarray, y: np.ndarray, n: int, p: int,
                   K: int = 5, step: int = 100) -> np.ndarray:
    """
    朴素 Lasso 交叉验证,用于初始化 NCL 的 R 参数。
    """
    Sigma_naive = (1 / n) * Z.T @ Z
    rho_naive = (1 / n) * Z.T @ y
    lam_max = np.max(np.abs(rho_naive))
    lam_min = 0.001 * lam_max
    lambda_list = _log_space(lam_max, lam_min, step)

    np.random.seed(42)
    folds = np.random.randint(1, K + 1, size=n)

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

            out = lasso_covariance(
                n_train, p, lam, Sigma_train, rho_train,
                beta_start=beta_start, penalty="lasso"
            )
            beta_cv = out["coefficients"]
            err = beta_cv @ Sigma_test @ beta_cv - 2 * rho_test @ beta_cv
            cv_errors.append(err)

        mean_cv = np.mean(cv_errors)
        if mean_cv < best_cv:
            best_cv = mean_cv
            out_full = lasso_covariance(
                n, p, lam, Sigma_naive, rho_naive,
                beta_start=beta_start, penalty="lasso"
            )
            best_beta = out_full["coefficients"]

        out_full = lasso_covariance(
            n, p, lam, Sigma_naive, rho_naive,
            beta_start=beta_start, penalty="lasso"
        )
        beta_start = out_full["coefficients"]

    return best_beta


def ncl_method(Z: np.ndarray, y: np.ndarray, n: int, p: int,
               tau: float, noise: str = "additive",
               K: int = 5, step: int = 100, n_R: int = 100,
               seed: int = 42) -> dict:
    """
    NCL 方法主函数。

    参数:
        Z: 观测到的带误差设计矩阵
        y: 响应变量
        n: 样本数
        p: 特征数
        tau: 误差标准差
        noise: 误差类型 ('additive' 或 'multiplicative')
        K: 交叉验证折数
        step: lambda 路径长度
        n_R: R 参数搜索点数
        seed: 随机种子

    返回:
        {'beta': 估计系数, 'lambda': 最优lambda, 'R': 最优R}
    """
    beta_init = naive_lasso_cv(Z, y, n, p, K=K, step=step)
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

    np.random.seed(seed)
    folds = np.random.randint(1, K + 1, size=n)

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
            if mean_cv < best_cv_error:
                best_cv_error = mean_cv
                best_lambda = lam
                best_R = R
                best_beta = ncl_coordinate_descent(
                    Gamma_full, rho_full, lam, R, p,
                    beta_start=beta_start_r
                )

            beta_start_r = ncl_coordinate_descent(
                Gamma_full, rho_full, lam, R, p,
                beta_start=beta_start_r
            )

    return {"beta": best_beta, "lambda": best_lambda, "R": best_R}
