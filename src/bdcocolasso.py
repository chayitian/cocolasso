"""
BDCoCoLasso（二块下降 CoCoLasso）估计器。

设计矩阵 Z 的前 p1 列为无误差特征，后 p2 列为含误差特征，
通过交替块坐标下降与 K 折交叉验证选择最优正则化参数。

基于 R 语言包 BDcocolasso (https://github.com/celiaescribe/BDcocolasso) 的 Python 复现。
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Optional, Dict

from ._utils import (
    _admm_proj,
    _hm_proj,
    _lasso_covariance,
    _lasso_sklearn,
    _log_space,
    _preprocess_data,
)


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
    inner_max_iter: int = 200,
    opt_tol: float = 1e-5,
    zero_threshold: float = 1e-6,
    solver: str = "coordinate_descent",
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

        if solver == "sklearn":
            beta1 = _lasso_sklearn(
                n=n, p=p1, lambda_val=lambda_val, XX=sigma1, Xy=Xy1,
                beta_start=beta1_old,
            )["coefficients"]
        else:
            beta1 = _lasso_covariance(
                n=n, p=p1, lambda_val=lambda_val, XX=sigma1, Xy=Xy1,
                beta_start=beta1_old, penalty=penalty, max_iter=inner_max_iter,
                opt_tol=opt_tol, zero_threshold=zero_threshold,
            )["coefficients"]

        if noise == "additive":
            Xy2 = (1 / n) * Z2.T @ (y - X1 @ beta1)
        else:
            Xy2 = (1 / n) * (Z2.T @ (y - X1 @ beta1)) / np.diag(ratio_matrix)

        if solver == "sklearn":
            beta2 = _lasso_sklearn(
                n=n, p=p2, lambda_val=lambda_val, XX=sigma2, Xy=Xy2,
                beta_start=beta2_old,
            )["coefficients"]
        else:
            beta2 = _lasso_covariance(
                n=n, p=p2, lambda_val=lambda_val, XX=sigma2, Xy=Xy2,
                beta_start=beta2_old, penalty=penalty, max_iter=inner_max_iter,
                opt_tol=opt_tol, zero_threshold=zero_threshold,
            )["coefficients"]

        if (np.sum(np.abs(beta1 - beta1_old)) < opt_tol and
                np.sum(np.abs(beta2 - beta2_old)) < opt_tol):
            break
        m += 1

    beta1[np.abs(beta1) < zero_threshold] = 0
    beta2[np.abs(beta2) < zero_threshold] = 0
    return {"coefficients_beta1": beta1, "coefficients_beta2": beta2, "num_it": m}


def _cv_covariance_matrices_block(
    K: int,
    mat: np.ndarray,
    y: np.ndarray,
    p: int,
    p1: int,
    p2: int,
    mu: float = 1.0,
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
    mu: float = 1.0,
    tau: Optional[float] = None,
    etol: float = 1e-4,
    optTol: float = 1e-5,
    earlyStopping_max: int = 10,
    noise: str = "additive",
    penalty: str = "lasso",
    mode: str = "ADMM",
    solver: str = "coordinate_descent",
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
                solver=solver,
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
            if not np.isfinite(err):
                err = 1e10
            cv_errors.append(err)

        cv_errors_arr = np.array(cv_errors)
        cv_errors_arr = np.where(np.isfinite(cv_errors_arr), cv_errors_arr, 1e10)
        error = np.mean(cv_errors_arr)
        error_list[i, 0] = error
        error_list[i, 1] = np.quantile(cv_errors_arr, 0.1)
        error_list[i, 2] = np.quantile(cv_errors_arr, 0.9)
        error_list[i, 3] = np.std(cv_errors_arr)

        out = _lasso_covariance_block(
            n=n, p1=p1, p2=p2, X1=X1, Z2=Z2, y=y_proc,
            sigma1=sigma1, sigma2=sigma2, lambda_val=lambda_step,
            noise=noise, ratio_matrix=ratio_matrix,
            beta1_start=beta1_start, beta2_start=beta2_start, penalty=penalty,
            solver=solver,
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

    errors_clean = np.where(np.isfinite(df_error["error"]), df_error["error"], 1e10)
    step_min = int(np.argmin(errors_clean))
    sd_best = df_error["error_sd"][step_min]
    if not np.isfinite(sd_best):
        sd_best = 0.0
    threshold = errors_clean[step_min] + sd_best
    candidates = np.where(
        (errors_clean <= threshold) &
        (df_error["lambda"] >= df_error["lambda"][step_min])
    )[0]
    step_sd = int(np.min(candidates)) if len(candidates) > 0 else step_min
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


class BDCoCoLasso(BaseEstimator, RegressorMixin):
    """
    二块下降 CoCoLasso（BD-CoCoLasso）估计器。

    设计矩阵 Z 的前 p1 列为无误差特征，后 p2 列为含误差特征，
    通过交替块坐标下降与 K 折交叉验证选择最优正则化参数。

    参数
    ----------
    alpha : float or None, 默认=None
        正则化强度参数。当 alpha=None 时，通过交叉验证自动选取。
    p1 : int, 默认=0
        无误差特征的列数。
    p2 : int, 默认=0
        含误差特征的列数。
    noise : str, 默认="additive"
        噪声类型，可选 "additive" 或 "missing"。
    tau : float or None, 默认=None
        加性误差的标准差。
    penalty : str, 默认="lasso"
        惩罚类型，可选 "lasso" 或 "SCAD"。
    mode : str, 默认="ADMM"
        PSD 投影模式，可选 "ADMM" 或 "HM"。
    mu : float, 默认=1.0
        ADMM 惩罚参数。
    max_iter : int, 默认=100
        lambda 路径上的最大搜索步数。
    cv_folds : int, 默认=4
        交叉验证折数。
    tol : float, 默认=1e-5
        收敛容差。
    etol : float, 默认=1e-4
        ADMM 收敛容差。
    early_stopping_max : int, 默认=10
        早停最大连续递增次数。
    lambda_factor : float or None, 默认=None
        lambda_min / lambda_max 的比率。
    center_Z : bool, 默认=True
        是否对设计矩阵 Z 中心化。
    scale_Z : bool, 默认=True
        是否对设计矩阵 Z 标准化。
    center_y : bool, 默认=True
        是否对响应 y 中心化。
    scale_y : bool, 默认=True
        是否对响应 y 标准化。

    属性
    ----------
    coef_ : ndarray, shape (p,)
        拟合后的系数向量（前 p1 + 后 p2）。
    intercept_ : float
        拟合后的截距。
    coef_sd_ : ndarray, shape (p,)
        1-std 准则下的系数向量。
    lambda_opt_ : float
        交叉验证选出的最优 lambda。
    lambda_sd_ : float
        1-std 准则下的 lambda。
    cv_results_ : dict
        交叉验证误差详情。
    coef_path_ : dict
        系数路径。
    n_iter_ : int
        实际迭代步数。
    mean_Z_ : ndarray
        Z 的列均值。
    sd_Z_ : ndarray
        Z 的列标准差。
    mean_y_ : float
        y 的均值。
    sd_y_ : float
        y 的标准差。

    示例
    ----------
    >>> from src import BDCoCoLasso
    >>> model = BDCoCoLasso(p1=50, p2=200, tau=0.75, noise="additive")
    >>> model.fit(Z, y)
    >>> print(model.coef_)
    """

    def __init__(
        self,
        alpha=None,
        p1=0,
        p2=0,
        noise="additive",
        tau=None,
        penalty="lasso",
        mode="ADMM",
        mu=1.0,
        max_iter=100,
        cv_folds=4,
        tol=1e-5,
        etol=1e-4,
        early_stopping_max=10,
        lambda_factor=None,
        center_Z=True,
        scale_Z=True,
        center_y=True,
        scale_y=True,
        solver="coordinate_descent",
    ):
        self.alpha = alpha
        self.p1 = p1
        self.p2 = p2
        self.noise = noise
        self.tau = tau
        self.penalty = penalty
        self.mode = mode
        self.mu = mu
        self.max_iter = max_iter
        self.cv_folds = cv_folds
        self.tol = tol
        self.etol = etol
        self.early_stopping_max = early_stopping_max
        self.lambda_factor = lambda_factor
        self.center_Z = center_Z
        self.scale_Z = scale_Z
        self.center_y = center_y
        self.scale_y = scale_y
        self.solver = solver

    def fit(self, Z, y):
        """
        拟合 BD-CoCoLasso 模型。

        参数
        ----------
        Z : ndarray, shape (n, p)
            设计矩阵（前 p1 列无误差，后 p2 列含误差）。
        y : ndarray, shape (n,)
            响应向量。

        返回
        -------
        self : BDCoCoLasso
            拟合后的估计器实例。
        """
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = Z.shape

        if self.p1 + self.p2 != p:
            raise ValueError(
                f"p1 + p2 = {self.p1 + self.p2} 不等于 Z 的列数 {p}"
            )

        result = _blockwise_coordinate_descent(
            Z=Z, y=y, n=n, p=p, p1=self.p1, p2=self.p2,
            center_Z=self.center_Z, scale_Z=self.scale_Z,
            center_y=self.center_y, scale_y=self.scale_y,
            lambda_factor=self.lambda_factor,
            step=self.max_iter,
            K=self.cv_folds,
            mu=self.mu,
            tau=self.tau,
            etol=self.etol,
            optTol=self.tol,
            earlyStopping_max=self.early_stopping_max,
            noise=self.noise,
            penalty=self.penalty,
            mode=self.mode,
            solver=self.solver,
        )

        self.coef_ = result["beta_opt"]
        self.coef_sd_ = result["beta_sd"]
        self.lambda_opt_ = result["lambda_opt"]
        self.lambda_sd_ = result["lambda_sd"]
        self.cv_results_ = result["data_error"]
        self.coef_path_ = result["data_beta"]
        self.n_iter_ = result["early_stopping"]
        self.mean_Z_ = result["mean_Z"]
        self.sd_Z_ = result["sd_Z"]
        self.mean_y_ = result["mean_y"]
        self.sd_y_ = result["sd_y"]

        sd_Z_safe = np.where(self.sd_Z_ != 0, self.sd_Z_, 1.0)
        coef_original = self.coef_ * self.sd_y_ / sd_Z_safe
        self.intercept_ = (
            self.mean_y_
            - np.dot(self.mean_Z_, coef_original)
        )

        return self

    def predict(self, Z):
        Z = np.asarray(Z, dtype=float)
        Z_clean = np.where(np.isnan(Z), 0.0, Z)
        return Z_clean @ self.coef_ + self.intercept_

    def score(self, Z, y):
        """
        计算 R² 决定系数。

        参数
        ----------
        Z : ndarray, shape (n_samples, p)
            设计矩阵。
        y : ndarray, shape (n_samples,)
            真实响应值。

        返回
        -------
        r2 : float
            R² 决定系数。
        """
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(Z)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1.0 - ss_res / ss_tot
