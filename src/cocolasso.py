"""
CoCoLasso（凸校正 Lasso）估计器。

标准 CoCoLasso 使用路径坐标下降与 K 折交叉验证选择最优正则化参数，
支持加性误差、缺失数据、乘性误差三种噪声类型。

基于 R 语言包 BDcocolasso (https://github.com/celiaescribe/BDcocolasso) 的 Python 复现。
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Optional, Dict

from ._utils import (
    _additive_noise_variance,
    _admm_proj,
    _apply_preprocess_data,
    _hm_proj,
    _lasso_covariance,
    _lasso_sklearn,
    _corrected_covariance_multiplicative,
    _log_space,
    _make_cv_folds,
    _preprocess_data,
    _ratio_matrix_from_mask,
    _restore_coefficient_path,
    _restore_coefficients,
    _restore_intercept,
    _validate_common_options,
    _validate_ratio_matrix,
)


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
        raise ValueError(f"未知的噪声类型: {noise}")
    return float(np.max(np.abs(rho_tilde)))


def _cv_covariance_matrices(
    K: int,
    Z: np.ndarray,
    y: np.ndarray,
    p: int,
    center_Z: bool = True,
    scale_Z: bool = True,
    center_y: bool = True,
    scale_y: bool = True,
    mu: float = 1.0,
    tau: Optional[float] = None,
    global_preprocessed: Optional[Dict] = None,
    etol: float = 1e-4,
    noise: str = "additive",
    mode: str = "ADMM",
    random_state: Optional[int] = None,
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
    Z : (n, p) ndarray, 原始设计矩阵
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
    n = Z.shape[0]
    fold_indices = _make_cv_folds(n, K, random_state=random_state)

    if global_preprocessed is None:
        global_preprocessed = _preprocess_data(
            Z, y, n, p, center_Z, scale_Z, center_y, scale_y, noise,
        )

    mat = global_preprocessed["Z"]
    y_proc = global_preprocessed["y"]
    ratio_matrix = global_preprocessed["ratio_matrix"]
    observed_mask = global_preprocessed["observed_mask"]
    additive_noise_diag = None
    if noise == "additive":
        additive_noise_diag = _additive_noise_variance(
            tau, global_preprocessed["sd_Z"], scale_Z,
        )

    def _project(cov_mat, R=None):
        if mode == "ADMM":
            return _admm_proj(cov_mat, mu=mu, etol=etol)["mat"]
        else:
            return _hm_proj(cov_mat, R=R, mu=mu, tolerance=etol)

    if noise == "additive":
        if additive_noise_diag is None:
            additive_noise_diag = np.full(p, tau ** 2)
        cov_modified = (1 / n) * mat.T @ mat - np.diag(additive_noise_diag)
        sigma_global = _project(cov_modified)
        rho_global = (1 / n) * mat.T @ y_proc

    elif noise == "missing":
        _validate_ratio_matrix(ratio_matrix)
        if observed_mask is None:
            raise ValueError("缺失数据需要 observed_mask")
        cov_modified = (1 / n) * mat.T @ mat / ratio_matrix
        sigma_global = _project(cov_modified, R=ratio_matrix)
        rho_global = (1 / n) * mat.T @ y_proc / np.diag(ratio_matrix)

    elif noise == "multiplicative":
        Gamma_global, rho_global = _corrected_covariance_multiplicative(
            mat, y_proc, n, p, tau)
        sigma_global = _project(Gamma_global)

    else:
        raise ValueError(f"未知的噪声类型: {noise}")

    list_matrices_lasso = []
    list_matrices_error = []
    list_rho_lasso = []
    list_rho_error = []

    for index in fold_indices:
        train_index = np.setdiff1d(np.arange(n), index, assume_unique=False)
        n_without_fold = len(train_index)
        n_one_fold = len(index)

        train_preprocessed = _preprocess_data(
            Z[train_index], y[train_index], n_without_fold, p,
            center_Z, scale_Z, center_y, scale_y, noise,
        )
        test_preprocessed = _apply_preprocess_data(
            Z[index], y[index],
            train_preprocessed["mean_Z"], train_preprocessed["sd_Z"],
            train_preprocessed["mean_y"], train_preprocessed["sd_y"],
            center_Z, scale_Z, center_y, scale_y, noise,
        )

        mat_train = train_preprocessed["Z"]
        y_train = train_preprocessed["y"]
        mat_test = test_preprocessed["Z"]
        y_test = test_preprocessed["y"]

        if noise == "additive":
            additive_noise_diag_fold = _additive_noise_variance(
                tau, train_preprocessed["sd_Z"], scale_Z,
            )
            cov_train = (1 / n_without_fold) * mat_train.T @ mat_train - np.diag(additive_noise_diag_fold)
            cov_test = (1 / n_one_fold) * mat_test.T @ mat_test - np.diag(additive_noise_diag_fold)
            rho_train = (1 / n_without_fold) * mat_train.T @ y_train
            rho_test = (1 / n_one_fold) * mat_test.T @ y_test
        elif noise == "missing":
            ratio_train = train_preprocessed["ratio_matrix"]
            ratio_test = _ratio_matrix_from_mask(test_preprocessed["observed_mask"])
            _validate_ratio_matrix(ratio_train, "training ratio_matrix")
            _validate_ratio_matrix(ratio_test, "test ratio_matrix")
            cov_train = (1 / n_without_fold) * mat_train.T @ mat_train / ratio_train
            cov_test = (1 / n_one_fold) * mat_test.T @ mat_test / ratio_test
            rho_train = (1 / n_without_fold) * mat_train.T @ y_train / np.diag(ratio_train)
            rho_test = (1 / n_one_fold) * mat_test.T @ y_test / np.diag(ratio_test)
        elif noise == "multiplicative":
            cov_train, rho_train = _corrected_covariance_multiplicative(
                mat_train, y_train, n_without_fold, p, tau)
            cov_test, rho_test = _corrected_covariance_multiplicative(
                mat_test, y_test, n_one_fold, p, tau)

        list_matrices_lasso.append(_project(cov_train, R=ratio_train if noise == "missing" else None))
        list_matrices_error.append(_project(cov_test, R=ratio_test if noise == "missing" else None))
        list_rho_lasso.append(rho_train)
        list_rho_error.append(rho_test)

    return {
        "sigma_global": sigma_global,
        "rho_global": rho_global,
        "list_matrices_lasso": list_matrices_lasso,
        "list_matrices_error": list_matrices_error,
        "list_rho_lasso": list_rho_lasso,
        "list_rho_error": list_rho_error,
        "fold_indices": fold_indices,
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
    mu: float = 1.0,
    tau: Optional[float] = None,
    etol: float = 1e-4,
    optTol: float = 1e-10,
    earlyStopping_max: int = 10,
    noise: str = "additive",
    penalty: str = "lasso",
    mode: str = "ADMM",
    solver: str = "coordinate_descent",
    alpha: Optional[float] = None,
    random_state: Optional[int] = None,
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
    solver : str, 'coordinate_descent' 或 'sklearn'
        Lasso 求解器。'coordinate_descent' 为协方差形式坐标下降（支持 SCAD），
        'sklearn' 为 Cholesky 分解 + sklearn Lasso（仅支持 lasso 惩罚）
    random_state : int or None, 交叉验证折分随机种子

    返回
    ----------
    dict, 包含:
        'lambda_opt', 'lambda_sd', 'beta_opt', 'beta_sd',
        'data_error', 'data_beta', 'early_stopping',
        'mean_Z', 'sd_Z', 'mean_y', 'sd_y'
    """
    _validate_common_options(
        noise=noise,
        allowed_noises={"additive", "missing", "multiplicative"},
        penalty=penalty,
        mode=mode,
        solver=solver,
        tau=tau,
        tau_required=noise in {"additive", "multiplicative"},
    )

    preprocessed = _preprocess_data(Z, y, n, p, center_Z, scale_Z, center_y, scale_y, noise)
    Z_proc = preprocessed["Z"]
    y_proc = preprocessed["y"]
    ratio_matrix = preprocessed["ratio_matrix"]

    if alpha is not None:
        if alpha < 0:
            raise ValueError("alpha 必须非负")
        step = 1
        lam_max = float(alpha)
        lambda_list = np.array([float(alpha)])
    else:
        if lambda_factor is None:
            lambda_factor = 0.01 if n < p else 0.001
        lam_max = _lambda_max(Z_proc, y_proc, n, ratio_matrix, noise, tau)
        if not np.isfinite(lam_max) or lam_max <= 0:
            lambda_list = np.zeros(step)
        else:
            lam_min = lambda_factor * lam_max
            lambda_list = _log_space(lam_max, lam_min, step)

    earlyStopping = step

    beta_start = np.zeros(p)
    best_lambda = lam_max
    beta_opt = beta_start.copy()
    best_error = np.inf
    error_list = np.zeros((step, 4))
    error = 1000.0
    earlyStopping_high = 0
    matrix_beta = np.zeros((step, p))

    output = _cv_covariance_matrices(
        K=K, Z=Z, y=y, p=p,
        center_Z=center_Z, scale_Z=scale_Z, center_y=center_y, scale_y=scale_y,
        mu=mu, tau=tau, global_preprocessed=preprocessed,
        etol=etol, noise=noise, mode=mode, random_state=random_state,
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
            if solver == "sklearn":
                coef_lambda = _lasso_sklearn(
                    n=n - len(output["fold_indices"][k]), p=p, lambda_val=lambda_step, XX=sigma_train, Xy=rho_train,
                    beta_start=beta_start,
                )["coefficients"]
            else:
                coef_lambda = _lasso_covariance(
                    n=n - len(output["fold_indices"][k]), p=p, lambda_val=lambda_step, XX=sigma_train, Xy=rho_train,
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

        if solver == "sklearn":
            coef_tot = _lasso_sklearn(
                n=n, p=p, lambda_val=lambda_step, XX=ZZ, Xy=Zy,
                beta_start=beta_start,
            )["coefficients"]
        else:
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


class CoCoLasso(BaseEstimator, RegressorMixin):
    """
    CoCoLasso（凸校正 Lasso）估计器。

    针对协变量中存在测量误差（加性误差、缺失数据、乘性误差）的
    高维误差变量回归问题，使用路径坐标下降与 K 折交叉验证
    选择最优正则化参数。

    参数
    ----------
    alpha : float or None, 默认=None
        固定正则化强度；当 alpha=None 时，通过交叉验证自动选取。
    noise : str, 默认="additive"
        噪声类型，可选 "additive"（加性）、"missing"（缺失）、
        "multiplicative"（乘性）。
    tau : float or None, 默认=None
        加性误差的标准差（noise="additive" 或 "multiplicative" 时必填）。
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
        lambda_min / lambda_max 的比率，None 时自动选取。
    center_Z : bool, 默认=True
        是否对设计矩阵 Z 中心化。
    scale_Z : bool, 默认=True
        是否对设计矩阵 Z 标准化。
    center_y : bool, 默认=True
        是否对响应 y 中心化。
    scale_y : bool, 默认=True
        是否对响应 y 标准化。
    random_state : int or None, 默认=None
        交叉验证折分随机种子；None 时沿用全局 numpy 随机状态。

    属性
    ----------
    coef_ : ndarray, shape (p,)
        拟合后的系数向量（对应最优 lambda）。
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
        系数路径（各 lambda 下的系数）。
    n_iter_ : int
        实际迭代步数。
    mean_Z_ : ndarray
        Z 的列均值（预处理用）。
    sd_Z_ : ndarray
        Z 的列标准差（预处理用）。
    mean_y_ : float
        y 的均值（预处理用）。
    sd_y_ : float
        y 的标准差（预处理用）。

    示例
    ----------
    >>> from src import CoCoLasso
    >>> model = CoCoLasso(tau=0.75, noise="additive")
    >>> model.fit(Z, y)
    >>> print(model.coef_)
    >>> y_pred = model.predict(Z_new)
    """

    def __init__(
        self,
        alpha=None,
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
        random_state=None,
    ):
        self.alpha = alpha
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
        self.random_state = random_state

    def fit(self, Z, y):
        """
        拟合 CoCoLasso 模型。

        参数
        ----------
        Z : ndarray, shape (n, p)
            含测量误差的设计矩阵。
        y : ndarray, shape (n,)
            响应向量。

        返回
        -------
        self : CoCoLasso
            拟合后的估计器实例。
        """
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = Z.shape

        result = _pathwise_coordinate_descent(
            Z=Z, y=y, n=n, p=p,
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
            alpha=self.alpha,
            random_state=self.random_state,
        )

        self.coef_scaled_ = result["beta_opt"]
        self.coef_sd_scaled_ = result["beta_sd"]
        self.lambda_opt_ = result["lambda_opt"]
        self.lambda_sd_ = result["lambda_sd"]
        self.cv_results_ = result["data_error"]
        self.coef_path_scaled_ = result["data_beta"]
        self.n_iter_ = result["early_stopping"]
        self.mean_Z_ = result["mean_Z"]
        self.sd_Z_ = result["sd_Z"]
        self.mean_y_ = result["mean_y"]
        self.sd_y_ = result["sd_y"]

        feature_scale = self.sd_Z_ if self.scale_Z else np.ones_like(self.sd_Z_)
        response_scale = self.sd_y_ if self.scale_y else 1.0
        self.coef_ = _restore_coefficients(self.coef_scaled_, feature_scale, response_scale)
        self.coef_sd_ = _restore_coefficients(self.coef_sd_scaled_, feature_scale, response_scale)
        self.coef_path_ = _restore_coefficient_path(self.coef_path_scaled_, feature_scale, response_scale)
        self.intercept_ = _restore_intercept(
            self.mean_Z_, self.mean_y_, self.coef_, self.center_Z, self.center_y,
        )

        return self

    def predict(self, Z):
        Z = np.asarray(Z, dtype=float)
        Z_clean = np.where(np.isnan(Z), self.mean_Z_, Z)
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
