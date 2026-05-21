"""
GeneralCoCoLasso（三块混合 CoCoLasso）估计器。

设计矩阵 Z 按列分为三块：无误差特征、含加性误差特征、含缺失数据特征，
通过交替块坐标下降与 K 折交叉验证选择最优正则化参数。

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
    inner_max_iter: int = 200,
    opt_tol: float = 1e-5,
    zero_threshold: float = 1e-6,
    solver: str = "coordinate_descent",
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

            Xy2 = (1 / n) * Z2.T @ (y - X1 @ beta1 - Z3_tilde @ beta3)
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

            Xy3 = (1 / n) * (Z3.T @ (y - X1 @ beta1 - Z2 @ beta2)) / np.diag(ratio_matrix)
            if solver == "sklearn":
                beta3 = _lasso_sklearn(
                    n=n, p=p3, lambda_val=lambda_val, XX=sigma3, Xy=Xy3,
                    beta_start=beta3_old,
                )["coefficients"]
            else:
                beta3 = _lasso_covariance(
                    n=n, p=p3, lambda_val=lambda_val, XX=sigma3, Xy=Xy3,
                    beta_start=beta3_old, penalty=penalty, max_iter=inner_max_iter,
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

            Xy3 = (1 / n) * (Z3.T @ (y - Z2 @ beta2)) / np.diag(ratio_matrix)
            if solver == "sklearn":
                beta3 = _lasso_sklearn(
                    n=n, p=p3, lambda_val=lambda_val, XX=sigma3, Xy=Xy3,
                    beta_start=beta3_old,
                )["coefficients"]
            else:
                beta3 = _lasso_covariance(
                    n=n, p=p3, lambda_val=lambda_val, XX=sigma3, Xy=Xy3,
                    beta_start=beta3_old, penalty=penalty, max_iter=inner_max_iter,
                    opt_tol=opt_tol, zero_threshold=zero_threshold,
                )["coefficients"]

            if (np.sum(np.abs(beta2 - beta2_old)) < opt_tol and
                    np.sum(np.abs(beta3 - beta3_old)) < opt_tol):
                break
            m += 1

        beta2[np.abs(beta2) < zero_threshold] = 0
        beta3[np.abs(beta3) < zero_threshold] = 0
        return {"coefficients_beta2": beta2, "coefficients_beta3": beta3, "num_it": m}


def _cv_covariance_matrices_block_general(
    K: int,
    Z: np.ndarray,
    y: np.ndarray,
    p: int,
    p1: int,
    p2: int,
    p3: int,
    center_Z: bool = True,
    scale_Z: bool = True,
    center_y: bool = True,
    scale_y: bool = True,
    mu: float = 1.0,
    tau: Optional[float] = None,
    global_preprocessed: Optional[Dict] = None,
    etol: float = 1e-4,
    mode: str = "ADMM",
    random_state: Optional[int] = None,
) -> Dict:
    """
    为三块 BD-CoCoLasso 交叉验证创建投影后的 PSD 协方差矩阵。

    参数
    ----------
    K : int, 交叉验证折数
    Z : (n, p) ndarray, 原始设计矩阵
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
    n = Z.shape[0]
    fold_indices = _make_cv_folds(n, K, random_state=random_state)

    if global_preprocessed is None:
        global_preprocessed = _preprocess_data(
            Z, y, n, p, center_Z, scale_Z, center_y, scale_y,
            noise="missing", p1=p1 + p2, p2=p3,
        )

    mat = global_preprocessed["Z"]
    ratio_matrix = global_preprocessed["ratio_matrix"]
    observed_mask = global_preprocessed["observed_mask"]
    additive_noise_diag = _additive_noise_variance(
        tau, global_preprocessed["sd_Z"][p1:p1 + p2], scale_Z,
    )

    def _project(cov_mat, R=None):
        if mode == "ADMM":
            return _admm_proj(cov_mat, mu=mu, etol=etol)["mat"]
        else:
            return _hm_proj(cov_mat, R=R, mu=mu, tolerance=etol)

    mat_uncorrupted = mat[:, :p1] if p1 > 0 else None
    mat_corrupted_additive = mat[:, p1:p1 + p2]
    mat_corrupted_missing = mat[:, p1 + p2:p]

    if additive_noise_diag is None:
        additive_noise_diag = np.full(p2, tau ** 2)
    _validate_ratio_matrix(ratio_matrix)
    if observed_mask is None:
        raise ValueError("缺失数据需要 observed_mask")

    cov_modified_additive = (1 / n) * mat_corrupted_additive.T @ mat_corrupted_additive - np.diag(additive_noise_diag)
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
    list_ratio_lasso_missing = []
    list_ratio_error_missing = []
    list_X1_lasso = []
    list_Z2_lasso = []
    list_Z3_lasso = []
    list_y_lasso = []
    list_X1_error = []
    list_Z2_error = []
    list_Z3_error = []
    list_y_error = []

    for index in fold_indices:
        train_index = np.setdiff1d(np.arange(n), index, assume_unique=False)
        n_without_fold = len(train_index)
        n_one_fold = len(index)

        train_preprocessed = _preprocess_data(
            Z[train_index], y[train_index], n_without_fold, p,
            center_Z, scale_Z, center_y, scale_y,
            noise="missing", p1=p1 + p2, p2=p3,
        )
        test_preprocessed = _apply_preprocess_data(
            Z[index], y[index],
            train_preprocessed["mean_Z"], train_preprocessed["sd_Z"],
            train_preprocessed["mean_y"], train_preprocessed["sd_y"],
            center_Z, scale_Z, center_y, scale_y,
            noise="missing", p1=p1 + p2, p2=p3,
        )

        mat_train = train_preprocessed["Z"]
        mat_test = test_preprocessed["Z"]
        mat_train_unc = mat_train[:, :p1] if p1 > 0 else None
        mat_test_unc = mat_test[:, :p1] if p1 > 0 else None
        mat_train_add = mat_train[:, p1:p1 + p2]
        mat_test_add = mat_test[:, p1:p1 + p2]
        mat_train_miss = mat_train[:, p1 + p2:p]
        mat_test_miss = mat_test[:, p1 + p2:p]

        additive_noise_diag_fold = _additive_noise_variance(
            tau, train_preprocessed["sd_Z"][p1:p1 + p2], scale_Z,
        )
        cov_train_add = (1 / n_without_fold) * mat_train_add.T @ mat_train_add - np.diag(additive_noise_diag_fold)
        cov_test_add = (1 / n_one_fold) * mat_test_add.T @ mat_test_add - np.diag(additive_noise_diag_fold)
        list_PSD_lasso_additive.append(_project(cov_train_add))
        list_PSD_error_additive.append(_project(cov_test_add))

        ratio_train = train_preprocessed["ratio_matrix"]
        ratio_test = _ratio_matrix_from_mask(test_preprocessed["observed_mask"])
        _validate_ratio_matrix(ratio_train, "training ratio_matrix")
        _validate_ratio_matrix(ratio_test, "test ratio_matrix")
        cov_train_miss = (1 / n_without_fold) * mat_train_miss.T @ mat_train_miss / ratio_train
        cov_test_miss = (1 / n_one_fold) * mat_test_miss.T @ mat_test_miss / ratio_test
        list_PSD_lasso_missing.append(_project(cov_train_miss, R=ratio_train))
        list_PSD_error_missing.append(_project(cov_test_miss, R=ratio_test))
        list_ratio_lasso_missing.append(ratio_train)
        list_ratio_error_missing.append(ratio_test)

        if p1 > 0:
            list_sigma_lasso.append((1 / n_without_fold) * mat_train_unc.T @ mat_train_unc)
            list_sigma_error.append((1 / n_one_fold) * mat_test_unc.T @ mat_test_unc)
        list_X1_lasso.append(mat_train_unc)
        list_Z2_lasso.append(mat_train_add)
        list_Z3_lasso.append(mat_train_miss)
        list_y_lasso.append(train_preprocessed["y"])
        list_X1_error.append(mat_test_unc)
        list_Z2_error.append(mat_test_add)
        list_Z3_error.append(mat_test_miss)
        list_y_error.append(test_preprocessed["y"])

    result = {
        "sigma_global_corrupted_additive": sigma_global_corrupted_additive,
        "sigma_global_corrupted_missing": sigma_global_corrupted_missing,
        "list_PSD_lasso_additive": list_PSD_lasso_additive,
        "list_PSD_error_additive": list_PSD_error_additive,
        "list_PSD_lasso_missing": list_PSD_lasso_missing,
        "list_PSD_error_missing": list_PSD_error_missing,
        "list_ratio_lasso_missing": list_ratio_lasso_missing,
        "list_ratio_error_missing": list_ratio_error_missing,
        "list_X1_lasso": list_X1_lasso,
        "list_Z2_lasso": list_Z2_lasso,
        "list_Z3_lasso": list_Z3_lasso,
        "list_y_lasso": list_y_lasso,
        "list_X1_error": list_X1_error,
        "list_Z2_error": list_Z2_error,
        "list_Z3_error": list_Z3_error,
        "list_y_error": list_y_error,
        "fold_indices": fold_indices,
    }
    if p1 > 0:
        result["sigma_global_uncorrupted"] = sigma_global_uncorrupted
        result["list_sigma_lasso"] = list_sigma_lasso
        result["list_sigma_error"] = list_sigma_error
    return result


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
    mu: float = 1.0,
    tau: Optional[float] = None,
    etol: float = 1e-4,
    optTol: float = 1e-5,
    earlyStopping_max: int = 10,
    penalty: str = "lasso",
    mode: str = "ADMM",
    solver: str = "coordinate_descent",
    alpha: Optional[float] = None,
    random_state: Optional[int] = None,
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
    random_state : int or None, 交叉验证折分随机种子

    返回
    ----------
    dict, 包含:
        'lambda_opt', 'lambda_sd', 'beta_opt', 'beta_sd',
        'data_error', 'data_beta', 'early_stopping',
        'mean_Z', 'sd_Z', 'mean_y', 'sd_y'
    """
    _validate_common_options(
        noise=None,
        allowed_noises=None,
        penalty=penalty,
        mode=mode,
        solver=solver,
        tau=tau,
        tau_required=True,
    )
    if p1 < 0 or p2 <= 0 or p3 <= 0:
        raise ValueError("p1 必须非负，且 p2 和 p3 必须为正")

    preprocessed = _preprocess_data(
        Z, y, n, p, center_Z, scale_Z, center_y, scale_y,
        noise="missing", p1=p1 + p2, p2=p3,
    )
    Z_proc = preprocessed["Z"]
    y_proc = preprocessed["y"]
    ratio_matrix = preprocessed["ratio_matrix"]
    observed_mask = preprocessed["observed_mask"]
    additive_noise_diag = _additive_noise_variance(
        tau, preprocessed["sd_Z"][p1:p1 + p2], scale_Z,
    )

    X1 = Z_proc[:, :p1] if p1 > 0 else None
    Z2 = Z_proc[:, p1:p1 + p2]
    Z3 = Z_proc[:, p1 + p2:p]
    if alpha is not None:
        if alpha < 0:
            raise ValueError("alpha 必须非负")
        step = 1
        lam_max = float(alpha)
        lambda_list = np.array([float(alpha)])
    else:
        if lambda_factor is None:
            lambda_factor = 0.01 if n < p else 0.001
        rho_tilde1 = (1 / n) * Z_proc[:, :p1 + p2].T @ y_proc
        rho_tilde2 = (1 / n) * Z3.T @ y_proc / np.diag(ratio_matrix)
        lam_max = float(max(
            np.max(np.abs(rho_tilde1)) if len(rho_tilde1) > 0 else 0,
            np.max(np.abs(rho_tilde2)) if len(rho_tilde2) > 0 else 0,
        ))
        if not np.isfinite(lam_max) or lam_max <= 0:
            lambda_list = np.zeros(step)
        else:
            lam_min = lambda_factor * lam_max
            lambda_list = _log_space(lam_max, lam_min, step)

    earlyStopping = step

    beta1_start = np.zeros(p1)
    beta2_start = np.zeros(p2)
    beta3_start = np.zeros(p3)
    beta_start = (np.concatenate([beta1_start, beta2_start, beta3_start])
                  if p1 > 0 else np.concatenate([beta2_start, beta3_start]))
    best_lambda = lam_max
    beta_opt = beta_start.copy()
    best_error = np.inf
    error_list = np.zeros((step, 4))
    error = 0.0
    earlyStopping_high = 0
    matrix_beta = np.zeros((step, p))

    output = _cv_covariance_matrices_block_general(
        K=K, Z=Z, y=y, p=p, p1=p1, p2=p2, p3=p3,
        center_Z=center_Z, scale_Z=scale_Z, center_y=center_y, scale_y=scale_y,
        mu=mu, tau=tau, global_preprocessed=preprocessed, etol=etol, mode=mode,
        random_state=random_state,
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
    list_ratio_lasso_missing = output["list_ratio_lasso_missing"]
    list_ratio_error_missing = output["list_ratio_error_missing"]
    list_X1_lasso = output["list_X1_lasso"]
    list_Z2_lasso = output["list_Z2_lasso"]
    list_Z3_lasso = output["list_Z3_lasso"]
    list_y_lasso = output["list_y_lasso"]
    list_X1_error = output["list_X1_error"]
    list_Z2_error = output["list_Z2_error"]
    list_Z3_error = output["list_Z3_error"]
    list_y_error = output["list_y_error"]
    fold_indices = output["fold_indices"]

    for i in range(step):
        lambda_step = lambda_list[i]
        error_old = error

        cv_errors = []
        for k in range(len(fold_indices)):
            n_without_fold = list_y_lasso[k].shape[0]
            n_one_fold = list_y_error[k].shape[0]

            sigma_corrupted_train_additive = list_PSD_lasso_additive[k]
            sigma_corrupted_train_missing = list_PSD_lasso_missing[k]
            sigma_uncorrupted_train = list_sigma_lasso[k] if list_sigma_lasso is not None else None
            ratio_train_missing = list_ratio_lasso_missing[k]
            ratio_test_missing = list_ratio_error_missing[k]

            X1_cv_train = list_X1_lasso[k]
            Z2_cv_train = list_Z2_lasso[k]
            Z3_cv_train = list_Z3_lasso[k]
            y_cv_train = list_y_lasso[k]

            out = _lasso_covariance_block_general(
                n=n_without_fold, p1=p1, p2=p2, p3=p3,
                X1=X1_cv_train, Z2=Z2_cv_train, Z3=Z3_cv_train, y=y_cv_train,
                sigma1=sigma_uncorrupted_train, sigma2=sigma_corrupted_train_additive,
                sigma3=sigma_corrupted_train_missing, lambda_val=lambda_step,
                ratio_matrix=ratio_train_missing,
                beta1_start=beta1_start, beta2_start=beta2_start, beta3_start=beta3_start,
                penalty=penalty, solver=solver,
            )

            beta1_lambda = out.get("coefficients_beta1", np.zeros(p1))
            beta2_lambda = out["coefficients_beta2"]
            beta3_lambda = out["coefficients_beta3"]

            sigma_corrupted_test_additive = list_PSD_error_additive[k]
            sigma_corrupted_test_missing = list_PSD_error_missing[k]

            X1_cv_test = list_X1_error[k]
            Z2_cv_test = list_Z2_error[k]
            Z3_cv_test = list_Z3_error[k]
            y_cv_test = list_y_error[k]

            Z3_tilde_test = Z3_cv_test / np.diag(ratio_test_missing)[np.newaxis, :]

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

        out = _lasso_covariance_block_general(
            n=n, p1=p1, p2=p2, p3=p3, X1=X1, Z2=Z2, Z3=Z3, y=y_proc,
            sigma1=sigma1, sigma2=sigma2, sigma3=sigma3, lambda_val=lambda_step,
            ratio_matrix=ratio_matrix,
            beta1_start=beta1_start, beta2_start=beta2_start, beta3_start=beta3_start,
            penalty=penalty, solver=solver,
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


class GeneralCoCoLasso(BaseEstimator, RegressorMixin):
    """
    三块混合 CoCoLasso 估计器。

    设计矩阵 Z 按列分为三块：
      - 第 0 ~ p1-1 列：无误差特征
      - 第 p1 ~ p1+p2-1 列：含加性误差的特征
      - 第 p1+p2 ~ p-1 列：含缺失数据的特征

    通过交替块坐标下降与 K 折交叉验证选择最优正则化参数。

    参数
    ----------
    alpha : float or None, 默认=None
        正则化强度参数。当 alpha=None 时，通过交叉验证自动选取。
    p1 : int, 默认=0
        无误差特征的列数。
    p2 : int, 默认=0
        含加性误差特征的列数。
    p3 : int, 默认=0
        含缺失数据特征的列数。
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
    random_state : int or None, 默认=None
        交叉验证折分随机种子；None 时沿用全局 numpy 随机状态。

    属性
    ----------
    coef_ : ndarray, shape (p,)
        拟合后的系数向量。
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
    >>> from src import GeneralCoCoLasso
    >>> model = GeneralCoCoLasso(p1=20, p2=100, p3=130, tau=0.75)
    >>> model.fit(Z, y)
    >>> print(model.coef_)
    """

    def __init__(
        self,
        alpha=None,
        p1=0,
        p2=0,
        p3=0,
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
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
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
        拟合三块混合 CoCoLasso 模型。

        参数
        ----------
        Z : ndarray, shape (n, p)
            设计矩阵（p = p1 + p2 + p3）。
        y : ndarray, shape (n,)
            响应向量。

        返回
        -------
        self : GeneralCoCoLasso
            拟合后的估计器实例。
        """
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = Z.shape

        if self.p1 + self.p2 + self.p3 != p:
            raise ValueError(
                f"p1 + p2 + p3 = {self.p1 + self.p2 + self.p3} "
                f"不等于 Z 的列数 {p}"
            )

        result = _blockwise_coordinate_descent_general(
            Z=Z, y=y, n=n, p=p,
            p1=self.p1, p2=self.p2, p3=self.p3,
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
