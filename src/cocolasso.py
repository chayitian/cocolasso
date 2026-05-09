"""
CoCoLasso 的 sklearn 风格估计器封装。

提供三个与 scikit-learn 兼容的估计器类：
- CoCoLasso：标准 CoCoLasso（路径坐标下降 + 交叉验证）
- BDCoCoLasso：二块下降 CoCoLasso（BD-CoCoLasso）
- GeneralCoCoLasso：三块混合 CoCoLasso（加性误差 + 缺失数据）

调用方式与 sklearn.linear_model.Lasso 一致：
    model = CoCoLasso(alpha=1.0, noise="additive", tau=0.75)
    model.fit(Z, y)
    y_pred = model.predict(Z_new)
    score = model.score(Z_test, y_test)
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from ._core import (
    _pathwise_coordinate_descent,
    _blockwise_coordinate_descent,
    _blockwise_coordinate_descent_general,
)


class CoCoLasso(BaseEstimator, RegressorMixin):
    """
    CoCoLasso（凸校正 Lasso）估计器。

    针对协变量中存在测量误差（加性误差、缺失数据、乘性误差）的
    高维误差变量回归问题，使用路径坐标下降与 K 折交叉验证
    选择最优正则化参数。

    参数
    ----------
    alpha : float, 默认=1.0
        正则化强度参数。当 alpha=None 时，通过交叉验证自动选取。
    noise : str, 默认="additive"
        噪声类型，可选 "additive"（加性）、"missing"（缺失）、
        "multiplicative"（乘性）。
    tau : float or None, 默认=None
        加性误差的标准差（noise="additive" 或 "multiplicative" 时必填）。
    penalty : str, 默认="lasso"
        惩罚类型，可选 "lasso" 或 "SCAD"。
    mode : str, 默认="ADMM"
        PSD 投影模式，可选 "ADMM" 或 "HM"。
    mu : float, 默认=10
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
    >>> from cocolasso import CoCoLasso
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
        mu=10,
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
        """
        使用拟合模型进行预测。

        参数
        ----------
        Z : ndarray, shape (n_samples, p)
            设计矩阵。

        返回
        -------
        y_pred : ndarray, shape (n_samples,)
            预测值。
        """
        Z = np.asarray(Z, dtype=float)
        return Z @ self.coef_ + self.intercept_

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
    mu : float, 默认=10
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
    >>> from cocolasso import BDCoCoLasso
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
        mu=10,
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
        """
        使用拟合模型进行预测。

        参数
        ----------
        Z : ndarray, shape (n_samples, p)
            设计矩阵。

        返回
        -------
        y_pred : ndarray, shape (n_samples,)
            预测值。
        """
        Z = np.asarray(Z, dtype=float)
        return Z @ self.coef_ + self.intercept_

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
    mu : float, 默认=10
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
    >>> from cocolasso import GeneralCoCoLasso
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
        mu=10,
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
        """
        使用拟合模型进行预测。

        参数
        ----------
        Z : ndarray, shape (n_samples, p)
            设计矩阵。

        返回
        -------
        y_pred : ndarray, shape (n_samples,)
            预测值。
        """
        Z = np.asarray(Z, dtype=float)
        return Z @ self.coef_ + self.intercept_

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
