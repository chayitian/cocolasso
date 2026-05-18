"""
BDcocolasso 源代码包入口。
"""

import numpy as np
from typing import Optional, Dict

from .cocolasso import CoCoLasso, _pathwise_coordinate_descent
from .bdcocolasso import BDCoCoLasso, _blockwise_coordinate_descent
from .generalcocolasso import GeneralCoCoLasso, _blockwise_coordinate_descent_general


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
    mu: float = 1.0,
    tau: Optional[float] = None,
    etol: float = 1e-4,
    optTol: float = 1e-5,
    earlyStopping_max: int = 10,
    noise: str = "additive",
    block: bool = True,
    penalty: str = "lasso",
    mode: str = "ADMM",
    solver: str = "coordinate_descent",
    alpha: Optional[float] = None,
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
    solver : str, 'coordinate_descent' 或 'sklearn'
    alpha : float or None, 固定正则化强度；None 时通过交叉验证选择

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
            noise=noise, penalty=penalty, mode=mode, solver=solver, alpha=alpha,
        )
    else:
        return _pathwise_coordinate_descent(
            Z=Z, y=y, n=n, p=p,
            center_Z=center_Z, scale_Z=scale_Z, center_y=center_y, scale_y=scale_y,
            lambda_factor=lambda_factor, step=step, K=K, mu=mu, tau=tau,
            etol=etol, optTol=optTol, earlyStopping_max=earlyStopping_max,
            noise=noise, penalty=penalty, mode=mode, solver=solver, alpha=alpha,
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
    mu: float = 1.0,
    tau: Optional[float] = None,
    etol: float = 1e-4,
    optTol: float = 1e-5,
    earlyStopping_max: int = 10,
    penalty: str = "lasso",
    mode: str = "ADMM",
    solver: str = "coordinate_descent",
    alpha: Optional[float] = None,
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
    solver : str, 'coordinate_descent' 或 'sklearn'
    alpha : float or None, 固定正则化强度；None 时通过交叉验证选择

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
        penalty=penalty, mode=mode, solver=solver, alpha=alpha,
    )


__all__ = [
    "CoCoLasso",
    "BDCoCoLasso",
    "GeneralCoCoLasso",
    "coco",
    "generalcoco",
]
