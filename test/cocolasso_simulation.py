"""
CoCoLasso 模拟实验

基于 Datta & Zou (2017) 的模拟设置。

实验设置：
- n=100, p=250, β*=(3,1.5,0,0,2,0,...,0), σ=3
- 协方差结构：AR (自回归) 和 CS (复合对称)
- 误差类型：加性测量误差 和 乘性测量误差
- 评价指标：C, IC, PE, SE
- 100次蒙特卡洛重复，报告中位数及bootstrap标准误
"""

import numpy as np
import pandas as pd
import os
import sys
import time
sys.stdout.reconfigure(line_buffering=True)

# 将 src 目录加入到系统路径中，以便能够正确导入 cocolasso
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import coco


def cov_autoregressive(p: int, rho: float = 0.5) -> np.ndarray:
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = rho ** abs(i - j)
    return cov


# ============================================================
# 常量定义
# ============================================================

N_MONTE_CARLO = 100
N_BOOTSTRAP = 500
N_SAMPLES = 100
N_FEATURES = 250
SIGMA_NOISE = 3.0
BETA_TRUE = np.zeros(N_FEATURES)
BETA_TRUE[:5] = [3, 1.5, 0, 0, 2]
K_FOLDS = 5
NONZERO_IDX = np.where(BETA_TRUE != 0)[0]
ZERO_IDX = np.where(BETA_TRUE == 0)[0]

ADDITIVE_TAUS = [0.75, 1.0, 1.25]
MULTIPLICATIVE_TAUS = [0.25, 0.5, 0.75]

COV_TYPES = ["AR", "CS"]

ZERO_THRESHOLD = 1e-6


# ============================================================
# 数据生成
# ============================================================

def generate_cs_covariance(p: int, rho: float = 0.5) -> np.ndarray:
    Sigma = np.full((p, p), rho)
    np.fill_diagonal(Sigma, rho + 0.5)
    return Sigma


def generate_covariance(p: int, cov_type: str = "AR", rho: float = 0.5) -> np.ndarray:
    if cov_type == "AR":
        return cov_autoregressive(p, rho)
    elif cov_type == "CS":
        return generate_cs_covariance(p, rho)
    else:
        raise ValueError(f"Unknown cov_type: {cov_type}")


def generate_data(n: int, p: int, beta: np.ndarray, sigma: float,
                  Sigma_X: np.ndarray, seed: int) -> tuple:
    np.random.seed(seed)
    X = np.random.multivariate_normal(np.zeros(p), Sigma_X, size=n)
    X = X - X.mean(axis=0)
    col_norms = np.sqrt(np.sum(X ** 2, axis=0) / n)
    col_norms[col_norms == 0] = 1.0
    X = X / col_norms[np.newaxis, :]
    y = X @ beta + np.random.normal(0, sigma, size=n)
    return X, y


def add_additive_error(X: np.ndarray, tau: float, seed: int) -> np.ndarray:
    np.random.seed(seed)
    A = np.random.normal(0, tau, size=X.shape)
    return X + A


def add_multiplicative_error(X: np.ndarray, tau: float, seed: int) -> np.ndarray:
    np.random.seed(seed)
    log_M = np.random.normal(0, tau, size=X.shape)
    M = np.exp(log_M)
    return X * M


def unscale_coefficients(beta_proc: np.ndarray,
                          sd_Z: np.ndarray, sd_y: float) -> np.ndarray:
    sd_Z_safe = np.where(sd_Z != 0, sd_Z, 1.0)
    return beta_proc * sd_y / sd_Z_safe


# ============================================================
# 评价指标
# ============================================================

def compute_metrics(beta_hat: np.ndarray, beta_true: np.ndarray,
                    Sigma_X: np.ndarray) -> dict:
    beta_hat_binary = (np.abs(beta_hat) > ZERO_THRESHOLD).astype(int)

    C = int(np.sum(beta_hat_binary[NONZERO_IDX] == 1))
    IC = int(np.sum(beta_hat_binary[ZERO_IDX] == 1))
    diff = beta_true - beta_hat
    PE = float(diff @ Sigma_X @ diff)
    SE = float(np.sum(diff ** 2))

    return {"C": C, "IC": IC, "PE": PE, "SE": SE}


def bootstrap_se(values: np.ndarray, n_bootstrap: int = 500) -> float:
    n = len(values)
    boot_medians = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        boot_medians[b] = np.median(sample)
    return float(np.std(boot_medians))


# ============================================================
# 单次实验运行
# ============================================================

def run_single_experiment(error_type: str,
                          tau: float, seed: int, Sigma_X: np.ndarray) -> dict:
    n, p = N_SAMPLES, N_FEATURES
    X, y = generate_data(n, p, BETA_TRUE, SIGMA_NOISE, Sigma_X, seed)

    if error_type == "additive":
        Z = add_additive_error(X, tau, seed + 10000)
    else:
        Z = add_multiplicative_error(X, tau, seed + 10000)

    result = coco(
        Z=Z, y=y, n=n, p=p,
        step=100, K=K_FOLDS, mu=10, tau=tau,
        etol=1e-4, noise=error_type, block=False,
        penalty="lasso", mode="ADMM"
    )
    beta_proc = result["beta_opt"]
    beta_hat = unscale_coefficients(
        beta_proc, result["sd_Z"], result["sd_y"])

    metrics = compute_metrics(beta_hat, BETA_TRUE, Sigma_X)
    return metrics


# ============================================================
# 主仿真流程
# ============================================================

def run_simulation(n_mc: int = N_MONTE_CARLO,
                   n_bootstrap: int = N_BOOTSTRAP,
                   cov_types: list = None,
                   additive_taus: list = None,
                   multiplicative_taus: list = None) -> pd.DataFrame:
    if cov_types is None:
        cov_types = COV_TYPES
    if additive_taus is None:
        additive_taus = ADDITIVE_TAUS
    if multiplicative_taus is None:
        multiplicative_taus = MULTIPLICATIVE_TAUS

    scenarios = []
    for cov_type in cov_types:
        for error_type, taus in [("additive", additive_taus),
                                  ("multiplicative", multiplicative_taus)]:
            for tau in taus:
                scenarios.append({
                    "cov_type": cov_type,
                    "error_type": error_type,
                    "tau": tau,
                })

    all_results = []
    total = len(scenarios)
    start_time = time.time()

    for si, scenario in enumerate(scenarios):
        cov_type = scenario["cov_type"]
        error_type = scenario["error_type"]
        tau = scenario["tau"]

        print(f"\n[{si+1}/{total}] CoCoLasso | {cov_type} | {error_type} | τ={tau}")
        Sigma_X = generate_covariance(N_FEATURES, cov_type)

        mc_results = {"C": [], "IC": [], "PE": [], "SE": []}

        for mc in range(n_mc):
            seed = mc + 1
            t0 = time.time()
            try:
                metrics = run_single_experiment(
                    error_type, tau, seed, Sigma_X
                )
            except Exception as e:
                print(f"  MC={mc+1} ERROR: {e}")
                continue

            for key in mc_results:
                mc_results[key].append(metrics[key])

            elapsed = time.time() - t0
            if (mc + 1) % 10 == 0:
                print(f"  MC={mc+1}/{n_mc}  C={metrics['C']}  IC={metrics['IC']}  "
                      f"PE={metrics['PE']:.4f}  SE={metrics['SE']:.4f}  "
                      f"({elapsed:.1f}s)")

        if not mc_results["C"]:
            continue

        row = {
            "Method": "CoCoLasso",
            "CovType": cov_type,
            "ErrorType": error_type,
            "Tau": tau,
        }

        for key in ["C", "IC", "PE", "SE"]:
            vals = np.array(mc_results[key])
            row[f"{key}_median"] = np.median(vals)
            row[f"{key}_se"] = bootstrap_se(vals, n_bootstrap)

        all_results.append(row)

        elapsed_total = time.time() - start_time
        print(f"  => C_median={row['C_median']:.1f}  IC_median={row['IC_median']:.1f}  "
              f"PE_median={row['PE_median']:.4f}  SE_median={row['SE_median']:.4f}  "
              f"[总耗时: {elapsed_total/60:.1f}min]")

    df = pd.DataFrame(all_results)
    return df


def run_quick_test() -> pd.DataFrame:
    print("=" * 60)
    print("快速测试模式 (n_mc=3)")
    print("=" * 60)
    return run_simulation(
        n_mc=3,
        n_bootstrap=50,
        cov_types=["AR"],
        additive_taus=[0.75],
        multiplicative_taus=[0.25],
    )


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CoCoLasso 模拟实验")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "quick"],
                        help="运行模式: full=完整实验, quick=快速测试")
    parser.add_argument("--n_mc", type=int, default=N_MONTE_CARLO,
                        help="蒙特卡洛重复次数")
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP,
                        help="Bootstrap次数")
    args = parser.parse_args()

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

    if args.mode == "quick":
        df = run_quick_test()
    else:
        df = run_simulation(
            n_mc=args.n_mc, n_bootstrap=args.n_bootstrap,
        )

    output_path = os.path.join(output_dir, "simulation_results.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存至: {output_path}")
    print(df.to_string(index=False))
