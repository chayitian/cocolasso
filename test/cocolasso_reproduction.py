"""
CoCoLasso 论文复现实验

复现 Datta & Zou (2017) "Covariance-assisted Lasso for high-dimensional 
measurement error models" 的模拟实验。

同时运行 CoCoLasso 和 NCL 方法进行对比。

实验设置：
- n=100, p=250, β*=(3,1.5,0,0,2,0,...,0), σ=3
- 协方差结构：AR (自回归) 和 CS (复合对称)
- 误差类型：加性测量误差 和 乘性测量误差
- 评价指标：C, IC, SE, PE
- 100次蒙特卡洛重复，报告中位数及bootstrap标准误
"""

import numpy as np
import pandas as pd
import os
import sys
import time
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from cocolasso import coco, cov_autoregressive
from ncl import ncl_method


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
    SE = float(np.sum((beta_true - beta_hat) ** 2))
    diff = beta_true - beta_hat
    PE = float(diff @ Sigma_X @ diff)

    return {"C": C, "IC": IC, "SE": SE, "PE": PE}


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

def run_cocolasso(Z: np.ndarray, y: np.ndarray, n: int, p: int,
                  tau: float, error_type: str) -> np.ndarray:
    result = coco(
        Z=Z, y=y, n=n, p=p,
        step=100, K=K_FOLDS, mu=10, tau=tau,
        etol=1e-4, noise=error_type, block=False,
        penalty="lasso", mode="ADMM"
    )
    beta_proc = result["beta_opt"]
    return unscale_coefficients(beta_proc, result["sd_Z"], result["sd_y"])


def run_ncl(Z: np.ndarray, y: np.ndarray, n: int, p: int,
            tau: float, error_type: str) -> np.ndarray:
    ncl_result = ncl_method(
        Z=Z, y=y, n=n, p=p, tau=tau,
        noise=error_type, K=K_FOLDS, step=100, n_R=100
    )
    return ncl_result["beta"]


def run_single_experiment(error_type: str,
                          tau: float, seed: int, Sigma_X: np.ndarray,
                          methods: list = None) -> dict:
    n, p = N_SAMPLES, N_FEATURES
    X, y = generate_data(n, p, BETA_TRUE, SIGMA_NOISE, Sigma_X, seed)

    if error_type == "additive":
        Z = add_additive_error(X, tau, seed + 10000)
    else:
        Z = add_multiplicative_error(X, tau, seed + 10000)

    results = {}
    if "CoCoLasso" in methods:
        beta_cocolasso = run_cocolasso(Z, y, n, p, tau, error_type)
        results["CoCoLasso"] = compute_metrics(beta_cocolasso, BETA_TRUE, Sigma_X)
    if "NCL" in methods:
        beta_ncl = run_ncl(Z, y, n, p, tau, error_type)
        results["NCL"] = compute_metrics(beta_ncl, BETA_TRUE, Sigma_X)

    return results


# ============================================================
# 主仿真流程
# ============================================================

def run_simulation(n_mc: int = N_MONTE_CARLO,
                   n_bootstrap: int = N_BOOTSTRAP,
                   cov_types: list = None,
                   additive_taus: list = None,
                   multiplicative_taus: list = None,
                   method: str = "full") -> pd.DataFrame:
    """
    主仿真流程。

    参数:
        method: "cocolasso_only", "ncl_only", 或 "full"
    """
    if cov_types is None:
        cov_types = COV_TYPES
    if additive_taus is None:
        additive_taus = ADDITIVE_TAUS
    if multiplicative_taus is None:
        multiplicative_taus = MULTIPLICATIVE_TAUS

    methods = []
    if method == "cocolasso_only":
        methods = ["CoCoLasso"]
    elif method == "ncl_only":
        methods = ["NCL"]
    else:
        methods = ["CoCoLasso", "NCL"]

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

        method_label = " | ".join(methods)
        print(f"\n[{si+1}/{total}] {method_label} | {cov_type} | {error_type} | τ={tau}")
        Sigma_X = generate_covariance(N_FEATURES, cov_type)

        mc_results = {m: {"C": [], "IC": [], "SE": [], "PE": []} for m in methods}

        for mc in range(n_mc):
            seed = mc + 1
            t0 = time.time()
            try:
                metrics = run_single_experiment(
                    error_type, tau, seed, Sigma_X, methods
                )
            except Exception as e:
                print(f"  MC={mc+1} ERROR: {e}")
                continue

            for m in methods:
                for key in mc_results[m]:
                    mc_results[m][key].append(metrics[m][key])

            elapsed = time.time() - t0
            if (mc + 1) % 10 == 0:
                parts = []
                for m in methods:
                    short = "Coco" if m == "CoCoLasso" else "NCL"
                    parts.append(f"{short} C={metrics[m]['C']} IC={metrics[m]['IC']}")
                print(f"  MC={mc+1}/{n_mc}  {'  '.join(parts)}  ({elapsed:.1f}s)")

        if not mc_results[methods[0]]["C"]:
            continue

        for m in methods:
            row = {
                "Method": m,
                "CovType": cov_type,
                "ErrorType": error_type,
                "Tau": tau,
            }

            for key in ["C", "IC", "SE", "PE"]:
                vals = np.array(mc_results[m][key])
                row[f"{key}_median"] = np.median(vals)
                row[f"{key}_se"] = bootstrap_se(vals, n_bootstrap)

            all_results.append(row)

        elapsed_total = time.time() - start_time
        summary_parts = []
        for m in methods:
            short = "Coco" if m == "CoCoLasso" else "NCL"
            c_val = np.median(mc_results[m]["C"])
            ic_val = np.median(mc_results[m]["IC"])
            summary_parts.append(f"{short} C={c_val:.1f} IC={ic_val:.1f}")
        print(f"  => {'  '.join(summary_parts)}  [总耗时: {elapsed_total/60:.1f}min]")

    df = pd.DataFrame(all_results)
    return df


def run_quick_test(method: str = "full") -> pd.DataFrame:
    print("=" * 60)
    print("快速测试模式 (n_mc=3)")
    print("=" * 60)
    return run_simulation(
        n_mc=3,
        n_bootstrap=50,
        cov_types=["AR"],
        additive_taus=[0.75],
        multiplicative_taus=[0.25],
        method=method,
    )


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CoCoLasso 论文复现实验")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "quick"],
                        help="运行模式: full=完整实验, quick=快速测试")
    parser.add_argument("--method", type=str, default="full",
                        choices=["cocolasso_only", "ncl_only", "full"],
                        help="方法选择: cocolasso_only=仅CoCoLasso, ncl_only=仅NCL, full=两者对比")
    parser.add_argument("--n_mc", type=int, default=N_MONTE_CARLO,
                        help="蒙特卡洛重复次数")
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP,
                        help="Bootstrap次数")
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(__file__))

    if args.mode == "quick":
        df = run_quick_test(method=args.method)
    else:
        df = run_simulation(
            n_mc=args.n_mc, n_bootstrap=args.n_bootstrap,
            method=args.method,
        )

    if args.method == "cocolasso_only":
        output_name = "cocolasso_only_results.csv"
    elif args.method == "ncl_only":
        output_name = "ncl_only_results.csv"
    else:
        output_name = "cocolasso_vs_ncl_results.csv"

    output_path = os.path.join(output_dir, output_name)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存至: {output_path}")
    print(df.to_string(index=False))
