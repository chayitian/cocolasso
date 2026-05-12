"""
_utils 模块单元测试

覆盖: _l1_proj, _admm_proj, _hm_proj, _scad_weight,
      _lasso_covariance, _lasso_sklearn, _compute_ratio_matrix,
      _log_space, _corrected_covariance_multiplicative, _preprocess_data
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src._utils import (
    _l1_proj,
    _admm_proj,
    _hm_proj,
    _scad_weight,
    _lasso_covariance,
    _lasso_sklearn,
    _compute_ratio_matrix,
    _log_space,
    _corrected_covariance_multiplicative,
    _preprocess_data,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

results = []
all_passed = True


def record(test_name, passed, detail=""):
    global all_passed
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    msg = f"[{status}] {test_name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append({"test": test_name, "status": status, "detail": detail})


def test_l1_proj():
    v = np.array([3.0, -1.0, 2.0])
    w = _l1_proj(v, b=2.0)
    record("l1_proj - L1范数约束",
           np.sum(np.abs(w)) <= 2.0 + 1e-10,
           f"L1={np.sum(np.abs(w)):.6f} <= 2.0")

    w_zero = _l1_proj(np.zeros(5), b=1.0)
    record("l1_proj - 零向量输入",
           np.allclose(w_zero, 0.0),
           f"output={w_zero}")

    v_large = np.array([10.0, -10.0])
    w_large = _l1_proj(v_large, b=1.0)
    record("l1_proj - 大向量投影",
           np.sum(np.abs(w_large)) <= 1.0 + 1e-10,
           f"L1={np.sum(np.abs(w_large)):.6f}")


def test_admm_proj():
    np.random.seed(1)
    p = 10
    A = np.random.randn(p, p)
    mat = A.T @ A + 0.1 * np.eye(p)

    result = _admm_proj(mat, mu=1.0, etol=1e-4)
    R = result["mat"]

    eigvals = np.linalg.eigvalsh(R)
    record("admm_proj - 输出PSD",
           np.all(eigvals >= -1e-8),
           f"min_eigval={np.min(eigvals):.2e}")

    record("admm_proj - 对称性",
           np.allclose(R, R.T, atol=1e-10),
           f"max_asym={np.max(np.abs(R - R.T)):.2e}")

    mat_psd = np.eye(5) * 3.0
    R_psd = _admm_proj(mat_psd, mu=1.0, etol=1e-4)["mat"]
    record("admm_proj - 已PSD矩阵不变",
           np.allclose(R_psd, mat_psd, atol=0.1),
           f"max_diff={np.max(np.abs(R_psd - mat_psd)):.2e}")

    mat_neg = -np.eye(5)
    R_neg = _admm_proj(mat_neg, mu=1.0, etol=1e-4)["mat"]
    eig_neg = np.linalg.eigvalsh(R_neg)
    record("admm_proj - 负定矩阵投影到PSD",
           np.all(eig_neg >= -1e-8),
           f"min_eigval={np.min(eig_neg):.2e}")


def test_hm_proj():
    np.random.seed(2)
    p = 8
    A = np.random.randn(p, p)
    mat = A.T @ A + 0.1 * np.eye(p)

    R = _hm_proj(mat, mu=1.0, tolerance=1e-4)
    eigvals = np.linalg.eigvalsh(R)
    record("hm_proj - 输出PSD",
           np.all(eigvals >= -1e-8),
           f"min_eigval={np.min(eigvals):.2e}")

    record("hm_proj - 对称性",
           np.allclose(R, R.T, atol=1e-10),
           f"max_asym={np.max(np.abs(R - R.T)):.2e}")

    R_ratio = _hm_proj(mat, R=np.eye(p), mu=1.0, tolerance=1e-4)
    eig_r = np.linalg.eigvalsh(R_ratio)
    record("hm_proj - 带R矩阵输出PSD",
           np.all(eig_r >= -1e-8),
           f"min_eigval={np.min(eig_r):.2e}")


def test_scad_weight():
    lam = 1.0
    a = 3.7

    w1 = _scad_weight(0.5, lam, a)
    record("scad_weight - |beta|<=lambda",
           abs(w1 - 1.0) < 1e-10,
           f"w={w1:.6f}, expected=1.0")

    w2 = _scad_weight(2.0, lam, a)
    expected = (a * lam - 2.0) / (lam * (a - 1))
    record("scad_weight - lambda<|beta|<=a*lambda",
           abs(w2 - expected) < 1e-10,
           f"w={w2:.6f}, expected={expected:.6f}")

    w3 = _scad_weight(5.0, lam, a)
    record("scad_weight - |beta|>a*lambda",
           abs(w3 - 0.0) < 1e-10,
           f"w={w3:.6f}, expected=0.0")


def test_lasso_covariance():
    np.random.seed(3)
    n, p = 50, 10
    X = np.random.randn(n, p)
    y = np.random.randn(n)
    XX = (1 / n) * X.T @ X
    Xy = (1 / n) * X.T @ y

    beta_start = np.zeros(p)
    res = _lasso_covariance(n=n, p=p, lambda_val=0.1, XX=XX, Xy=Xy,
                            beta_start=beta_start, penalty="lasso")
    record("lasso_covariance - 返回结构",
           "coefficients" in res and "num_it" in res,
           f"keys={list(res.keys())}")

    record("lasso_covariance - 系数维度",
           res["coefficients"].shape == (p,),
           f"shape={res['coefficients'].shape}")

    lam_large = 100.0
    res_large = _lasso_covariance(n=n, p=p, lambda_val=lam_large, XX=XX, Xy=Xy,
                                  beta_start=np.zeros(p), penalty="lasso")
    record("lasso_covariance - 大lambda全零",
           np.allclose(res_large["coefficients"], 0.0, atol=1e-4),
           f"max_coef={np.max(np.abs(res_large['coefficients'])):.2e}")

    res_scad = _lasso_covariance(n=n, p=p, lambda_val=0.1, XX=XX, Xy=Xy,
                                 beta_start=np.zeros(p), penalty="SCAD")
    record("lasso_covariance - SCAD惩罚",
           res_scad["coefficients"].shape == (p,),
           f"nnz={np.sum(np.abs(res_scad['coefficients']) > 1e-6)}")


def test_lasso_sklearn():
    np.random.seed(4)
    n, p = 50, 10
    X = np.random.randn(n, p)
    y = np.random.randn(n)
    XX = (1 / n) * X.T @ X
    Xy = (1 / n) * X.T @ y

    res = _lasso_sklearn(n=n, p=p, lambda_val=0.1, XX=XX, Xy=Xy)
    record("lasso_sklearn - 返回结构",
           "coefficients" in res and "num_it" in res,
           f"keys={list(res.keys())}")

    record("lasso_sklearn - 系数维度",
           res["coefficients"].shape == (p,),
           f"shape={res['coefficients'].shape}")

    lam_large = 100.0
    res_large = _lasso_sklearn(n=n, p=p, lambda_val=lam_large, XX=XX, Xy=Xy)
    record("lasso_sklearn - 大lambda全零",
           np.allclose(res_large["coefficients"], 0.0, atol=0.05),
           f"max_coef={np.max(np.abs(res_large['coefficients'])):.2e}")

    res_warm = _lasso_sklearn(n=n, p=p, lambda_val=0.1, XX=XX, Xy=Xy,
                              beta_start=np.zeros(p))
    record("lasso_sklearn - warm_start",
           res_warm["coefficients"].shape == (p,),
           f"nnz={np.sum(np.abs(res_warm['coefficients']) > 1e-6)}")


def test_solver_consistency():
    np.random.seed(5)
    n, p = 50, 10
    X = np.random.randn(n, p)
    y = np.random.randn(n)
    XX = (1 / n) * X.T @ X
    Xy = (1 / n) * X.T @ y

    diffs = []
    for lam in [0.01, 0.05, 0.1, 0.5]:
        res_cd = _lasso_covariance(n=n, p=p, lambda_val=lam, XX=XX, Xy=Xy,
                                   beta_start=np.zeros(p), penalty="lasso")
        res_sk = _lasso_sklearn(n=n, p=p, lambda_val=lam, XX=XX, Xy=Xy,
                                beta_start=np.zeros(p))
        diff = np.max(np.abs(res_cd["coefficients"] - res_sk["coefficients"]))
        diffs.append(diff)

    max_diff = max(diffs)
    record("solver_consistency - CD vs sklearn",
           max_diff < 0.05,
           f"max_diff={max_diff:.2e}, diffs={[f'{d:.2e}' for d in diffs]}")


def test_compute_ratio_matrix():
    np.random.seed(6)
    n, p = 30, 5
    Z = np.random.randn(n, p)
    Z[0, 0] = np.nan
    Z[1, 1] = np.nan
    Z[2, 0] = np.nan

    R = _compute_ratio_matrix(Z, p)
    record("ratio_matrix - 形状",
           R.shape == (p, p),
           f"shape={R.shape}")

    record("ratio_matrix - 对称性",
           np.allclose(R, R.T),
           f"max_asym={np.max(np.abs(R - R.T)):.2e}")

    record("ratio_matrix - 对角线<=1",
           np.all(np.diag(R) <= 1.0 + 1e-10),
           f"max_diag={np.max(np.diag(R)):.4f}")

    record("ratio_matrix - 非负",
           np.all(R >= 0),
           f"min_val={np.min(R):.4f}")

    R_full = _compute_ratio_matrix(np.random.randn(n, p), p)
    record("ratio_matrix - 无缺失时全为1",
           np.allclose(R_full, 1.0, atol=1e-10),
           f"max_diff_from_1={np.max(np.abs(R_full - 1.0)):.2e}")


def test_log_space():
    seq = _log_space(1.0, 0.001, 100)
    record("log_space - 长度",
           len(seq) == 100,
           f"len={len(seq)}")

    record("log_space - 起止值",
           abs(seq[0] - 1.0) < 1e-10 and abs(seq[-1] - 0.001) < 1e-10,
           f"start={seq[0]:.6f}, end={seq[-1]:.6f}")

    record("log_space - 单调递减",
           np.all(np.diff(seq) < 0),
           f"first_diff={seq[0]-seq[1]:.6f}")


def test_corrected_covariance_multiplicative():
    np.random.seed(7)
    n, p = 100, 5
    tau = 0.5
    X = np.random.randn(n, p)
    log_M = np.random.normal(0, tau, (n, p))
    Z = X * np.exp(log_M)
    y = np.random.randn(n)

    Gamma, rho = _corrected_covariance_multiplicative(Z, y, n, p, tau)
    record("multiplicative_corr - Gamma形状",
           Gamma.shape == (p, p),
           f"shape={Gamma.shape}")

    record("multiplicative_corr - rho形状",
           rho.shape == (p,),
           f"shape={rho.shape}")

    record("multiplicative_corr - Gamma对称",
           np.allclose(Gamma, Gamma.T),
           f"max_asym={np.max(np.abs(Gamma - Gamma.T)):.2e}")

    exp_tau2 = np.exp(tau ** 2)
    exp_2tau2 = np.exp(2 * tau ** 2)
    Sigma_raw = (1 / n) * Z.T @ Z
    expected_offdiag = Sigma_raw / exp_tau2
    expected_diag = np.diag(Sigma_raw) / exp_2tau2
    expected_Gamma = expected_offdiag.copy()
    np.fill_diagonal(expected_Gamma, expected_diag)
    record("multiplicative_corr - 校正公式正确",
           np.allclose(Gamma, expected_Gamma, atol=1e-10),
           f"max_diff={np.max(np.abs(Gamma - expected_Gamma)):.2e}")


def test_preprocess_data():
    np.random.seed(8)
    n, p = 50, 10
    Z = np.random.randn(n, p)
    y = np.random.randn(n)

    res = _preprocess_data(Z, y, n, p, center_Z=True, scale_Z=True,
                           center_y=True, scale_y=True, noise="additive")
    record("preprocess - 加性噪声返回键",
           all(k in res for k in ["Z", "y", "mean_Z", "sd_Z", "mean_y", "sd_y", "ratio_matrix"]),
           f"keys={list(res.keys())}")

    record("preprocess - Z中心化后均值近似零",
           np.allclose(np.mean(res["Z"], axis=0), 0, atol=1e-10),
           f"max_mean={np.max(np.abs(np.mean(res['Z'], axis=0))):.2e}")

    record("preprocess - y中心化后均值近似零",
           abs(np.mean(res["y"])) < 1e-10,
           f"mean_y={np.mean(res['y']):.2e}")

    Z_miss = np.random.randn(n, p)
    Z_miss[0, 0] = np.nan
    Z_miss[1, 1] = np.nan
    res_miss = _preprocess_data(Z_miss, y, n, p, center_Z=True, scale_Z=True,
                                center_y=True, scale_y=True, noise="missing")
    record("preprocess - 缺失数据ratio_matrix非None",
           res_miss["ratio_matrix"] is not None,
           f"ratio_matrix shape={res_miss['ratio_matrix'].shape if res_miss['ratio_matrix'] is not None else None}")

    record("preprocess - 缺失数据Z无NaN",
           not np.any(np.isnan(res_miss["Z"])),
           f"has_nan={np.any(np.isnan(res_miss['Z']))}")


def main():
    t0 = time.time()
    print("=" * 60)
    print("_utils 模块单元测试")
    print("=" * 60)

    test_l1_proj()
    test_admm_proj()
    test_hm_proj()
    test_scad_weight()
    test_lasso_covariance()
    test_lasso_sklearn()
    test_solver_consistency()
    test_compute_ratio_matrix()
    test_log_space()
    test_corrected_covariance_multiplicative()
    test_preprocess_data()

    elapsed = time.time() - t0
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = total - passed

    summary = {
        "module": "_utils",
        "total": total,
        "passed": passed,
        "failed": failed,
        "elapsed_sec": round(elapsed, 2),
        "all_passed": all_passed,
        "details": results,
    }

    output_path = os.path.join(RESULTS_DIR, "test_utils_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"总计: {total}, 通过: {passed}, 失败: {failed}, 耗时: {elapsed:.2f}s")
    print(f"结果已保存: {output_path}")
    print(f"{'=' * 60}")

    return all_passed


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
