"""
求解器一致性测试

验证 coordinate_descent 和 sklearn 两种求解器在相同输入下产生一致的结果。
覆盖: CoCoLasso、BDCoCoLasso、GeneralCoCoLasso、
      加性/缺失/乘性噪声、底层 _lasso_covariance vs _lasso_sklearn
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import CoCoLasso, BDCoCoLasso, GeneralCoCoLasso
from src._utils import _lasso_covariance, _lasso_sklearn

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

results = []
all_passed = True

COEF_TOL = 0.1
LAMBDA_TOL_RATIO = 0.3


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


def generate_data(n, p, beta_true, sigma, seed):
    np.random.seed(seed)
    X = np.random.randn(n, p)
    X = X - X.mean(axis=0)
    col_norms = np.sqrt(np.sum(X ** 2, axis=0) / n)
    col_norms[col_norms == 0] = 1.0
    X = X / col_norms[np.newaxis, :]
    y = X @ beta_true + np.random.normal(0, sigma, size=n)
    return X, y


def test_lasso_solvers_direct():
    print("\n--- 底层求解器直接对比 ---")
    np.random.seed(1)
    n, p = 50, 20
    X = np.random.randn(n, p)
    y = np.random.randn(n)
    XX = (1 / n) * X.T @ X
    Xy = (1 / n) * X.T @ y

    for lam in [0.01, 0.05, 0.1, 0.5, 1.0]:
        res_cd = _lasso_covariance(n=n, p=p, lambda_val=lam, XX=XX, Xy=Xy,
                                   beta_start=np.zeros(p), penalty="lasso")
        res_sk = _lasso_sklearn(n=n, p=p, lambda_val=lam, XX=XX, Xy=Xy,
                                beta_start=np.zeros(p))
        diff = np.max(np.abs(res_cd["coefficients"] - res_sk["coefficients"]))
        record(f"direct_lambda={lam:.2f}",
               diff < COEF_TOL,
               f"max_diff={diff:.2e}")


def test_lasso_warm_start():
    print("\n--- 底层求解器 warm_start 对比 ---")
    np.random.seed(2)
    n, p = 50, 20
    X = np.random.randn(n, p)
    y = np.random.randn(n)
    XX = (1 / n) * X.T @ X
    Xy = (1 / n) * X.T @ y

    res_no_warm = _lasso_sklearn(n=n, p=p, lambda_val=0.1, XX=XX, Xy=Xy,
                                 beta_start=None)
    res_warm = _lasso_sklearn(n=n, p=p, lambda_val=0.1, XX=XX, Xy=Xy,
                              beta_start=np.zeros(p))
    diff = np.max(np.abs(res_no_warm["coefficients"] - res_warm["coefficients"]))
    record("warm_start_consistency",
           diff < 1e-6,
           f"max_diff={diff:.2e}")


def test_cocolasso_additive():
    print("\n--- CoCoLasso 加性噪声 CD vs sklearn ---")
    n, p = 50, 20
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=10)
    tau = 0.5
    np.random.seed(110)
    Z = X + np.random.randn(n, p) * tau

    model_cd = CoCoLasso(tau=tau, noise="additive", max_iter=20, cv_folds=3,
                         solver="coordinate_descent")
    model_cd.fit(Z, y)

    model_sk = CoCoLasso(tau=tau, noise="additive", max_iter=20, cv_folds=3,
                         solver="sklearn")
    model_sk.fit(Z, y)

    diff = np.max(np.abs(model_cd.coef_ - model_sk.coef_))
    record("coco_additive_coef",
           diff < COEF_TOL,
           f"max_diff={diff:.6f}")

    lambda_ratio = model_sk.lambda_opt_ / model_cd.lambda_opt_ if model_cd.lambda_opt_ > 0 else 0
    record("coco_additive_lambda",
           abs(lambda_ratio - 1.0) < LAMBDA_TOL_RATIO,
           f"ratio={lambda_ratio:.4f}")


def test_cocolasso_multiplicative():
    print("\n--- CoCoLasso 乘性噪声 CD vs sklearn ---")
    n, p = 50, 15
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=30)
    tau = 0.3
    np.random.seed(130)
    Z = X * np.exp(np.random.normal(0, tau, (n, p)))

    model_cd = CoCoLasso(tau=tau, noise="multiplicative", max_iter=20, cv_folds=3,
                         solver="coordinate_descent")
    model_cd.fit(Z, y)

    model_sk = CoCoLasso(tau=tau, noise="multiplicative", max_iter=20, cv_folds=3,
                         solver="sklearn")
    model_sk.fit(Z, y)

    diff = np.max(np.abs(model_cd.coef_ - model_sk.coef_))
    record("coco_multiplicative_coef",
           diff < COEF_TOL,
           f"max_diff={diff:.6f}")


def test_cocolasso_missing():
    print("\n--- CoCoLasso 缺失数据 CD vs sklearn ---")
    n, p = 50, 15
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=20)
    np.random.seed(120)
    Z = X.copy()
    mask = np.random.rand(n, p) < 0.1
    Z[mask] = np.nan

    model_cd = CoCoLasso(noise="missing", max_iter=20, cv_folds=3,
                         solver="coordinate_descent")
    model_cd.fit(Z, y)

    model_sk = CoCoLasso(noise="missing", max_iter=20, cv_folds=3,
                         solver="sklearn")
    model_sk.fit(Z, y)

    diff = np.max(np.abs(model_cd.coef_ - model_sk.coef_))
    record("coco_missing_coef",
           diff < COEF_TOL,
           f"max_diff={diff:.6f}")


def test_bdcocolasso_additive():
    print("\n--- BDCoCoLasso 加性噪声 CD vs sklearn ---")
    n, p = 50, 20
    p1, p2 = 8, 12
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=40)
    tau = 0.5
    np.random.seed(140)
    Z = X.copy()
    Z[:, p1:] = Z[:, p1:] + np.random.randn(n, p2) * tau

    model_cd = BDCoCoLasso(p1=p1, p2=p2, tau=tau, noise="additive",
                           max_iter=20, cv_folds=3, solver="coordinate_descent")
    model_cd.fit(Z, y)

    model_sk = BDCoCoLasso(p1=p1, p2=p2, tau=tau, noise="additive",
                           max_iter=20, cv_folds=3, solver="sklearn")
    model_sk.fit(Z, y)

    diff = np.max(np.abs(model_cd.coef_ - model_sk.coef_))
    record("BD_additive_coef",
           diff < COEF_TOL,
           f"max_diff={diff:.6f}")


def test_bdcocolasso_missing():
    print("\n--- BDCoCoLasso 缺失数据 CD vs sklearn ---")
    n, p = 50, 15
    p1, p2 = 5, 10
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=50)
    np.random.seed(150)
    Z = X.copy()
    mask_miss = np.random.rand(n, p2) < 0.1
    Z_miss = Z[:, p1:].copy()
    Z_miss[mask_miss] = np.nan
    Z[:, p1:] = Z_miss

    model_cd = BDCoCoLasso(p1=p1, p2=p2, noise="missing",
                           max_iter=20, cv_folds=3, solver="coordinate_descent")
    model_cd.fit(Z, y)

    model_sk = BDCoCoLasso(p1=p1, p2=p2, noise="missing",
                           max_iter=20, cv_folds=3, solver="sklearn")
    model_sk.fit(Z, y)

    diff = np.max(np.abs(model_cd.coef_ - model_sk.coef_))
    record("BD_missing_coef",
           diff < COEF_TOL,
           f"max_diff={diff:.6f}")


def test_generalcocolasso():
    print("\n--- GeneralCoCoLasso CD vs sklearn ---")
    n, p = 50, 20
    p1, p2, p3 = 5, 8, 7
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=60)
    tau = 0.5
    np.random.seed(160)
    Z = X.copy()
    Z[:, p1:p1 + p2] = Z[:, p1:p1 + p2] + np.random.randn(n, p2) * tau
    mask_miss = np.random.rand(n, p3) < 0.1
    Z_miss = Z[:, p1 + p2:].copy()
    Z_miss[mask_miss] = np.nan
    Z[:, p1 + p2:] = Z_miss

    model_cd = GeneralCoCoLasso(p1=p1, p2=p2, p3=p3, tau=tau,
                                max_iter=20, cv_folds=3, solver="coordinate_descent")
    model_cd.fit(Z, y)

    model_sk = GeneralCoCoLasso(p1=p1, p2=p2, p3=p3, tau=tau,
                                max_iter=20, cv_folds=3, solver="sklearn")
    model_sk.fit(Z, y)

    diff = np.max(np.abs(model_cd.coef_ - model_sk.coef_))
    record("General_coef",
           diff < COEF_TOL,
           f"max_diff={diff:.6f}")


def test_timing():
    print("\n--- 性能对比 ---")
    n, p = 80, 40
    beta_true = np.zeros(p)
    beta_true[:5] = [3, 1.5, 0, 0, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=70)
    tau = 0.75
    np.random.seed(170)
    Z = X + np.random.randn(n, p) * tau

    t0 = time.time()
    model_cd = CoCoLasso(tau=tau, noise="additive", max_iter=30, cv_folds=3,
                         solver="coordinate_descent")
    model_cd.fit(Z, y)
    t_cd = time.time() - t0

    t0 = time.time()
    model_sk = CoCoLasso(tau=tau, noise="additive", max_iter=30, cv_folds=3,
                         solver="sklearn")
    model_sk.fit(Z, y)
    t_sk = time.time() - t0

    speedup = t_cd / t_sk if t_sk > 0 else float('inf')
    record("timing - sklearn可运行",
           t_sk > 0 and np.all(np.isfinite(model_sk.coef_)),
           f"CD={t_cd:.2f}s, sklearn={t_sk:.2f}s, speedup={speedup:.1f}x")

    diff = np.max(np.abs(model_cd.coef_ - model_sk.coef_))
    record("timing - 结果一致性",
           diff < COEF_TOL,
           f"max_diff={diff:.6f}")


def main():
    t0 = time.time()
    print("=" * 60)
    print("求解器一致性测试")
    print("=" * 60)

    test_lasso_solvers_direct()
    test_lasso_warm_start()
    test_cocolasso_additive()
    test_cocolasso_multiplicative()
    test_cocolasso_missing()
    test_bdcocolasso_additive()
    test_bdcocolasso_missing()
    test_generalcocolasso()
    test_timing()

    elapsed = time.time() - t0
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = total - passed

    summary = {
        "module": "solver_consistency",
        "total": total,
        "passed": passed,
        "failed": failed,
        "elapsed_sec": round(elapsed, 2),
        "all_passed": all_passed,
        "details": results,
    }

    output_path = os.path.join(RESULTS_DIR, "test_solver_consistency_results.json")
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
