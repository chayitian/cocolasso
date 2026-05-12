"""
CoCoLasso 集成测试

覆盖: 加性噪声、缺失数据、乘性噪声、sklearn求解器、
      sklearn接口兼容性、预测与评分、CV结果完整性
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import CoCoLasso

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


def generate_data(n, p, beta_true, sigma, seed):
    np.random.seed(seed)
    X = np.random.randn(n, p)
    X = X - X.mean(axis=0)
    col_norms = np.sqrt(np.sum(X ** 2, axis=0) / n)
    col_norms[col_norms == 0] = 1.0
    X = X / col_norms[np.newaxis, :]
    y = X @ beta_true + np.random.normal(0, sigma, size=n)
    return X, y


def test_additive_noise():
    print("\n--- 加性噪声 ---")
    n, p = 50, 20
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=10)
    tau = 0.5
    np.random.seed(110)
    Z = X + np.random.randn(n, p) * tau

    model = CoCoLasso(tau=tau, noise="additive", max_iter=20, cv_folds=3,
                      solver="coordinate_descent")
    model.fit(Z, y)

    record("additive - fit成功",
           hasattr(model, 'coef_'),
           f"coef_shape={model.coef_.shape}")

    record("additive - coef_维度",
           model.coef_.shape == (p,),
           f"shape={model.coef_.shape}")

    record("additive - intercept_存在",
           isinstance(model.intercept_, float),
           f"intercept={model.intercept_:.4f}")

    record("additive - lambda_opt_ > 0",
           model.lambda_opt_ > 0,
           f"lambda_opt={model.lambda_opt_:.6f}")

    record("additive - cv_results_完整",
           all(k in model.cv_results_ for k in ["lambda", "error", "error_inf", "error_sup", "error_sd"]),
           f"keys={list(model.cv_results_.keys())}")

    record("additive - coef_path_完整",
           "lambda" in model.coef_path_ and "beta" in model.coef_path_,
           f"path_len={len(model.coef_path_['lambda'])}")

    nonzero = np.sum(np.abs(model.coef_) > 1e-6)
    record("additive - 变量选择合理",
           nonzero >= 3,
           f"nonzero={nonzero}, expected>=3")

    y_pred = model.predict(Z)
    record("additive - predict维度",
           y_pred.shape == (n,),
           f"shape={y_pred.shape}")

    r2 = model.score(Z, y)
    record("additive - score(R²)有限",
           np.isfinite(r2),
           f"R²={r2:.4f}")


def test_missing_data():
    print("\n--- 缺失数据 ---")
    n, p = 50, 15
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=20)
    np.random.seed(120)
    Z = X.copy()
    mask = np.random.rand(n, p) < 0.1
    Z[mask] = np.nan

    model = CoCoLasso(noise="missing", max_iter=20, cv_folds=3,
                      solver="coordinate_descent")
    model.fit(Z, y)

    record("missing - fit成功",
           hasattr(model, 'coef_'),
           f"coef_shape={model.coef_.shape}")

    record("missing - coef_无NaN",
           not np.any(np.isnan(model.coef_)),
           f"has_nan={np.any(np.isnan(model.coef_))}")

    record("missing - lambda_opt_ > 0",
           model.lambda_opt_ > 0,
           f"lambda_opt={model.lambda_opt_:.6f}")

    y_pred = model.predict(X)
    record("missing - predict无NaN",
           not np.any(np.isnan(y_pred)),
           f"has_nan={np.any(np.isnan(y_pred))}")


def test_multiplicative_noise():
    print("\n--- 乘性噪声 ---")
    n, p = 50, 15
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=30)
    tau = 0.3
    np.random.seed(130)
    log_M = np.random.normal(0, tau, (n, p))
    Z = X * np.exp(log_M)

    model = CoCoLasso(tau=tau, noise="multiplicative", max_iter=20, cv_folds=3,
                      solver="coordinate_descent")
    model.fit(Z, y)

    record("multiplicative - fit成功",
           hasattr(model, 'coef_'),
           f"coef_shape={model.coef_.shape}")

    record("multiplicative - coef_无NaN/Inf",
           np.all(np.isfinite(model.coef_)),
           f"has_nan={np.any(np.isnan(model.coef_))}, has_inf={np.any(np.isinf(model.coef_))}")

    record("multiplicative - lambda_opt_ > 0",
           model.lambda_opt_ > 0,
           f"lambda_opt={model.lambda_opt_:.6f}")


def test_sklearn_solver():
    print("\n--- sklearn 求解器 ---")
    n, p = 50, 20
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=40)
    tau = 0.5
    np.random.seed(140)
    Z = X + np.random.randn(n, p) * tau

    model = CoCoLasso(tau=tau, noise="additive", max_iter=20, cv_folds=3,
                      solver="sklearn")
    model.fit(Z, y)

    record("sklearn_solver - fit成功",
           hasattr(model, 'coef_'),
           f"coef_shape={model.coef_.shape}")

    record("sklearn_solver - coef_无NaN/Inf",
           np.all(np.isfinite(model.coef_)),
           f"finite={np.all(np.isfinite(model.coef_))}")

    record("sklearn_solver - lambda_opt_ > 0",
           model.lambda_opt_ > 0,
           f"lambda_opt={model.lambda_opt_:.6f}")

    y_pred = model.predict(Z)
    record("sklearn_solver - predict维度",
           y_pred.shape == (n,),
           f"shape={y_pred.shape}")


def test_sklearn_api_compatibility():
    print("\n--- sklearn API 兼容性 ---")
    n, p = 50, 20
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=50)
    tau = 0.5
    np.random.seed(150)
    Z = X + np.random.randn(n, p) * tau

    model = CoCoLasso(tau=tau, noise="additive", max_iter=20, cv_folds=3)

    from sklearn.base import BaseEstimator, RegressorMixin
    record("sklearn_api - 继承BaseEstimator",
           isinstance(model, BaseEstimator),
           f"type={type(model).__mro__}")

    record("sklearn_api - 继承RegressorMixin",
           isinstance(model, RegressorMixin),
           "")

    model.fit(Z, y)
    record("sklearn_api - get_params可用",
           "tau" in model.get_params(),
           f"tau={model.get_params()['tau']}")

    model.set_params(max_iter=50)
    record("sklearn_api - set_params可用",
           model.max_iter == 50,
           f"max_iter={model.max_iter}")


def test_admm_vs_hm():
    print("\n--- ADMM vs HM 投影模式 ---")
    n, p = 40, 10
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=60)
    tau = 0.5
    np.random.seed(160)
    Z = X + np.random.randn(n, p) * tau

    model_admm = CoCoLasso(tau=tau, noise="additive", max_iter=15, cv_folds=3,
                           mode="ADMM", solver="coordinate_descent")
    model_admm.fit(Z, y)

    model_hm = CoCoLasso(tau=tau, noise="additive", max_iter=15, cv_folds=3,
                         mode="HM", solver="coordinate_descent")
    model_hm.fit(Z, y)

    record("ADMM_vs_HM - 两种模式均可运行",
           hasattr(model_admm, 'coef_') and hasattr(model_hm, 'coef_'),
           "")

    diff = np.max(np.abs(model_admm.coef_ - model_hm.coef_))
    record("ADMM_vs_HM - 系数差异合理",
           diff < 1.0,
           f"max_diff={diff:.4f}")


def test_early_stopping():
    print("\n--- 早停机制 ---")
    n, p = 50, 20
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=70)
    tau = 0.5
    np.random.seed(170)
    Z = X + np.random.randn(n, p) * tau

    model = CoCoLasso(tau=tau, noise="additive", max_iter=100, cv_folds=3,
                      early_stopping_max=5, solver="coordinate_descent")
    model.fit(Z, y)

    record("early_stopping - n_iter_ <= max_iter",
           model.n_iter_ <= 100,
           f"n_iter={model.n_iter_}")

    record("early_stopping - n_iter_ > 0",
           model.n_iter_ > 0,
           f"n_iter={model.n_iter_}")


def test_coef_sd():
    print("\n--- 1-std 准则 ---")
    n, p = 50, 20
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=80)
    tau = 0.5
    np.random.seed(180)
    Z = X + np.random.randn(n, p) * tau

    model = CoCoLasso(tau=tau, noise="additive", max_iter=20, cv_folds=3,
                      solver="coordinate_descent")
    model.fit(Z, y)

    record("coef_sd - 存在",
           hasattr(model, 'coef_sd_'),
           "")

    record("coef_sd - 维度",
           model.coef_sd_.shape == (p,),
           f"shape={model.coef_sd_.shape}")

    record("coef_sd - lambda_sd_ >= lambda_opt_",
           model.lambda_sd_ >= model.lambda_opt_ - 1e-10,
           f"lambda_sd={model.lambda_sd_:.6f} >= lambda_opt={model.lambda_opt_:.6f}")


def main():
    t0 = time.time()
    print("=" * 60)
    print("CoCoLasso 集成测试")
    print("=" * 60)

    test_additive_noise()
    test_missing_data()
    test_multiplicative_noise()
    test_sklearn_solver()
    test_sklearn_api_compatibility()
    test_admm_vs_hm()
    test_early_stopping()
    test_coef_sd()

    elapsed = time.time() - t0
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = total - passed

    summary = {
        "module": "CoCoLasso",
        "total": total,
        "passed": passed,
        "failed": failed,
        "elapsed_sec": round(elapsed, 2),
        "all_passed": all_passed,
        "details": results,
    }

    output_path = os.path.join(RESULTS_DIR, "test_cocolasso_results.json")
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
