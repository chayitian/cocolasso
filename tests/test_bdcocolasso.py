"""
BDCoCoLasso 集成测试

覆盖: 加性噪声、缺失数据、sklearn求解器、
      块坐标下降收敛性、sklearn API兼容性
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import BDCoCoLasso

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
    print("\n--- BD 加性噪声 ---")
    n, p = 50, 20
    p1, p2 = 8, 12
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=10)
    tau = 0.5
    np.random.seed(110)
    Z = X.copy()
    Z[:, p1:] = Z[:, p1:] + np.random.randn(n, p2) * tau

    model = BDCoCoLasso(p1=p1, p2=p2, tau=tau, noise="additive",
                        max_iter=20, cv_folds=3, solver="coordinate_descent")
    model.fit(Z, y)

    record("BD_additive - fit成功",
           hasattr(model, 'coef_'),
           f"coef_shape={model.coef_.shape}")

    record("BD_additive - coef_维度",
           model.coef_.shape == (p,),
           f"shape={model.coef_.shape}")

    record("BD_additive - intercept_存在",
           isinstance(model.intercept_, float),
           f"intercept={model.intercept_:.4f}")

    record("BD_additive - lambda_opt_ > 0",
           model.lambda_opt_ > 0,
           f"lambda_opt={model.lambda_opt_:.6f}")

    record("BD_additive - coef_无NaN/Inf",
           np.all(np.isfinite(model.coef_)),
           f"finite={np.all(np.isfinite(model.coef_))}")

    y_pred = model.predict(Z)
    record("BD_additive - predict维度",
           y_pred.shape == (n,),
           f"shape={y_pred.shape}")

    r2 = model.score(Z, y)
    record("BD_additive - score(R²)有限",
           np.isfinite(r2),
           f"R²={r2:.4f}")


def test_missing_data():
    print("\n--- BD 缺失数据 ---")
    n, p = 50, 15
    p1, p2 = 5, 10
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=20)
    np.random.seed(120)
    Z = X.copy()
    mask_miss = np.random.rand(n, p1 + p2) < 0.1
    mask_miss[:, :p1] = False
    Z[mask_miss] = np.nan

    model = BDCoCoLasso(p1=p1, p2=p2, noise="missing",
                        max_iter=20, cv_folds=3, solver="coordinate_descent")
    model.fit(Z, y)

    record("BD_missing - fit成功",
           hasattr(model, 'coef_'),
           f"coef_shape={model.coef_.shape}")

    record("BD_missing - coef_无NaN",
           not np.any(np.isnan(model.coef_)),
           f"has_nan={np.any(np.isnan(model.coef_))}")

    record("BD_missing - lambda_opt_ > 0",
           model.lambda_opt_ > 0,
           f"lambda_opt={model.lambda_opt_:.6f}")


def test_sklearn_solver():
    print("\n--- BD sklearn 求解器 ---")
    n, p = 50, 20
    p1, p2 = 8, 12
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=40)
    tau = 0.5
    np.random.seed(140)
    Z = X.copy()
    Z[:, p1:] = Z[:, p1:] + np.random.randn(n, p2) * tau

    model = BDCoCoLasso(p1=p1, p2=p2, tau=tau, noise="additive",
                        max_iter=20, cv_folds=3, solver="sklearn")
    model.fit(Z, y)

    record("BD_sklearn - fit成功",
           hasattr(model, 'coef_'),
           f"coef_shape={model.coef_.shape}")

    record("BD_sklearn - coef_无NaN/Inf",
           np.all(np.isfinite(model.coef_)),
           f"finite={np.all(np.isfinite(model.coef_))}")

    record("BD_sklearn - lambda_opt_ > 0",
           model.lambda_opt_ > 0,
           f"lambda_opt={model.lambda_opt_:.6f}")


def test_block_convergence():
    print("\n--- BD 块坐标下降收敛 ---")
    n, p = 40, 15
    p1, p2 = 5, 10
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=60)
    tau = 0.3
    np.random.seed(160)
    Z = X.copy()
    Z[:, p1:] = Z[:, p1:] + np.random.randn(n, p2) * tau

    model = BDCoCoLasso(p1=p1, p2=p2, tau=tau, noise="additive",
                        max_iter=30, cv_folds=3, solver="coordinate_descent")
    model.fit(Z, y)

    record("BD_convergence - n_iter_ > 0",
           model.n_iter_ > 0,
           f"n_iter={model.n_iter_}")

    record("BD_convergence - n_iter_ <= max_iter",
           model.n_iter_ <= 30,
           f"n_iter={model.n_iter_}")


def test_sklearn_api():
    print("\n--- BD sklearn API 兼容性 ---")
    n, p = 50, 20
    p1, p2 = 8, 12
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=50)
    tau = 0.5
    np.random.seed(150)
    Z = X.copy()
    Z[:, p1:] = Z[:, p1:] + np.random.randn(n, p2) * tau

    model = BDCoCoLasso(p1=p1, p2=p2, tau=tau, noise="additive",
                        max_iter=20, cv_folds=3)

    from sklearn.base import BaseEstimator
    record("BD_api - 继承BaseEstimator",
           isinstance(model, BaseEstimator),
           f"type={type(model).__mro__}")

    model.fit(Z, y)
    record("BD_api - get_params可用",
           "p1" in model.get_params() and "p2" in model.get_params(),
           f"p1={model.get_params()['p1']}, p2={model.get_params()['p2']}")

    model.set_params(max_iter=50)
    record("BD_api - set_params可用",
           model.max_iter == 50,
           f"max_iter={model.max_iter}")


def test_coef_sd():
    print("\n--- BD 1-std 准则 ---")
    n, p = 50, 20
    p1, p2 = 8, 12
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=80)
    tau = 0.5
    np.random.seed(180)
    Z = X.copy()
    Z[:, p1:] = Z[:, p1:] + np.random.randn(n, p2) * tau

    model = BDCoCoLasso(p1=p1, p2=p2, tau=tau, noise="additive",
                        max_iter=20, cv_folds=3, solver="coordinate_descent")
    model.fit(Z, y)

    record("BD_coef_sd - 存在",
           hasattr(model, 'coef_sd_'),
           "")

    record("BD_coef_sd - 维度",
           model.coef_sd_.shape == (p,),
           f"shape={model.coef_sd_.shape}")

    record("BD_coef_sd - lambda_sd_ >= lambda_opt_",
           model.lambda_sd_ >= model.lambda_opt_ - 1e-10,
           f"lambda_sd={model.lambda_sd_:.6f} >= lambda_opt={model.lambda_opt_:.6f}")


def main():
    t0 = time.time()
    print("=" * 60)
    print("BDCoCoLasso 集成测试")
    print("=" * 60)

    test_additive_noise()
    test_missing_data()
    test_sklearn_solver()
    test_block_convergence()
    test_sklearn_api()
    test_coef_sd()

    elapsed = time.time() - t0
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = total - passed

    summary = {
        "module": "BDCoCoLasso",
        "total": total,
        "passed": passed,
        "failed": failed,
        "elapsed_sec": round(elapsed, 2),
        "all_passed": all_passed,
        "details": results,
    }

    output_path = os.path.join(RESULTS_DIR, "test_bdcocolasso_results.json")
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
