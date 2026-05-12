"""
GeneralCoCoLasso 集成测试

覆盖: 三块混合噪声（无误差+加性+缺失）、sklearn求解器、
      sklearn API兼容性、1-std准则
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import GeneralCoCoLasso

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


def test_three_blocks():
    print("\n--- 三块混合噪声 ---")
    n, p = 60, 20
    p1, p2, p3 = 5, 8, 7
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=10)
    tau = 0.5
    np.random.seed(110)
    Z = X.copy()
    Z[:, p1:p1 + p2] = Z[:, p1:p1 + p2] + np.random.randn(n, p2) * tau
    mask_miss = np.random.rand(n, p3) < 0.1
    Z_miss = Z[:, p1 + p2:].copy()
    Z_miss[mask_miss] = np.nan
    Z[:, p1 + p2:] = Z_miss

    model = GeneralCoCoLasso(p1=p1, p2=p2, p3=p3, tau=tau,
                             max_iter=20, cv_folds=3, solver="coordinate_descent")
    model.fit(Z, y)

    record("General_3block - fit成功",
           hasattr(model, 'coef_'),
           f"coef_shape={model.coef_.shape}")

    record("General_3block - coef_维度",
           model.coef_.shape == (p,),
           f"shape={model.coef_.shape}")

    record("General_3block - intercept_存在",
           isinstance(model.intercept_, float),
           f"intercept={model.intercept_:.4f}")

    record("General_3block - lambda_opt_ > 0",
           model.lambda_opt_ > 0,
           f"lambda_opt={model.lambda_opt_:.6f}")

    record("General_3block - coef_无NaN/Inf",
           np.all(np.isfinite(model.coef_)),
           f"finite={np.all(np.isfinite(model.coef_))}")

    y_pred = model.predict(Z)
    record("General_3block - predict维度",
           y_pred.shape == (n,),
           f"shape={y_pred.shape}")

    r2 = model.score(Z, y)
    record("General_3block - score(R²)有限",
           np.isfinite(r2),
           f"R²={r2:.4f}")


def test_two_blocks_no_uncorrupted():
    print("\n--- 两块（无无误差特征）---")
    n, p = 50, 15
    p1, p2, p3 = 0, 8, 7
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=20)
    tau = 0.5
    np.random.seed(120)
    Z = X.copy()
    Z[:, :p2] = Z[:, :p2] + np.random.randn(n, p2) * tau
    mask_miss = np.random.rand(n, p3) < 0.1
    Z_miss = Z[:, p2:].copy()
    Z_miss[mask_miss] = np.nan
    Z[:, p2:] = Z_miss

    model = GeneralCoCoLasso(p1=p1, p2=p2, p3=p3, tau=tau,
                             max_iter=20, cv_folds=3, solver="coordinate_descent")
    model.fit(Z, y)

    record("General_2block - fit成功",
           hasattr(model, 'coef_'),
           f"coef_shape={model.coef_.shape}")

    record("General_2block - coef_无NaN/Inf",
           np.all(np.isfinite(model.coef_)),
           f"finite={np.all(np.isfinite(model.coef_))}")

    record("General_2block - lambda_opt_ > 0",
           model.lambda_opt_ > 0,
           f"lambda_opt={model.lambda_opt_:.6f}")


def test_sklearn_solver():
    print("\n--- General sklearn 求解器 ---")
    n, p = 50, 20
    p1, p2, p3 = 5, 8, 7
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=40)
    tau = 0.5
    np.random.seed(140)
    Z = X.copy()
    Z[:, p1:p1 + p2] = Z[:, p1:p1 + p2] + np.random.randn(n, p2) * tau
    mask_miss = np.random.rand(n, p3) < 0.1
    Z_miss = Z[:, p1 + p2:].copy()
    Z_miss[mask_miss] = np.nan
    Z[:, p1 + p2:] = Z_miss

    model = GeneralCoCoLasso(p1=p1, p2=p2, p3=p3, tau=tau,
                             max_iter=20, cv_folds=3, solver="sklearn")
    model.fit(Z, y)

    record("General_sklearn - fit成功",
           hasattr(model, 'coef_'),
           f"coef_shape={model.coef_.shape}")

    record("General_sklearn - coef_无NaN/Inf",
           np.all(np.isfinite(model.coef_)),
           f"finite={np.all(np.isfinite(model.coef_))}")

    record("General_sklearn - lambda_opt_ > 0",
           model.lambda_opt_ > 0,
           f"lambda_opt={model.lambda_opt_:.6f}")


def test_sklearn_api():
    print("\n--- General sklearn API 兼容性 ---")
    n, p = 50, 20
    p1, p2, p3 = 5, 8, 7
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=50)
    tau = 0.5
    np.random.seed(150)
    Z = X.copy()
    Z[:, p1:p1 + p2] = Z[:, p1:p1 + p2] + np.random.randn(n, p2) * tau
    mask_miss = np.random.rand(n, p3) < 0.1
    Z_miss = Z[:, p1 + p2:].copy()
    Z_miss[mask_miss] = np.nan
    Z[:, p1 + p2:] = Z_miss

    model = GeneralCoCoLasso(p1=p1, p2=p2, p3=p3, tau=tau,
                             max_iter=20, cv_folds=3)

    from sklearn.base import BaseEstimator
    record("General_api - 继承BaseEstimator",
           isinstance(model, BaseEstimator),
           f"type={type(model).__mro__}")

    model.fit(Z, y)
    record("General_api - get_params可用",
           "p1" in model.get_params() and "p2" in model.get_params() and "p3" in model.get_params(),
           f"p1={model.get_params()['p1']}, p2={model.get_params()['p2']}, p3={model.get_params()['p3']}")

    model.set_params(max_iter=50)
    record("General_api - set_params可用",
           model.max_iter == 50,
           f"max_iter={model.max_iter}")


def test_coef_sd():
    print("\n--- General 1-std 准则 ---")
    n, p = 50, 20
    p1, p2, p3 = 5, 8, 7
    beta_true = np.zeros(p)
    beta_true[:3] = [3, 1.5, 2]
    X, y = generate_data(n, p, beta_true, 1.0, seed=80)
    tau = 0.5
    np.random.seed(180)
    Z = X.copy()
    Z[:, p1:p1 + p2] = Z[:, p1:p1 + p2] + np.random.randn(n, p2) * tau
    mask_miss = np.random.rand(n, p3) < 0.1
    Z_miss = Z[:, p1 + p2:].copy()
    Z_miss[mask_miss] = np.nan
    Z[:, p1 + p2:] = Z_miss

    model = GeneralCoCoLasso(p1=p1, p2=p2, p3=p3, tau=tau,
                             max_iter=20, cv_folds=3, solver="coordinate_descent")
    model.fit(Z, y)

    record("General_coef_sd - 存在",
           hasattr(model, 'coef_sd_'),
           "")

    record("General_coef_sd - 维度",
           model.coef_sd_.shape == (p,),
           f"shape={model.coef_sd_.shape}")

    record("General_coef_sd - lambda_sd_ >= lambda_opt_",
           model.lambda_sd_ >= model.lambda_opt_ - 1e-10,
           f"lambda_sd={model.lambda_sd_:.6f} >= lambda_opt={model.lambda_opt_:.6f}")


def main():
    t0 = time.time()
    print("=" * 60)
    print("GeneralCoCoLasso 集成测试")
    print("=" * 60)

    test_three_blocks()
    test_two_blocks_no_uncorrupted()
    test_sklearn_solver()
    test_sklearn_api()
    test_coef_sd()

    elapsed = time.time() - t0
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = total - passed

    summary = {
        "module": "GeneralCoCoLasso",
        "total": total,
        "passed": passed,
        "failed": failed,
        "elapsed_sec": round(elapsed, 2),
        "all_passed": all_passed,
        "details": results,
    }

    output_path = os.path.join(RESULTS_DIR, "test_generalcocolasso_results.json")
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
