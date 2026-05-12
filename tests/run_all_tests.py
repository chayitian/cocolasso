"""
运行全部测试的主入口

依次执行: test_utils → test_cocolasso → test_bdcocolasso → test_generalcocolasso → test_solver_consistency
汇总结果保存至 tests/results/all_results.json
"""

import sys
import os
import json
import time
import subprocess

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(TESTS_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_FILES = [
    "test_utils.py",
    "test_cocolasso.py",
    "test_bdcocolasso.py",
    "test_generalcocolasso.py",
    "test_solver_consistency.py",
]


def run_test(test_file):
    test_path = os.path.join(TESTS_DIR, test_file)
    print(f"\n{'#' * 60}")
    print(f"# 运行: {test_file}")
    print(f"{'#' * 60}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, test_path],
        capture_output=True,
        text=True,
        cwd=TESTS_DIR,
    )
    elapsed = time.time() - t0

    print(result.stdout)
    if result.stderr:
        for line in result.stderr.splitlines():
            if "ConvergenceWarning" not in line and "cd_fast" not in line:
                print(f"[STDERR] {line}")

    return {
        "test_file": test_file,
        "returncode": result.returncode,
        "elapsed_sec": round(elapsed, 2),
        "passed": result.returncode == 0,
    }


def main():
    print("=" * 60)
    print("CoCoLasso 全部测试")
    print("=" * 60)

    t0 = time.time()
    run_results = []
    for tf in TEST_FILES:
        r = run_test(tf)
        run_results.append(r)

    total_elapsed = time.time() - t0

    all_json = {}
    for tf in TEST_FILES:
        name = tf.replace(".py", "_results.json")
        jpath = os.path.join(RESULTS_DIR, name)
        if os.path.exists(jpath):
            with open(jpath, "r", encoding="utf-8") as f:
                all_json[tf] = json.load(f)

    total_tests = sum(j.get("total", 0) for j in all_json.values())
    total_passed = sum(j.get("passed", 0) for j in all_json.values())
    total_failed = sum(j.get("failed", 0) for j in all_json.values())

    summary = {
        "total_test_files": len(TEST_FILES),
        "total_assertions": total_tests,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_elapsed_sec": round(total_elapsed, 2),
        "all_passed": total_failed == 0,
        "per_file": run_results,
        "details": all_json,
    }

    output_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"全部测试汇总")
    print(f"{'=' * 60}")
    for r in run_results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['test_file']}  ({r['elapsed_sec']}s)")

    print(f"\n断言总数: {total_tests}, 通过: {total_passed}, 失败: {total_failed}")
    print(f"总耗时: {total_elapsed:.2f}s")
    print(f"结果已保存: {output_path}")
    print(f"{'=' * 60}")

    return total_failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
