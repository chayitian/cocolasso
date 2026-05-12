# CoCoLasso 源代码评估指南

## 1. 项目概述

CoCoLasso（凸校正 Lasso）是针对协变量中存在测量误差的高维回归问题的 Python 实现，基于 R 语言包 BDcocolasso 的复现。

### 模块结构

| 模块 | 功能 |
|------|------|
| `src/_utils.py` | 底层工具函数（PSD投影、Lasso求解器、数据预处理） |
| `src/cocolasso.py` | 标准 CoCoLasso 估计器 |
| `src/bdcocolasso.py` | 二块下降 CoCoLasso（BD-CoCoLasso） |
| `src/generalcocolasso.py` | 三块混合 CoCoLasso（GeneralCoCoLasso） |
| `src/__init__.py` | 包入口，导出 `coco`/`generalcoco` 函数 |

---

## 2. 测试文件说明

| 测试文件 | 覆盖范围 | 预期耗时 |
|----------|----------|----------|
| `test_utils.py` | `_utils.py` 中所有函数的单元测试 | ~5s |
| `test_cocolasso.py` | CoCoLasso 三种噪声类型 + sklearn求解器 + API兼容性 | ~30s |
| `test_bdcocolasso.py` | BDCoCoLasso 加性/缺失噪声 + sklearn求解器 | ~30s |
| `test_generalcocolasso.py` | GeneralCoCoLasso 三块/两块 + sklearn求解器 | ~30s |
| `test_solver_consistency.py` | CD vs sklearn 求解器一致性 + 性能对比 | ~60s |
| `run_all_tests.py` | 运行全部测试并汇总结果 | ~3min |

### 运行方式

```bash
# 运行全部测试
cd /root/git_projects/cocolasso
python tests/run_all_tests.py

# 运行单个测试
python tests/test_utils.py
python tests/test_cocolasso.py
```

### 输出位置

所有测试结果（JSON格式）保存在 `tests/results/` 目录下：

| 文件 | 内容 |
|------|------|
| `test_utils_results.json` | _utils 模块测试详情 |
| `test_cocolasso_results.json` | CoCoLasso 测试详情 |
| `test_bdcocolasso_results.json` | BDCoCoLasso 测试详情 |
| `test_generalcocolasso_results.json` | GeneralCoCoLasso 测试详情 |
| `test_solver_consistency_results.json` | 求解器一致性测试详情 |
| `all_results.json` | 全部测试汇总 |

---

## 3. 评估维度与判断标准

将测试结果发给我后，我将从以下维度评估源代码质量：

### 3.1 正确性（Critical）

| 测试项 | 判断标准 | 失败意味着 |
|--------|----------|-----------|
| PSD投影输出半正定 | 所有特征值 ≥ -1e-8 | ADMM/HM投影算法有bug |
| Lasso求解器一致性 | CD与sklearn最大系数差 < 0.1 | alpha计算或Cholesky变换有误 |
| 大lambda全零 | lambda=100时系数≈0 | 惩罚项映射错误 |
| 协方差校正公式 | 乘性校正与手算公式一致 | `_corrected_covariance_multiplicative` 有误 |
| 缺失数据处理 | 输出无NaN | NaN处理逻辑有缺陷 |

### 3.2 数值稳定性（High）

| 测试项 | 判断标准 | 失败意味着 |
|--------|----------|-----------|
| 系数无NaN/Inf | 所有系数有限 | Cholesky分解遇到非PSD矩阵 |
| 预测无NaN | predict输出有限 | 系数或截距含异常值 |
| 对称性 | 投影矩阵对称 | 浮点精度问题未处理 |
| ratio_matrix对角线≤1 | 观测比率合理 | `_compute_ratio_matrix` 计算有误 |

### 3.3 API兼容性（Medium）

| 测试项 | 判断标准 | 失败意味着 |
|--------|----------|-----------|
| 继承BaseEstimator | isinstance检查通过 | sklearn接口未正确继承 |
| get_params/set_params | 参数可读写 | 构造函数签名不符合sklearn规范 |
| predict维度 | 输出shape=(n_samples,) | predict实现有误 |
| score(R²)有限 | 返回有限浮点数 | R²计算有除零等问题 |

### 3.4 功能完整性（Medium）

| 测试项 | 判断标准 | 失败意味着 |
|--------|----------|-----------|
| 三种噪声类型均可运行 | fit成功且coef_存在 | 某种噪声类型未实现或有bug |
| CV结果完整 | lambda/error/error_inf/error_sup/error_sd 均存在 | 交叉验证流程有遗漏 |
| coef_path_完整 | lambda和beta路径均存在 | 路径记录逻辑有误 |
| 1-std准则 | lambda_sd ≥ lambda_opt | 1-std准则选择逻辑有误 |
| 早停机制 | n_iter ≤ max_iter | 迭代控制有bug |

### 3.5 性能（Low）

| 测试项 | 判断标准 | 说明 |
|--------|----------|------|
| sklearn求解器可运行 | 无崩溃 | Cholesky + sklearn Lasso 集成正确 |
| sklearn加速 | sklearn耗时 ≤ CD耗时 | warm_start生效 |
| ADMM vs HM | 两种模式均可运行且结果相近 | HM投影实现正确 |

---

## 4. 结果提交格式

运行完测试后，请将以下内容发给我：

### 必需

1. **`tests/results/all_results.json`** 的完整内容（或截图）
2. **终端输出**中的 FAIL 项（如有）

### 可选

3. 各模块单独的 JSON 结果文件
4. 任何异常的警告信息（如 ConvergenceWarning）

### 示例提交

```
请评估以下测试结果：

all_results.json 内容：
{
  "total_assertions": 85,
  "total_passed": 83,
  "total_failed": 2,
  "all_passed": false,
  ...
}

FAIL 项：
[FAIL] solver_consistency - coco_multiplicative_coef (max_diff=0.15)
[FAIL] BD_missing_coef (max_diff=0.12)
```

---

## 5. 已知问题与修正记录

### 已修正

| 问题 | 修正 | 文件 |
|------|------|------|
| `_lasso_sklearn` alpha计算错误 | `alpha = lambda_val * n / p` → `alpha = lambda_val` | `_utils.py` |
| Cholesky缩放因子错误 | `sqrt(n)` → `sqrt(p)` | `_utils.py` |
| `_lasso_sklearn` 无warm_start | 添加 `beta_start` 参数和 `warm_start=True` | `_utils.py` |
| 调用处未传递beta_start | 所有 `_lasso_sklearn` 调用添加 `beta_start` 参数 | 三个模块 |

### 已知限制

1. sklearn求解器仅支持 `penalty="lasso"`，不支持 SCAD
2. 乘性噪声场景下两种求解器可能有稍大差异（CV过程放大数值误差）
3. 缺失数据场景下 sklearn 可能产生 ConvergenceWarning（可忽略，不影响结果正确性）

---

## 6. 评估结论模板

我将根据测试结果给出如下格式的评估：

```
## 评估结论

### 总体评级: [A/B/C/D]

### 正确性: [通过/存在问题]
- ...

### 数值稳定性: [通过/存在问题]
- ...

### API兼容性: [通过/存在问题]
- ...

### 需要修复的问题:
1. [Critical] ...
2. [High] ...
3. [Medium] ...

### 建议优化:
1. ...
2. ...
```
