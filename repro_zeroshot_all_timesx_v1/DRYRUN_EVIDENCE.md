# DRYRUN_EVIDENCE — S3/S4/S5 全链本地演练 v2（2026-10-10，GPU 执行前）

在 `/tmp/audev/repro`（本目录隔离副本 + 桩模块：predict 按冻结种子公式返回真实
旧缓存预测、torch/load_aurora/aurora 为最小桩）中，用**真实代码**跑通
S3 preflight --run → S4 infer_fill → S5a assemble → S5b evaluate → 复算 →
验收 → 报告。前向不经过 GPU，其余全部逻辑（指纹、双判据、verdict、集合算术、
评分、门）均为真实执行。演练副本已删除，本文件为记录。

本版针对预执行审计包评审意见修复后的代码：

1. **抽样范围修复**：compare 窗只从"实际可复用缓存窗口"（三 seed 均在验证分片
   中的 LN-test sid）均匀抽取；冻结时断言全部命中。修复前 190 个 compare 窗中
   115 个不在旧缓存（会在 S3 触发 KeyError）。独立复核：190/190 全部在缓存。
2. **final_vars 解包修复**：`"seed|var_key"` 字符串键显式 `split("|", 1)` +
   int 转换（修复前按二元组解包触发 ValueError）。
3. **零复用空集合**：infer_fill 的 `covered` 预置每 seed 空集（某 seed 零缓存
   复用时 fill = 全量 8,106，不再 KeyError）。
4. **LN 清单入包**：`split_manifest.json` 冻结进 `visionts_ref/`（sha 断言 ==
   旧 EXP-010 pin），服务器不再依赖本地绝对路径；各阶段 input_guards 同步纳入。

## 场景 A（cocoa 变量三 seed 全部超差 → 整 (var,seed) 重推）

- S3：checked=582（compare=570），failed_(seed,var)=**3**，c1_max=1.000e-03、
  c2_max=1.048e-02（阈值 1e-5/1e-4）→ reuse_vars=189，reuse_windows=2,457，
  fill=5,649/seed；reuse_final 解包与 totals 正确
- S4：16,947 前向、570 分片、manifest 指纹绑定校验通过
- S5a：24,318 行（cache=7,371 / new=16,947），补推集合 == 精确集合差
- S5b：VisionTS 重算门 rel_err **0.00e+00**；2,474 子集非纯缓存 →
  **NOT_EXECUTABLE 分支正确记录**（actual 4.293647±0.001592 / 1.200965±0.000304）
- 复算 PASS（d 4.05e-16 / 域表 1.58e-13 / 变量表 6.49e-16）；验收 R1–R7 ALL PASS

## 场景 B（全过 → 全量复用）

- S3：failed=0，c1_max=c2_max=0.000e+00 → 每 seed reuse 2,474 / fill 5,632
- S4：16,896 前向；S5a：24,318 行（cache=7,422 / new=16,896）
- S5b：VTS 门 0.00e+00；2,474 纯缓存 **4.293598±0.001592 / 1.200945±0.000305
  reproduced=True**（容差 5e-7，含 seed-std）
- 复算 PASS；验收 R1–R7 ALL PASS；REPORT.md 渲染成功

结论：修复后流水线在重推/部分复用/全量复用三种形态下端到端绿，S3 抽样全部
命中真实缓存，可进入 theta 真实 S3 + 单变量 S4 冒烟。
