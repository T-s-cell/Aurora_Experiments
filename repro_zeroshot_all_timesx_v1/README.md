# Aurora × TimesX 全量原生窗口 zero-shot（Aurora-timesx-all-native-A48-v1）

单模态 Aurora A48 zero-shot 于 **全部 8,106 个 TimesX 原生窗口**（190 变量 / 19 域，
past96→target12），三 base seed 2021/2022/2023，与冻结的 VisionTS EXP-014
（git `7b434dd`）逐窗对齐，同窗口同分母直接对照。

## 设计要点

- **只读引用**：VisionTS EXP-014 4 文件（inventory / frozen_arrays / vars /
  preds.jsonl.gz）sha256 钉住复制到 `visionts_ref/`；旧 EXP-010 Aurora 仓库
  （repo 根目录）path-import + md5 记录，绝不修改。
- **缓存复用**：EXP-010 旧 A48 缓存（57 分片 = 19 域 × 3 seed，LN test 键集恒为
  2,474）逐分片校验（header 指纹、content_sha256、旧代码 md5、aurora 包 md5、
  target/d/past96 与冻结逐值一致）后方可复用；缺失/不合格窗口转新推，**数量如实
  记录**。纯缓存复现门（EXP-010 4.293598±0.001592 / 1.200945±0.000305，5e-7）
  仅在子集完全由已验证旧缓存服务时执行，否则记 NOT_EXECUTABLE。
- **预检双判据**（S3，GPU）：抽样窗新旧前向对比
  c1 = max|new−old|/max(1e-8,max|old|) ≤ 1e-5 且 c2 = max|new−old|/d ≤ 1e-4；
  任一窗超差 → 整 (变量,seed) 重推；阈值绝不放宽。`forward_binding` 指纹
  （数据/权重/代码/包/env）绑定 S3→S4，不一致拒跑。
- **评分链**：窗（12 步点预测 vs target）→ 变量均值 → 域内变量等权（断言恰
  19 域）→ 19 域等权 overall；raw 只到变量级；分母 d = std(past96, ddof=0)，
  <1e-8 回退 vars.json calib_std（恰 4 常量窗）。三 seed 独立评分 →
  mean±std(ddof=1)，不平均预测、不选优。
- **门**：VisionTS 重算门（本目录评分实现重算 EXP-014 →
  4.2122984787894 / 0.856010666064058，rel ≤1e-12）先于一切 Aurora 数字；
  独立复算（recompute_check.py，纯 python 重写、独立进程）三级对账
  （d 1e-12 / 变量 1e-9 / 聚合 1e-12）；验收 R1–R7。

## 目录

```
protocol.py            冻结协议（全部常量/pin/容差）
common.py              路径/哈希/状态标记/输入守卫/前向指纹
stage_guard.py         .done 标记守卫（sha + input_guards 重算比对）
run_all.sh             本地阶段（local）与 S5 链（post）运行器
freeze_ref.py     S1a  冻结 VisionTS 引用 + d 重算 + LN 2,474 键集
verify_cache.py   S1b  57 旧分片逐一校验 -> reuse_plan.json
preflight.py      S2/S3 --sample-only 冻结抽样 / --run GPU 双判据预检
infer_fill.py     S4   补推（fill = 集合差），cache_all/ 分片 + manifest
assemble.py       S5a  cache+new 合并 -> results/aurora_preds.jsonl.gz
evaluate.py       S5b  VisionTS 重算门 + 三 seed 评分 + 19 域对照 + 子集报告
recompute_check.py     独立复算（standalone，失败 exit 1）
verify_acceptance.py   验收 R1–R7
make_report.py         REPORT.md 渲染
audit_all.sh           审计打包（--preexec / --final，<50MB -> temp/）
visionts_ref/          EXP-014 冻结引用（sha256 钉住）
cache_all/             S4 新推分片 + manifest.json（GPU 回传）
results/               S5 产物
.state/                阶段标记（sha + input_guards）
```

## 运行

见 `RUN_COMMANDS.md`。本地 S1/S2 → theta（env `/dev_data/wlt/conda/envs/aurora`）
S3/S4 → 回传后本地 S5。任何门失败即停止定位，不改阈值、不剔窗。
