# 文本预算与规则压缩诊断报告（analysis-text-budget-v1）

日期：2026-10-07　分支：`analysis-text-budget-v1`　基线：EXP-011 冻结 `2285a24`（只读）

## 0. 边界声明（先读）

- 本轮是**纯统计诊断**：无模型推理、无训练、无 LLM、不读取任何预测误差。所有结论只涉及"进入 BERT 输入的 token 与内容覆盖"。
- **覆盖率 ↑ 不必然改善预测**：覆盖=进入输入，不代表模型有效利用；EXP-011（M48T512 4.3065 vs A48 4.2936，配对 +0.0129 轻微负向）已说明文本通路整体收益存疑。本报告不得、也未用于依据测试性能挑选方案。
- 所有统计由 `stats_diag.py` 从 `outputs/trace_*.jsonl` 程序化生成；报告数字全部来自 `results/` 表，无手抄。

## 1. 方案与数据

| 方案 | 定义 |
|---|---|
| D0 | EXP-011 冻结构建器：四块固定预算 48/270/64/128（含块名），逐块截前缀，无预算转移 |
| D1 | 只回收闲置预算：基础分配 min(raw, budget)，剩余按 Events→Covariates→Calendar→Background 固定顺序补足；不变式 alloc_D1≥alloc_D0、Σ≤510 |
| D2 | 冻结规则清理（d2_rules.json：R1 前缀改写 / R2 引用删除 / R3 background 模板压缩 / R4 前缀压缩 / R5 前缀删除 / R6 空白修复）后按 D1 分配 |

数据：TimesX 全量 **8106** 窗（键唯一），按冻结 `split_manifest.json` 实际划分：
**train 4041 / val 895 / test 2474 / excluded 696**（排除原因：cross_train_val / cross_val_test）。逐窗 scope 与排除原因已写入 trace JSONL。

硬门槛（全量构建内建，`build_diag.py`）：
- D0 ids+mask 与冻结 `cache/text_tokens_M48T512.npz` **逐位一致 0/2474 失配**；
- domain（var_key 前缀）与冻结 `domains` **0/2474 失配**；
- tokenizer 四文件（config/tokenizer_json/**tokenizer_config.json**/vocab）md5 与 protocol_mm 锚定**全部一致**（缺失任一文件即失败）。

## 2. 主结果（全量 8106；测试 2474）

以下三表由 `gen_report_tables.py` 直接从 results/ CSV 生成（`report_tables.md`），不经手抄。

### 2.1 总体（stats_overall.csv）

| 指标 | D0 | D1 | D2 |
|---|---|---|---|
| content 中位（全量） | 451 | 510 | 510 |
| idle 中位（全量） | 59 | 0 | 0 |
| 仍截断窗比例（全量） | 100.0% | 99.8% | 99.7% |
| Events 完整 / 总事件（全量） | 10667/66815 (16.0%) | 13568/66815 (20.3%) | 16323/66815 (24.4%) |
| Covariates 完整条目（全量） | 16160/72586 | 17395/72586 | 17916/72586 |
| content 中位（测试） | 454 | 510 | 510 |
| idle 中位（测试） | 56 | 0 | 0 |
| 仍截断窗比例（测试） | 100.0% | 100.0% | 100.0% |
| Events 完整 / 总事件（测试） | 3841/22831 (16.8%) | 4836/22831 (21.2%) | 5690/22831 (24.9%) |
| Covariates 完整条目（测试） | 4908/22175 | 4908/22175 | 4951/22175 |

### 2.2 逐字段 alloc 中位（stats_fields.csv）

中位含 skipped 窗（0 值），与 CSV 口径一致。

| 字段 | scope | raw 中位 | D0 alloc | D1 alloc | D2 清理后 | D2 alloc |
|---|---|---|---|---|---|---|
| Background | full | 32 | 32 | 32 | 16 | 16 |
| Background | test | 32 | 32 | 32 | 16 | 16 |
| Events | full | 1327 | 270 | 327 | 1250 | 350 |
| Events | test | 1313 | 270 | 326 | 1253 | 349 |
| Calendar | full | 19 | 19 | 19 | 12 | 12 |
| Calendar | test | 21 | 21 | 21 | 14 | 14 |
| Covariates | full | 459 | 128 | 128 | 454 | 128 |
| Covariates | test | 458 | 128 | 128 | 452 | 128 |

### 2.3 D2 两指标分开（stats_d2_ratios.csv，逐窗中位）

| 字段 | scope | 清理缩短比例（1−cleaned/raw） | 清理后保留比例（alloc/cleaned） |
|---|---|---|---|
| Background | full | 0.500 | 1.000 |
| Background | test | 0.500 | 1.000 |
| Events | full | 0.062 | 0.279 |
| Events | test | 0.044 | 0.277 |
| Calendar | full | 0.241 | 1.000 |
| Calendar | test | 0.219 | 1.000 |
| Covariates | full | 0.011 | 0.300 |
| Covariates | test | 0.011 | 0.283 |

注意：token 减少 ≠ 信息无损（R3 模板套话除外，R2 删除的引用编号本身无内容）。本轮规则清理的收益有限：Events 仅缩短 4–6%，Background/Calendar 虽缩短明显但体量小，均远不能弥合第 3 节的预算缺口。

## 3. 四问回答

**Q1 D1 回收了多少预算？**
闲置位置中位 59（全量）/ 56（测试）→ 0：D1 把闲置预算全部回收（content 中位 451→510）。增益全部流向 Events（alloc 中位 270→326 测试；19 域一致 +42~+75 token），Covariates 在测试窗无增益（leftover 被 Events 按固定顺序先吃光——符合 budget_config 分配规则，非缺陷）。事件完整率相对 +27.2%（全量 10667→13568）。

**Q2 D2 在 D1 之上多保留多少？**
事件完整 13568→16323（全量，+20.3% 相对；测试 4836→5690，+17.7%）；Covariates 完整条目 +521（全量）/ +43（测试）。机制：清理省出的 token（Events 中位 1327→1250，Background 32→16，Calendar/Covariates 前缀各省几个）经 D1 分配回流 Events。

**Q3 还有多少窗口放不下？**
几乎所有窗口仍放不下：全量 still_truncated D1 99.8%、D2 99.7%（测试 100%）。结构性原因：2464/2474 测试窗仅 Events 原文就超过 510（原文中位 1313 token，为总预算 2.6 倍）。即便 D2，Events 只保留 27.9%（测试，按 token 和；逐窗中位保留比例 27.7%），Covariates 28.8%（按 token 和）。本轮规则清理的收益有限（Events 仅缩短 4–6%），远低于需求缺口。

**Q4 是否值得进一步设计本地摘要方案？**
空间存在但天花板明确：预算 510 固定，Events 需求中位 1313（测试），D2 后仍只保留 27.9%（按 token 和），缺口 ~72%；规则清理仅使原文缩短 4–6%，相对该缺口收益有限。若目标是让更多事件**完整**进入输入，两个有数据支撑的方向是（a）抽取式事件级摘要/选择（保留 `<N>` 边界，选完整事件子集替代头部截断——当前头部截断导致后半事件大量 absent：全量 43046/66815 未纳入），（b）协变量条目筛选。但按第 0 节边界：覆盖率↑不必然改善预测，任何摘要设计应先做同类统计预研、再以受控预测实验评估收益，本轮不启动。

## 4. 校验与确定性

| 检查 | 结果 |
|---|---|
| D0 vs 冻结缓存逐位（2474 测试窗） | 0 失配（构建内建 + verify_struct 独立复检双确认） |
| domain 对账 | 0/2474 失配 |
| 结构（8106×3：CLS/SEP/mask 前缀/pad/span 铺满/键集） | verify_struct ALL OK |
| 参考值复算（EXP-011 探索值 1313 / 56 / 2464） | 全部一致（reference_check.json，assert 强制） |
| tokenizer md5 vs protocol_mm（4 文件） | 全部一致；缺失即失败（修正后口径） |
| 键集完整唯一 | 8106 唯一；测试 2474 == 冻结键集 |
| 展示窗（21，全部 train/val） | 19 域各≥1、1D+1W、7 窗缺 Calendar；事实保留检查 ALL OK（日期/数值零丢失；背景三事实 21/21 保留；引用编号按规则豁免） |
| 确定性（修正后代码全管线两跑） | **PASS**：两跑六文件全部一致（npz 数组逐位、jsonl 字节级；determinism_double_run.txt） |
| 全新构建 vs 修补版 run1 | **PASS**：六文件全部一致——元信息修补（patch_meta.py）与修正后代码输出完全等价（determinism_fresh_vs_patched.txt） |

## 5. 审计修正记录（2026-10-07，用户复核发现）

| # | 问题 | 修正 |
|---|---|---|
| 1 | D2 记录 raw_tokens 误写为清理后计数 → 压缩率恒 0 | raw_tokens 恒取 D0 同窗计数；cleaned_tokens 单独保存；统计重算（patch_meta.py 对 run1 产物原位修正 npz 不动；新代码同步修正） |
| 2 | 展示用 D2 边界切 D0 ids；背景模板因大小写 21/21 未匹配却判通过 | field_text 按方案自身记录取 span；BACK_TPL 加 IGNORECASE；未匹配 → FAIL（不再是 OK） |
| 3 | scope 全部非测试窗标 trainval，混入 696 excluded | 按冻结 split_manifest 标注 train/val/test/excluded + excluded_reason；展示窗重选（全部 train/val） |
| 4 | tokenizer 校验误读 tokenizer_config.txt，缺失静默通过 | 改读 tokenizer_config.json；任一锚定文件缺失 → all_matched=False 并 assert 失败 |

（另：开发期内自行发现并修复 9 项，见审计包 STATUS.md 第 4 节；其中 #9 展示错位经溯源确认产物数据本身无误。）

## 6. 交付物

- 代码：`analysis/text_budget_v1/{build_diag,allocate,clean_rules,stats_diag,showcase,verify_struct,patch_meta,compare_traces,gen_report_tables}.py` + 冻结配置（d2_rules.json / budget_config.json，提交 `cadd2eb`）
- 产物（不入 git）：`outputs/trace_{D0,D1,D2}.{npz,jsonl}`（ids/mask + 全字段元信息、清理文本、删改记录、解码输入）
- 结果（入 git）：本报告 + 报告表格生成脚本产物（report_tables.md / report_numbers.json）+ 5 张统计表 + reference_check.json + showcase.{md,json} + determinism_*.txt + provenance.json
