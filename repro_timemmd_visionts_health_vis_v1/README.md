# EXP-023: repro_timemmd_visionts_health_vis_v1

EXP-021（VisionTS×TimeMMD 原始行序，token `06c003f25618`）冻结结果之上，对 **Health 域（weekly，选中 P=52）** 生成最优/最差测试窗口可视化，用于人工观察成功与失败模式。全部预测复用 EXP-021 冻结 npz，**不跑任何模型推理**；EXP-019/020/021 目录零改动。

## 来源实验与锚

- 基础实验：EXP-021 `repro_timemmd_visionts_zeroshot_v1`（add `d95004d4edb7`、freeze `c905d43db23c`）
- 协议文件 sha256 `06c003f25618665914939bf641d41f217edb6320db9baebda8c0a3dd63706eda`（本实验门 G1 复核）
- selected_p.json sha256 `91b0d5887938a32cf89272d148b5fde2a94a94065b8552913914c25c78e1e05a`（门 G2；Health 选中 P=52，候选 [1, 4, 52]）
- Health CSV sha256 `88a81d443533991b…`（门 G3）
- 4 个冻结预测 npz sha256 == EXP-021 done.json 钉（门 G4）

## 配置

- 数据：Health weekly，1389 行 1997-09-29..2024-05-06（原始行序=时间升序），切分 972/140/277，val_end=1112（测试首预测行，2019-01-21）；date 列有 5 处周界重复日期（1997/2003/2008/2014-12-29、2020-12-28，冻结 CSV 原样保留，门 G3 钉死）
- 窗口：L=96，H∈[12, 24, 36, 48]，窗数 266/254/242/230（共 992 窗），测试起点 s∈[1112, 1389-H]（含端点）
- 评估尺度：EXP-021 原评估口径 = standardized 空间；scaler = sklearn StandardScaler 拟合 OT[0:972)，scaled32 = float32((OT−mean)/scale)；逆变换 orig = scaled×scale+mean（绘图用）
- 指标门值（EXP-021 冻结 f32）：见下表；4H 等权均值 MSE 2.0210202038288116 / MAE 0.9590280205011368

## 门与复算结果（全部 PASS 才继续）

| 门 | 内容 | 结果 |
|---|---|---|
| `G1.protocol_sha256` | sha256=06c003f25618665914939bf641d41f217edb6320db9baebda8c0a3dd63706eda | PASS |
| `G1.protocol.Health.seq_len` | got 96 want 96 | PASS |
| `G1.protocol.Health.horizons` | got [12, 24, 36, 48] want [12, 24, 36, 48] | PASS |
| `G1.protocol.Health.windows` | got {'12': 266, '24': 254, '36': 242, '48': 230} want {'12': 266, '24': 254, '36': 242, '4 | PASS |
| `G1.protocol.Health.split_rows` | got [972, 140, 277] want [972, 140, 277] | PASS |
| `G1.protocol.Health.csv_sha256` | got '88a81d443533991b8e2b55da8db0a6496d68fc510077b1c0f568d6f186394cf8' want '88a81d4435339 | PASS |
| `G1.protocol.Health.selected_periodicity` | got 52 want 52 | PASS |
| `G1.protocol.Health.candidates` | got [1, 4, 52] want [1, 4, 52] | PASS |
| `G1.protocol.Health.dates` | got {'first': '1997-09-29', 'last': '2024-05-06', 'val_end': '2019-01-21', 'test_first': ' | PASS |
| `G2.selected_p_sha256` | sha256=91b0d5887938a32cf89272d148b5fde2a94a94065b8552913914c25c78e1e05a | PASS |
| `G2.Health_selected_P` | selected=52 | PASS |
| `G3.csv_sha256` | sha256=88a81d443533991b8e2b55da8db0a6496d68fc510077b1c0f568d6f186394cf8 | PASS |
| `G3.n_rows` | rows=1389 | PASS |
| `G3.dates_first_last` | 1997-09-29..2024-05-06 | PASS |
| `G3.dates_monotonic` | non-decreasing; exactly the 5 known duplicate week-boundary dates [(14, '1997-12-29'), (32 | PASS |
| `G3.split` | 972/140/277 val_end=1112 | PASS |
| `G3.ot_finite` | all finite | PASS |
| `G4.h12.npz_sha256` | sha256=54d2cad190d9b7302a56567f3c3a9ca01221f1103ea6169a68d7058c5d8a0b31 | PASS |
| `G4.h12.done_npz_pin` | done.json npz_sha256 == file | PASS |
| `G4.h24.npz_sha256` | sha256=83a1922f46abf41d3f5ee6bf74295047b3a8b1419f4c8c681bdae8aa7f5f38ba | PASS |
| `G4.h24.done_npz_pin` | done.json npz_sha256 == file | PASS |
| `G4.h36.npz_sha256` | sha256=3da5a41f67351ce8d17ef975b7ad32e904e49df0c8d8678e81283b656063f458 | PASS |
| `G4.h36.done_npz_pin` | done.json npz_sha256 == file | PASS |
| `G4.h48.npz_sha256` | sha256=146eb15e39bfd4466bfefa84c4ed7faff88259a268cb7a2eb5d33153b833e899 | PASS |
| `G4.h48.done_npz_pin` | done.json npz_sha256 == file | PASS |
| `G7.mean_4h_equal_weight` | mean(f32 per-H) mse=2.0210202038288116 mae=0.9590280205011368 (per-H first, then 4H equal- | PASS |

逐 H 复算（official float32 复刻，npz→f32）：

| H | 窗数 | MSE(f32 复算) | MAE(f32 复算) | 冻结 MSE | 冻结 MAE |
|---|---|---|---|---|---|
| 12 | 266 | 1.9547765255 | 0.9514692426 | 1.9547765255 | 0.9514692426 |
| 24 | 254 | 1.9851282835 | 0.9490209222 | 1.9851282835 | 0.9490209222 |
| 36 | 242 | 2.0321850777 | 0.9560883045 | 2.0321850777 | 0.9560883045 |
| 48 | 230 | 2.1119909286 | 0.9795336127 | 2.1119909286 | 0.9795336127 |
| **4H 等权均值** | — | **2.0210202038** | **0.9590280205** | 2.0210202038 | 0.9590280205 |

汇总指标计算方式（用户修正 #3）：先逐 H 复算组指标，再对四个 H **等权平均**与冻结值比较；**不是** 992 行窗口指标的直接均值（各 H 窗数不同）。逐窗身份门（G5）：每窗 row_pred_start==arange(1112, 1389-H+1)、date_pred_start==CSV dates[s]、scaled32==npz true 位级（dt 上 sklearn 1.7.2 实测位级一致），同时证明 scaler 与行对齐。

- scaler：mean_ot=1.5704967088477366，scale_ot=1.2079697109259782（拟合行 [0,972)，sklearn 1.7.2）

## 选窗规则（用户修正 #2：固定 H=12）

- 只在 h12 的 266 个测试窗内选：MSE 最小 5 个=best、最大 5 个=worst；**不混合四种预测长度排名**，避免把预测长度带来的难度差异误认为时序模式差异
- 并列时按窗口标识 `h{H:02d}_s{row:04d}` 升序
- 指标在 scaled 空间逐窗计算（f64 自冻结 npz 值）；其余三个 H 仅保留逐窗指标与组内排名（CSV 全 992 行）

## 入选 10 窗（h12）

| 排名 | 窗口标识 | 预测期（日期） | MSE(scaled) | MAE(scaled) | h12 组内排名 |
|---|---|---|---|---|---|
| best#1 | `h12_s1135` | 2019-07-01..2019-09-16 | 0.005127 | 0.061386 | 1/266 |
| best#2 | `h12_s1134` | 2019-06-24..2019-09-09 | 0.005503 | 0.067132 | 2/266 |
| best#3 | `h12_s1133` | 2019-06-17..2019-09-02 | 0.005686 | 0.069636 | 3/266 |
| best#4 | `h12_s1138` | 2019-07-22..2019-10-07 | 0.006207 | 0.070645 | 4/266 |
| best#5 | `h12_s1136` | 2019-07-08..2019-09-23 | 0.006677 | 0.067151 | 5/266 |
| worst#1 | `h12_s1212` | 2020-12-21..2021-03-01 | 14.649862 | 3.804000 | 266/266 |
| worst#2 | `h12_s1211` | 2020-12-14..2021-02-22 | 14.627923 | 3.795473 | 265/266 |
| worst#3 | `h12_s1215` | 2021-01-04..2021-03-22 | 14.042761 | 3.712962 | 264/266 |
| worst#4 | `h12_s1213` | 2020-12-28..2021-03-08 | 13.916456 | 3.706280 | 263/266 |
| worst#5 | `h12_s1214` | 2020-12-28..2021-03-15 | 13.775350 | 3.685037 | 262/266 |

## 窗口重叠情况

- 十窗两两 45 对中 20 对预测区间有行重叠；最大预测段重叠 11 行
- 全跨（历史+预测）最小重叠 26 行
- 1 row = 1 week (Health is weekly); same-H windows start stride 1, so heavy overlap is expected

## 图目（figures/，10 幅，dpi 150）

- `Health_best1_h12_s1135_2019-07-01.png`（best#1，h12_s1135，2019-07-01）
- `Health_best2_h12_s1134_2019-06-24.png`（best#2，h12_s1134，2019-06-24）
- `Health_best3_h12_s1133_2019-06-17.png`（best#3，h12_s1133，2019-06-17）
- `Health_best4_h12_s1138_2019-07-22.png`（best#4，h12_s1138，2019-07-22）
- `Health_best5_h12_s1136_2019-07-08.png`（best#5，h12_s1136，2019-07-08）
- `Health_worst1_h12_s1212_2020-12-21.png`（worst#1，h12_s1212，2020-12-21）
- `Health_worst2_h12_s1211_2020-12-14.png`（worst#2，h12_s1211，2020-12-14）
- `Health_worst3_h12_s1215_2021-01-04.png`（worst#3，h12_s1215，2021-01-04）
- `Health_worst4_h12_s1213_2020-12-28.png`（worst#4，h12_s1213，2020-12-28）
- `Health_worst5_h12_s1214_2020-12-28.png`（worst#5，h12_s1214，2020-12-28）
- 图内文字语言：en（未找到 CJK 字体，英文回退）

配色（用户指定）：历史真值蓝、预测区间真值橙、模型预测红虚线、后续真值灰；黑虚线竖线=预测起止。全局面板：完整 1389 行 OT 原值 vs 日期，测试区浅灰，本窗历史段浅蓝/预测段浅橙高亮。局部面板：预测由 npz pred 逆变换还原为原始单位，预测段真值直读 CSV。

## 复现

1. 依赖：diag_vts_env（python 3.10.20 / numpy 2.2.6 / sklearn 1.7.2）+ matplotlib（本实验补装，版本见 logs/pip_freeze_after.txt）
2. `bash run_vis_dt.sh all`：env_guard → verify_and_select.py （全部门+选窗）→ plot_windows.py（10 PNG）→ make_report.py → audit_pack.py（单审计包）
3. 或离线复算：包内 series_date_ot.csv（1389 行 date+OT）、scaler.json、4 个冻结 npz 副本、health_window_metrics.csv 足以核查原始曲线、逆变换与窗口位置

## 解释边界

- 结果说明只记录**图中可见现象**；机制归因（周期捕捉、幅值、分布漂移等）一律为**待验证假设**，后续再讨论。
- best/worst 集中于某些时段/幅值属于可见事实，仅记录不外推；单域（Health）单 H（h12）观察不能推广。
- 指标与排名均在 scaled 空间；图还原为 OT 原始单位（% WEIGHTED ILI，见 CSV 表头），两种纵轴勿混读。

---
*Generated by EXP-023 make_report.py; anchors: EXP-021 token 06c003f25618.*
