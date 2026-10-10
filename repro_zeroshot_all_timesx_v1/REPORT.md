# Aurora × TimesX 全量原生窗口 zero-shot（Aurora-timesx-all-native-A48-v1）

- 生成时间：2026-10-10 14:41:14；主机：`LaptopWlt`
- 模型：Aurora A48 单模态 zero-shot（aurora-model 0.2.0，HF revision `c495b02c1b15…`，itl=48，num_samples=100，revin，FP32 确定性）；三 base seed [2021, 2022, 2023]
- 窗口对齐 EXP-014（git 7b434dd）：8106 窗 / 190 变量 / 19 域，past96→target12；评分分母 = 冻结 d（std(past96)，<1e-8 回退 calib_std）
- 对照：VisionTS EXP-014 全量（单值）4.2122984788 / 0.856010666064

## 1. 阶段状态

| 阶段 | 状态 | 关键产出 |
|---|---|---|
| S1a 冻结 VisionTS EXP-014 引用 | done | windows=8106; ln_test=2474 |
| S1b 校验旧 EXP-010 A48 缓存（57 分片） | done | shards_ok=19; shards_bad_or_missing=0 |
| S2 冻结预检抽样 | done | n_windows=194; n_forward_total=582 |
| S3 GPU 预检（双判据新旧对比） | done | n_checked=582; failed_(seed,var)=0 |
| S4 GPU 补推 | done | fill_windows_total=16746; entries=570 |
| S5a 合并预测 | done | rows=24318; cache=7422; new=16896 |
| S5b 评分与对照 | done | aurora_std_mse_mean=4.539154389756772; aurora_std_mae_mean=0.9849643167182355; vts_gate=PASS; subset_reproduced=True |
| 独立复算 | done |  |
| 验收 R1-R7 | done | passed=True; subset_reproduced=True |

## 2. 引用与缓存复用

- VisionTS 4 文件 sha256 全部钉住一致；LN manifest sha 与旧 EXP-010 pin 一致；d 重算一致，回退（calib_std）恰 4 窗
- 旧缓存：57/57 分片通过（header 指纹 + content_sha256 + code_md5 + aurora_package_md5 一致 + target/d/past96 与冻结逐值一致），每 seed 可复用 2474 窗 / 190 变量
- 预检抽样 194 窗 × 3 seed = 582 次前向；实测 checked=582，失败 (seed,var)=0，c1_max=0.000e+00，c2_max=0.000e+00（阈值 1e-5 / 1e-4，未放宽）

| seed | 复用变量 | 复用窗 | 补推窗 |
|---|---|---|---|
| 2021 | 190 | 2474 | 5632 |
| 2022 | 190 | 2474 | 5632 |
| 2023 | 190 | 2474 | 5632 |

- S4 实际执行前向 16896 次（=Σ补推窗）；S5a 合并 24318 行（cache=7422，new=16896），补推集合 == 精确集合差

## 3. 主结果（19 域等权 overall）

- VisionTS 重算门：rel_err MSE 0.00e+00 / MAE 0.00e+00（≤1e-12）→ **PASS**
- Aurora（3 seed 独立评分，mean±std ddof=1）：std-MSE **4.539154 ± 0.004104**，std-MAE **0.984964 ± 0.000269**
- VisionTS：std-MSE 4.212298，std-MAE 0.856011
- Δ（Aurora − VisionTS）：std-MSE +0.326856，std-MAE +0.128954（负 = Aurora 更优）
- 改善计数：域 1/19 (MSE)、0/19 (MAE)；变量 27/190 (MSE)、28/190 (MAE)

### 19 域对照表

| 域 | n_vars | VisionTS MSE | Aurora MSE | ΔMSE | VisionTS MAE | Aurora MAE | ΔMAE |
|---|---|---|---|---|---|---|---|
| CropsAndStaples | 11 | 1.223960 | 1.462972±0.002143 | +0.2390 | 0.709330 | 0.843958±0.001282 | +0.1346 |
| Currency | 10 | 1.124216 | 1.309726±0.002310 | +0.1855 | 0.714195 | 0.819546±0.001216 | +0.1054 |
| EnergyAndFuels | 10 | 1.498044 | 1.762294±0.005866 | +0.2642 | 0.770415 | 0.871077±0.000143 | +0.1007 |
| LivestockAndFoodProducts | 10 | 1.291079 | 1.516972±0.004066 | +0.2259 | 0.718931 | 0.862902±0.000594 | +0.1440 |
| RawMaterialsAndConstruction | 10 | 1.215918 | 1.472356±0.001859 | +0.2564 | 0.756661 | 0.866839±0.001200 | +0.1102 |
| SpecialtyAndAdvancedMaterials | 8 | 2.952636 | 3.307706±0.002248 | +0.3551 | 0.614503 | 0.803908±0.000306 | +0.1894 |
| StrategicAndHighValueMaterials | 10 | 9.884354 | 8.022373±0.060423 | -1.8620 | 0.815568 | 0.994190±0.000606 | +0.1786 |
| arts | 10 | 2.087604 | 2.266749±0.001881 | +0.1791 | 0.834441 | 0.913010±0.001187 | +0.0786 |
| climate | 10 | 2.329152 | 2.807551±0.005776 | +0.4784 | 0.814267 | 0.959744±0.001190 | +0.1455 |
| economy | 10 | 1.655928 | 2.014569±0.001097 | +0.3586 | 0.863740 | 0.959479±0.000414 | +0.0957 |
| electronic_technology | 10 | 13.817808 | 15.002580±0.012167 | +1.1848 | 1.253648 | 1.404474±0.001968 | +0.1508 |
| finance | 10 | 3.614412 | 4.355358±0.008304 | +0.7409 | 1.103004 | 1.273287±0.002408 | +0.1703 |
| pets | 10 | 2.163914 | 2.722333±0.004766 | +0.5584 | 0.899938 | 1.019827±0.001226 | +0.1199 |
| public_health | 10 | 1.766582 | 2.081422±0.005475 | +0.3148 | 0.803933 | 0.921071±0.001027 | +0.1171 |
| public_policy | 11 | 20.965990 | 21.498721±0.011174 | +0.5327 | 1.121904 | 1.204124±0.000743 | +0.0822 |
| science | 10 | 5.445912 | 5.967878±0.003272 | +0.5220 | 0.973408 | 1.064165±0.001254 | +0.0908 |
| shopping | 10 | 0.821706 | 1.439244±0.000853 | +0.6175 | 0.495677 | 0.687490±0.000831 | +0.1918 |
| society | 10 | 2.764258 | 3.346837±0.002090 | +0.5826 | 0.946042 | 1.115566±0.000493 | +0.1695 |
| traffic | 10 | 3.410198 | 3.886292±0.010055 | +0.4761 | 1.054597 | 1.129665±0.000962 | +0.0751 |
| 平均值（19 域等权） | 19 | 4.212298 | 4.539154±0.004104 | +0.3269 | 0.856011 | 0.984964±0.000269 | +0.1290 |

### 三 seed 独立 overall

| seed | std-MSE | std-MAE |
|---|---|---|
| 2021 | 4.536664 | 0.984727 |
| 2022 | 4.536909 | 0.984910 |
| 2023 | 4.543891 | 0.985256 |

### 旧 LN-test 2,474 子集（key set 恒为 2,474）

- 纯缓存：4.293598±0.001592 / 1.200945±0.000305；复现 EXP-010 4.293598±0.001592 / 1.200945±0.000305 → **复现**（容差 5e-7，含 seed-std）

## 4. 独立复算与验收

- 独立复算（纯 python 重写，独立进程）：**PASS**；d 最大相对偏差 4.05e-16，域表 2.59e-13，变量表 6.59e-16（容差 d 1e-12 / 变量 1e-9 / 聚合 1e-12）
- 验收 R1–R7：**ALL PASS**；常量历史窗 denom_source=calib_std 每 seed {'2021': 4, '2022': 4, '2023': 4}（冻结 4）

## 5. 前向绑定（S3→S4 指纹）

- 权重 sha256 `df2fb96852a59515…`（843564328 B）；HF revision `c495b02c1b15…`；itl=48；num_samples=100
- 种子规则：`w = int.from_bytes(sha256(f'aurora-timesx-v1|{base_seed}|{var_key}|{sample_id}'.encode()).digest()[:8], 'big') % (2**31 - 1); torch.manual_seed(w) immediately before each window's generate(); no other RNG consumption in between`
- env：torch 2.12.0+cu126 / host theta / CUBLAS_WORKSPACE_CONFIG=:4096:8；aurora 包 md5 8 模块全部一致

## 6. 结果文件 sha256

- `results/aurora_preds.jsonl.gz` `6692471368cedfed55023d21659d9c07bb6850f1b52a8f5ceff63c0a3424d6f7`
- `results/overall.json` `b30eb1db6c4c42784b19b34711565899f5795737e9cbe7e9f4020b09a61f9575`
- `results/domain_level.csv` `3aba143a113f64ea5b32d9e88198dd8bf443234d179ff7c0fbf83324b8b3f2d6`
- `results/variable_level.csv` `c24dc7598f9284734968809317412266ffc5e7399cd9b45296effc8ad33aff5b`
- `results/subset_2474.json` `bcc41ccd6ccd10167ecd430be8f409c66ca5e946368a17ae176f820910bc47dc`
- `results/recompute_check.json` `2ec78dabb88df3e944f19140f868b7f8262c21705e095df0721f4995558aa267`
- `results/acceptance.json` `3ec5af8eb84621e58e0a2f900677ee2820a88d5f31af62dfc92cab47154e4e15`
- `reuse_final.json` `7c016d43bf2d0fe9f8e83f0b09c40699eec826ac8210b36decfe749b8d92025c`
- `preflight_report.json` `940af5b74ef14cf7827c81504527de9b79932a4fc64ed3567c2ab68fbecd4701`
