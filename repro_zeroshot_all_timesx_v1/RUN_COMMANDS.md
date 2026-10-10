# RUN_COMMANDS — Aurora-timesx-all-native-A48-v1

## 本地（wlt 工作站）

```bash
cd /home/wlt/MMTS/Aurora_Experiments/repro_zeroshot_all_timesx_v1
PY=/home/wlt/miniconda3/envs/aurora_test/bin/python

# S1+S2：冻结引用 / 校验 57 旧分片 / 冻结预检抽样
bash run_all.sh local          # = freeze_ref + verify + sample（已有标记则走 stage_guard 跳过）
```

## theta GPU（env /dev_data/wlt/conda/envs/aurora，GPU1）

```bash
# rsync 新目录（含 visionts_ref、reuse_plan、preflight_sample）上 theta 后：
cd <theta:path>/repro_zeroshot_all_timesx_v1
PY=/dev_data/wlt/conda/envs/aurora/bin/python
CUDA_VISIBLE_DEVICES=1 $PY preflight.py --run        # S3：582 前向，双判据 -> reuse_final.json
CUDA_VISIBLE_DEVICES=1 $PY infer_fill.py             # S4：指纹绑定校验后补推（≈5,632×3 窗）
```

同一 tmux 会话连续执行 S3→S4（forward_binding 必须一致，否则 S4 拒跑）。

## 回传本地后

```bash
bash run_all.sh post          # = assemble + evaluate + recompute + accept
$PY make_report.py            # REPORT.md
bash audit_all.sh --final     # 审计包 -> /home/wlt/MMTS/temp/（<50MB）
```

## 预执行审计包（GPU 前，用户审核用）

```bash
bash audit_all.sh --preexec   # 代码 + 冻结引用 + S1/S2 证据 -> temp/
```

## 强制重跑

```bash
FORCE_VERIFY=1 bash run_all.sh verify     # 单阶段（FORCE_<STAGE大写>=1）
FORCE_FILL=1 CUDA_VISIBLE_DEVICES=1 $PY infer_fill.py   # GPU 阶段直接 --force 语义见脚本
```

## rsync 清单（theta → 本地）

`{cache_all/, results/, reuse_final.json, preflight_report.json, .state/, logs/, nohup.out}`
（`cache_all` 为新增 npz + manifest.json；predictions/ 旧分片不动。）
