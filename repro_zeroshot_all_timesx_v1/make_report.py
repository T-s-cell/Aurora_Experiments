#!/usr/bin/env python3
"""Render REPORT.md from the frozen stage artifacts (S1-S5, recompute,
acceptance). Read-only over results; fails loudly if a required artifact is
missing."""
import os
import time

import common as C
import protocol as P

STAGES = [
    ("s1a_freeze_ref", "S1a 冻结 VisionTS EXP-014 引用"),
    ("s1b_verify_cache", "S1b 校验旧 EXP-010 A48 缓存（57 分片）"),
    ("s2_sample_frozen", "S2 冻结预检抽样"),
    ("s3_preflight", "S3 GPU 预检（双判据新旧对比）"),
    ("s4_infer_fill", "S4 GPU 补推"),
    ("s5a_assemble", "S5a 合并预测"),
    ("s5b_evaluate", "S5b 评分与对照"),
    ("recompute", "独立复算"),
    ("accept", "验收 R1-R7"),
]


def num(x, nd=6):
    return f"{x:.{nd}f}"


def main():
    state = {s: C.state_of(s) for s, _ in STAGES}
    missing = [s for s, v in state.items() if v is None]
    if missing:
        raise SystemExit(f"missing stage markers: {missing}")
    ref = C.load_json(C.REF_REPORT)
    plan = C.load_json(C.REUSE_PLAN)
    sample = C.load_json(C.PREFLIGHT_SAMPLE)
    pf = C.load_json(os.path.join(C.HERE, "preflight_report.json"))
    fin = C.load_json(C.REUSE_FINAL)
    asm = C.load_json(os.path.join(C.RESULTS_DIR, "assemble_report.json"))
    gate = C.load_json(os.path.join(C.RESULTS_DIR, "vts_gate.json"))
    overall = C.load_json(os.path.join(C.RESULTS_DIR, "overall.json"))
    subset = C.load_json(os.path.join(C.RESULTS_DIR, "subset_2474.json"))
    rec = C.load_json(os.path.join(C.RESULTS_DIR, "recompute_check.json"))
    acc = C.load_json(os.path.join(C.RESULTS_DIR, "acceptance.json"))

    L = []
    w = L.append
    au = overall["aurora"]
    delta = overall["delta"]

    w(f"# Aurora × TimesX 全量原生窗口 zero-shot（{P.PROTOCOL_VERSION}）")
    w("")
    w(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；主机：`{C.env_info()['host']}`")
    w(f"- 模型：Aurora A48 单模态 zero-shot（aurora-model {P.AURORA_PACKAGE_VERSION}，"
      f"HF revision `{P.HF_REVISION[:12]}…`，itl={P.ITL}，num_samples={P.NUM_SAMPLES}，"
      f"revin，FP32 确定性）；三 base seed {list(P.SEEDS)}")
    w(f"- 窗口对齐 EXP-014（git 7b434dd）：{P.TOTAL_WINDOWS} 窗 / {P.N_VARS} 变量 / "
      f"{P.N_DOMAINS} 域，past96→target12；评分分母 = 冻结 d（std(past96)，<1e-8 回退 calib_std）")
    w(f"- 对照：VisionTS EXP-014 全量（单值）{num(P.REF_VTS_STD_MSE, 10)} / "
      f"{num(P.REF_VTS_STD_MAE, 12)}")
    w("")
    w("## 1. 阶段状态")
    w("")
    w("| 阶段 | 状态 | 关键产出 |")
    w("|---|---|---|")
    for s, label in STAGES:
        v = state[s]
        keys = [k for k in v if k not in ("status", "sha", "input_guards")]
        detail = "; ".join(f"{k}={v[k]}" for k in keys if not isinstance(v[k], dict))
        w(f"| {label} | {v['status']} | {detail[:160]} |")
    w("")
    w("## 2. 引用与缓存复用")
    w("")
    w(f"- VisionTS 4 文件 sha256 全部钉住一致；LN manifest sha 与旧 EXP-010 pin 一致；"
      f"d 重算一致，回退（calib_std）恰 {ref['calib_fallback_windows']} 窗")
    w(f"- 旧缓存：57/57 分片通过（header 指纹 + content_sha256 + code_md5 + "
      f"aurora_package_md5 一致 + target/d/past96 与冻结逐值一致），"
      f"每 seed 可复用 {plan['per_seed']['2021']['n_reuse_windows']} 窗 / "
      f"{plan['per_seed']['2021']['n_reuse_vars']} 变量")
    w(f"- 预检抽样 {len(sample['windows'])} 窗 × 3 seed = {sample['n_forward_total']} 次前向；"
      f"实测 checked={pf['stats']['n_checked']}，失败 (seed,var)="
      f"{pf['stats']['n_failed_(seed,var)']}，c1_max={pf['stats']['c1_max']:.3e}，"
      f"c2_max={pf['stats']['c2_max']:.3e}（阈值 1e-5 / 1e-4，未放宽）")
    w("")
    w("| seed | 复用变量 | 复用窗 | 补推窗 |")
    w("|---|---|---|---|")
    for s in P.SEEDS:
        t = fin["totals"]["per_seed"][str(s)]
        w(f"| {s} | {t['reuse_vars']} | {t['reuse_windows']} | {t['fill_windows']} |")
    w("")
    w(f"- S4 实际执行前向 {asm['new_rows']} 次（=Σ补推窗）；S5a 合并 "
      f"{asm['rows']} 行（cache={asm['cache_rows']}，new={asm['new_rows']}），"
      f"补推集合 == 精确集合差")
    w("")
    w("## 3. 主结果（19 域等权 overall）")
    w("")
    w(f"- VisionTS 重算门：rel_err MSE {gate['rel_err']['std_mse']:.2e} / "
      f"MAE {gate['rel_err']['std_mae']:.2e}（≤1e-12）→ **{'PASS' if gate['passed'] else 'FAIL'}**")
    w(f"- Aurora（3 seed 独立评分，mean±std ddof=1）：std-MSE "
      f"**{num(au['std_mse_mean'])} ± {num(au['std_mse_std'])}**，std-MAE "
      f"**{num(au['std_mae_mean'])} ± {num(au['std_mae_std'])}**")
    w(f"- VisionTS：std-MSE {num(overall['visionts']['std_mse'])}，"
      f"std-MAE {num(overall['visionts']['std_mae'])}")
    w(f"- Δ（Aurora − VisionTS）：std-MSE {delta['std_mse']:+.6f}，"
      f"std-MAE {delta['std_mae']:+.6f}（负 = Aurora 更优）")
    ic = overall["improvement_counts"]
    w(f"- 改善计数：域 {ic['domains']['mse']}/19 (MSE)、{ic['domains']['mae']}/19 (MAE)；"
      f"变量 {ic['variables']['mse']}/190 (MSE)、{ic['variables']['mae']}/190 (MAE)")
    w("")
    w("### 19 域对照表")
    w("")
    w("| 域 | n_vars | VisionTS MSE | Aurora MSE | ΔMSE | VisionTS MAE | Aurora MAE | ΔMAE |")
    w("|---|---|---|---|---|---|---|---|")
    import csv
    with open(os.path.join(C.RESULTS_DIR, "domain_level.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        name = r["domain"]
        w(f"| {name} | {r['n_vars']} | {num(float(r['vts_std_mse']))} | "
          f"{num(float(r['aurora_std_mse_mean']))}±{num(float(r['aurora_std_mse_std']))} | "
          f"{float(r['delta_std_mse']):+.4f} | {num(float(r['vts_std_mae']))} | "
          f"{num(float(r['aurora_std_mae_mean']))}±{num(float(r['aurora_std_mae_std']))} | "
          f"{float(r['delta_std_mae']):+.4f} |")
    w("")
    w("### 三 seed 独立 overall")
    w("")
    w("| seed | std-MSE | std-MAE |")
    w("|---|---|---|")
    for s in P.SEEDS:
        v = au["per_seed"][str(s)]
        w(f"| {s} | {num(v['std_mse'])} | {num(v['std_mae'])} |")
    w("")
    w(f"### 旧 LN-test 2,474 子集（key set 恒为 2,474）")
    w("")
    if subset.get("pure_cache"):
        w(f"- 纯缓存：{num(subset['std_mse_mean'])}±{num(subset['std_mse_std'])} / "
          f"{num(subset['std_mae_mean'])}±{num(subset['std_mae_std'])}；"
          f"复现 EXP-010 4.293598±0.001592 / 1.200945±0.000305 → "
          f"**{'复现' if subset.get('reproduced') else '未复现'}**（容差 5e-7，含 seed-std）")
    else:
        w(f"- {subset.get('reproduction')}")
        w(f"- 实际（含重推来源）：{num(subset['std_mse_mean'])}±{num(subset['std_mse_std'])} / "
          f"{num(subset['std_mae_mean'])}±{num(subset['std_mae_std'])}")
    w("")
    w("## 4. 独立复算与验收")
    w("")
    rc = rec["checks"]
    w(f"- 独立复算（纯 python 重写，独立进程）：**{'PASS' if rec['passed'] else 'FAIL'}**；"
      f"d 最大相对偏差 {rc['max_rel_d_vs_pred_files']:.2e}，域表 {rc['max_rel_domain_csv']:.2e}，"
      f"变量表 {rc['max_rel_variable_csv']:.2e}（容差 d 1e-12 / 变量 1e-9 / 聚合 1e-12）")
    w(f"- 验收 R1–R7：**{'ALL PASS' if acc['passed'] else 'FAILED'}**；"
      f"常量历史窗 denom_source=calib_std 每 seed "
      f"{acc['counts']['constant_rows_per_seed']}（冻结 4）")
    w("")
    w("## 5. 前向绑定（S3→S4 指纹）")
    w("")
    fb = fin["forward_binding"]
    w(f"- 权重 sha256 `{fb['weights_sha256'][:16]}…`（{fb['weights_bytes']} B）；"
      f"HF revision `{fb['hf_revision'][:12]}…`；itl={fb['itl']}；"
      f"num_samples={fb['num_samples']}")
    w(f"- 种子规则：`{P.SEED_RULE}`")
    w(f"- env：torch {fb['env'].get('torch')} / host {fb['env'].get('host')} / "
      f"CUBLAS_WORKSPACE_CONFIG={fb['env'].get('cublas_workspace_config')}；"
      f"aurora 包 md5 {len(fb['aurora_package_md5'])} 模块全部一致")
    w("")
    w("## 6. 结果文件 sha256")
    w("")
    for rel in ("results/aurora_preds.jsonl.gz", "results/overall.json",
                "results/domain_level.csv", "results/variable_level.csv",
                "results/subset_2474.json", "results/recompute_check.json",
                "results/acceptance.json", "reuse_final.json",
                "preflight_report.json"):
        w(f"- `{rel}` `{C.sha256_file(os.path.join(C.HERE, rel))}`")
    w("")

    out = os.path.join(C.HERE, "REPORT.md")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print(f"[report] -> {out}")
    if not (gate["passed"] and rec["passed"] and acc["passed"]):
        raise SystemExit("report rendered but a gate FAILED — do not ship")


if __name__ == "__main__":
    main()
