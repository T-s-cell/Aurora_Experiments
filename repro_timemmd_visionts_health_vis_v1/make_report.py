#!/usr/bin/env python3
"""EXP-023 report: generate README.md from the actual run artifacts
(gates.json / selection.json / scaler.json / health_window_metrics.csv).
Requires verify_and_select.py to have passed."""
import json
import os
import sys

import numpy as np

import anchors as A

RESULTS = A.RESULTS_DIR


def main():
    for p in ("gates.json", "selection.json", "scaler.json"):
        if not os.path.isfile(os.path.join(RESULTS, p)):
            sys.exit(f"[report] missing {p} — run verify_and_select.py first")
    gates = json.load(open(os.path.join(RESULTS, "gates.json"),
                           encoding="utf-8"))
    selection = json.load(open(os.path.join(RESULTS, "selection.json"),
                               encoding="utf-8"))
    scaler = json.load(open(os.path.join(RESULTS, "scaler.json"),
                            encoding="utf-8"))
    sel = selection["windows"]
    best = sorted([w for w in sel if w["kind"] == "best"],
                  key=lambda w: w["rank"])
    worst = sorted([w for w in sel if w["kind"] == "worst"],
                   key=lambda w: w["rank"])
    ov = selection["overlap"]["summary"]

    def wrow(w):
        return (f"| {w['kind']}#{w['rank']} | `{w['window_id']}` "
                f"| {w['date_pred_start']}..{w['date_pred_end']} "
                f"| {w['mse']:.6f} | {w['mae']:.6f} "
                f"| {w['group_rank']}/266 |")

    lines = [
        f"# {A.EXP_ID}: {A.PROTOCOL_NAME}",
        "",
        "EXP-021（VisionTS×TimeMMD 原始行序，token "
        f"`{A.BASE['protocol_token']}`）冻结结果之上，对 **Health 域"
        f"（weekly，选中 P={A.SELECTED_P}）** 生成最优/最差测试窗口可视化，"
        "用于人工观察成功与失败模式。全部预测复用 EXP-021 冻结 npz，"
        "**不跑任何模型推理**；EXP-019/020/021 目录零改动。",
        "",
        "## 来源实验与锚",
        "",
        f"- 基础实验：EXP-021 `repro_timemmd_visionts_zeroshot_v1`"
        f"（add `{A.BASE['add_commit'][:12]}`、freeze "
        f"`{A.BASE['freeze_commit'][:12]}`）",
        f"- 协议文件 sha256 `{A.BASE['protocol_sha256']}`（本实验门 G1 复核）",
        f"- selected_p.json sha256 `{A.BASE['selected_p_sha256']}`"
        f"（门 G2；Health 选中 P={A.SELECTED_P}，候选 {A.CANDIDATES}）",
        f"- Health CSV sha256 `{A.CSV_SHA256[:16]}…`（门 G3）",
        f"- 4 个冻结预测 npz sha256 == EXP-021 done.json 钉（门 G4）",
        "",
        "## 配置",
        "",
        f"- 数据：Health weekly，{A.N_ROWS} 行 "
        f"{A.DATES['first']}..{A.DATES['last']}（原始行序=时间升序），"
        f"切分 {A.SPLIT_ROWS[0]}/{A.SPLIT_ROWS[1]}/{A.SPLIT_ROWS[2]}，"
        f"val_end={A.VAL_END}（测试首预测行，"
        f"{A.DATES['test_first']}）；date 列有 5 处周界重复日期"
        "（1997/2003/2008/2014-12-29、2020-12-28，冻结 CSV 原样保留，"
        "门 G3 钉死）",
        f"- 窗口：L={A.SEQ_LEN}，H∈{A.HORIZONS}，窗数 "
        f"{A.WINDOWS[12]}/{A.WINDOWS[24]}/{A.WINDOWS[36]}/{A.WINDOWS[48]}"
        f"（共 {A.TOTAL_TEST_WINDOWS} 窗），测试起点 "
        f"s∈[{A.VAL_END}, {A.N_ROWS}-H]（含端点）",
        "- 评估尺度：EXP-021 原评估口径 = standardized 空间；"
        "scaler = sklearn StandardScaler 拟合 OT[0:972)，"
        "scaled32 = float32((OT−mean)/scale)；逆变换 orig = "
        "scaled×scale+mean（绘图用）",
        f"- 指标门值（EXP-021 冻结 f32）：见下表；4H 等权均值 "
        f"MSE {A.FROZEN_4H_MEAN_F32['mse']} / MAE "
        f"{A.FROZEN_4H_MEAN_F32['mae']}",
        "",
        "## 门与复算结果（全部 PASS 才继续）",
        "",
        "| 门 | 内容 | 结果 |",
        "|---|---|---|",
    ]
    for g in gates["gates"]:
        name = g["gate"]
        if name.startswith(("G5.", "G6.", "G4.")) and \
                not name.endswith(("npz_sha256", "done_npz_pin")):
            continue  # per-group noise; summarised below
        lines.append(f"| `{name}` | {g['detail'][:90]} | "
                     f"{'PASS' if g['ok'] else 'FAIL'} |")
    lines += [
        "",
        "逐 H 复算（official float32 复刻，npz→f32）：",
        "",
        "| H | 窗数 | MSE(f32 复算) | MAE(f32 复算) | 冻结 MSE | 冻结 MAE |",
        "|---|---|---|---|---|---|",
    ]
    for h in A.HORIZONS:
        ph = gates["per_h"][str(h)]
        fm = A.FROZEN_METRICS_F32[h]
        lines.append(
            f"| {h} | {ph['windows']} | {ph['metrics_f32']['mse']:.10f} "
            f"| {ph['metrics_f32']['mae']:.10f} | {fm['mse']:.10f} "
            f"| {fm['mae']:.10f} |")
    m = gates["mean_4h_f32"]
    lines += [
        f"| **4H 等权均值** | — | **{m['mse']:.10f}** | "
        f"**{m['mae']:.10f}** | {A.FROZEN_4H_MEAN_F32['mse']:.10f} "
        f"| {A.FROZEN_4H_MEAN_F32['mae']:.10f} |",
        "",
        "汇总指标计算方式（用户修正 #3）：先逐 H 复算组指标，再对四个 H "
        "**等权平均**与冻结值比较；**不是** 992 行窗口指标的直接均值"
        "（各 H 窗数不同）。逐窗身份门（G5）：每窗 row_pred_start=="
        f"arange({A.VAL_END}, {A.N_ROWS}-H+1)、date_pred_start==CSV "
        "dates[s]、scaled32==npz true 位级（dt 上 sklearn 1.7.2 实测位级"
        "一致），同时证明 scaler 与行对齐。",
        "",
        f"- scaler：mean_ot={scaler['mean_ot']!r}，"
        f"scale_ot={scaler['scale_ot']!r}（拟合行 [0,{A.N_TRAIN})，"
        f"sklearn {scaler['sklearn_version']}）",
        "",
        "## 选窗规则（用户修正 #2：固定 H=12）",
        "",
        f"- 只在 h{A.SEL_H} 的 {A.WINDOWS[A.SEL_H]} 个测试窗内选："
        f"MSE 最小 {A.K} 个=best、最大 {A.K} 个=worst；**不混合四种预测"
        "长度排名**，避免把预测长度带来的难度差异误认为时序模式差异",
        "- 并列时按窗口标识 `h{H:02d}_s{row:04d}` 升序",
        "- 指标在 scaled 空间逐窗计算（f64 自冻结 npz 值）；其余三个 H "
        "仅保留逐窗指标与组内排名（CSV 全 992 行）",
        "",
        f"## 入选 10 窗（h{A.SEL_H}）",
        "",
        "| 排名 | 窗口标识 | 预测期（日期） | MSE(scaled) | MAE(scaled) "
        "| h12 组内排名 |",
        "|---|---|---|---|---|---|",
    ]
    lines += [wrow(w) for w in best + worst]
    lines += [
        "",
        "## 窗口重叠情况",
        "",
        f"- 十窗两两 {ov['n_pairs']} 对中 {ov['pairs_with_pred_overlap']} "
        f"对预测区间有行重叠；最大预测段重叠 {ov['max_pred_overlap_rows']} 行",
        f"- 全跨（历史+预测）最小重叠 {ov['min_full_span_overlap_rows']} 行",
        f"- {ov['note']}",
        "",
        "## 图目（figures/，10 幅，dpi 150）",
        "",
    ]
    manifest = json.load(open(os.path.join(
        A.FIG_DIR, "plot_manifest.json"), encoding="utf-8"))
    for f in manifest["figures"]:
        lines.append(f"- `{f['figure']}`（{f['kind']}#{f['rank']}，"
                     f"{f['window_id']}，{f['date_pred_start']}）")
    lines += [
        f"- 图内文字语言：{manifest['labels']}"
        + (f"（字体 {', '.join(manifest['cjk_fonts'][:2])}）"
           if manifest["cjk_fonts"] else "（未找到 CJK 字体，英文回退）"),
        "",
        "配色（用户指定）：历史真值蓝、预测区间真值橙、模型预测红虚线、"
        "后续真值灰；黑虚线竖线=预测起止。全局面板：完整 1389 行 OT 原值 "
        "vs 日期，测试区浅灰，本窗历史段浅蓝/预测段浅橙高亮。局部面板："
        "预测由 npz pred 逆变换还原为原始单位，预测段真值直读 CSV。",
        "",
        "## 复现",
        "",
        "1. 依赖：diag_vts_env（python 3.10.20 / numpy 2.2.6 / sklearn "
        "1.7.2）+ matplotlib（本实验补装，版本见 logs/pip_freeze_after.txt）",
        "2. `bash run_vis_dt.sh all`：env_guard → verify_and_select.py "
        "（全部门+选窗）→ plot_windows.py（10 PNG）→ make_report.py → "
        "audit_pack.py（单审计包）",
        "3. 或离线复算：包内 series_date_ot.csv（1389 行 date+OT）、"
        "scaler.json、4 个冻结 npz 副本、health_window_metrics.csv 足以"
        "核查原始曲线、逆变换与窗口位置",
        "",
        "## 解释边界",
        "",
        "- 结果说明只记录**图中可见现象**；机制归因（周期捕捉、幅值、"
        "分布漂移等）一律为**待验证假设**，后续再讨论。",
        "- best/worst 集中于某些时段/幅值属于可见事实，仅记录不外推；"
        "单域（Health）单 H（h12）观察不能推广。",
        "- 指标与排名均在 scaled 空间；图还原为 OT 原始单位（% WEIGHTED "
        "ILI，见 CSV 表头），两种纵轴勿混读。",
        "",
        f"---\n*Generated by {A.EXP_ID} make_report.py; anchors: "
        f"EXP-021 token {A.BASE['protocol_token']}.*",
    ]
    out = os.path.join(os.path.dirname(A.RESULTS_DIR.rstrip("/")),
                       "README.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] wrote {out} ({len(lines)} lines)")
    print("REPORT OK")


if __name__ == "__main__":
    main()
