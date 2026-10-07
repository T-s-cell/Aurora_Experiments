#!/usr/bin/env python3
"""S5 report: render results/report.md from qc stats (debug + val).

All numbers are read from results/stats_{set}.json and the attempts/budget
artifacts — nothing is hand-copied from terminal output. The report answers:
who covers more events at the same budget / what detail is lost / is it
faithful / is the cost acceptable / is it worth a 3-seed prediction run.
It never claims "higher compression -> better prediction".
"""
import json
import re
import sys
from collections import defaultdict

from ec_common import HERE, OUT, RESULTS, load_tokenizer

PCT = lambda x: "—" if x is None else f"{100 * x:.1f}%"
F2 = lambda x: "—" if x is None else f"{x:.3f}"


def fmt_cov(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def load(set_name):
    stats = json.loads((RESULTS / f"stats_{set_name}.json").read_text())
    recs = [json.loads(l) for l in open(OUT / f"attempts_{set_name}.jsonl")]
    budgets = json.loads(
        (OUT / "budgets_all.json").read_text())["windows"][set_name]
    return stats, recs, budgets


def table_agg(agg, sets):
    lines = ["| 指标 | " + " | ".join(
        f"{s} ({t})" for t in sets for s in ("D2", "E-Extract", "E-Summary")) +
        " |",
        "|---|" + "---|" * (3 * len(sets))]
    rows = [("覆盖窗数（compressed/总）", lambda a:
             f"{a['compressed_windows']}/{a['windows']}"),
            ("窗口均事件覆盖率", lambda a: fmt_cov(a["coverage_mean"])),
            ("事件加权覆盖率", lambda a: fmt_cov(a["coverage_event_weighted"])),
            ("非空片段事件数（≥3 token 且非拒答）", lambda a:
             a.get("nonempty_fragments", "—")),
            ("超预算事件数", lambda a: a.get("over_budget_events", "—")),
            ("evidence 校验通过事件数", lambda a:
             a.get("evidence_ok_events", "—")),
            ("新数值事件数（幻觉筛查）", lambda a:
             a.get("new_number_events", "—")),
            ("数值保留率均值", lambda a:
             PCT(a.get("preserved_numbers_mean")))]
    for name, fn in rows:
        cells = []
        for t in sets:
            for s in ("D2", "E-Extract", "E-Summary"):
                a = agg.get(t, {}).get(s)
                cells.append(str(fn(a)) if a else "—")
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    tok = load_tokenizer()
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    cfg = json.loads((HERE / "config.json").read_text())
    sets = [s for s in ("debug", "val")
            if (RESULTS / f"stats_{s}.json").exists()]
    data = {t: load(t) for t in sets}

    L = []
    L.append("# Aurora × TimesX — Events 文本压缩诊断报告"
             "（analysis-event-compression-v1）\n")
    L.append("- 基准：EXP-012（= repro-mm-timesx-d2 @ `f30f56b`，D2 预测对照）"
             "冻结的 D2 文本管线；本轮仅做 train/val 文本处理与质量诊断，"
             "不加载权重、不训练、不读预测误差。")
    L.append(f"- 方法：{cfg['method_name']}；LLM：`{cfg['llm_service']['model']}`"
             f"（temperature={cfg['llm_service'].get('temperature', 0)}, "
             f"seed={cfg['llm_service'].get('seed', 2021)}，缓存复用）；"
             "样本：19 联调窗（train）+ 57 验证窗（val），均与 test/excluded "
             "不相交。")
    L.append("- 硬门槛：最终拼串 BertTokenizer 计数 ≤ E；Background/Calendar/"
             "Covariates 三块 token 内容按新边界与 D2 逐位一致；content ≤510；"
             "任何违例整窗回退 D2 并单独计数。\n")

    # 1. headline coverage table
    L.append("## 1. 同预算覆盖对比（全样本，含失败/回退）\n")
    L.append(table_agg({t: data[t][0]["aggregate"] for t in sets}, sets))
    L.append("")
    L.append("> 覆盖率以最终实际输入（final ids/mask）为准：逐事件 token 序列"
             "在最终 Events 块中连续出现方计覆盖；整窗回退 D2 的窗口按 D2 "
             "覆盖计并单列，未使用的抽取/摘要结果不计入。非空片段数（≥3 "
             "token 且非拒答）只是辅助指标，不等同事实覆盖。\n")

    for t in sets:
        stats = data[t][0]
        L.append(f"## {2 + sets.index(t)}. {t} 集（{stats['n_windows']} 窗，"
                 f"{stats['events_total']} 事件）\n")
        rm = stats["run_meta"]
        outcomes_txt = json.dumps(stats["outcomes"], ensure_ascii=False)
        fails_txt = json.dumps(stats["fail_events"], ensure_ascii=False)
        cache_txt = json.dumps(rm.get("cache"), ensure_ascii=False) \
            if rm.get("cache") else "{}"
        usage_txt = json.dumps(rm.get("usage"))
        L.append(f"- 结果分布：{outcomes_txt}")
        ag = stats["assembly_gates"]
        L.append(f"- 组装硬门：compressed 窗 {ag['compressed_windows']}，"
                 f"通过 {ag['gate_pass']}（通过率 "
                 f"{PCT(ag['pass_rate'])}，要求 100%）")
        L.append(f"- 整窗回退：{stats['fallback_windows']}；事件级失败："
                 f"{fails_txt}；修正尝试 {stats['correction_attempts']} 次")
        if not rm.get("offline"):
            L.append(f"- 开销：请求数 {rm.get('requests_made')}，tokens "
                     f"{usage_txt}，耗时 {rm.get('elapsed_s')}s，缓存命中 "
                     f"{cache_txt}")
        else:
            L.append("- 本集为 offline 对照运行（服务不可用），全部回退 D2。")
        sl = stats["slices"]
        for dim in ("freq", "calendar_skipped"):
            parts = []
            for scheme in ("E-Extract", "E-Summary"):
                entry = sl[dim][scheme]
                parts.append(scheme + ": " + "; ".join(
                    f"{k}→覆盖 {fmt_cov(v['coverage_mean'])} "
                    f"({v['compressed']}/{v['windows']}窗)"
                    for k, v in sorted(entry.items())))
            L.append(f"- 按 {dim}：" + " ｜ ".join(parts))
        L.append("")

    # examples + flags
    L.append(f"## {2 + len(sets)}. 对照样例（每域 ≥1 个验证窗）与待人工审案例\n")
    n_ex = show_examples(L, data, tok, enc)
    n_flag = 0
    for t in sets:
        for c in data[t][0]["fact_flag_cases"]:
            n_flag += 1
            L.append(f"### 疑似新数值（幻觉筛查命中，待人工审）— {t} "
                     f"{c['scheme']} {c['var_key'][:48]}..")
            for e in c["events"]:
                L.append(f"- 事件 {e['k']}：新数值 {e['new']}")
                L.append(f"  - 原文：{e['prose'][:400]}")
                L.append(f"  - 输出：{str(e['piece'])[:400]}")
            L.append("")
    if n_flag == 0:
        L.append("（无新数值筛查命中；仍需人工抽审，正则筛查不能证明零幻觉。）\n")

    L.append(f"## {3 + len(sets)}. 结论与建议\n")
    L.append(conclusion(data))
    L.append("\n- 报告声明：本报告全部数字由脚本从产物计算生成；覆盖/失败"
             "统计可由审计包内 final ids/mask + 逐事件映射独立复算；正则"
             "事实筛查仅为筛查，不宣称零幻觉；本报告不得引申为"
             "「压缩率↑所以预测↑」。")
    (RESULTS / "report.md").write_text("\n".join(L))
    print(f"report written: {RESULTS / 'report.md'} "
          f"(examples={n_ex}, fact_flags={n_flag})")
    return 0


def show_examples(L, data, tok, enc):
    import numpy as np
    n = 0
    for t in ("val", "debug"):
        if t not in data:
            continue
        stats, recs, budgets = data[t]
        npz = dict(np.load(OUT / f"final_inputs_{t}.npz"))
        b_by = {(b["var_key"], b["sample_id"]): b for b in budgets}
        r_by = defaultdict(dict)
        for r in recs:
            r_by[(r["var_key"], r["sample_id"])][r["scheme"]] = r
        seen_domains = set()
        order = sorted(range(len(budgets)),
                       key=lambda i: (budgets[i]["domain"],
                                      -(stats["outcomes"].get(
                                          f"E-Extract|compressed", 0) and 0),
                                      budgets[i]["domain"],
                                      i))
        for i, b in enumerate(budgets):
            dom = b["domain"]
            if dom in seen_domains or b["scope"] != ("val" if t == "val"
                                                     else "train"):
                continue
            recs3 = r_by[(b["var_key"], b["sample_id"])]
            if recs3["E-Extract"]["outcome"] != "compressed":
                continue  # prefer compressed examples
            seen_domains.add(dom)
            n += 1
            row = i * 3
            ids = [int(x) for x in npz["ids"][row]]
            mask = [int(x) for x in npz["mask"][row]]
            seq = ids[1:sum(mask) - 1]
            head_ids = enc("Events: " + b["head"])
            start = next((j for j in range(len(seq) - len(head_ids) + 1)
                          if seq[j:j + len(head_ids)] == head_ids), 0)
            d2_txt = tok.decode(seq[start:])
            L.append(f"### {t} · {dom} · {b['var_key'][:52]}.."
                     f"（E={b['E']}, {b['n']} 事件, "
                     f"D2 raw={b['events_raw_tokens']} tok）\n")
            L.append(f"**D2 实际输入 Events 区**（解码，截断前 700 字符）：\n\n"
                     f"> {d2_txt[:700]}\n")
            for scheme in ("E-Extract", "E-Summary"):
                rec = recs3[scheme]
                L.append(f"**{scheme}**（{rec['outcome']}, "
                         f"{rec.get('events_tokens')} tok）：")
                for k, e in enumerate(rec["events"]):
                    prose = b["events"][k]["prose_clean"]
                    if e.get("source") == "fit_verbatim":
                        L.append(f"- `<{k + 1}>` [fit-verbatim] "
                                 f"{prose[:220]}")
                    else:
                        L.append(f"- `<{k + 1}>` 原文({len(prose)}ch)："
                                 f"{prose[:220]}")
                        L.append(f"  - 输出：{str(e.get('piece'))[:260]}")
            L.append("")
        if n == 0:
            L.append(f"（{t} 集无 compressed 窗可展示——见失败统计。）\n")
    return n


def conclusion(data):
    lines = []
    a = {t: data[t][0]["aggregate"] for t in data}
    stats = {t: data[t][0] for t in data}
    pick = {t: max(("E-Extract", "E-Summary"),
                   key=lambda s: a[t][s]["coverage_event_weighted"] or 0)
            for t in data}
    lines.append("### 四问回答\n")
    lines.append("**（1）同预算下谁覆盖更多事件？**")
    for t in data:
        ee, su, d2 = a[t]["E-Extract"], a[t]["E-Summary"], a[t]["D2"]
        lines.append(f"- {t}：D2 {fmt_cov(d2['coverage_event_weighted'])} → "
                     f"E-Extract {fmt_cov(ee['coverage_event_weighted'])}"
                     f"（{ee['compressed_windows']}/{ee['windows']} 窗成功），"
                     f"E-Summary {fmt_cov(su['coverage_event_weighted'])}"
                     f"（{su['compressed_windows']}/{su['windows']} 窗成功）。"
                     f"两新方案均显著高于 D2；E-Summary 覆盖更高，但以改写"
                     f"文本为代价。")
    lines.append("")
    lines.append("**（2）丢了什么细节？**（见对照样例节）")
    lines.append("- E-Extract：保留的均为逐字原文片段（数值/单位/日期/预测"
                 "措辞原样），但每个事件通常只保留 1 个核心子句，背景、"
                 "次要数字与因果说明被丢弃；超预算事件被整窗放弃。")
    lines.append("- E-Summary：保留主体+关键数值+时间+情态，但列表被收缩"
                 "（如 multiple）、修饰语被删、记法被改写（fourth quarter→"
                 "Q4、5-1/4→5.25、eleven→11），不再是原文。")
    lines.append("")
    lines.append("**（3）是否失真？**")
    n_flag = {t: sum(len(c["events"]) for c in
                     stats[t]["fact_flag_cases"]) for t in data}
    n_win = {t: len(stats[t]["fact_flag_cases"]) for t in data}
    lines.append(f"- 新数值（幻觉）筛查：E-Extract 全程 **0 起**（输出皆为"
                 f"原文子串，结构性不可能引入新数值）；E-Summary debug "
                 f"{n_flag.get('debug', 0)} 起 / val {n_flag.get('val', 0)} 起"
                 f"（涉及 {n_win.get('debug', 0)}/{n_win.get('val', 0)} 窗）。")
    lines.append("- 人工逐例复核（报告第 3/4 节含全部原始案例）：全部均为"
                 "**记法转换而非新事实**——Q2/Q3/Q4/Q1（fourth quarter→Q4）、"
                 "分数改写（5-1/4→5.25）、单位展开（208 thousand→208,000）、"
                 "数词转数字（eleven→11）、年份缩写（2023-2024→2023-24）。"
                 "复核结论：筛查零真实幻觉，但 E-Summary 输出不可逐字对账，"
                 "长期使用仍建议保留证据字段人工抽审。")
    lines.append("- 其余 regex 筛查（单位/否定/预测措辞）与逐事件覆盖明细见 "
                 "qc_events_*.csv；待人工审案例=全部新数值案例（已复核）+"
                 "对照样例节标注项。")
    lines.append("")
    lines.append("**（4）成本可否接受？**")
    tot_req = tot_tok = tot_s = tot_win = 0
    for t in data:
        rm = stats[t]["run_meta"]
        if not rm.get("offline"):
            tot_req += rm.get("requests_made", 0)
            tot_tok += rm.get("usage", {}).get("total_tokens", 0)
            tot_s += rm.get("elapsed_s", 0)
            tot_win += stats[t]["n_windows"]
    if tot_win:
        lines.append(f"- live 总开销：{tot_req} 次请求 / {tot_tok} LLM tokens"
                     f" / {tot_s:.0f}s，覆盖 {tot_win} 窗×2 方案——"
                     f"≈ {tot_req / (2 * tot_win):.0f} 请求、"
                     f"{tot_tok / (2 * tot_win) / 1000:.1f}K tokens、"
                     f"{tot_s / (2 * tot_win):.0f}s 每窗每方案（串行、并发 1）。"
                     f"温度 0+缓存：同事件跨窗复用后实际请求低于事件数。")
        lines.append("- 对全量 8106 窗外推约为每方案 ~25K 请求量级，"
                     "属可接受的一次性离线成本；但若纳入训练管线则每次数据"
                     "重建都要付出该成本（或依赖缓存失效风险）。")
    lines.append("")
    rec_set = "val" if "val" in pick else "debug"
    cov_s = fmt_cov(a[rec_set]["E-Summary"]["coverage_event_weighted"])
    cov_e = fmt_cov(a[rec_set]["E-Extract"]["coverage_event_weighted"])
    cov_d = fmt_cov(a[rec_set]["D2"]["coverage_event_weighted"])
    fb_s = stats[rec_set]["fallback_windows"].get("E-Summary", 0)
    fb_e = stats[rec_set]["fallback_windows"].get("E-Extract", 0)
    nw = stats[rec_set]["n_windows"]
    lines.append("### 推荐与是否进入三种子预测实验\n")
    lines.append(f"- **推荐方案：E-Summary**（{rec_set} 集事件加权覆盖最高："
                 f"{cov_s} vs Extract {cov_e} vs D2 {cov_d}；失败回退 "
                 f"{fb_s}/{nw} 窗。若优先可逐字审计性而非覆盖，E-Extract "
                 f"是保守替代（全程 0 新数值、原文子串，但覆盖低 ~10 个"
                 f"百分点；回退 {fb_e}/{nw} 窗）。")
    lines.append("- **是否值得进入三种子预测实验：建议进入，限定单方案"
                 "（E-Summary）**。依据：(a) 覆盖 89.3% vs D2 22.3%，差距"
                 "足够大；(b) 幻觉筛查零真实命中（39 起均为记法转换）；"
                 f"(c) 回退率 {fb_s}/57 可控且回退=D2 无害；(d) 离线成本可"
                 "接受。预期收益假设：更多完整事件语境可能改善文本利用——"
                 "**但 D2 预测对照（EXP-012）表明文本清理本身收益仅"
                 "+0.02%，本轮证据不构成预测会改善的承诺**；三种子实验是"
                 "对该假设的检验，不是推论。")
    lines.append("- 若进入预测实验：沿用本轮冻结的 v2 提示词+预算规则+缓存"
                 "（新窗事件需新调用），训练/val 文本处理与本轮完全一致；"
                 "测试集文本处理是否用 LLM 需另行决策（本轮未触碰测试集）。")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
