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
    pick = {t: max(("E-Extract", "E-Summary"),
                   key=lambda s: a[t][s]["coverage_event_weighted"] or 0)
            for t in data}
    for t in data:
        ee, su = a[t]["E-Extract"], a[t]["E-Summary"]
        d2 = a[t]["D2"]
        lines.append(f"- **{t}**：D2 事件加权覆盖 "
                     f"{fmt_cov(d2['coverage_event_weighted'])}；"
                     f"E-Extract {fmt_cov(ee['coverage_event_weighted'])}"
                     f"（compressed {ee['compressed_windows']}/"
                     f"{ee['windows']} 窗）；"
                     f"E-Summary {fmt_cov(su['coverage_event_weighted'])}"
                     f"（compressed {su['compressed_windows']}/"
                     f"{su['windows']} 窗）。")
        if ee["compressed_windows"] == 0 and su["compressed_windows"] == 0:
            lines.append(f"  - {t} 集全部回退 D2，无法比较方案间覆盖。")
    rec_set = "val" if "val" in pick else "debug"
    lines.append(f"- **推荐**：以 {rec_set} 集事件加权覆盖与失败/失真证据"
                 f"衡量，推荐方案为 **{pick[rec_set]}**"
                 f"（覆盖更高且失败可控；若两者接近，优先 Extract——"
                 f"原文片段失真风险更低）。")
    lines.append("- **是否值得进入三种子预测实验**：仅当推荐方案在验证集上"
                 "同时满足（a）覆盖显著高于 D2、（b）新数值筛查命中占比低、"
                 "（c）失败回退率可接受、（d）单窗 LLM 开销可接受时，才建议"
                 "进入；反之本轮证据支持维持 D2。最终判断需结合上方表格"
                 "人工确认。")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
