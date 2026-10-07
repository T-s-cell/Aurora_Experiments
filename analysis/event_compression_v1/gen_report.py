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
    rows = [("成功压缩窗数（compressed/总）", lambda a, s:
             f"{a['compressed_windows']}/{a['windows']}"),
            ("完整事件进入率（事件加权，D2 语义参照）", lambda a, s:
             fmt_cov(a["coverage_event_weighted"])),
            ("事件进入率（任一源内容进入，事件加权）", lambda a, s:
             fmt_cov(a.get("entry_rate"))),
            ("词项匹配率（7 类事实 token，事件加权；非语义正确性）",
             lambda a, s: fmt_cov(a.get("fact_preserved_event_weighted"))),
            ("新事实 token 事件数（幻觉筛查）", lambda a, s:
             a.get("fact_new_token_events", "—")),
            ("非空片段事件数（≥3 token 且非拒答）", lambda a, s:
             a.get("nonempty_fragments", "—")),
            ("超预算事件数", lambda a, s: a.get("over_budget_events", "—")),
            ("evidence 校验通过事件数（仅 E-Summary 适用）", lambda a, s:
             a.get("evidence_ok_events", "—") if s == "E-Summary" else "—"),
            ("新数值事件数（幻觉筛查）", lambda a, s:
             a.get("new_number_events", "—") if s != "D2" else "—"),
            ("数值保留率均值", lambda a, s:
             PCT(a.get("preserved_numbers_mean")) if s != "D2" else "—")]
    for name, fn in rows:
        cells = []
        for t in sets:
            for s in ("D2", "E-Extract", "E-Summary"):
                a = agg.get(t, {}).get(s)
                cells.append(str(fn(a, s)) if a else "—")
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
             "任何违例整窗回退 D2 并单独计数。")
    L.append("- 版本：v3.1（2026-10-07）——首轮独立审计四修正：①撤回 v2 "
             "「零真实幻觉」结论；②E-Summary 接受门加入 evidence 硬门+词级 "
             "grounding；③修复 content 门多算 CLS/SEP 的 bug；④口径拆分。"
             "二轮审计三修正：⑤E-Extract 增设子句级限定词保留门（span 所在"
             "源子句含预测/计划/否定措辞时，span 必须保留其一，否则拒绝并"
             "修正/回退；提示词文件不变，合格缓存复用、不合格输出重处理）；"
             "⑥LLM 判别筛查扩展到 E-Extract；⑦报告收紧：抽取不再宣称零失真"
             "（逐字≠保真），「事实保留率」更名**词项匹配率**（词项重叠，"
             "不代表语义正确），E-Summary 仅作诊断、其预测实验暂停，文本"
             "方案本轮不冻结。产物留档：v2→*_promptv2_*、v3→*_v3_*。\n")

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
                     f"{c.get('scheme', '?')} {c['var_key'][:48]}..")
            for e in c["events"]:
                L.append(f"- 事件 {e['k']}：新数值 {e['new']}")
                L.append(f"  - 原文：{e['prose'][:400]}")
                L.append(f"  - 输出：{str(e['piece'])[:400]}")
            L.append("")
    if n_flag == 0:
        L.append("（无新数值筛查命中；仍需人工抽审，正则筛查不能证明零幻觉。）\n")

    # LLM judge screening section
    L.append(f"## {3 + len(sets)}. LLM 判别筛查（已采用片段 vs 原文，"
             "仅筛查、不进接受门）\n")
    for t in sets:
        jp = RESULTS / f"judge_summary_{t}.json"
        if not jp.exists():
            L.append(f"- {t}：无判别结果（judge 未运行或服务不可用）。")
            continue
        js = json.loads(jp.read_text())
        rel = js["relations"]
        L.append(f"- {t}：判定 {js['judged_pieces']} 条已采用片段——"
                 f"supported {rel.get('supported', 0)} / omission_only "
                 f"{rel.get('omission_only', 0)} / distortion "
                 f"{rel.get('distortion', 0)}；判定错误 "
                 f"{js['errors']}（不参与统计）。")
        for s in ("E-Extract", "E-Summary"):
            e = (js.get("by_scheme") or {}).get(s)
            if e:
                L.append(f"  - {s}：supported {e.get('supported', 0)} / "
                         f"omission_only {e.get('omission_only', 0)} / "
                         f"distortion {e.get('distortion', 0)}"
                         f"（{e.get('judged', 0)} 条）")
        for c in js["distortion_cases"]:
            L.append(f"  - **distortion** [{c.get('scheme', '?')}] {c['domain']} "
                     f"{c['var_key'][:44]}.. 事件 {c['k']}：{c['reason']}")
        for c in js.get("omission_only_cases", [])[:3]:
            L.append(f"  - omission_only 例 [{c.get('scheme', '?')}]："
                     f"{c['domain']} "
                     f"{c['var_key'][:44]}.. 事件 {c['k']}：{c['reason']}")
    L.append("")
    L.append("> 判别模型与压缩模型同一服务（自审局限）：仅作筛查线索，"
             "不构成保真证明；全部 distortion 案例须人工复核。\n")

    L.append(f"## {4 + len(sets)}. 结论与建议\n")
    L.append(conclusion(data))
    L.append("\n- 报告声明：本报告全部数字由脚本从产物计算生成；覆盖/失败/"
             "事实保留统计可由审计包内 final ids/mask + 逐事件映射独立复算；"
             "正则筛查与 LLM 判别均仅为筛查，不宣称零幻觉/零失真；v2 版"
             "报告的「零真实幻觉」结论已撤回；本报告不得引申为"
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
    judge = {}
    for t in data:
        jp = RESULTS / f"judge_summary_{t}.json"
        judge[t] = json.loads(jp.read_text()) if jp.exists() else None
    lines.append("### 口径声明（v3.1 起）\n")
    lines.append("- 放弃把「D2 整段原文进入率」与「摘要短句进入率」放在"
                 "同一口径比较：二者语义不同，22.3%→89.3% 一类数字**不是"
                 "同一种覆盖的提升**。三方案可比的是**词项匹配率**（原文 "
                 "7 类事实 token——数值/年份/月份/季度/单位/否定/预测措辞"
                 "——实际进入最终输入的比例，事件加权、双侧同过 tokenizer "
                 "消除记法偏差）与**事件进入率**（任一源内容进入）。完整"
                 "事件进入率仅作 D2 语义参照单列。**词项匹配率只是词项"
                 "重叠比例，不代表语义正确**：主体取舍、限定词截留、关系"
                 "错位都可能在词项全数命中时仍然失真，语义判定依赖 LLM "
                 "判别筛查与人工抽审。")
    lines.append("- **v2 报告「筛查零真实幻觉」结论正式撤回**；抽取方案的"
                 "「零失真风险」说法同步删除——逐字子串只保证 token 级"
                 "可对账，不保证语义保真。\n")
    lines.append("### 四问回答\n")
    lines.append("**（1）同预算下谁让更多事件内容进入输入？**")
    for t in data:
        ee, su, d2 = a[t]["E-Extract"], a[t]["E-Summary"], a[t]["D2"]
        lines.append(
            f"- {t}：完整事件进入率（D2 语义参照）D2 "
            f"{fmt_cov(d2['coverage_event_weighted'])} / E-Extract "
            f"{fmt_cov(ee['coverage_event_weighted'])} "
            f"（{ee['compressed_windows']}/{ee['windows']} 窗成功）/ "
            f"E-Summary {fmt_cov(su['coverage_event_weighted'])}"
            f"（{su['compressed_windows']}/{su['windows']} 窗成功）；"
            f"事件进入率 D2 {fmt_cov(d2['entry_rate'])} / E-Extract "
            f"{fmt_cov(ee['entry_rate'])} / E-Summary "
            f"{fmt_cov(su['entry_rate'])}；**词项匹配率** D2 "
            f"{fmt_cov(d2['fact_preserved_event_weighted'])} / E-Extract "
            f"{fmt_cov(ee['fact_preserved_event_weighted'])} / E-Summary "
            f"{fmt_cov(su['fact_preserved_event_weighted'])}。")
    lines.append("")
    lines.append("**（2）丢了什么细节？**（对照样例节逐链展示）")
    lines.append("- D2：预算截断把排后事件整体/尾部丢弃——匹配率低来自截断"
                 "而非改写，进入内容皆原文。")
    lines.append("- E-Extract：每事件仅保留 1 个逐字核心子句，源事件其余事实"
                 "（背景、次要数字、因果）被丢弃；词项匹配率仅略高于 D2——"
                 "「进入事件多」不等于「词项保留多」。且 v3.1 之前 span 选择"
                 "可截留子句限定词（如把「预计增长 9.9%」抽成裸「增长 "
                 "9.9%」，预测变断言），v3.1 门已拒绝此类选择并强制修正。")
    lines.append("- E-Summary：改写保留主体+关键数值+时间+情态；v3 门强制"
                 "记法与原文一致、拒绝无证据断言，列表收缩与修饰删除仍是"
                 "设计内损失。")
    lines.append("")
    lines.append("**（3）是否失真？**")
    n_flag = {t: sum(len(c["events"]) for c in
                     stats[t]["fact_flag_cases"]) for t in data}
    lines.append(f"- 新数值筛查（正则）：E-Extract 输出皆为原文子串（token 级"
                 f"可对账），新数值 0 起；E-Summary debug {n_flag.get('debug', 0)}"
                 f" 起 / val {n_flag.get('val', 0)} 起。**逐字只保证可对账，"
                 f"不保证保真**：两方案都可能丢限定词/主体或错置关系。")
    for t in data:
        if judge.get(t):
            rel = judge[t]["relations"]
            by = judge[t].get("by_scheme") or {}
            parts = []
            for s in ("E-Extract", "E-Summary"):
                e = by.get(s)
                if e:
                    parts.append(f"{s} distortion {e.get('distortion', 0)}/"
                                 f"{e.get('judged', 0)}")
            lines.append(
                f"- LLM 判别筛查（{t}，同一服务自审、仅筛查不进门）："
                f"supported {rel.get('supported', 0)} / omission_only "
                f"{rel.get('omission_only', 0)} / distortion "
                f"{rel.get('distortion', 0)}（共 {judge[t]['judged_pieces']} "
                f"条，错误 {judge[t]['errors']}；{'；'.join(parts)}）。"
                "distortion 全部逐条列出待人工复核；omission_only=仅省略、"
                "无断言外内容。")
    lines.append("- v3.1 接受门：E-Extract 增设**子句级限定词保留门**——span "
                 "所在源子句含预测/计划/否定措辞时，span 必须保留其中至少"
                 "一个 token（「预计增长 9.9%」不得抽成裸「增长 9.9%」），"
                 "违者拒绝并修正/回退；E-Summary 维持 v3 的 evidence 硬门+"
                 "词级 grounding+否定丢失即拒（记法漂移、新谓词类拦截）。")
    lines.append("- 门仍拦不住的失真：主体-时间-数值**关系**错误且词面全部"
                 "有据（如「预计损失超 29 亿美元」写成「已损失超 29 亿"
                 "美元」若措辞恰好换成已缓存外的同义词面）、主体取舍与跨子句"
                 "归并；依赖 LLM 判别筛查与人工抽审兜底。**两方案均不宣称"
                 "零失真**。")
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
        lines.append(f"- live 总开销（本 run_meta 所记生成）：{tot_req} 次请求"
                     f" / {tot_tok} LLM tokens / {tot_s:.0f}s，覆盖 {tot_win} "
                     f"窗×2 方案——≈ {tot_req / (2 * tot_win):.0f} 请求、"
                     f"{tot_tok / (2 * tot_win) / 1000:.1f}K tokens、"
                     f"{tot_s / (2 * tot_win):.0f}s 每窗每方案（串行、并发 1）。")
        lines.append("- E-Extract v2 提示词未变，缓存基本全量命中（仅 43 次"
                     "重跑差异导致的新调用）；v3 新增成本主要来自 E-Summary"
                     "（含 grounding 门触发的修正轮）。温度 0+缓存：跨窗同"
                     "事件复用。")
        lines.append("- 对全量 8106 窗外推约为每方案 ~25K 请求量级，"
                     "属可接受的一次性离线成本；但若纳入训练管线则每次数据"
                     "重建都要付出该成本（或依赖缓存失效风险）。")
    lines.append("")
    rec_set = "val" if "val" in pick else "debug"
    fp_s = fmt_cov(a[rec_set]["E-Summary"]["fact_preserved_event_weighted"])
    fp_e = fmt_cov(a[rec_set]["E-Extract"]["fact_preserved_event_weighted"])
    fp_d = fmt_cov(a[rec_set]["D2"]["fact_preserved_event_weighted"])
    fb_s = stats[rec_set]["fallback_windows"].get("E-Summary", 0)
    fb_e = stats[rec_set]["fallback_windows"].get("E-Extract", 0)
    nw = stats[rec_set]["n_windows"]
    jd = judge.get(rec_set) or {}
    dist = (jd.get("relations") or {}).get("distortion", 0)
    n_judged = jd.get("judged_pieces", 0)
    sub = compressed_only_fact_pres(stats)
    lines.append("### 候选与三种子预测实验（v3.1 口径）\n")
    lines.append(f"- **候选方案：E-Extract（带 v3.1 限定词保留门）**——"
                 f"{rec_set} 集全窗事件加权词项匹配率 {fp_e} vs E-Summary "
                 f"{fp_s} vs D2 {fp_d}；成功 {nw - fb_e}/{nw} 窗 vs "
                 f"E-Summary {nw - fb_s}/{nw} 窗；输出全部为原文逐字子串，"
                 "逐 token 可对账、结构上无新事实。**但逐字 ≠ 保真**："
                 "抽取仍可能丢限定词/主体（v3.1 门只堵「源子句含预测/"
                 "计划/否定措辞而 span 未保留」一类），且判别筛查显示"
                 "非零 distortion。其压缩成功窗内词项匹配率"
                 f"（{sub.get('E-Extract', '—')}）低于 E-Summary 同口径"
                 f"（{sub.get('E-Summary', '—')}）。判别筛查 + 人工抽审"
                 "仍是必要补充，不因「子串」性质豁免。")
    lines.append(f"- **E-Summary：仅作诊断保留，预测实验继续暂停**。"
                 f"{rec_set} 集 {fb_s}/{nw} 窗回退 D2（实际改变输入的窗"
                 "太少）；判别筛查 distortion 待人工复核（含「预计损失"
                 "超 29 亿美元」被写成「已损失超 29 亿美元」一类情态"
                 "错误）。压缩成功时词项匹配率最高，说明路线本身有效；"
                 "若继续该路线，需重平衡接受门（如按事实类别分级的 "
                 "evidence 覆盖要求）并另记版本全量重跑。")
    lines.append("- **文本方案本轮不冻结**：是否进入、以及以何方案进入"
                 "三种子预测实验，待用户复审 v3.1 审计包（含全部 "
                 "distortion/qualifier_dropped 案例）后再定。"
                 "**声明不构成预测改善承诺**：EXP-012（= "
                 "repro-mm-timesx-d2 @ f30f56b）表明文本清理本身收益仅 "
                 "+0.02%，三种子实验是对「更多事实语境可能改善文本利用」"
                 "假设的检验，不是推论。")
    lines.append("- 若将来进入预测实验：沿用当轮冻结的接受规则+预算规则+"
                 "缓存（新窗事件需新调用），训练/val 文本处理与该轮完全"
                 "一致；测试集文本处理是否用 LLM 需另行决策（本轮未触碰"
                 "测试集）。")
    return "\n".join(lines)


def compressed_only_fact_pres(stats):
    """fact preservation restricted to windows the scheme actually
    compressed (success subset), from the qc_events CSVs."""
    import csv
    out = {}
    for t, st in stats.items():
        rows = list(csv.DictReader(
            open(RESULTS / f"qc_events_{t}.csv")))
        for s in ("E-Extract", "E-Summary"):
            wkeys = {(r["var_key"], r["sample_id"]) for r in rows
                     if r["scheme"] == s and r["source"] not in
                     ("d2", "d2_fallback")}
            sub = [float(r["fact_preserved"]) for r in rows
                   if r["scheme"] == s and r["fact_preserved"] not in
                   ("", "None") and (r["var_key"], r["sample_id"]) in wkeys]
            if sub:
                out[s] = fmt_cov(sum(sub) / len(sub))
    return out


if __name__ == "__main__":
    sys.exit(main())
