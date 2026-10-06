#!/usr/bin/env python3
"""Generate report summary tables + key numbers DIRECTLY from results CSVs
(audit 2026-10-07: report numbers must be CSV-generated, never hand-copied;
the previous report misread alloc_p90 as alloc_median for test Events).

Writes results/report_tables.md (markdown snippets for report.md sections
2.1-2.3) and results/report_numbers.json (values cited in the narrative).
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SCHEMES = ("D0", "D1", "D2")
FIELDS = ("Background", "Events", "Calendar", "Covariates")


def read(name):
    return list(csv.DictReader(open(RESULTS / name, encoding="utf-8")))


def f(x):
    return float(x)


def i(x):
    return int(float(x))


def fmt_med(x):
    return f"{f(x):.0f}"


def main():
    overall = read("stats_overall.csv")
    fields = read("stats_fields.csv")
    ratios = read("stats_d2_ratios.csv")

    ov = {(r["scheme"], r["scope"]): r for r in overall}
    fl = {(r["scheme"], r["scope"], r["field"]): r for r in fields}
    rt = {(r["scope"], r["field"]): r for r in ratios}

    lines = []
    lines.append("### 2.1 总体（stats_overall.csv，脚本生成）\n")
    lines.append("| 指标 | D0 | D1 | D2 |")
    lines.append("|---|---|---|---|")
    for scope, tag in (("full", "全量"), ("test", "测试")):
        r = {s: ov[(s, scope)] for s in SCHEMES}
        lines.append(f"| content 中位（{tag}） | " + " | ".join(
            fmt_med(r[s]["content_median"]) for s in SCHEMES) + " |")
        lines.append(f"| idle 中位（{tag}） | " + " | ".join(
            fmt_med(r[s]["idle_median"]) for s in SCHEMES) + " |")
        lines.append(f"| 仍截断窗比例（{tag}） | " + " | ".join(
            f"{f(r[s]['still_truncated_frac']) * 100:.1f}%"
            for s in SCHEMES) + " |")
        lines.append(f"| Events 完整 / 总事件（{tag}） | " + " | ".join(
            f"{i(r[s]['ev_complete'])}/{i(r[s]['ev_total'])} "
            f"({f(r[s]['ev_complete_frac']) * 100:.1f}%)"
            for s in SCHEMES) + " |")
        lines.append(f"| Covariates 完整条目（{tag}） | " + " | ".join(
            f"{i(r[s]['cv_complete'])}/{i(r[s]['cv_entries'])}"
            for s in SCHEMES) + " |")
    lines.append("")

    lines.append("### 2.2 逐字段 alloc 中位（stats_fields.csv，脚本生成）\n")
    lines.append("中位含 skipped 窗（0 值），与 CSV 口径一致。")
    lines.append("| 字段 | scope | raw 中位 | D0 alloc | D1 alloc | D2 清理后 | D2 alloc |")
    lines.append("|---|---|---|---|---|---|---|")
    for field in FIELDS:
        for scope in ("full", "test"):
            def g(s, col):
                v = fl[(s, scope, field)][col]
                return fmt_med(v) if v else "0"
            lines.append(
                f"| {field} | {scope} | {g('D0', 'raw_median')} | "
                f"{g('D0', 'alloc_median')} | {g('D1', 'alloc_median')} | "
                f"{g('D2', 'cleaned_median')} | {g('D2', 'alloc_median')} |")
    lines.append("")

    lines.append("### 2.3 D2 两指标分开（stats_d2_ratios.csv，脚本生成）\n")
    lines.append("| 字段 | scope | 清理缩短比例（逐窗中位） | 清理后保留比例（逐窗中位） |")
    lines.append("|---|---|---|---|")
    for field in FIELDS:
        for scope in ("full", "test"):
            r = rt[(scope, field)]
            lines.append(
                f"| {field} | {scope} | {f(r['shortening_median_w']):.3f} | "
                f"{f(r['retention_median_w']):.3f} |")
    lines.append("")

    # by-sum D2 retentions for the narrative (kept = entered input)
    def by_sum_ret(scope, field):
        r = fl[("D2", scope, field)]
        return f(r["alloc_sum"]) / f(r["cleaned_sum"])

    nums = {
        "events_alloc_median_test": {s: f(fl[(s, "test", "Events")]["alloc_median"])
                                     for s in SCHEMES},
        "events_alloc_median_full": {s: f(fl[(s, "full", "Events")]["alloc_median"])
                                     for s in SCHEMES},
        "ev_rel_gain_full_d1_vs_d0": (f(ov[("D1", "full")]["ev_complete"])
                                      / f(ov[("D0", "full")]["ev_complete"]) - 1),
        "ev_rel_gain_full_d2_vs_d1": (f(ov[("D2", "full")]["ev_complete"])
                                      / f(ov[("D1", "full")]["ev_complete"]) - 1),
        "ev_rel_gain_test_d2_vs_d1": (f(ov[("D2", "test")]["ev_complete"])
                                      / f(ov[("D1", "test")]["ev_complete"]) - 1),
        "cv_complete_gain_full_d2_vs_d1": (i(ov[("D2", "full")]["cv_complete"])
                                           - i(ov[("D1", "full")]["cv_complete"])),
        "cv_complete_gain_test_d2_vs_d1": (i(ov[("D2", "test")]["cv_complete"])
                                           - i(ov[("D1", "test")]["cv_complete"])),
        "d2_events_retention_by_sum_test": by_sum_ret("test", "Events"),
        "d2_cov_retention_by_sum_test": by_sum_ret("test", "Covariates"),
        "d2_events_absent_full": i(ov[("D2", "full")]["ev_absent"]),
        "d2_events_shortening_median_w": {
            scope: f(rt[(scope, "Events")]["shortening_median_w"])
            for scope in ("full", "test")},
    }
    (RESULTS / "report_numbers.json").write_text(json.dumps(nums, indent=2))

    out = RESULTS / "report_tables.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\n[numbers] " + json.dumps(nums, indent=1))


if __name__ == "__main__":
    main()
