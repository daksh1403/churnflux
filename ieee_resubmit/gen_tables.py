#!/usr/bin/env python3
"""Generate LaTeX tables/figures from paper_results.json for the Results section."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'paper_results.json'


def load():
    return json.loads(RES.read_text())


def fmt(v, nd=3):
    if v is None:
        return '--'
    return f'{v:.{nd}f}'


def ci_str(c, nd=3, clamp=None):
    if c is None:
        return '--'
    if c.get('ci') is None:
        v = c['mean']
        if clamp:
            v = min(max(v, clamp[0]), clamp[1])
        return f"{v:.{nd}f}"
    m, lo, hi = c['mean'], c['ci'][0], c['ci'][1]
    if clamp:
        m, lo, hi = min(max(m, clamp[0]), clamp[1]), min(max(lo, clamp[0]), clamp[1]), min(max(hi, clamp[0]), clamp[1])
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def tab_churn_auc(data):
    """AUC of instability predictor vs churn rate, per dataset."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{AUC of the instability predictor (mean [95\% CI]) across churn rates.}")
    lines.append(r"\label{tab:auc}")
    lines.append(r"\begin{tabular}{l" + "c" * 6 + "}")
    lines.append(r"\toprule")
    lines.append("Dataset & " + " & ".join(f"{c}\\%" for c in ['5','10','20','30','40','50']) + r" \\")
    lines.append(r"\midrule")
    for dname in ['SIFT-1M', 'GloVe-200', 'Fashion-MNIST']:
        key = f'HNSW_{dname}'
        if key not in data:
            continue
        row = [dname]
        for c in ['5', '10', '20', '30', '40', '50']:
            row.append(ci_str(data[key]['summary'][c]['auc'], clamp=(0, 1)))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def tab_churn_recall(data):
    """Recall, p1, fail count under churn, and adaptive vs uniform recovery, per dataset."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Effect of churn on HNSW retrieval and ChurnFlux adaptive recovery at matched cost (mean [95\% CI]).}")
    lines.append(r"\label{tab:churn}")
    lines.append(r"\begin{tabular}{lllllll}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Churn & Avg Recall & p1 & Fail & ChurnFlux p1 & Uniform p1 \\")
    lines.append(r"\midrule")
    for dname in ['SIFT-1M', 'GloVe-200', 'Fashion-MNIST']:
        key = f'HNSW_{dname}'
        if key not in data:
            continue
        first = True
        for c in ['5', '10', '20', '30', '40', '50']:
            s = data[key]['summary'][c]
            ch = s['churn']
            ad = s['adapt']
            un = s['uniform']
            row = [dname if first else '', f"{c}\\%",
                   ci_str(ch['avg'], clamp=(0, 1)), ci_str(ch['p1'], clamp=(0, 1)),
                   ci_str(ch['n_fail'], 1),
                   ci_str(ad['p1'], clamp=(0, 1)), ci_str(un['p1'], clamp=(0, 1))]
            lines.append(" & ".join(row) + r" \\")
            first = False
        lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def tab_ivfpq(data):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{IVF-PQ recall under churn and quantization-error analysis.}")
    lines.append(r"\label{tab:ivfpq}")
    lines.append(r"\begin{tabular}{lllll}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Churn & Recall & QE (new) & QE (old) \\")
    lines.append(r"\midrule")
    for dname in ['SIFT-1M', 'GloVe-200', 'Fashion-MNIST']:
        key = f'IVFPQ_{dname}'
        if key not in data:
            continue
        first = True
        for c in ['5', '10', '20', '30', '40', '50']:
            s = data[key]['summary'][c]
            row = [dname if first else '', f"{c}\\%",
                   ci_str(s['recall']), ci_str(s['qe_churned'], 1), ci_str(s['qe_unchanged'], 1)]
            lines.append(" & ".join(row) + r" \\")
            first = False
        lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def tab_rag(data):
    rag = data.get('RAG_downstream', {})
    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Downstream retrieval quality: ChurnFlux adaptive vs matched-cost uniform recovery (10 seeds).}")
    lines.append(r"\label{tab:rag}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Recall@10 & p1 \\")
    lines.append(r"\midrule")
    if 'adaptive_recall' in rag.get('summary', {}):
        lines.append(r"ChurnFlux (adaptive) & " + ci_str(rag['summary']['adaptive_recall']) +
                     r" & " + ci_str(rag['summary']['adaptive_p1']) + r" \\")
        lines.append(r"Uniform (matched ef) & " + ci_str(rag['summary']['uniform_recall']) +
                     r" & " + ci_str(rag['summary']['uniform_p1']) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    data = load()
    out = []
    out.append("%% ===== GENERATED TABLES (do not edit by hand) =====")
    out.append(tab_churn_auc(data))
    out.append("")
    out.append(tab_churn_recall(data))
    out.append("")
    out.append(tab_ivfpq(data))
    out.append("")
    out.append(tab_rag(data))
    print("\n\n".join(out))


if __name__ == '__main__':
    main()
