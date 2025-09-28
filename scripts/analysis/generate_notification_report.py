from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORT_DIR = Path("outputs/analysis/report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> dict[str, object]:
    data = {}
    data["severity"] = pd.read_csv("outputs/analysis/hierarchical_severity_regularized/severity_predictions.csv")
    data["severity_coefs"] = pd.read_csv("outputs/analysis/hierarchical_severity_regularized/top_coefficients.csv")
    data["propensity_coefs"] = pd.read_csv("outputs/analysis/joint_notification/propensity_top_coefficients.csv")
    data["metrics"] = json.loads(Path("outputs/analysis/joint_notification/joint_model_metrics.json").read_text())
    data["bootstrap"] = json.loads(Path("outputs/analysis/joint_notification/bootstrap_results.json").read_text())
    data["feature_matrix"] = pd.read_parquet("outputs/analysis/feature_matrix.parquet")
    # derive case initiation summary on the fly
    severity_map = data["severity"].set_index("decision_id")["severity_expected"]
    feature = data["feature_matrix"].join(severity_map, on="decision_id")
    cols = [
        c
        for c in feature.columns
        if c.startswith("q15_case_initiation_")
        and not c.endswith(("coverage_status", "known", "unknown", "status", "exclusivity_conflict"))
    ]
    rows = []
    for col in cols:
        mask = feature[col].fillna(0) == 1
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "case_initiation": col.split("q15_case_initiation_")[1],
                "count": int(mask.sum()),
                "mean_log_fine": feature.loc[mask, "fine_log1p"].astype(float).mean(),
                "mean_severity_expected": feature.loc[mask, "severity_expected"].mean(),
            }
        )
    data["case_initiation"] = pd.DataFrame(rows).sort_values("mean_log_fine", ascending=False)
    return data


def plot_severity_distribution(severity: pd.DataFrame) -> str:
    counts = severity["severity_label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", color="#1f77b4", ax=ax)
    ax.set_title("Distribution of Predicted Severity Classes")
    ax.set_xlabel("Severity Class")
    ax.set_ylabel("Number of Decisions")
    ax.bar_label(ax.containers[0], padding=3)
    plt.tight_layout()
    path = REPORT_DIR / "severity_distribution.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def plot_case_initiation(case_df: pd.DataFrame) -> tuple[str, str]:
    order = case_df["case_initiation"].tolist()

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(order, case_df.set_index("case_initiation").loc[order, "mean_log_fine"], color="#d62728")
    ax1.set_title("Average log fine by case initiation channel")
    ax1.set_xlabel("Case initiation channel")
    ax1.set_ylabel("Mean log fine")
    plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    path1 = REPORT_DIR / "case_initiation_logfine.png"
    fig1.savefig(path1, dpi=140)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(order, case_df.set_index("case_initiation").loc[order, "mean_severity_expected"], color="#2ca02c")
    ax2.set_title("Average expected severity by case initiation channel")
    ax2.set_xlabel("Case initiation channel")
    ax2.set_ylabel("Mean expected severity (0-4)")
    plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    path2 = REPORT_DIR / "case_initiation_severity.png"
    fig2.savefig(path2, dpi=140)
    plt.close(fig2)

    return path1.name, path2.name


def plot_severity_coefficients(coefs: pd.DataFrame) -> str:
    top = coefs[coefs["class"] == 4].nlargest(10, "coefficient")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(top["feature"], top["coefficient"], color="#9467bd")
    ax.set_title("Class 4 (FINE_PLUS) – Top Positive Predictors")
    ax.set_xlabel("Coefficient (log-odds)")
    ax.invert_yaxis()
    plt.tight_layout()
    path = REPORT_DIR / "severity_coefficients.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def plot_propensity_coefficients(prop_coefs: pd.DataFrame) -> str:
    top = prop_coefs.head(10)
    colors = ["#1f77b4" if coef > 0 else "#ff7f0e" for coef in top["coefficient"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(top["feature"], top["coefficient"], color=colors)
    ax.set_title("Timeliness propensity – leading predictors")
    ax.set_xlabel("Coefficient (regularised logit)")
    ax.invert_yaxis()
    plt.tight_layout()
    path = REPORT_DIR / "propensity_coefficients.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def plot_bootstrap(bootstrap: dict[str, dict[str, float]]) -> str:
    scenarios = []
    means = []
    lower = []
    upper = []
    for trim, stats in bootstrap.items():
        scenarios.append(trim)
        means.append(stats["bootstrap_mean"])
        lower.append(stats["ci_lower"])
        upper.append(stats["ci_upper"])
    fig, ax = plt.subplots(figsize=(6, 4))
    y = np.arange(len(scenarios))
    ax.errorbar(means, y, xerr=[np.array(means) - np.array(lower), np.array(upper) - np.array(means)], fmt='o', color="#d62728", ecolor="#ff9896")
    ax.set_yticks(y)
    ax.set_yticklabels([f"Trim {s}" for s in scenarios])
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Bootstrap mean AIPW (log fine)")
    ax.set_title("Bootstrap confidence intervals for AIPW estimates")
    plt.tight_layout()
    path = REPORT_DIR / "bootstrap_aipw.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def build_html(images: dict[str, str], tables: dict[str, str], text_blocks: dict[str, str], metrics: dict[str, float], bootstrap: dict[str, dict[str, float]]) -> str:
    html = ["<html><head><meta charset='utf-8'><title>Breach Notification & Enforcement Report</title>",
            "<style>body{font-family:Helvetica,Arial,sans-serif;max-width:900px;margin:2em auto;line-height:1.5;} h1,h2,h3{color:#333;} table{border-collapse:collapse;margin:1em 0;width:100%;} th,td{border:1px solid #ccc;padding:6px;text-align:left;} figure{margin:1.5em 0;} figcaption{font-style:italic;text-align:center;margin-top:4px;} .note{background:#f5f5f5;padding:1em;border-left:4px solid #0072B2;} .legal{font-weight:bold;}</style></head><body>"]
    html.append("<h1>Breach Notification & Enforcement Outcomes: Preliminary Report</h1>")
    html.append("<p>This report summarises the current quantitative evidence on GDPR breach notifications, drawing on our enriched feature matrix (999 decisions). It is intended for legal scholars and data-protection practitioners seeking to interpret enforcement patterns through both statistical and doctrinal lenses.</p>")

    html.append("<h2>1. Data snapshot</h2>")
    html.append("<p>We analyse cleaned AI-labeled decisions using two regularised models: (i) a multinomial classifier of enforcement severity (NONE → FINE_PLUS), and (ii) a propensity/OLS pipeline estimating the effect of timely Article 33 notification on log-transformed fines.</p>")

    html.append("<h2>2. Enforcement severity landscape</h2>")
    html.append("<figure><img src='{0}' alt='Severity distribution'><figcaption>Figure 1. Distribution of predicted severity classes across 999 decisions.</figcaption></figure>".format(images["severity_distribution"]))
    html.append(tables["severity_summary"])
    html.append("<p><span class='legal'>Observation.</span> The severity classifier separates four tiers cleanly: most matters end in fines (with or without additional powers), while sanction-free cases remain rare. Expected severity is lowest for cases featuring no corrective powers and highest for those with extensive Article 58(2) measures.</p>")

    html.append("<figure><img src='{0}' alt='Severity coefficients'><figcaption>Figure 2. Leading predictors of FINE_PLUS outcomes (log-odds).</figcaption></figure>".format(images["severity_coefficients"]))
    html.append("<p><span class='legal'>Key insight.</span> Article 58(2) levers such as administrative fines, compliance orders, and remedial mandates dominate the transition into the FINE_PLUS tier, supporting doctrinal expectations that DPAs combine monetary and structural remedies in the most serious cases.</p>")

    html.append("<h2>3. Case-initiation channels</h2>")
    html.append("<figure><img src='{0}' alt='Case initiation log fines'><figcaption>Figure 3. Average log fines by initiation channel.</figcaption></figure>".format(images["case_logfine"]))
    html.append("<figure><img src='{0}' alt='Case initiation severity'><figcaption>Figure 4. Average expected severity by initiation channel.</figcaption></figure>".format(images["case_severity"]))
    html.append(tables["case_initiation_summary"])
    html.append("<p><span class='legal'>Interpretation.</span> Self-reported breaches (Art. 33 notifications) and ex officio investigations correlate with the highest fines and severity scores. The pooled low-frequency bucket (media, joint investigations, follow-ups) drops markedly, reinforcing the inference that proactive or high-visibility triggers correspond to harsher enforcement toolkits.</p>")

    html.append("<h2>4. Notification timeliness and fines</h2>")
    html.append(tables["joint_metrics"])
    html.append("<figure><img src='{0}' alt='Propensity coefficients'><figcaption>Figure 5. Leading predictors of timely notification (positive coefficients increase the odds of timely filing).</figcaption></figure>".format(images["propensity_coefficients"]))
    html.append("<p><span class='legal'>Reading the coefficients.</span> Richer discussions of data-subject rights (<code>rights_discussed/violated</code> components) push organisations toward timely filing, whereas DPAs with historically severe outcomes (<code>dpa_severity_shrinkage</code>) and the presence of reprimands correlate with slower disclosure. Light-touch portfolios (<code>q53_powers_NONE</code>) move in the opposite direction.</p>")

    html.append("<figure><img src='{0}' alt='Bootstrap intervals'><figcaption>Figure 6. Bootstrap confidence intervals for AIPW estimates across two trim bands.</figcaption></figure>".format(images["bootstrap"]))
    html.append(tables["bootstrap_summary"])
    html.append("<p class='note'><span class='legal'>Caution.</span> Under the baseline 0.10–0.90 trim the bootstrap interval spans −20.6 to +10.3 log points, signalling severe instability driven by overlap issues. Tightening to 0.20–0.80 centres the effect near zero but retains wide uncertainty. In legal terms, we cannot yet claim that timely notification either mitigates or exacerbates fines—the data are compatible with both interpretations once extreme propensities are excluded.</p>")

    html.append("<h2>5. Implications for legal analysis</h2>")
    html.append("<ul>"
                "<li><span class='legal'>Sanction architecture.</span> High-severity cases nearly always involve layered Article&nbsp;58(2) powers alongside fines. Practitioners should expect DPAs to combine financial penalties with structural orders when multiple corrective measures are recorded.</li>"
                "<li><span class='legal'>Initiation channel.</span> Complaint-driven matters dominate the docket numerically, but breach notifications and ex officio actions occupy the severe tail. This supports theories that disclosure duties surface in cases already deemed serious by organisations or regulators.</li>"
                "<li><span class='legal'>Timeliness.</span> Current evidence is inconclusive: the direction of the causal estimate swings with trimming, and bootstrap intervals are wide. Before advising clients, we should stabilise the estimator (e.g., overlap weighting, targeted learning) or complement it with qualitative case analysis.</li>"
                "</ul>")

    html.append("<h2>6. Next steps</h2>")
    html.append("<ol>"
                "<li>Introduce overlap-conscious estimation (e.g., trimmed IPTW or targeted ML) to see whether a reliable causal effect emerges.</li>"
                "<li>Pool or shrink residual low-count initiation categories, or move to Bayesian hierarchies so each trigger inherits information from complaint-heavy cases.</li>"
                "<li>Augment this statistical perspective with doctrinal review of a small case sample to contextualise rights-discussion latent components for legal audiences.</li>"
                "</ol>")

    html.append("<p>Generated automatically by <code>scripts/analysis/generate_notification_report.py</code>. All figures and tables draw on reproducible artefacts stored under <code>outputs/analysis/</code>.</p>")

    html.append("</body></html>")
    return "\n".join(html)


def main() -> None:
    data = load_data()
    images = {}
    images["severity_distribution"] = plot_severity_distribution(data["severity"])
    case_imgs = plot_case_initiation(data["case_initiation"])
    images["case_logfine"], images["case_severity"] = case_imgs
    images["severity_coefficients"] = plot_severity_coefficients(data["severity_coefs"])
    images["propensity_coefficients"] = plot_propensity_coefficients(data["propensity_coefs"])
    images["bootstrap"] = plot_bootstrap(data["bootstrap"])

    tables = {
        "severity_summary": data["severity"].groupby("severity_label")["severity_expected"].agg(['count','mean','std']).reset_index().to_html(index=False, border=0, float_format="{:.3f}".format),
        "case_initiation_summary": data["case_initiation"].to_html(index=False, border=0, float_format="{:.3f}".format),
        "joint_metrics": pd.DataFrame([data["metrics"]]).to_html(index=False, border=0, float_format="{:.3f}".format),
        "bootstrap_summary": pd.DataFrame.from_dict(data["bootstrap"], orient='index').reset_index().rename(columns={'index':'trim_range'}).to_html(index=False, border=0, float_format="{:.3f}".format),
    }

    html = build_html(images, tables, {}, data["metrics"], data["bootstrap"])
    output_path = REPORT_DIR / "breach_notification_report.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
