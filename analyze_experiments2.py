import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = ROOT / "experiments2"
ANALYSIS_DIR = ROOT / "analysis_results" / "experiments2"
WHITEBOX_PATH = EXPERIMENT_DIR / "whitebox_results.json"
BLACKBOX_PATH = EXPERIMENT_DIR / "records.jsonl"


def load_whitebox():
    with WHITEBOX_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_blackbox():
    if not BLACKBOX_PATH.exists():
        return pd.DataFrame()
    return pd.read_json(BLACKBOX_PATH, lines=True, dtype={"image_id": str})


def plot_whitebox(data):
    df = pd.DataFrame.from_dict(data, orient="index").reset_index(names="experiment")
    df = df.dropna(subset=["whitebox_asr"])
    df["whitebox_asr_percent"] = df["whitebox_asr"] * 100

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x="experiment", y="whitebox_asr_percent", hue="surrogate", dodge=False)
    ax.set_title("White-box Attack Success Rate")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=35)
    for patch in ax.patches:
        height = patch.get_height()
        if pd.notna(height):
            ax.annotate(f"{height:.1f}%", (patch.get_x() + patch.get_width() / 2, height), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "whitebox_asr.png", dpi=160)
    plt.close()


def summarize_blackbox(df):
    if df.empty:
        return pd.DataFrame()
    summary = (
        df.groupby(["surrogate", "eps", "target_model"], as_index=False)
        .agg(
            samples=("image_id", "nunique"),
            records=("attack_success", "size"),
            attack_success_rate=("attack_success", "mean"),
        )
        .sort_values(["surrogate", "eps", "target_model"])
    )
    summary["attack_success_percent"] = summary["attack_success_rate"] * 100
    return summary


def plot_blackbox(summary):
    if summary.empty:
        return
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=summary, x="eps", y="attack_success_percent", hue="target_model")
    ax.set_title("Black-box Transfer ASR - Ensemble Surrogate")
    ax.set_xlabel("Perturbation Budget")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, 105)
    for patch in ax.patches:
        height = patch.get_height()
        if pd.notna(height):
            ax.annotate(f"{height:.1f}%", (patch.get_x() + patch.get_width() / 2, height), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "blackbox_ensemble_asr.png", dpi=160)
    plt.close()

    pivot = summary.pivot_table(index="eps", columns="target_model", values="attack_success_percent")
    plt.figure(figsize=(8, 5))
    ax = sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_title("Black-box Transfer ASR Heatmap (%)")
    ax.set_xlabel("Target Model")
    ax.set_ylabel("Perturbation Budget")
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "blackbox_ensemble_heatmap.png", dpi=160)
    plt.close()


def write_report(whitebox, blackbox_summary):
    lines = [
        "# Experiments2 Evaluation Report",
        "",
        "## Scope",
        "",
        "- Modality: image classification.",
        "- Attack type: black-box transfer attack.",
        "- Remaining surrogate evaluated in this run: ensemble.",
        "- Target models: InceptionV3, DenseNet121, MobileNetV2, EfficientNet-B0.",
        "- Perturbation budgets: eps4, eps8, eps16.",
        "",
        "## White-box Baseline",
        "",
    ]

    wb_df = pd.DataFrame.from_dict(whitebox, orient="index").reset_index(names="experiment")
    for _, row in wb_df.iterrows():
        asr = row.get("whitebox_asr")
        asr_text = "N/A" if pd.isna(asr) else f"{asr * 100:.2f}%"
        lines.append(f"- {row['experiment']}: total={int(row['total'])}, ASR={asr_text}")

    lines.extend(["", "## Black-box Transfer Summary", ""])
    if blackbox_summary.empty:
        lines.append("No black-box records found.")
    else:
        for _, row in blackbox_summary.iterrows():
            lines.append(
                f"- {row['surrogate']}_{row['eps']} -> {row['target_model']}: "
                f"samples={int(row['samples'])}, ASR={row['attack_success_percent']:.2f}%"
            )

        overall = blackbox_summary["attack_success_rate"].mean() * 100
        best = blackbox_summary.loc[blackbox_summary["attack_success_rate"].idxmax()]
        lines.extend(
            [
                "",
                "## Findings",
                "",
                f"- Average transfer ASR across target models and eps settings: {overall:.2f}%.",
                f"- Best transfer setting: {best['surrogate']}_{best['eps']} on {best['target_model']} "
                f"with ASR={best['attack_success_percent']:.2f}%.",
                "- Query cost: 1 target-model query per adversarial sample during evaluation.",
                "- Semantic consistency is not applicable to pure image classification labels in this run; perturbation budget is represented by eps.",
            ]
        )

    report_path = ANALYSIS_DIR / "experiments2_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    whitebox = load_whitebox()
    blackbox_df = load_blackbox()
    blackbox_summary = summarize_blackbox(blackbox_df)

    plot_whitebox(whitebox)
    plot_blackbox(blackbox_summary)
    blackbox_summary.to_csv(ANALYSIS_DIR / "blackbox_ensemble_summary.csv", index=False, encoding="utf-8-sig")
    write_report(whitebox, blackbox_summary)

    print(f"Analysis written to: {ANALYSIS_DIR}")
    if not blackbox_summary.empty:
        print(blackbox_summary.to_string(index=False))


if __name__ == "__main__":
    main()
