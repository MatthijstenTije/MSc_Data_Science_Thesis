import logging
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("model_temp_summary")

def generate_model_temp_summary(input_path="Notebooks/Phase_02/output/sentences_cleaned.csv", encoding="utf-8"):
    """
    Generate and save summary by model × temperature × noun_gender × adjective_gender.
    Also creates a heatmap visualization.
    """
    # Load the data
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, encoding=encoding)
    
    # Summary by model × noun_gender × adjective_gender × temperature
    summary = (
        df
        .groupby(['model', 'temperature', 'noun_gender', 'adjective_gender'])
        .size()
        .reset_index(name='sentence_count')
    )
    
    # Compute percentage within each model-temperature group
    summary['percentage'] = (
        summary
        .groupby(['model', 'temperature'])['sentence_count']
        .transform(lambda x: x / x.sum() * 100)
    )
    
    # Sort for readability
    summary = summary.sort_values(['model', 'temperature', 'sentence_count'], 
                                 ascending=[True, True, False])
    
    # Display the summary
    logger.info("Sentence counts by model, temperature, noun_gender & adjective_gender")
    logger.info("\n" + str(summary))
    
    # Save to CSV
    output_path = Path("Notebooks/Phase_02/output/summary_by_model_temp_gender.csv")
    summary.to_csv(output_path, index=False, encoding=encoding)
    logger.info(f"Saved summary to {output_path}")
    
    # Create heatmap visualization
    create_heatmap(summary)
    
    return summary


model_display_names = {
    "llama3:8b": "LLaMA 3 - Chat (RLHF)",
    "llama3-chatqa:8b": "LLaMA 3 - ChatQA (RLHF)",
    "llama3:text": "LLaMA 3 - Text (Pretrained Only)",
    "Llama2-Uncensored": "LLaMA 2 - Uncensored"
}

aligned_models = ["llama3:8b", "llama3-chatqa:8b"]
non_aligned_models = ["llama3:text", "Llama2-Uncensored"]
models = aligned_models + non_aligned_models  # this order is guaranteed

def create_heatmap(summary):
    import numpy as np
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as patches

    summary = summary.copy()
    summary["sentence_count"] = summary["sentence_count"].round().astype(int)
    summary["gender_pair"] = (
        summary["noun_gender"].str[0].str.upper() + " - " +
        summary["adjective_gender"].str[0].str.upper()
    )

    gp_order = ["M - M", "F - F", "M - F", "F - M"]
    summary["gender_pair"] = pd.Categorical(summary["gender_pair"], categories=gp_order, ordered=True)
    summary["temperature"] = pd.to_numeric(summary["temperature"])
    temps = sorted(summary["temperature"].unique())

    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    aligned_models_order = ["llama3:8b", "llama3-chatqa:8b"]
    non_aligned_models_order = ["llama3:text", "Llama2-Uncensored"]
    all_models = aligned_models_order + non_aligned_models_order
    models = [m for m in all_models if m in summary["model"].unique()]
    n_models = len(models)

    fig = plt.figure(figsize=(6 * n_models, 6.8))
    gs = gridspec.GridSpec(1, n_models, figure=fig, wspace=0.4)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_models)]

    vmin = summary["sentence_count"].min()
    vmax = summary["sentence_count"].max()
    mid_idx = n_models // 2

    for i, (ax, model) in enumerate(zip(axes, models)):
        dfm = summary[summary["model"] == model]
        bg_color = '#f0f8ff' if model in aligned_models else '#fff0f0'

        bbox = ax.get_position()
        fig.add_artist(patches.FancyBboxPatch(
            (bbox.x0 - 0.01, bbox.y0 - 0.01),
            bbox.width + 0.02,
            bbox.height + 0.02,
            boxstyle="round,pad=0.02",
            facecolor=bg_color,
            edgecolor='none',
            linewidth=0,
            transform=fig.transFigure,
            zorder=-1
        ))

        count_tbl = (
            dfm.pivot_table(index="temperature", columns="gender_pair", values="sentence_count", aggfunc="sum", fill_value=0)
            .reindex(index=temps, columns=gp_order)
            .astype(int)
        )
        perc_tbl = count_tbl.div(count_tbl.sum(axis=1), axis=0) * 100
        annot = count_tbl.astype(str) + "\n(" + perc_tbl.round(1).astype(str) + "%)"

        sns.heatmap(
            count_tbl,
            annot=annot,
            fmt="",
            cmap=cmap,
            linewidths=0.5,
            linecolor="white",
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            square=True,
            ax=ax
        )

        # Manually place model name above card (with padding)
        fig.text(
            bbox.x0 + bbox.width / 2,
            bbox.y1 + 0.04,
            model_display_names.get(model, model),
            ha='center', va='bottom',
            fontsize=12
        )

        # No xlabel, no ylabel inside plot
        ax.set_xticklabels(gp_order, rotation=0, fontstyle="italic")
        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.invert_yaxis()
        ax.axvline(2, color="black", linewidth=1.5, zorder=10)

        # Add y-axis label manually outside the leftmost card
        if i == 0:
            fig.text(bbox.x0 - 0.05, bbox.y0 + bbox.height / 2,
                     "Temperature", ha="center", va="center", rotation="vertical", fontsize=11)

        for text in ax.texts:
            text.set_fontsize(8)

    # Group Labels: larger, bold, centered between groups
    aligned_count = sum(m in aligned_models for m in models)
    nonaligned_count = n_models - aligned_count
    if aligned_count:
        fig.text(0.5 * aligned_count / n_models, 1.045, "RLHF-Aligned Models",
                 fontsize=14, ha='center', weight='bold')
    if nonaligned_count:
        fig.text((aligned_count + 0.5 * nonaligned_count) / n_models, 1.045,
                 "Non-Aligned Models", fontsize=14, ha='center', weight='bold')

    # X label manually placed above colorbar
    fig.text(0.5, 0.08, "Noun / Adjective Gender", ha="center", fontsize=11)

    # Shared horizontal colorbar — pushed further down
    cbar_ax = fig.add_axes([0.35, 0.035, 0.3, 0.012])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=9)
    # No label for the colorbar
    cbar.ax.set_xlabel("")

    plt.tight_layout(rect=[0.02, 0.1, 0.9, 0.99])
    plt.savefig("Notebooks/Phase_02/visualizations/heatmap_gender_temp.png", dpi=300, bbox_inches="tight")
    plt.savefig("Notebooks/Phase_02/visualizations/heatmap_gender_temp.pdf", dpi=300, bbox_inches="tight")
    plt.show()



if __name__ == "__main__":
    generate_model_temp_summary()