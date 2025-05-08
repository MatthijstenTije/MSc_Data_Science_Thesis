import logging
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np


from config import OUTPUT_FOLDER, VISUALIZATION_FOLDER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("model_temp_summary")



def generate_model_temp_summary(input_path=Path(OUTPUT_FOLDER) / "sentences_final.csv", encoding="utf-8"):
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
    summary.to_csv( Path(OUTPUT_FOLDER) / "summary_by_model_temp_gender.csv", index=False, encoding=encoding)
    logger.info(f"Saved summary")
    
    # Create heatmap visualization
    create_heatmap(summary)
    
    return summary


model_display_names = {
    "llama3:8b": "LLaMA 3 - Chat (RLHF)",
    "llama3-chatqa:8b": "LLaMA 3 - ChatQA (RLHF)",
    "llama3:text": "LLaMA 3 - Text (Pretrained Only)",
    "llama2-uncensored": "LLaMA 2 - Uncensored"
}

aligned_models = ["llama3:8b", "llama3-chatqa:8b"]
non_aligned_models = ["llama3:text", "llama2-uncensored"]
models = aligned_models + non_aligned_models  # this order is guaranteed

def create_heatmap(summary):

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

    base_cmap = LinearSegmentedColormap.from_list(
    "pastel_blue_red",
    [
        "#4E79A7",  # blue
        "#F28E2B",  # orange
        "#E15759"   # red
    ]
)

    # Sample colors and add alpha
    colors = base_cmap(np.linspace(0, 1, 256))  # 256 colors
    colors[:, -1] = 0.7  # Set alpha (transparency) to 70%

    # Create a new ListedColormap with alpha
    cmap = ListedColormap(colors)
    
    aligned_models_order = ["llama3:8b", "llama3-chatqa:8b"]
    non_aligned_models_order = ["llama3:text", "llama2-uncensored"]
    all_models = aligned_models_order + non_aligned_models_order
    models = [m for m in all_models if m in summary["model"].unique()]
    n_models = len(models)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.4, hspace=0.5)
    axes = [fig.add_subplot(gs[i//2, i%2]) for i in range(n_models)]

    vmin = summary["sentence_count"].min()
    vmax = summary["sentence_count"].max()

    for i, (ax, model) in enumerate(zip(axes, models)):
        dfm = summary[summary["model"] == model]
        bg_color = 'white'

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
            zorder=-1,
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
            ax=ax,
            annot_kws={"color": "white", "fontsize": 10}
        )

        fig.text(
            bbox.x0 + bbox.width / 2,
            bbox.y1 + 0.04,
            model_display_names.get(model, model),
            ha='center', va='bottom',
            fontsize=20  # larger titles
        )

        ax.set_xticklabels(gp_order, rotation=0, fontstyle="italic")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.invert_yaxis()
        ax.axvline(2, color="black", linewidth=1.5, zorder=10)

        for text in ax.texts:
            text.set_fontsize(12)  # larger cell font

    # Labels for left-most plots only
    for idx, ax in enumerate(axes):
        if idx % 2 == 0:  # left column
            ax.set_ylabel("Temperature", fontsize=12)

    # Shared x-label
    fig.text(0.5, 0.05, "Noun / Adjective Gender", ha="center", fontsize=12)

    # Colorbar
    cbar_ax = fig.add_axes([0.35, 0.02, 0.3, 0.02])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    plt.savefig(Path(VISUALIZATION_FOLDER) / "heatmap_gender_temp.png", dpi=300, bbox_inches="tight")
    plt.savefig(Path(VISUALIZATION_FOLDER) / "heatmap_gender_temp.pdf", dpi=300, bbox_inches="tight")
    plt.show()




if __name__ == "__main__":
    generate_model_temp_summary()