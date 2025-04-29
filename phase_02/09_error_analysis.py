import logging
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


from config import LOG_FOLDER, VISUALIZATION_FOLDER, OUTPUT_FOLDER

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("generate_wordclouds")

model_display_names = {
    "llama3:8b": "LLaMA 3 - Chat (RLHF)",
    "llama3-chatqa:8b": "LLaMA 3 - ChatQA (RLHF)",
    "llama3:text": "LLaMA 3 - Text (Pretrained Only)",
    "llama2-uncensored": "LLaMA 2 - Uncensored"
}

aligned_models = ["llama3:8b", "llama3-chatqa:8b"]
non_aligned_models = ["llama3:text", "llama2-uncensored"]
models_order = aligned_models + non_aligned_models

def generate_wordclouds_from_jsonl(input_path=Path(LOG_FOLDER) / "quality_issues.jsonl", encoding="utf-8"):
    """
    Generate and save wordclouds by model based on unexpected words.
    """
    logger.info(f"Loading data from {input_path}")
    
    duplicates_skipped_sum = defaultdict(int)
    wordmap = defaultdict(lambda: defaultdict(int))
    
    with open(input_path, 'r', encoding=encoding) as f:
        for line in f:
            data = json.loads(line)
            model = data['model']
            duplicates_skipped_sum[model] += data['duplicates_skipped']
            for word in data['unique_unexpected_words']:
                wordmap[model][word] += 1
    
    duplicates_skipped_sum = dict(duplicates_skipped_sum)
    wordmap = {model: dict(words) for model, words in wordmap.items()}
    
    logger.info("Calculated duplicates skipped per model:")
    logger.info(duplicates_skipped_sum)
    
    models_present = [m for m in models_order if m in wordmap]
    n_models = len(models_present)
    if n_models == 0:
        logger.error("No models with words found.")
        return
    
    logger.info(f"Generating one figure with {n_models} WordClouds.")

    # --- Setup Matplotlib ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8

    # Manually create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()  # easy indexing

    for ax, model in zip(axes, models_present):
        words = wordmap[model]
        top_words = dict(sorted(words.items(), key=lambda item: item[1], reverse=True)[:30])

        if not top_words:
            logger.warning(f"No words for model {model}, skipping.")
            continue
        
        # Define your base color map
        base_cmap = LinearSegmentedColormap.from_list(
            "pastel_blue_red",
            [
                "#4E79A7",  # blue
                "#F28E2B",  # orange
                "#E15759"   # red
            ]
        )

        # Sample colors from base_cmap
        colors = base_cmap(np.linspace(0, 1, 256))

        # Apply alpha = 0.7
        colors[:, -1] = 0.7

        # Create a final cmap with alpha
        pastel_cmap = ListedColormap(colors)
        
        wc = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap=pastel_cmap,
            max_words=30,
            relative_scaling=0.5
        ).generate_from_frequencies(top_words)
        
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        
        model_name_display = model_display_names.get(model, model)
        ax.set_title(
            model_name_display,
            fontsize=24,      
            pad=20,           
            fontweight='bold',
            loc='center'
        )

        # White rounded background
        bbox = ax.get_position()
        fig.add_artist(patches.FancyBboxPatch(
            (bbox.x0 - 0.015, bbox.y0 - 0.015),
            bbox.width + 0.03,
            bbox.height + 0.03,
            boxstyle="round,pad=0.02",
            facecolor='white',
            edgecolor='none',
            linewidth=0,
            transform=fig.transFigure,
            zorder=-1
        ))

    # Remove extra empty axes if any
    for i in range(len(models_present), len(axes)):
        fig.delaxes(axes[i])

    plt.subplots_adjust(wspace=0.2, hspace=0.25, left=0.05, right=0.95, top=0.92, bottom=0.08)

    # Save figure
    output_folder = Path(VISUALIZATION_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    out_png = output_folder / "wordclouds_models.png"
    out_pdf = output_folder / "wordclouds_models.pdf"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    
    logger.info(f"Saved WordCloud figure to {out_png} and {out_pdf}")
    plt.show()

    return wordmap, duplicates_skipped_sum

if __name__ == "__main__":
    generate_wordclouds_from_jsonl()