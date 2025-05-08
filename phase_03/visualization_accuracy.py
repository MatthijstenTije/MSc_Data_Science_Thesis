import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from pathlib import Path

# Set up output directory
base_dir = Path(__file__).resolve().parent
output_dir = base_dir / 'visualizations'
output_dir.mkdir(parents=True, exist_ok=True)

# Publication-style plot formatting
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino', 'Computer Modern Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300
})

# Load data
csv_path = Path("phase_03/results/RobBERT/results_by_model_temp_DTAI-KULeuven_robbert-2023-dutch-large.csv")
df = pd.read_csv(csv_path, skiprows=[1])

# Normalize and clean model names
df['llm_model'] = df['llm_model'].str.lower().str.replace("llama2-uncensored", "Llama2-Uncensored")
df['llm_model'] = df['llm_model'].str.replace("llama3-chatqa:8b", "llama3-chatqa:8b")
df['llm_model'] = df['llm_model'].str.replace("llama3:8b", "llama3:8b")
df['llm_model'] = df['llm_model'].str.replace("llama3:text", "llama3:text")

# Model display names
model_display_names = {
    "llama3:8b": "LLaMA 3 - Chat (RLHF)",
    "llama3-chatqa:8b": "LLaMA 3 - ChatQA (RLHF)",
    "llama3:text": "LLaMA 3 - Text (Pretrained Only)",
    "Llama2-Uncensored": "LLaMA 2 - Uncensored (No Alignment)"
}

aligned_models = ["llama3:8b", "llama3-chatqa:8b"]
non_aligned_models = ["Llama2-Uncensored", "llama3:text"]
sorted_models = aligned_models + non_aligned_models

# Define base color map
base_cmap = LinearSegmentedColormap.from_list(
    "pastel_blue_red", ["#4E79A7", "#F28E2B", "#E15759"]
)

# Normalize temperature values to [0, 1] for colormap
norm_temp = (df['temperature'] - df['temperature'].min()) / (df['temperature'].max() - df['temperature'].min())
temp_to_color = dict(zip(df['temperature'].unique(), base_cmap(norm_temp.unique())))

# Create subplots
fig, axes = plt.subplots(1, len(sorted_models), figsize=(14, 5), sharey=True)

for j, (model_name, ax) in enumerate(zip(sorted_models, axes)):
    model_data = df[df['llm_model'] == model_name].sort_values(by='temperature')
    temperatures = model_data['temperature']
    accuracies = model_data['accuracy']

    bars = ax.bar(
        range(len(temperatures)),
        accuracies,
        width=0.6,
        color=[temp_to_color[temp] for temp in temperatures],
        edgecolor='black'
    )

    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}",
                    ha='center', va='bottom', fontsize=8)

    ax.set_title(model_display_names.get(model_name, model_name), fontsize=12)
    ax.set_xticks(range(len(temperatures)))
    ax.set_xticklabels([f"{t:.2f}" for t in temperatures])
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    if model_name in aligned_models:
        ax.set_facecolor('#f0f8ff')
    else:
        ax.set_facecolor('#fff0f0')

    if j == 0:
        ax.set_ylabel("Accuracy", fontsize=14)

# Add general x-axis label
fig.text(0.5, 0.04, "Temperature", ha='center', fontsize=16)

# Add alignment group labels
fig.text(0.25, 0.96, "RLHF-Aligned Models", fontsize=12, ha='center', fontweight="bold")
fig.text(0.75, 0.96, "Non-Aligned Models", fontsize=12, ha='center', fontweight="bold")


# Final layout tweaks and export
plt.tight_layout()
plt.subplots_adjust(top=0.85, bottom=0.18)
fig.savefig(output_dir / 'accuracy_visualization.pdf', dpi=300)
plt.show()