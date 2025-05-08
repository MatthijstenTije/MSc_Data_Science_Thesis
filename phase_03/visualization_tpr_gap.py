import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
sns.set_style("whitegrid")

base_dir = Path(__file__).resolve().parent
output_dir = base_dir / 'visualizations'
output_dir.mkdir(parents=True, exist_ok=True)

# Load the CSV (assumes it's in the same directory as your script)
df = pd.read_csv("phase_03/results/RobBERT/results_by_model_temp_DTAI-KULeuven_robbert-2023-dutch-large.csv", skiprows=1)

# Rename columns explicitly
df.columns = [
    'llm_model', 'temperature', 'accuracy_mean', 'accuracy_std',
    'f1_mean', 'f1_std', 'tpr_gap_s', 'tpr_gap_s_std',
    'tpr_gap_contra', 'tpr_gap_contra_std', 'sample_size'
]

# Convert data types
df['temperature'] = df['temperature'].astype(float)
df['tpr_gap_s'] = df['tpr_gap_s'].astype(float)
df['tpr_gap_s_std'] = df['tpr_gap_s_std'].astype(float)
df['tpr_gap_contra'] = df['tpr_gap_contra'].astype(float)
df['tpr_gap_contra_std'] = df['tpr_gap_contra_std'].astype(float)

# Define clean pastel colors
# Define varied marker styles and line styles for better distinction
marker_styles = ['o', 's', '^', 'D']  # circle, square, triangle, diamond
line_styles = ['-', '--', '-.', ':']  # solid, dashed, dashdot, dotted

# Set clean, minimalistic style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white'
})

# Create subplots with clean layout - removed main title
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)


# Get unique models
unique_models = df['llm_model'].unique()

for ax in axes:
    ax.set_yticks([-0.30, -0.20, -0.10, 0, 0.10, 0.20, 0.30])# Use the specified color scheme (blue, red, orange) plus an appropriate addition (teal)
set_colors = ["#4E79A7",  # blue
              "#E15759",  # red
              "#F28E2B",  # orange
              "#76B7B2"]  # teal (complementary addition)# Set symmetrical y-axis range for fair visual perception
for ax in axes:
    ax.set_ylim(-0.30, 0.30)
    ax.grid(True, linestyle='--', alpha=0.3, color='#cccccc')
    
# Plot TPR Gap (S) with varied markers and line styles for better distinction
for i, model in enumerate(unique_models):
    model_data = df[df['llm_model'] == model]
    axes[0].plot(model_data['temperature'], model_data['tpr_gap_s'], 
                marker=marker_styles[i % len(marker_styles)], 
                linestyle=line_styles[i % len(line_styles)],
                linewidth=2.5, markersize=8, 
                label=model, color=set_colors[i % len(set_colors)])
    

# Plot TPR Gap (Contra) with varied markers and line styles for better distinction
for i, model in enumerate(unique_models):
    model_data = df[df['llm_model'] == model]
    axes[1].plot(model_data['temperature'], model_data['tpr_gap_contra'], 
                marker=marker_styles[i % len(marker_styles)], 
                linestyle=line_styles[i % len(line_styles)],
                linewidth=2.5, markersize=8, 
                label=model, color=set_colors[i % len(set_colors)])
    

# Add horizontal line at y=0 for both plots with clearer styling
for ax in axes:
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7, zorder=0)
    
    # Very subtle background shading with colors that complement the new color scheme
    ax.axhspan(0, 0.30, alpha=0.05, color='#4E79A7', label='_nolegend_')  # Very light version of the red
    ax.axhspan(-0.30, 0, alpha=0.05, color='#E15759', label='_nolegend_')  # Very light version of the blue

# Set titles for individual plots - updated as requested
axes[0].set_title('TPR Gap (S)', fontweight='bold', fontsize=13)
axes[1].set_title('TPR Gap (Contra)', fontweight='bold', fontsize=13)

axes[0].set_xlabel('Temperature')
axes[1].set_xlabel('Temperature')

axes[0].set_ylabel('TPR Gap', fontweight='normal')

# Set specific x-tick values to match data points
for ax in axes:
    ax.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5])
    ax.set_xlim(0.45, 1.55)

# Improved legend placement - horizontal layout with better spacing
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.08),
           frameon=True, ncol=4, title="Model", 
           edgecolor='lightgray', fancybox=True, fontsize=10)


# Add subtle annotations
axes[0].text(0.02, 0.95, "Bias towards males", transform=axes[0].transAxes, 
             fontsize=10, verticalalignment='top', color='#666666',
             bbox=dict(facecolor='white', alpha=0.7, pad=3, edgecolor='none'))
axes[0].text(0.02, 0.05, "Bias towards females", transform=axes[0].transAxes, 
             fontsize=10, verticalalignment='bottom', color='#666666',
             bbox=dict(facecolor='white', alpha=0.7, pad=3, edgecolor='none'))

# Clean layout with appropriate spacing - adjusted for legend below plot
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

# Save the figure with high resolution
fig.savefig(output_dir / 'tpr_gap.pdf', dpi=300)
# Show the plot
plt.show()