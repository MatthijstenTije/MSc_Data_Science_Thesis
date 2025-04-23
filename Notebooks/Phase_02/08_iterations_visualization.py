import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl

# Set up publication-quality formatting
plt.rcParams.update({
    'font.family': 'serif',  # Standard serif fonts for academic publications
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

model_display_names = {
    "llama3:8b": "LLaMA 3 - Chat (RLHF)",
    "llama3-chatqa:8b": "LLaMA 3 - ChatQA (RLHF)",
    "llama3:text": "LLaMA 3 - Text (Pretrained Only)",
    "Llama2-Uncensored": "LLaMA 2 - Uncensored (No Alignment)"
}


# Base path where logs are stored
logs_base_dir = "Notebooks/Phase_02/logs"

# Grab all JSONL log files in model-specific subdirectories
log_files = glob.glob(os.path.join(logs_base_dir, "*", "*.jsonl"))

# Prepare a list to collect statistics
run_stats = []

# Loop over each log file
for log_file in log_files:
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                log_data = json.loads(line)
                
                # Extract metadata
                model = log_data.get("model")
                temperature = log_data.get("temperature")
                noun_gender = log_data.get("noun_gender")
                adjective_gender = log_data.get("adjective_gender")
                
                # Extract run statistics
                total_runs = log_data.get("total_runs", 0)
                
                # Create a configuration label for the chart
                config_name = f"{noun_gender}-{adjective_gender}"
                
                # Add run statistics to our list
                run_stats.append({
                    "model": model,
                    "temperature": temperature,
                    "noun_gender": noun_gender,
                    "adjective_gender": adjective_gender,
                    "config": config_name,
                    "total_runs": total_runs
                })
            
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON in file: {log_file}")
                continue

# Convert to DataFrame
df_runs = pd.DataFrame(run_stats)
# Create directory for visualizations
os.makedirs("Notebooks/Phase_02/visualizations", exist_ok=True)

# Simplified color palette - more distinct and less overwhelming
config_colors = {
    "male-female": "#e6550d",  # Orange
    "female-male": "#e6550d",  # Green
    "female-female": "#3182bd", # Purple
    "male-male": "#3182bd",    # Blue
}

# Create a list of unique configurations, temperatures, and models
configurations = [
    "male-female",   # orange
    "female-male",   # orange
    "female-female", # blue/purple
    "male-male"      # blue
]

temperatures = sorted(df_runs['temperature'].unique())
models = df_runs['model'].unique()
print(models)
# Group models
aligned_models = ["llama3:8b", "llama3-chatqa:8b"]
non_aligned_models = ["llama3:text", "Llama2-Uncensored"]

# Sort models: aligned first
sorted_models = aligned_models + non_aligned_models

fig, axes = plt.subplots(1, len(sorted_models), figsize=(12, 5), sharey=True)

for j, (model_name, ax) in enumerate(zip(sorted_models, axes)):
    model_data = df_runs[df_runs['model'] == model_name]
    
    # Set up bar positions
    bar_positions = np.arange(len(temperatures))
    
    for k, config in enumerate(configurations):
        config_data = model_data[model_data['config'] == config]
        
        # Prepare data for this configuration across temperatures
        values = []
        for temp in temperatures:
            temp_data = config_data[config_data['temperature'] == temp]
            values.append(temp_data['total_runs'].sum() if not temp_data.empty else 0)
        
        # Add this layer to the stacked bar
        bottom = np.zeros(len(temperatures))
        if k > 0:
            for prev_k in range(k):
                prev_config = configurations[prev_k]
                for i, temp in enumerate(temperatures):
                    prev_config_data = model_data[(model_data['config'] == prev_config) & 
                                                 (model_data['temperature'] == temp)]
                    bottom[i] += prev_config_data['total_runs'].sum() if not prev_config_data.empty else 0
        
        # Plot the bars
        bars = ax.bar(bar_positions, values, 0.8, 
                     bottom=bottom,
                     color=config_colors.get(config),
                     edgecolor='black',
                     linewidth=0.5,
                     label=config if j == 0 else "")
        
        # Add text labels for significant values
        for i, v in enumerate(values):
            if v > max(df_runs['total_runs'])/15:
                ax.text(bar_positions[i], bottom[i] + v/2, 
                       str(int(v)), ha='center', va='center', 
                       fontsize=8, color='white')
    
    # Customize subplot
    ax.set_title(model_display_names.get(model_name, model_name), fontsize=12)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels([f"{t}" for t in temperatures])
    ax.grid(axis='y', linestyle='--', alpha=0.2)
    
    # Background shading by alignment group
    if model_name in aligned_models:
        ax.set_facecolor('#f0f8ff')  # Light blue
    else:
        ax.set_facecolor('#fff0f0')  # Light red

    
    # Only add y-label to first subplot
    if j == 0:
        ax.set_ylabel("Number of Iterations")
    
    # Add x-label only to middle subplot
    if j == len(sorted_models)//2:
        ax.set_xlabel("Temperature")

# Add alignment group labels above subplots
fig.text(0.25, 1.02, "RLHF-Aligned Models", fontsize=12, ha='center')
fig.text(0.75, 1.02, "Non-Aligned Models", fontsize=12, ha='center')

# Legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Prompt Configuration",
           loc='upper center', bbox_to_anchor=(0.5, 0.02),
           ncol=len(configurations), frameon=True)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

# Save small multiples version
plt.savefig("Notebooks/Phase_02/visualizations/sentence_gen_model_comparison_v1.png", bbox_inches='tight')
plt.savefig("Notebooks/Phase_02/visualizations/sentence_gen_model_comparison__v1.pdf", dpi=300, bbox_inches='tight')

print("Small multiples visualization created successfully.")

