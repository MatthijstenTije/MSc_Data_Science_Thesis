import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects
import pandas as pd
import numpy as np
from matplotlib_venn import venn3
import scipy.stats as stats
from config import FIGURES_DIR

def plot_bias_comparison(df_compare):
    """Create a scatter plot comparing bias across two models."""
    # Set publication-quality parameters
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    
    # Create figure with appropriate dimensions for academic paper
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Professional color palette (colorblind-friendly)
    palette = {
        "Significant in both (agree)": "#4E79A7",   # blue
        "Significant in both (oppose)": "#E15759",   # orange
        "Only Word2Vec": "#3A923A",                 # green
        "Only FastText": "#C03D3E",                 # red
        "Non-significant": "#999999"                # gray
    }
    
    # Create the scatterplot with professional styling
    scatter = sns.scatterplot(
        data=df_compare,
        x='bias_w2v',
        y='bias_ft',
        hue='tag',
        palette=palette,
        s=60,  # Slightly smaller points for clarity
        edgecolor='white',  # White edges help distinguish overlapping points
        alpha=0.8,
        linewidth=0.5
    )
    
    # Add reference lines with improved styling
    plt.axhline(0, color='black', linestyle='--', linewidth=0.7, alpha=0.6)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.7, alpha=0.6)
    plt.plot([-0.15, 0.15], [-0.15, 0.15], linestyle=':', color='dimgray', linewidth=0.8, alpha=0.7)  # y = x
    
    # Quadrant labels with better positioning and formatting
    quadrant_labels = [
        {"text": "Consistent\nMale Bias", "pos": (0.08, 0.12), "ha": "center", "fontweight": "bold"},
        {"text": "Contradictory Bias\n(FastText → Male\nWord2Vec → Female)", 
         "pos": (-0.1, 0.1), "ha": "center", "fontweight": "normal"},
        {"text": "Consistent\nFemale Bias", "pos": (-0.1, -0.1), "ha": "center", "fontweight": "bold"},
    ]
    
    for label in quadrant_labels:
        text = plt.text(
            label["pos"][0], label["pos"][1],
            label["text"],
            fontsize=10,
            color='dimgray',
            ha=label["ha"],
            va="center",
            fontweight=label.get("fontweight", "normal"),
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
        )
        # Add subtle shadow effect for better readability
        text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
    
    # Axis labels with more professional formatting
    plt.xlabel("Gender Bias Score (Word2Vec)", fontsize=12, labelpad=10)
    plt.ylabel("Gender Bias Score (FastText)", fontsize=12, labelpad=10)
    
    # Legend with better formatting and positioning
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [
        "Significant in both (agree)",
        "Significant in both (oppose)",
        "Only Word2Vec",
        "Only FastText",
        "Non-significant"
    ]
    ordered = sorted(zip(labels, handles), key=lambda x: order.index(x[0]) if x[0] in order else 99)
    labels, handles = zip(*ordered)
    legend = plt.legend(
        handles, labels,
        title="Bias Significance Category",
        loc='lower right',
        fontsize=10,
        title_fontsize=11,
        frameon=True,
        framealpha=0.95,
        edgecolor='lightgray',
    )
    legend.get_frame().set_linewidth(0.8)
    
    # Set consistent axis limits
    plt.xlim(-0.15, 0.15)
    plt.ylim(-0.15, 0.15)
    
    # Add subtle grid for readability
    plt.grid(True, linestyle=':', alpha=0.3, color='gray', linewidth=0.5)
    
    # Make tick marks more readable
    plt.tick_params(axis='both', which='major', labelsize=10, width=0.8, length=4)
    
    # Add subtle axes spines
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('gray')
    
    # Improve overall layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save high-resolution figure for publication
    plt.savefig(os.path.join(FIGURES_DIR, 'gender_bias_agreement_plot.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'gender_bias_agreement_plot.png'), bbox_inches='tight', dpi=300)
    
    # Display the figure
    plt.show()

def plot_bias_barplot_abs(df_subset, title):
    """
    Plot absolute Z-scores for bias (e.g., for female-biased words),
    so all bars go left to right.
    """
    df_plot = df_subset.copy()
    
    # Take absolute Z-scores
    df_plot['z_score_w2v'] = df_plot['z_score_w2v'].abs()
    df_plot['z_score_ft'] = df_plot['z_score_ft'].abs()
    
    # Sort by average
    df_plot['avg_z'] = (df_plot['z_score_w2v'] + df_plot['z_score_ft']) / 2
    df_plot = df_plot.sort_values('avg_z', ascending=False)
    
    df_melted = df_plot.melt(
        id_vars='word',
        value_vars=['z_score_w2v', 'z_score_ft'],
        var_name='Model',
        value_name='Z-score',
    )
    
    df_melted['Model'] = df_melted['Model'].map({
        'z_score_w2v': 'Word2Vec',
        'z_score_ft': 'FastText'
    })
    
    df_melted['word'] = pd.Categorical(df_melted['word'], categories=df_plot['word'], ordered=True)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_melted,
        y='word',
        x='Z-score',
        hue='Model',
        orient='h'
    )
    
    plt.title(title)
    plt.xlabel("Bias Z-score (|standardized|)")
    plt.ylabel("Adjective")
    plt.legend(title="Embedding Model", loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{title.lower().replace(' ', '_')}.png"), dpi=300)
    plt.show()

def plot_bias_correlation(df, model_name):
    """Plot correlation between cosine bias and RIPA score."""
    corr, p_val = stats.pearsonr(df['bias'], df['RIPA_score'])
    plt.figure(figsize=(8, 6))
    sns.regplot(
        x='bias',
        y='RIPA_score',
        data=df,
        scatter_kws={'alpha': 0.6, 's': 40},
        line_kws={'color': 'black'}
    )
    
    plt.title(
        f"{model_name}: Correlation Between Cosine Bias and RIPA\n"
        f"Pearson r = {corr:.3f}",
        fontsize=13
    )
    plt.xlabel("Cosine Bias (Mean Similarity Difference)", fontsize=11)
    plt.ylabel("RIPA Score (Mean Association Strength)", fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"bias_correlation_{model_name.lower()}.png"), dpi=300)
    plt.show()

def plot_ripa_vs_length_with_stats(df, model_name, color):
    """Plot RIPA Z-score vs adjective length with regression line."""
    # Get regression stats
    slope, intercept, r, p, _ = stats.linregress(df['adjective_length'], df['ripa_z'])
    
    plt.figure(figsize=(12, 6))
    # --- Scatter plot
    sns.scatterplot(
        data=df,
        x='adjective_length',
        y='ripa_z',
        alpha=0.7,
        color=color,
        s=60,
        edgecolor='black'
    )
    
    # --- Regression line manually
    sns.regplot(
        data=df,
        x='adjective_length',
        y='ripa_z',
        scatter=False,
        color='black'
    )
    
    # --- Add regression stats to legend
    plt.plot([], [], ' ', label=f"Slope = {slope:.3f}")
    plt.plot([], [], ' ', label=f"r = {r:.2f}, p = {p:.4f}")
    
    # --- Reference line
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    # --- Annotate strong outliers
    outliers = df[df['ripa_z'].abs() > 5]
    for _, row in outliers.iterrows():
        plt.text(
            row['adjective_length'] + 0.2,
            row['ripa_z'],
            row['word'],
            fontsize=9,
            alpha=0.8
        )
    
    # --- Layout and labels
    plt.title(f"{model_name} – RIPA Z-score vs. Adjective Length", fontsize=14)
    plt.xlabel("Adjective Length (characters)", fontsize=12)
    plt.ylabel("RIPA Z-score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"ripa_vs_length_{model_name.lower()}.png"), dpi=300)
    plt.show()

def plot_bias_venn_diagram(df_combined_w2v, df_combined_ft, df_combined_ripa, top_n=250):
    """Create a Venn diagram showing overlap between different bias detection methods."""
    # Ensure we have top N words
    top_cosine_w2v = set(df_combined_w2v.sort_values('cosine_bias_z', key=abs, ascending=False).head(top_n)['word'])
    top_cosine_ft = set(df_combined_ft.sort_values('cosine_bias_z', key=abs, ascending=False).head(top_n)['word'])
    top_ripa_w2v = set(df_combined_ripa.sort_values('ripa_z', key=abs, ascending=False).head(top_n)['word'])
    
    all_words = top_cosine_w2v | top_cosine_ft | top_ripa_w2v
    total_words = len(all_words)
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    venn = venn3(
        [top_cosine_w2v, top_cosine_ft, top_ripa_w2v],
        set_labels=("Word2Vec\nCosine Similarity", "FastText\nCosine Similarity", "RIPA"),
        set_colors=("#4E79A7", "#E15759", "#F28E2B"),
        alpha=0.7,
        ax=ax
    )

    # Forcefully center diagram within the axis by adjusting limits
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.7, 0.7)

    for patch in venn.patches:
        if patch:
            patch.set_linewidth(1.8)
            patch.set_edgecolor('black')

    for text in venn.set_labels:
        if text:
            text.set_fontweight('bold')
            text.set_fontsize(12)

    for subset_id in ['100', '010', '110', '001', '101', '011', '111']:
        subset = venn.get_label_by_id(subset_id)
        if subset:
            count = int(subset.get_text())
            perc = 100 * count / total_words
            subset.set_text(f"{count}\n({perc:.1f}%)")
            subset.set_fontsize(10)

    plt.grid(False)
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    plt.savefig(os.path.join(FIGURES_DIR, 'gender_bias_venn_diagram.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'gender_bias_venn_diagram.png'), bbox_inches='tight', dpi=300)

    plt.show()
