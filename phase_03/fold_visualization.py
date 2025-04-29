import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'axes.grid': False,
    'figure.dpi': 300
})

# Color set and transparency
c_train, c_val, c_test, c_extra = ("#4E79A7", "#E15759", "#F28E2B", "#A6C8E2")

alpha = 0.7

# Fold settings\outer_folds = 5
inner_folds = 5
train_ratio = 0.8
val_ratio = 0.16
test_ratio = 0.20
inner_width = train_ratio / inner_folds

# Create figure
fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

def draw_outer_fold(ax, idx):
    y = outer_folds - idx - 1
    # Outer train block
    ax.add_patch(Rectangle((0, y), train_ratio, 0.8,
                           facecolor=c_train, alpha=alpha,
                           edgecolor='k', linewidth=0.8, zorder=1))
    # Outer test block
    ax.add_patch(Rectangle((train_ratio, y), test_ratio, 0.8,
                           facecolor=c_test, alpha=alpha,
                           edgecolor='k', linewidth=0.8, zorder=1))
    # Inner CV folds
    for j in range(inner_folds):
        x0 = j * inner_width
        facecol = c_val if j == idx else c_train
        # Draw inner fold
        ax.add_patch(Rectangle((x0, y + 0.45), inner_width, 0.3,
                               facecolor=facecol, alpha=1,
                               edgecolor='k', linewidth=0.6, zorder=2))
        # Percentage label
        pct = val_ratio * 100 if j == idx else ((train_ratio - val_ratio) / (inner_folds - 1)) * 100
        ax.text(x0 + inner_width / 2, y + 0.6,
                f"{pct:.1f}%", ha='center', va='center',
                fontsize=8, color='white' if j == idx else 'white', zorder=3)
    # Outer fold label
    ax.text(-0.02, y + 0.4, f"Fold {idx + 1}", ha='right', va='center', fontsize=9)

# Draw all outer folds
for i in range(outer_folds):
    draw_outer_fold(ax, i)

# Axis labels and ticks
ax.set_xlim(0, 1.3)
ax.set_ylim(0, outer_folds)
ax.set_xticks([0, train_ratio, train_ratio + test_ratio])
ax.set_xticklabels(['0%', '80%', '100%'])
ax.set_yticks([])

# Add x-axis label
ax.set_xlabel('Data Partition', fontsize=12, labelpad=10)


# Legenda
legend_items = [
    Patch(facecolor=c_train, edgecolor='black', label='Train (outer)',alpha=alpha),
    Patch(facecolor=c_extra, edgecolor='black', label='Train (inner)',alpha=alpha),
    Patch(facecolor=c_val, edgecolor='black', label='Validation (HP-tuning)', alpha=alpha),
    Patch(facecolor=c_test, edgecolor='black', label='Test (outer)', alpha=alpha)
]
ax.legend(
    handles=legend_items,
    loc='upper right',
    bbox_to_anchor=(1, 1),       
    bbox_transform=ax.transAxes,
    frameon=True,
    fontsize=10
)
plt.tight_layout()
fig.subplots_adjust(right=0.9)
plt.show()