import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd

# Load all ppl_distribution files that start with ppl_distribution_hidden
ppl_files = sorted(glob.glob('/scratch2/mrenaudin/GTH/ppl_distribution_gpt2*.npz'))

# Create a dataframe with all distributions (sample for performance)
dfs = []
for file_path in ppl_files:
    data = np.load(file_path)
    perplexities = data['perplexities']
    label = file_path.split('/')[-1].replace('ppl_distribution_', '').replace('.npz', '')
    
    # Sample data for faster KDE computation (keep up to 10k points per dataset)
    if len(perplexities) > 10000:
        indices = np.random.choice(len(perplexities), 10000, replace=False)
        perplexities = perplexities[indices]
    
    # Create a DataFrame for this distribution
    df = pd.DataFrame({
        'PPL': perplexities,
        'Dataset': label
    })
    dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Create the plot with KDE overlay
fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(data=combined_df, x='PPL', hue='Dataset', fill=True, alpha=0.3, linewidth=2, ax=ax, palette='husl')
ax.set_xlabel('Perplexity (PPL)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('PPL Distributions Comparison', fontsize=14, fontweight='bold')
ax.set_xlim(0, 5000)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('/scratch2/mrenaudin/GTH/ppl_overlap_plot_gpt2.png', dpi=300, bbox_inches='tight')
print("Plot saved to ppl_overlap_plot_gpt2.png")
