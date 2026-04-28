import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import glob

# Load all ppl_distribution files
ppl_files = sorted(glob.glob('/scratch2/mrenaudin/GTH/ppl_distribution_*.npz'))

# Create subplots
n_files = len(ppl_files)
n_cols = 3
n_rows = (n_files + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
axes = axes.flatten()  # Flatten to make indexing easier

# Plot each distribution
for idx, file_path in enumerate(ppl_files):
    data = np.load(file_path)
    perplexities = data['perplexities']
    
    # Get a nice label from the filename
    label = file_path.split('/')[-1].replace('ppl_distribution_', '').replace('.npz', '')
    
    # Plot histogram using matplotlib
    ax = axes[idx]
    ax.hist(perplexities, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Perplexity (PPL)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

# Remove extra subplots
for idx in range(len(ppl_files), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('/scratch2/mrenaudin/GTH/ppl_distributions_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved to ppl_distributions_plot.png")
plt.show()

# Print summary statistics
print("\n" + "="*80)
print("Summary Statistics:")
print("="*80)
for file_path in ppl_files:
    data = np.load(file_path)
    perplexities = data['perplexities']
    label = file_path.split('/')[-1].replace('ppl_distribution_', '').replace('.npz', '')
    
    print(f"\n{label}")
    print(f"  Count: {len(perplexities)}")
    print(f"  Mean: {perplexities.mean():.2f}")
    print(f"  Median: {np.median(perplexities):.2f}")
    print(f"  Std: {perplexities.std():.2f}")
    print(f"  Min: {perplexities.min():.2f}")
    print(f"  Max: {perplexities.max():.2f}")
