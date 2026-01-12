"""
Timing Breakdown Visualization for MNIST in CUDA
Morandi vintage color palette with clean, spacious design.
"""
import matplotlib.pyplot as plt
import numpy as np

# ============ SciPainter Color Palette ============
COLORS = {
    'bg': '#FAFAFA',           # Clean white background
    'text': '#3D505A',         # Dark gray-blue text
    'grid': '#E0E0E0',         # Light grid
    'blue': '#699ECA',         # Sky blue
    'orange': '#FF8C00',       # Bright orange
    'pink': '#F898CB',         # Candy pink
    'green': '#4DAF4A',        # Fresh green
    'yellow': '#FFCB5B',       # Sunny yellow
    'cyan': '#0098B2',         # Cyan/teal
}

colors = [COLORS['green'], COLORS['blue'], COLORS['pink'], 
          COLORS['yellow'], COLORS['orange'], COLORS['cyan'], '#9B59B6', '#CCCCCC']  # 8 colors (+ gray for Other)

# Set global style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.facecolor': COLORS['bg'],
    'figure.facecolor': COLORS['bg'],
    'axes.edgecolor': COLORS['grid'],
    'axes.labelcolor': COLORS['text'],
    'xtick.color': COLORS['text'],
    'ytick.color': COLORS['text'],
    'text.color': COLORS['text'],
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Data (v5-v8 have different timing: async operations measured as kernel launch + stream sync)
versions = ['v1 PyTorch', 'v2 NumPy', 'v3 C', 'v4 CUDA', 'v5 cuBLAS', 'v6 Streams', 'v7 Custom', 'v8 FP16']
totals = [3.3, 22.4, 384.6, 1.6, 0.76, 0.47, 0.81, 0.38]

# Calculate "Other" to make each version sum to 100%
# Other includes: Python interpreter overhead, CUDA init, epoch loops, print, etc.
raw_data = {
    'Data Loading': [0.065, 0.016, 0.000, 0.135, 0.129, 0.013, 0.013, 0.014],
    'Forward': [0.666, 5.655, 272.2, 0.731, 0.00, 0.00, 0.00, 0.00],
    'Loss': [0.327, 0.597, 0.002, 0.002, 0.00, 0.00, 0.00, 0.00],
    'Backward': [1.449, 10.219, 106.4, 0.436, 0.00, 0.00, 0.00, 0.00],
    'Updates': [0.592, 5.919, 3.38, 0.168, 0.00, 0.00, 0.00, 0.00],
    'Kernel Launch': [0.00, 0.00, 0.00, 0.00, 0.00, 0.21, 0.56, 0.32],   # v6-v8: CPU time to submit async ops
    'Stream Sync': [0.00, 0.00, 0.00, 0.00, 0.631, 0.24, 0.24, 0.05],    # v5-v8: wait for GPU completion
}

# Calculate Other = total - sum(all categories)
other = []
for i, t in enumerate(totals):
    measured = sum(raw_data[k][i] for k in raw_data)
    other.append(max(0, t - measured))  # Ensure non-negative

data = raw_data.copy()
data['Other'] = other  # Python/CUDA overhead, epoch loops, print, etc.

# ============ Figure 1: Percentage Breakdown (Horizontal) ============
fig, ax = plt.subplots(figsize=(12, 8))  # Taller for 8 versions

# Convert to percentages
percentages = {k: [v / t * 100 for v, t in zip(vals, totals)] for k, vals in data.items()}

y = np.arange(len(versions))
height = 0.6
left = np.zeros(len(versions))

for i, (label, values) in enumerate(percentages.items()):
    bars = ax.barh(y, values, height, left=left, label=label, color=colors[i], edgecolor='white', linewidth=0.5)
    left += np.array(values)

# Add total time labels on left (before version name)
for i, (v, t) in enumerate(zip(versions, totals)):
    # Format time nicely
    if t >= 100:
        time_str = f'{t:.0f}s'
    elif t >= 1:
        time_str = f'{t:.1f}s'
    else:
        time_str = f'{t:.2f}s'
    ax.text(103, i, time_str, va='center', ha='left', fontsize=11, fontweight='bold', color=COLORS['text'])

ax.set_xlim(0, 120)
ax.set_xlabel('Time Distribution (%)', fontsize=12)
ax.set_yticks(y)
ax.set_yticklabels(versions, fontsize=11)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=7, frameon=False, fontsize=9)
ax.set_title('MNIST MLP Training · Time Breakdown (% of total)', fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, color=COLORS['grid'])

# Add percentage markers
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])

plt.tight_layout()
plt.savefig('assets/timing_analysis.png', dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
print("Saved: timing_analysis.png")

# ============ Figure 2: Version Progression Flowchart (Two Rows) ============
fig2, ax2 = plt.subplots(figsize=(14, 8))
ax2.set_xlim(0, 12)
ax2.set_ylim(0, 8)
ax2.axis('off')

# Version data (updated with latest benchmark results, including v7 and v8)
flow_data = [
    # Row 1: v1-v4 (left to right)
    {'name': 'v1.py', 'tech': 'PyTorch', 'time': '3.3s', 'speedup': '117×', 'color': COLORS['blue']},
    {'name': 'v2.py', 'tech': 'NumPy', 'time': '22.4s', 'speedup': '17×', 'color': COLORS['green']},
    {'name': 'v3.c', 'tech': 'C CPU', 'time': '384.6s', 'speedup': '1× (base)', 'color': COLORS['orange']},
    {'name': 'v4.cu', 'tech': 'CUDA', 'time': '1.6s', 'speedup': '240×', 'color': COLORS['pink']},
    # Row 2: v5-v8 (right to left, so store in reverse display order)
    {'name': 'v5.cu', 'tech': 'cuBLAS', 'time': '0.76s', 'speedup': '506×', 'color': COLORS['yellow']},
    {'name': 'v6.cu', 'tech': 'Streams', 'time': '0.47s', 'speedup': '818×', 'color': COLORS['cyan']},
    {'name': 'v7.cu', 'tech': 'Fused', 'time': '0.81s', 'speedup': '475×', 'color': '#9370DB'},
    {'name': 'v8.cu', 'tech': 'FP16', 'time': '0.38s', 'speedup': '1012×', 'color': '#FF6B6B'},
]

# Box positions: 4 boxes per row
x_row = [1.5, 4.0, 6.5, 9.0]
x_row_rev = [9.0, 6.5, 4.0, 1.5]  # Reversed for row 2
y_row1 = 5.5  # Top row
y_row2 = 2.0  # Bottom row
box_w, box_h = 2.0, 2.2

# Draw boxes and content
for i, d in enumerate(flow_data):
    if i < 4:
        x, y = x_row[i], y_row1  # Row 1: left to right
    else:
        x, y = x_row_rev[i - 4], y_row2  # Row 2: right to left
    
    # Box
    rect = plt.Rectangle((x - box_w/2, y - box_h/2), box_w, box_h,
                          facecolor=d['color'], edgecolor='white', linewidth=2, 
                          alpha=0.85, zorder=2)
    ax2.add_patch(rect)
    
    # Version name
    ax2.text(x, y + 0.6, d['name'], ha='center', va='center', 
             fontsize=13, fontweight='bold', color='white', zorder=3)
    # Tech
    ax2.text(x, y + 0.15, d['tech'], ha='center', va='center', 
             fontsize=11, color='white', alpha=0.9, zorder=3)
    # Time
    ax2.text(x, y - 0.35, d['time'], ha='center', va='center', 
             fontsize=14, fontweight='bold', color='white', zorder=3)
    # Speedup
    ax2.text(x, y - 0.8, d['speedup'], ha='center', va='center', 
             fontsize=10, color='white', alpha=0.85, zorder=3)

# Draw horizontal arrows within rows
arrow_style = dict(arrowstyle='->', color=COLORS['text'], lw=2, mutation_scale=15)

# Row 1 arrows (v1->v2->v3->v4, left to right)
for i in range(3):
    x_start = x_row[i] + box_w/2 + 0.05
    x_end = x_row[i+1] - box_w/2 - 0.05
    ax2.annotate('', xy=(x_end, y_row1), xytext=(x_start, y_row1),
                 arrowprops=arrow_style, zorder=1)

# Row 2 arrows (v5->v6->v7->v8, right to left visually)
for i in range(3):
    x_start = x_row_rev[i] - box_w/2 - 0.05  # Start from right side
    x_end = x_row_rev[i+1] + box_w/2 + 0.05  # End at left side
    ax2.annotate('', xy=(x_end, y_row2), xytext=(x_start, y_row2),
                 arrowprops=arrow_style, zorder=1)

# Vertical arrow from v4 to v5 (row transition, both on right side)
ax2.annotate('', xy=(x_row[3], y_row2 + box_h/2 + 0.1), 
             xytext=(x_row[3], y_row1 - box_h/2 - 0.1),
             arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=2, 
                           mutation_scale=15),
             zorder=1)

# Labels below arrows - Row 1 (left to right)
transitions_row1 = [('Remove autograd', 0), ('Remove BLAS', 1), ('GPU parallel', 2)]
for label, idx in transitions_row1:
    x_mid = (x_row[idx] + x_row[idx+1]) / 2
    ax2.text(x_mid, y_row1 - 1.5, label, ha='center', va='top', 
             fontsize=9, color=COLORS['text'], alpha=0.8)

# Labels below arrows - Row 2 (right to left: v5->v6->v7->v8)
transitions_row2 = [('Async Streams', 0), ('Custom GEMM', 1), ('Half Precision', 2)]
for label, idx in transitions_row2:
    x_mid = (x_row_rev[idx] + x_row_rev[idx+1]) / 2
    ax2.text(x_mid, y_row2 - 1.5, label, ha='center', va='top', 
             fontsize=9, color=COLORS['text'], alpha=0.8)

# Label for v4->v5 transition (vertical)
ax2.text(x_row[3] + 0.8, (y_row1 + y_row2) / 2, 'Use cuBLAS', ha='left', va='center', 
         fontsize=9, color=COLORS['text'], alpha=0.8)

# Title
ax2.text(5.25, 7.5, 'Version Progression · MNIST MLP Training', 
         ha='center', va='center', fontsize=14, fontweight='bold', color=COLORS['text'])

# Subtitle
ax2.text(5.25, 7.0, '784 → 1024 → 10  |  10 epochs  |  batch 32', 
         ha='center', va='center', fontsize=10, color=COLORS['text'], alpha=0.7)

plt.tight_layout()
plt.savefig('assets/speedup_comparison.png', dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
print("Saved: speedup_comparison.png")

