"""
MNIST MLP Training Loop Flowchart
Hand-drawn sketch style with warm paper background
"""
import warnings
import logging
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# ============ Color Palette ============
M = {
    'bg': '#F5EFE6',          # Warm paper background
    'text': '#3D3D3D',
    'dim': '#8B8680',
    'sage': '#7BA05B',        # Green
    'blue': '#5B8FA8',        # Blue  
    'terra': '#C27D4E',       # Orange/terracotta
    'mauve': '#8B7BA0',       # Purple
    'rose': '#B86B6B',        # Red/rose
}

plt.xkcd(scale=0.5, length=150, randomness=1.5)

fig, ax = plt.subplots(figsize=(24, 14))
ax.set_xlim(-1, 23)
ax.set_ylim(-0.5, 13.5)
ax.axis('off')
fig.patch.set_facecolor(M['bg'])
ax.set_facecolor(M['bg'])

def box(x, y, w, h, title, sub='', c=M['text'], fs_title=18, fs_sub=18):
    r = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,rounding_size=0.15",
                       fc=M['bg'], ec=c, lw=2.5)
    ax.add_patch(r)
    if sub:
        ax.text(x, y+h*0.22, title, ha='center', va='center', fontsize=fs_title, color=c, fontweight='bold')
        ax.text(x, y-h*0.18, sub, ha='center', va='center', fontsize=fs_sub, color=c, alpha=0.9)
    else:
        ax.text(x, y, title, ha='center', va='center', fontsize=fs_title, color=c, fontweight='bold')

def dual_box(x, y, w, h, t1, s1, t2, s2, c=M['text']):
    """Draw a single box divided into two columns"""
    r = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,rounding_size=0.15",
                       fc=M['bg'], ec=c, lw=2.5)
    ax.add_patch(r)
    ax.plot([x, x], [y-h/2+0.15, y+h/2-0.15], color=c, lw=1.5, alpha=0.5)
    ax.text(x-w/4, y+h*0.25, t1, ha='center', va='center', fontsize=20, color=c, fontweight='bold')
    ax.text(x-w/4, y-h*0.1, s1, ha='center', va='center', fontsize=18, color=c, alpha=0.9)
    ax.text(x+w/4, y+h*0.25, t2, ha='center', va='center', fontsize=20, color=c, fontweight='bold')
    ax.text(x+w/4, y-h*0.1, s2, ha='center', va='center', fontsize=18, color=c, alpha=0.9)

def arr(x1, y1, x2, y2, c=M['dim']):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=2.5))

# ============ TITLE (bottom-left) ============
ax.text(12.5, 6.5, 'MNIST MLP Training', ha='center', va='center', fontsize=32, color=M['text'], fontweight='bold')
ax.text(12.5, 5.5, 'Loop in CUDA C/C++', ha='center', va='center', fontsize=32, color=M['text'], fontweight='bold')

# ============ FORWARD PASS (TOP) ============
# Step 1: Input
ax.text(2.2, 12.2, 'Inputs', ha='center', va='center', fontsize=20, color=M['text'], fontweight='bold')
r = FancyBboxPatch((0.8, 9.6), 2.8, 2.2, boxstyle="round,rounding_size=0.15", fc=M['bg'], ec=M['sage'], lw=2.5)
ax.add_patch(r)
ax.text(2.2, 11.1, '28×28 px', ha='center', va='center', fontsize=20, color=M['sage'], fontweight='bold')
ax.text(2.2, 10.5, 'Image', ha='center', va='center', fontsize=18, color=M['sage'])
ax.text(2.2, 9.95, 'batched', ha='center', va='center', fontsize=18, color=M['sage'], alpha=0.8)
ax.text(0.7, 12.0, 'Step 1', ha='left', va='center', fontsize=18, color=M['dim'], style='italic')

# Arrow down
arr(2.2, 9.5, 2.2, 8.9)

# Step 2: Flatten
box(2.2, 7.8, 4.2, 1.6, 'Flattened Images - batched', '(B, 784)', M['sage'], fs_title=18, fs_sub=18)
ax.text(0.3, 6.75, 'Step 2', ha='left', va='center', fontsize=18, color=M['dim'], style='italic')

# Arrow right
arr(4.4, 7.8, 5.2, 7.8)

# Step 3: X1 @ W1
box(7.0, 7.8, 3.2, 1.6, 'X1 @ W1', '(B, 784) @ (784, 256)\n= (B, 256)', M['blue'])
ax.text(5.3, 6.75, 'Step 3', ha='left', va='center', fontsize=18, color=M['dim'], style='italic')

# Arrow up
arr(7.0, 8.65, 7.0, 9.6)

# Step 4: ReLU
box(7.0, 10.6, 2.8, 1.3, 'ReLU', '(B, 256) -> x2 = relu(x)', M['blue'], fs_sub=18)
ax.text(5.5, 9.75, 'Step 4', ha='left', va='center', fontsize=18, color=M['dim'], style='italic')

# Arrow right
arr(8.45, 10.6, 9.4, 10.6)

# Step 5: X2 @ W2
box(10.8, 10.6, 2.8, 1.4, 'X2 @ W2', '(B, 256) @ (256, 10)\n= (B, 10)', M['mauve'])
ax.text(9.3, 11.5, 'Step 5', ha='left', va='center', fontsize=18, color=M['dim'], style='italic')

# Arrow right
arr(12.25, 10.6, 13.2, 10.6)

# Step 6: Loss (larger box)
box(15.0, 10.6, 4.0, 1.6, 'loss = -ln(0.1) = ~2.3', 'CrossEntropyLoss(prob dist, labels)', M['rose'], fs_title=18, fs_sub=18)
ax.text(12.9, 11.6, 'Step 6', ha='left', va='center', fontsize=18, color=M['dim'], style='italic')

# Arrow right
arr(17.05, 10.6, 17.8, 10.6)

# dLoss - Step 7
box(19.7, 10.6, 3.6, 1.5, 'dLoss =', 'element-wise...\nsoftmax_probs - y_true_one_hot', M['rose'], fs_sub=18)
ax.text(21.5, 11.55, 'Step 7', ha='left', va='center', fontsize=18, color=M['dim'], style='italic')


# ============ BACKWARD PASS ============
# Arrow down
arr(19.7, 9.8, 19.7, 9.1)

# Step 8: dW2 + dX2 (combined box)
dual_box(19.7, 7.8, 5.6, 2.0, 
         'dW2 =', 'X2.T @ dLoss\n(256, B) @ (B, 10)\n= (256, 10)',
         'dX2 =', 'dLoss @ W2.T\n(B, 10) @ (10, 256)\n= (B, 256)', M['terra'])
ax.text(22.4, 9.0, 'Step 8', ha='left', va='center', fontsize=18, color=M['dim'], style='italic')

# Arrow down (Step 9 directly below Step 8)
arr(19.7, 6.75, 19.7, 6.0)

# Step 9: d_ReLU_out (centered below Step 8)
box(19.7, 4.9, 3.2, 1.6, 'd_ReLU_out', 'dX2 * dReLU(x)\n= (B, 256)', M['terra'])
ax.text(21.4, 5.9, 'Step 9', ha='left', va='center', fontsize=18, color=M['dim'], style='italic')

# Arrow down from d_ReLU_out to dW1/dX1
arr(19.7, 4.05, 19.7, 3.4)

# Step 10: dW1 + dX1 (combined box, centered below Step 9)
dual_box(19.7, 2.2, 5.6, 2.0,
         'dW1 =', 'X1.T @ d_ReLU_out\n(784, B) @ (B, 256)\n= (784, 256)',
         'dX1 =', 'd_ReLU_out @ W1.T\n(B, 256) @ (256, 784)\n= (B, 784)', M['sage'])
ax.text(22.4, 3.4, 'Step 10', ha='left', va='center', fontsize=18, color=M['dim'], style='italic')

# Arrow left
arr(16.65, 2.2, 12.2, 2.2)

# Step 11: Weight update
box(9.5, 2.2, 5.3, 2.0, 'Element-wise gradient updates.', 'w_i -= dw_i * learning_rate\n\nSet grads back to zero.', M['text'], fs_title=18)
ax.text(9.5, 0.85, 'Step 11', ha='center', va='center', fontsize=18, color=M['dim'], style='italic')

# Loop arrow back up to X1 @ W1 (straight line)
arr(7.0, 3.5, 7.0, 6.8)

plt.tight_layout()
plt.savefig('assets/mlp_training_flow.png', dpi=150, bbox_inches='tight', facecolor=M['bg'])
print("Saved: mlp_training_flow.png")
plt.rcdefaults()

