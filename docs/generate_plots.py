#!/usr/bin/env python3
"""
Performance visualization script for CUDA kernels
Generates plots for SGEMM optimization progress and memory bandwidth utilization
Saves plots to the profiling directory for documentation
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

def plot_sgemm_optimization():
    """Plot SGEMM optimization progress"""
    versions = ['naive', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']
    gflops = [2094, 2611, 6405, 2830, 2828, 9933, 15309]
    cublas_percent = [10.8, 13.5, 33.0, 14.6, 14.6, 51.0, 83.4]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # GFLOPS progression
    bars1 = ax1.bar(versions, gflops, color='skyblue', alpha=0.8)
    ax1.axhline(y=18356, color='red', linestyle='--', label='cuBLAS (18,356 GFLOPS)')
    ax1.set_ylabel('GFLOPS')
    ax1.set_title('SGEMM Performance Optimization Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, gflops):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{val:,}', ha='center', va='bottom', fontweight='bold')
    
    # Percentage of cuBLAS
    bars2 = ax2.bar(versions, cublas_percent, color='lightcoral', alpha=0.8)
    ax2.axhline(y=100, color='red', linestyle='--', label='cuBLAS (100%)')
    ax2.axhline(y=65, color='orange', linestyle=':', label='Original Target (65%)')
    ax2.set_ylabel('% of cuBLAS Performance')
    ax2.set_title('SGEMM Optimization: % of cuBLAS Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars2, cublas_percent):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sgemm_optimization_progress.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_memory_bandwidth():
    """Plot memory bandwidth utilization for elementwise operations"""
    kernels = ['ADD\n(float4)', 'SIGMOID', 'RELU']
    bandwidth = [846.67, 832.00, 836.90]
    peak_bandwidth = 936.10
    efficiency = [bw/peak_bandwidth*100 for bw in bandwidth]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bandwidth comparison
    bars1 = ax1.bar(kernels, bandwidth, color='lightgreen', alpha=0.8)
    ax1.axhline(y=peak_bandwidth, color='red', linestyle='--', 
                label=f'Peak Bandwidth ({peak_bandwidth:.1f} GB/s)')
    ax1.set_ylabel('Bandwidth (GB/s)')
    ax1.set_title('Memory Bandwidth Utilization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, bandwidth):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Efficiency percentage
    bars2 = ax2.bar(kernels, efficiency, color='gold', alpha=0.8)
    ax2.axhline(y=100, color='red', linestyle='--', label='Peak Efficiency (100%)')
    ax2.axhline(y=85, color='orange', linestyle=':', label='Target (85%)')
    ax2.set_ylabel('Efficiency (% of Peak)')
    ax2.set_title('Memory Bandwidth Efficiency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, efficiency):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('memory_bandwidth_utilization.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_scalability_analysis():
    """Plot SGEMM scalability across different matrix sizes"""
    matrix_sizes = ['256³', '512³', '1024³', '2048³', '4096³']
    v7_gflops = [0, 3838, 15309, 18973, 19707]  # v7 performance
    cublas_gflops = [3818, 10793, 18356, 24045, 24275]  # cuBLAS reference
    v7_percent = [0, 35.6, 83.4, 78.9, 81.2]  # % of cuBLAS
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Absolute performance
    x = np.arange(len(matrix_sizes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, v7_gflops, width, label='Custom v7', color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, cublas_gflops, width, label='cuBLAS', color='lightcoral', alpha=0.8)
    
    ax1.set_ylabel('GFLOPS')
    ax1.set_title('SGEMM Scalability: Absolute Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(matrix_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Percentage of cuBLAS
    bars3 = ax2.bar(matrix_sizes[1:], v7_percent[1:], color='gold', alpha=0.8)
    ax2.axhline(y=100, color='red', linestyle='--', label='cuBLAS (100%)')
    ax2.axhline(y=80, color='orange', linestyle=':', label='Excellent (80%)')
    ax2.set_ylabel('% of cuBLAS Performance')
    ax2.set_title('SGEMM Scalability: Efficiency vs Matrix Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars3, v7_percent[1:]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sgemm_scalability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_optimization_techniques():
    """Plot the impact of different optimization techniques"""
    techniques = ['Baseline\n(naive)', 'Shared\nMemory', '1D Thread\nTiling', 
                  'Vectorized\nLoads', 'Double\nBuffering']
    gflops = [2094, 2611, 6405, 9933, 15309]
    speedup = [x/2094 for x in gflops]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # GFLOPS progression
    bars1 = ax1.bar(techniques, gflops, color=['gray', 'lightblue', 'lightgreen', 
                                               'gold', 'lightcoral'], alpha=0.8)
    ax1.set_ylabel('GFLOPS')
    ax1.set_title('Impact of Optimization Techniques')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, gflops):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{val:,}', ha='center', va='bottom', fontweight='bold')
    
    # Speedup factors
    bars2 = ax2.bar(techniques, speedup, color=['gray', 'lightblue', 'lightgreen', 
                                                'gold', 'lightcoral'], alpha=0.8)
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Cumulative Speedup from Optimizations')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, speedup):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}×', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('optimization_techniques_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all performance plots in the profiling directory"""
    print("Generating CUDA kernels performance plots...")
    print(f"Working directory: {os.getcwd()}")
    
    plot_sgemm_optimization()
    print("✓ SGEMM optimization progress plot saved")
    
    plot_memory_bandwidth()
    print("✓ Memory bandwidth utilization plot saved")
    
    plot_scalability_analysis()
    print("✓ Scalability analysis plot saved")
    
    plot_optimization_techniques()
    print("✓ Optimization techniques impact plot saved")
    
    print("\nAll plots generated successfully!")
    print("Files saved in current directory:")
    for filename in ['sgemm_optimization_progress.png', 'memory_bandwidth_utilization.png', 
                     'sgemm_scalability_analysis.png', 'optimization_techniques_impact.png']:
        if os.path.exists(filename):
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} (missing)")

if __name__ == "__main__":
    main()