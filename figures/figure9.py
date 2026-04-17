import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

# ===================== （+） =====================
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['legend.frameon'] = False
plt.rcParams['hatch.linewidth'] = 0.5
# 
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'


datasets = ['Reddit', 'Wikipedia']  # 
systems = ['TGL', 'ETC', 'SIMPLE', 'PipeTGL', 'DOLPHIN']
# x（：x）
x = np.array([0, 1, 2, 3, 4])  
width = 0.6


data = [
    {
        'Loading': np.array([3925.49, np.nan, 1195.96, 3048.88, 179.36]),
        'Sample': np.array([352.62, np.nan, 356.88, 362.81, 58.2]),
        'Training': np.array([1432.81, np.nan, 1342.96, 1364.82, 1251.64]),
        'status': ['', 'OOM', '', '', '']
    },
    {
        'Loading': np.array([np.nan, np.nan, np.nan, np.nan, 1836.16]),
        'Sample': np.array([np.nan, np.nan, np.nan, np.nan, 838.74]),
        'Training': np.array([np.nan, np.nan, np.nan, np.nan, 5119.42]),
        'status': ['TLE', 'OOM', 'TLE', 'TLE', '']
    }
]

# ===================== 、 =====================
colors = ['lightgrey', 'grey', 'darkgrey']
hatches = ['', '//', 'xx']
labels = ['Loading', 'Sample', 'Training']
border_color = 'black'
border_width = 1.0
border_linestyle = '-'

# =====================  =====================
def format_scientific(x, pos):
    val = x / 100
    return f'{val:.0f}'

scientific_formatter = mticker.FuncFormatter(format_scientific)

# ===================== （1×2） =====================
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
fig.subplots_adjust(wspace=0.2, top=0.9)
l = 1

for i, (ax, dataset, d) in enumerate(zip(axes, datasets, data)):
    bottom = np.zeros(len(systems))
    # Loading→Sample→Training
    for j, phase in enumerate(['Loading', 'Sample', 'Training']):
        values = np.array(d[phase])
        ax.bar(x, values, width, color=colors[j], hatch=hatches[j],
               edgecolor=border_color, linewidth=border_width, linestyle=border_linestyle,
               bottom=bottom, label=labels[j])
        bottom += np.nan_to_num(values)
    
    # OOM/TLE（）
    for j, sys in enumerate(systems):
        if d['status'][j]:
            # x，
            ax.text(x[j], 0, d['status'][j], ha='center', va='bottom',
                    fontsize=22, color='black', weight='bold')
    
    # ========== ：x ==========
    # x，
    ax.set_xticks(x)
    # ，
    ax.set_xticklabels(systems, fontsize=20, fontweight='bold')
    # x，
    ax.set_xlim(-0.5, 4.5)
    
    # 
    if l == 1:
        ax.set_ylabel('Runtime (s)', fontsize=30, color='black', fontweight='bold')
    ax.tick_params(axis='y', labelsize=30)
    ax.set_xlabel(f"T-GNN System(L = {l})", fontsize=28, color='black', fontweight='bold')
    l += 1
    
    # y
    if dataset == 'Reddit':
        y_max = 6000
        y_ticks = [0, 3000, 6000]
    else:
        y_max = 8000
        y_ticks = [0, 4000, 8000]
    
    ax.text(0.1, 1.01, r'$\times 10^2$', transform=ax.transAxes,
            fontsize=21, va='bottom', ha='right')
    ax.set_ylim(0, y_max)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(scientific_formatter)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
    ax.spines[:].set_color('black')

# =====================  =====================
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.09),
           ncol=3, fontsize=30, frameon=False, labelcolor='black')

# =====================  =====================
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figure9.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()