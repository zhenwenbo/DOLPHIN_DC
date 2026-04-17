import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker


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


datasets = ['Bitcoin', 'GDELT']  # 
systems = ['TGL', 'ETC', 'SIMPLE', 'PipeTGL', 'DOLPHIN']
x = np.arange(len(systems))
width = 0.6


data = [
    
    {
        'Loading': np.array([689.38, 463.03, 1261.43, 837.85, 126.52]),
        'Sample': np.array([111.1, 80.98, 124.36, 164.13, 18.29]),
        'Training': np.array([504.44, 464.22, 499.4, 474.55, 397.21]),
        'status': ['', '', '', '', '']
    },
    
    {
        'Loading': np.array([9886.03, np.nan, 2076.89, 3562.48, 353.78]),
        'Sample': np.array([531.88, np.nan, 512.81, 525.62, 100.4]),
        'Training': np.array([2281.76, np.nan, 2056.22, 2146.33, 1692.83]),
        'status': ['', 'OOM', '', '', '']
    }
]

colors = ['lightgrey', 'grey', 'darkgrey']
hatches = ['', '//', 'xx']  # Loading:, Sample://, Training:xx
labels = ['Loading', 'Sample', 'Training']
border_color = 'black'
border_width = 1.0
border_linestyle = '-'


def format_scientific(x, pos):
    # 10²，
    val = x / 100
    return f'{val:.0f}'

scientific_formatter = mticker.FuncFormatter(format_scientific)

fig, axes = plt.subplots(1, 2, figsize=(15, 4)) 
fig.subplots_adjust(wspace=0.2, top=0.9)
l=1
for i, (ax, dataset, d) in enumerate(zip(axes, datasets, data)):
    bottom = np.zeros(len(systems))
    # Loading→Sample→Training
    for j, phase in enumerate(['Loading', 'Sample', 'Training']):
        values = np.array(d[phase])
        ax.bar(x, values, width, color=colors[j], hatch=hatches[j],
               edgecolor=border_color, linewidth=border_width, linestyle=border_linestyle,
               bottom=bottom, label=labels[j])
        bottom += np.nan_to_num(values)
    

    for j, sys in enumerate(systems):
        if d['status'][j]:
            ax.text(j, 0, d['status'][j], ha='center', va='bottom',
                    fontsize=22, color='black', weight='bold')
    
    # 
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=20,fontweight='bold')
    if(l == 1):
        ax.set_ylabel('Runtime (s)', fontsize=30, color='black', fontweight='bold')
    ax.tick_params(axis='y', labelsize=30)
    ax.set_xlabel(f"T-GNN System(L = {l})",fontsize=28, color='black', fontweight='bold') 
    l=l+1
    # ========== ：y10²+4 ==========
    if dataset == 'Bitcoin':
        y_max = 2000  # 20×10²
        y_ticks = [0, 1000, 2000]
        # y ×10² 
        ax.text(0.1, 1.01, r'$\times 10^2$', transform=ax.transAxes,
                fontsize=21, va='bottom', ha='right')
    else:  # GDELT
        y_max = 14000  # 140×10²
        y_ticks = [0, 7000, 14000]
        # y ×10² 
        ax.text(0.1, 1.01, r'$\times 10^2$', transform=ax.transAxes,
                fontsize=21, va='bottom', ha='right')
    
    ax.set_ylim(0, y_max)
    ax.set_yticks(y_ticks)  # 4
    ax.yaxis.set_major_formatter(scientific_formatter)  # 10²
    
    ax.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
    ax.spines[:].set_color('black')  # 

# ===================== （） =====================
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.09),
           ncol=3, fontsize=30, frameon=False, labelcolor='black')

# =====================  =====================
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figure8.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()