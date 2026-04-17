import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec

throughput_data = {
    "Datasets": ["WT","SO","BC","GDELT"],
    "w/o Planing": [112.93,1304.92,1976.81,2895.8],
    "DOLPHIN": [63,542.02,1079.82,1489.2],
}
throughput_df = pd.DataFrame(throughput_data)

# I/O
io_time_data_dict = {
    "Datasets": ["WT","SO","BC","GDELT"],
    "w/o Planing":[359.91,3245.92,3489.22,12443.72],
    "DOLPHIN":  [225.32,2147.01,2048.13,7794.32],
}
io_time_df = pd.DataFrame(io_time_data_dict)

# 
methods = ["w/o Planing", "DOLPHIN"]
datasets = throughput_df["Datasets"].tolist()
bar_width = 0.25
x = np.arange(len(datasets))
colors = ["white",  "black"]
hatches = ['/',  "o"] 

# （）
io_volume_data = throughput_df[methods].to_numpy().T
tgl_index = methods.index("DOLPHIN")
normalized_volume = np.zeros_like(io_volume_data, dtype=np.float64)
for col in range(io_volume_data.shape[1]):
    tgl_value = io_volume_data[tgl_index, col]
    normalized_volume[:, col] = io_volume_data[:, col] / 1 if tgl_value != 0 else np.nan

io_time_data = io_time_df[methods].to_numpy().T
dolphin_index = methods.index("DOLPHIN")
normalized_time = np.zeros_like(io_time_data, dtype=np.float64)
for col in range(io_time_data.shape[1]):
    dolphin_value = io_time_data[dolphin_index, col]
    normalized_time[:, col] = io_time_data[:, col] / 1 if dolphin_value != 0 else np.nan


fig = plt.figure(figsize=(14, 10))  


gs = gridspec.GridSpec(
    nrows=3, ncols=2, 
    height_ratios=[1, 4, 4], 
    width_ratios=[1, 1],     
    wspace=0.2,  
    hspace=0.2  
)

ax_legend = fig.add_subplot(gs[0, :])  
ax_legend.axis('off')  # ，

handles = [plt.Rectangle((0, 0), 1.5, 1.5, 
                         facecolor=colors[i], 
                         edgecolor='black', 
                         hatch=hatches[i],
                         linewidth=1.5)  # ，
           for i in range(len(methods))]


legend = ax_legend.legend(
    handles, methods,
    loc='center',              
    bbox_to_anchor=(0.45, 0.15), 
    ncol=len(methods),         
    fontsize=29,               
    frameon=False,              
    labelspacing=0.2,          
    columnspacing=1.5,        
    handlelength=1.8,          
    handletextpad=0.6 ,        
    borderaxespad=0.5    
)

ax_left = fig.add_subplot(gs[1, 0])
ax_left.set_title(
    r"Layer = 1 $\downarrow$", 
    fontsize=30,          
    y = -0.3,    
    pad=10          
)
 
for spine in ax_left.spines.values():
    spine.set_linewidth(2)  # 
for i, method in enumerate(methods):
    offset = x - 1 * bar_width + i * bar_width  
    valid_indices = ~np.isnan(normalized_volume[i])
    ax_left.bar(
        offset[valid_indices], 
        normalized_volume[i][valid_indices], 
        width=bar_width, 
        color=colors[i], 
        hatch=hatches[i], 
        edgecolor="black",
        linewidth=1.5  # 
    )
ax_left.set_ylabel("Runtime(s)", fontweight='bold', fontsize=30)


ax_left.set_xticks(x)
ax_left.set_xticklabels(datasets, fontweight='bold')
ax_left.set_yscale('log')
ax_left.tick_params(
    axis='both',          
    labelsize=25,     
    width=2,          
    length=6         
)
# 5. （Stalling Time）：（gs[1, 1]）
ax_right = fig.add_subplot(gs[1, 1])
ax_right.set_title(
    r"Layer = 2 $\downarrow$", 
    fontsize=30,         
    pad=10,   
    y = -0.3                       
)
for spine in ax_right.spines.values():
    spine.set_linewidth(2)  # 
for i, method in enumerate(methods):
    offset = x - 1 * bar_width + i * bar_width  
    valid_indices = ~np.isnan(normalized_time[i])
    ax_right.bar(
        offset[valid_indices], 
        normalized_time[i][valid_indices], 
        width=bar_width, 
        color=colors[i], 
        hatch=hatches[i], 
        edgecolor="black",
        linewidth=1.3  # 
    )

ax_right.tick_params(
    axis='y',          #
    labelsize=16,      # 
    width=2,           # 
    length=6           # 
)

ax_right.set_xticks(x)
ax_right.set_yscale('log') 
ax_right.set_xticklabels(datasets, fontsize=16,fontweight='bold')
ax_right.tick_params(
    axis='both',         
    labelsize=25,    
    width=2,       
    length=6          
)


plt.savefig(
    "figure14.pdf", 
    dpi=600,           
    bbox_inches='tight', 
    pad_inches=0.05    
)
plt.close(fig)
plt.show()