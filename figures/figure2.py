import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 
throughput_data = {
    "Datasets": ["LFM", "WT", "SO", "BC", "GDELT"],
    "TGL": [43983.12925, 55982.99028, 48659.72627, 56506.84665, 33495.63345],
    "ETC": [67349.16667, 81476.3886, 62993.10516, 68703.01191, 53303.67069],
    "SIMPLE": [73849.45745, 69362.79111, 33682.04266, 37344.49544, 66058.04372],
    "DOLPHIN": [128284.127, 124335.5556, 117148.9059, 113859.8674, 128452.1105]
}
throughput_df = pd.DataFrame(throughput_data)


io_time_data_dict = {
    "batch_num": ["1,000", "3,000", "5,000", "7,000", "9,000"],
    "DOLPHIN": [8.66, 8.73, 8.86, 8.92, 9.11],
    "TGL": [12.71, 13.44, 15.42, 17.53, 21.82],
    "ETC": [11.76, 12.35, 13.28, 15.2, 17.04],
    "SIMPLE": [12.31, 12.99, 13.95, 16.54, 19.95]
}
io_time_df = pd.DataFrame(io_time_data_dict)

# ：-
methods = ["TGL", "ETC", "SIMPLE", "DOLPHIN"]
style_map = {
    "TGL": {"color": "white", "hatch": "\\", "marker": "o"},      
    "ETC": {"color": "gray", "hatch": "x", "marker": "s"},        
    "SIMPLE": {"color": "gray", "hatch": "/", "marker": "^"},      
    "DOLPHIN": {"color": "black", "hatch": "o", "marker": "D"}    
}
datasets = throughput_df["Datasets"].tolist()
batch_nums = io_time_df["batch_num"].tolist()
bar_width = 0.18
x = np.arange(len(datasets))



io_volume_data = throughput_df[methods].to_numpy().T
tgl_index = methods.index("TGL")
normalized_volume = np.zeros_like(io_volume_data, dtype=np.float64)
for col in range(io_volume_data.shape[1]):
    tgl_value = io_volume_data[tgl_index, col]
    normalized_volume[:, col] = io_volume_data[:, col] / tgl_value if tgl_value != 0 else np.nan


raw_time_data = io_time_df[methods].to_numpy().T



fig = plt.figure(figsize=(15, 5))  
gs = gridspec.GridSpec(
    nrows=3, ncols=2,  
    height_ratios=[1, 4, 0.8],  
    width_ratios=[1, 1],
    wspace=0.35,  
    hspace=0.05 
)

ax_left_legend = fig.add_subplot(gs[0, 0])
ax_left_legend.axis('off')


line_handles = []
for i, method in enumerate(methods):
    line = plt.Line2D(
        [], [],  # ，
        color="black",
    
        marker=style_map[method]["marker"],
        markersize=16,
        linewidth=3.5,
        markerfacecolor=style_map[method]["color"],
        markeredgecolor='black',
        markeredgewidth=2,
        label=method
    )
    line_handles.append(line)

ax_left_legend.legend(
    line_handles, methods,
    loc='center',
    ncol=2,
    fontsize=26,
    frameon=False,
    edgecolor='black',
    labelspacing=0.3,
    columnspacing=1.8,
    handlelength=2.2,
    handletextpad=0.8,
    numpoints=1
)

for text in ax_left_legend.get_legend().get_texts():
    text.set_fontweight('bold')



ax_left = fig.add_subplot(gs[1, 0])
# 
for spine in ax_left.spines.values():
    spine.set_linewidth(2.5)

for i, method in enumerate(methods):
    ax_left.plot(
        batch_nums,
        raw_time_data[i],
        color="black",
        marker=style_map[method]["marker"],
        markersize=13,
        linewidth=3.5,
        markerfacecolor=style_map[method]["color"],
        markeredgecolor='black',
        markeredgewidth=2
    )

# 
ax_left.set_ylabel("GPU Idle Time (s)", fontweight='bold', fontsize=26)
ax_left.set_xlabel("Number of Batches", fontweight='bold', fontsize=26)
ax_left.set_yticks([8, 12, 16, 20])
ax_left.set_yticklabels([8, 12, 16, 20], fontsize=26, fontweight='bold')
ax_left.set_xticks(range(len(batch_nums)))
ax_left.set_xticklabels(batch_nums, fontsize=23, fontweight='bold')
ax_left.set_ylim(7.5, 22.5)


ax_left_title = fig.add_subplot(gs[2, 0])
ax_left_title.axis('off')
# （）
ax_left_title.text(
    0.5, -0.75,  # ：
    r"(a) GPU Idle Time (WT)",
    ha='center', va='center',  # +
    fontsize=26,
    fontweight='bold'
)

ax_right_legend = fig.add_subplot(gs[0, 1])
ax_right_legend.axis('off')


bar_handles = []
for method in methods:
    bar = plt.Rectangle(
        (0, 0), 1.5, 1.5,  # 
        facecolor=style_map[method]["color"],
        hatch=style_map[method]["hatch"],
        edgecolor="black",
        linewidth=2,
        alpha=0.9,
        label=method
    )
    bar_handles.append(bar)

# 
ax_right_legend.legend(
    bar_handles, methods,
    loc='center',
    ncol=2,
    fontsize=26,
    frameon=False,
    edgecolor='black',
    labelspacing=0.3,
    columnspacing=1.8,
    handlelength=2.2,
    handletextpad=0.8
)
# 
for text in ax_right_legend.get_legend().get_texts():
    text.set_fontweight('bold')


#
ax_right = fig.add_subplot(gs[1, 1])
# 
for spine in ax_right.spines.values():
    spine.set_linewidth(2.5)

# 
for i, method in enumerate(methods):
    offset = x - 1.5 * bar_width + i * bar_width
    valid_indices = ~np.isnan(normalized_volume[i])
    ax_right.bar(
        offset[valid_indices],
        normalized_volume[i][valid_indices],
        width=bar_width,
        color=style_map[method]["color"],
        hatch=style_map[method]["hatch"],
        edgecolor="black",
        linewidth=2,
        alpha=0.9
    )


ax_right.set_ylabel("Norm. Throughput", fontweight='bold', fontsize=26)
ax_right.set_xlabel("datasets", fontweight='bold', fontsize=26)
ax_right.set_yticks([0, 1, 2, 3])
ax_right.set_yticklabels([0, 1, 2, 3], fontsize=26, fontweight='bold')
ax_right.set_xticks(x)
ax_right.set_xticklabels(datasets, fontsize=26, fontweight='bold')
ax_right.set_ylim(0, 4)

ax_right_title = fig.add_subplot(gs[2, 1])
ax_right_title.axis('off')
# （）
ax_right_title.text(
    0.5, -0.75,
    r"(b) Throughput Volume$\uparrow$",
    ha='center', va='center',
    fontsize=26,
    fontweight='bold'
)

plt.savefig(
    "figure2.pdf",
    dpi=600,
    bbox_inches='tight',  # 、、
    pad_inches=0.05
)
print(": figure2.pdf")
plt.show()