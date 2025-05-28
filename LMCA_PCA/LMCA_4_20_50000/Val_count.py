import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
raw_data = pd.read_csv("data/train_1_avg.csv")

# 统计标签分布
label_counts = raw_data[' Label'].value_counts()

# 打印类别及数量到终端
print("Label Distribution:")
print(label_counts)

# 可视化标签分布
plt.figure(figsize=(10, 6))
bars = plt.bar(label_counts.index, label_counts.values)

# 在柱状图上标注具体数量
for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(bar.get_height()),
        ha='center',
        va='bottom',
        fontsize=10
    )

# 设置图形的其他细节
plt.xticks(rotation=90)
plt.title("Label Distribution", fontsize=14)
plt.xlabel("Labels", fontsize=12)
plt.ylabel("Count", fontsize=12)

# 保存柱状图
plt.tight_layout()
plt.savefig("data/plots/label_counts_bar_chartextrain_1_avg.png")
plt.close()
