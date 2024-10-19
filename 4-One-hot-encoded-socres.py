
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 确保使用适用于无界面环境的后端
import matplotlib.pyplot as plt
import numpy as np

# 读取包含重新映射的 STU_ID 和平均分数的 CSV 文件
file_path = './data/average_scores_with_stu_id.csv'
df = pd.read_csv(file_path)

# 计算平均分和标准差
mean_score = df['Average Score'].mean()
std_dev = df['Average Score'].std()

# 基于均值和标准差来定义分数区间
bins = [mean_score - 2 * std_dev,
        mean_score - std_dev,
        mean_score,
        mean_score + std_dev,
        mean_score + 2 * std_dev]
bins = [0] + bins + [100]  # 保证分数区间在 0-100 范围内

# 创建分数区间的标签
bin_labels = ['< -2σ', '-2σ to -1σ', '-1σ to Mean', 'Mean to +1σ', '+1σ to +2σ', '> +2σ']

# 将分数映射到区间
df['Score Range'] = pd.cut(df['Average Score'], bins=bins, labels=bin_labels, include_lowest=True)

# 计算每个分数段的人数
score_range_counts = df['Score Range'].value_counts().sort_index()

# # 绘制柱状图
plt.figure(figsize=(10, 6))
score_range_counts.plot(kind='bar', color='skyblue', edgecolor='black')

# 设置图表标题和标签
plt.title('Number of Students in Each Score Range Based on Mean and Standard Deviation')
plt.xlabel('Score Range')
plt.ylabel('Number of Students')
plt.xticks(rotation=45)

# 保存图表
plt.tight_layout()
plt.savefig('./data/score_distribution.png')
plt.close()

# 绘制柱状图
# plt.figure(figsize=(10, 6))
# ax = score_range_counts.plot(kind='bar', color='skyblue', edgecolor='black', label='Number of Students')
#
# # 添加轨迹线
# # 使用 ax.twinx() 创建一个共享 x 轴的第二个 y 轴，以便在柱状图上叠加折线图
# ax2 = ax.twinx()
# ax2.plot(score_range_counts.index, score_range_counts.values, color='red', marker='o', linestyle='-', linewidth=2, label='Trend Line')
#
# # 设置图表标题和标签
# ax.set_title('Number of Students in Each Score Range Based on Mean and Standard Deviation')
# ax.set_xlabel('Score Range')
# ax.set_ylabel('Number of Students')
# ax2.set_ylabel('Trend Line')
#
# # 设置 x 轴的标签旋转角度
# plt.xticks(rotation=45)
#
# # 添加图例
# ax.legend(loc='upper left')
# ax2.legend(loc='upper right')
#
# # 保存图表
# plt.tight_layout()
# plt.savefig('./data/score_distribution_with_trend_line.png')
# plt.close()
#
# print("柱状图和轨迹线已保存到 './data/score_distribution_with_trend_line.png'")

# One-Hot 编码分数区间
one_hot_mapping = {
    '< -2σ': 0,
    '-2σ to -1σ': 1,
    '-1σ to Mean': 2,
    'Mean to +1σ': 3,
    '+1σ to +2σ': 4,
    '> +2σ': 5
}

# 创建 One-Hot 编码的 DataFrame
one_hot_encoded = pd.DataFrame(0, index=df.index, columns=one_hot_mapping.values())

# 填充 One-Hot 编码
for score_range, code in one_hot_mapping.items():
    one_hot_encoded.loc[df['Score Range'] == score_range, code] = 1

# 将 One-Hot 编码添加回原始 DataFrame
df = pd.concat([df, one_hot_encoded], axis=1)

# 重命名 One-Hot 编码列
df.columns = list(df.columns[:-len(one_hot_mapping)]) + list(one_hot_mapping.keys())

# 保存包含 One-Hot 编码的 DataFrame 到 CSV 文件
one_hot_output_path = './data/one_hot_encoded_scores.csv'
df.to_csv(one_hot_output_path, index=False)

# 现在 df 中包含 One-Hot 编码的分数区间，可以进行进一步处理或分析
print(f"One-Hot 编码已成功保存到: {one_hot_output_path}")

