####本文件的作用是：读取转换为数字的学生答题记录文件，进行低记录的筛选，并对筛选后的记录保存为用户练习日志的json文件

import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # 或者使用 'TkAgg'
import matplotlib.pyplot as plt

# 读取已经转换为数字的 CSV 文件
file_path = "./data/pisa-science-2015-transform_to_number.csv"
df = pd.read_csv(file_path)

# 排除指定的字段
excluded_columns = ['STU_ID']
question_ids = [col for col in df.columns if col not in excluded_columns]

# 清理数据，删除没有有效回答的列
df_question = df[question_ids]
column_counts = df_question.count()
columns_with_zero_count = column_counts[column_counts == 0].index
df_question_cleaned = df_question.drop(columns=columns_with_zero_count)

# 计算每个学生的答题数量
df['Question Count'] = df_question_cleaned.notnull().sum(axis=1)

# 筛选答题数量大于或等于 30条有效记录的学生
df_student_filtered = df[df['Question Count'] >= 30]

# 计算筛选后学生的有效总答题数量
total_filtered_question_count = df_student_filtered['Question Count'].sum()

# 计算每个学生的平均得分
# average_scores = df_student_filtered[question_ids].mean(axis=1)
# 计算每个学生的平均得分
# df_student_filtered['Average Score'] = df_student_filtered[question_ids].mean(axis=1)
#
#
# # 获取 STU_ID 和 Average Score
# average_scores_with_id = df_student_filtered[['STU_ID', 'Average Score']]
#
# # 将 STU_ID 和平均得分保存到磁盘
# average_scores_with_id.to_csv('./data/average_scores_with_id.csv', index=False)

# # 将所有学生的平均得分存储在列表中
# average_scores_list = average_scores.tolist()
#
# # 将平均得分保存到磁盘
# with open('./data/average_scores.txt', 'w', encoding='utf8') as f:
#     for score in average_scores_list:
#         f.write(f"{score}\n")



# 输出结果
print("删除学习序列较短的学生后，还有{}人".format(len(df_student_filtered)))
print("删除学习序列较短的学生后，还有{}个有效答题数".format(total_filtered_question_count))
# 计算筛选后 df[q] 中值为 1 的题目数量
count_of_zeros_in_filtered = (df_student_filtered[question_ids] == 0).sum()
count_of_ones_in_filtered = (df_student_filtered[question_ids] == 1).sum()
# count_of_twos_in_filtered = (df_student_filtered[question_ids] == 2).sum()  #对于考虑半对情况的三分类使用

# 输出结果
print(f"筛选后为 0 的题目数量: {count_of_zeros_in_filtered.sum()}")
print(f"筛选后 为 1 的题目数量: {count_of_ones_in_filtered.sum()}")
# print(f"筛选后 为 2 的题目数量: {count_of_twos_in_filtered.sum()}")

print("-----------------------------------------------------------------------------------------------")

print("开始处理筛选后的数据，保存为 JSON 格式……")

# 加载 exer_id_mapping.json
with open('./data/exer_id_mapping.json', 'r') as file:
    exer_id_mapping = json.load(file)

# 加载 Q_matrix
Q_matrix = np.load('./data/Q_matrix.npy')

# 重新映射 STU_ID
stu_id_mapping = {stu_id: i + 1 for i, stu_id in enumerate(df_student_filtered['STU_ID'].unique())}
df_student_filtered['STU_ID'] = df_student_filtered['STU_ID'].map(stu_id_mapping)

# 计算每个学生的平均得分
df_student_filtered['Average Score'] = df_student_filtered[question_ids].mean(axis=1)


# 获取 STU_ID 和 Average Score
average_scores_with_stu_id = df_student_filtered[['STU_ID', 'Average Score']]
# 将 STU_ID 列减去 1
average_scores_with_stu_id['STU_ID'] = average_scores_with_stu_id['STU_ID'] - 1


# 将 STU_ID 和平均得分保存到磁盘
average_scores_with_stu_id.to_csv('./data/average_scores_with_stu_id.csv', index=False)

# # 处理每一行数据
# logs = []
# for index, row in df_student_filtered.iterrows():
#     user_id = row['STU_ID']
#     user_logs = []
#     valid_answer_count = 0  # 计数有效答案
#
#     for column in question_ids:  # 只处理题目相关列
#         exer_id = exer_id_mapping.get(column)  # 获取题目编号
#         score = row[column]  # 已经是数字
#
#         if exer_id is not None and not np.isnan(score):
#             # 获取知识点
#             knowledge_points = np.where(Q_matrix[exer_id - 1] > 0)[0] + 1  # 获取知识点编号
#             user_logs.append({
#                 "exer_id": exer_id,
#                 "score": score,
#                 "knowledge_code": knowledge_points.tolist()  # 转换为列表
#             })
#             # if score in [0, 1, 2]:  # 计算有效的答题个数
#             valid_answer_count += 1
#
#     logs.append({
#         "user_id": int(user_id),
#         "log_num": valid_answer_count,  # 有效答题个数
#         "logs": user_logs
#     })
#
# # 保存到 JSON 文件
# with open('./data/log_data-3-class.json', 'w') as outfile:
#     json.dump(logs, outfile, indent=4)
#
# print("处理完成，数据已保存为 JSON 格式。")
