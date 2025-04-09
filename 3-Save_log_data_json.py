# This file's purpose is to read the CSV file with converted numerical student answer records, perform filtering based on the number of records, and save the filtered records as a user exercise log in a JSON file.

import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # Alternatively, use 'TkAgg'
import matplotlib.pyplot as plt

# Read the CSV file with converted numerical answers
file_path = "./data/pisa-science-2015-transform_to_number.csv"
df = pd.read_csv(file_path)

# Exclude specified columns
excluded_columns = ['STU_ID']
question_ids = [col for col in df.columns if col not in excluded_columns]

# Clean the data by removing columns with no valid answers
df_question = df[question_ids]
column_counts = df_question.count()
columns_with_zero_count = column_counts[column_counts == 0].index
df_question_cleaned = df_question.drop(columns=columns_with_zero_count)

# Calculate the number of answers per student
df['Question Count'] = df_question_cleaned.notnull().sum(axis=1)

# Filter students who have answered 30 or more valid questions
df_student_filtered = df[df['Question Count'] >= 30]

# Calculate the total number of valid answers for the filtered students
total_filtered_question_count = df_student_filtered['Question Count'].sum()

# Output results
print("After deleting students with shorter learning sequences, there are {} students left.".format(len(df_student_filtered)))
print("After deleting students with shorter learning sequences, there are {} valid answers left.".format(total_filtered_question_count))

# Count the number of '0' and '1' answers in the filtered data
count_of_zeros_in_filtered = (df_student_filtered[question_ids] == 0).sum()
count_of_ones_in_filtered = (df_student_filtered[question_ids] == 1).sum()

# Output the results
print(f"Number of questions with 0 answers after filtering: {count_of_zeros_in_filtered.sum()}")
print(f"Number of questions with 1 answer after filtering: {count_of_ones_in_filtered.sum()}")

print("-----------------------------------------------------------------------------------------------")

print("Processing the filtered data and saving it in JSON format...")

# Load the exer_id_mapping.json
with open('./data/exer_id_mapping.json', 'r') as file:
    exer_id_mapping = json.load(file)

# Load the Q_matrix
Q_matrix = np.load('./data/Q_matrix.npy')

# Remap STU_ID
stu_id_mapping = {stu_id: i + 1 for i, stu_id in enumerate(df_student_filtered['STU_ID'].unique())}
df_student_filtered['STU_ID'] = df_student_filtered['STU_ID'].map(stu_id_mapping)

# Calculate each student's average score
df_student_filtered['Average Score'] = df_student_filtered[question_ids].mean(axis=1)

# Get STU_ID and Average Score
average_scores_with_stu_id = df_student_filtered[['STU_ID', 'Average Score']]
# Subtract 1 from the STU_ID column
average_scores_with_stu_id['STU_ID'] = average_scores_with_stu_id['STU_ID'] - 1

# Save the STU_ID and average score to a CSV file
average_scores_with_stu_id.to_csv('./data/average_scores_with_stu_id.csv', index=False)

# Process each row (this section is commented out for now)
# logs = []
# for index, row in df_student_filtered.iterrows():
#     user_id = row['STU_ID']
#     user_logs = []
#     valid_answer_count = 0  # Count valid answers
#
#     for column in question_ids:  # Process only question-related columns
#         exer_id = exer_id_mapping.get(column)  # Get exercise ID
#         score = row[column]  # This is already a number
#
#         if exer_id is not None and not np.isnan(score):
#             # Get knowledge points
#             knowledge_points = np.where(Q_matrix[exer_id - 1] > 0)[0] + 1  # Get knowledge point IDs
#             user_logs.append({
#                 "exer_id": exer_id,
#                 "score": score,
#                 "knowledge_code": knowledge_points.tolist()  # Convert to a list
#             })
#             valid_answer_count += 1
#
#     logs.append({
#         "user_id": int(user_id),
#         "log_num": valid_answer_count,  # Valid answer count
#         "logs": user_logs
#     })
#
# # Save to JSON file
# with open('./data/log_data-3-class.json', 'w') as outfile:
#     json.dump(logs, outfile, indent=4)
#
# print("Processing complete, data has been saved as JSON format.")
