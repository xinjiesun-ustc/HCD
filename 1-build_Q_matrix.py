import pandas as pd
import numpy as np
import json

# This file's purpose is to generate the Q matrix, knowledge point mapping, exercise ID mapping, and the relationship between exercises and knowledge points

# Read the CSV file, trying different encodings
file_path = "./data/pisa2015-science-kcs.csv"
kc_list = []
exercises_data = {}

try:
    df = pd.read_csv(file_path, encoding='ISO-8859-1')  # Or try 'utf-16'

    # Create exercise ID mapping dictionary
    exer_id_mapping = {row['ID']: i + 1 for i, row in df.iterrows()}  # ID starts from 1

    # Extract all unique knowledge points
    for column in df.columns:
        if column != 'ID':
            unique_values = df[column].unique()  # Get the unique values for each column
            kc_list.extend(unique_values)  # Directly add to kc_list

    # Create unique knowledge point set and mapping
    unique_kc = set(kc_list)
    knowledge_points_map = {kc: i for i, kc in enumerate(sorted(unique_kc))}

    # Store the knowledge points for each exercise
    for index, row in df.iterrows():
        exer_id = exer_id_mapping[row['ID']]
        knowledge_list = []
        for column in df.columns:
            if column != 'ID':
                if row[column] in knowledge_points_map:
                    knowledge_list.append(row[column])
        exercises_data[exer_id] = knowledge_list

    # Retain only the specific data from kc_list
    kc_list = list(unique_kc)
    print("Knowledge points KC mapping dictionary:", knowledge_points_map)
    print("Knowledge points KC:", kc_list)
    print("Number of knowledge points KC:", len(kc_list))

except Exception as e:
    print(f"An error occurred: {e}")

# Build the Q matrix
num_exercises = len(exercises_data)
num_knowledge_points = len(knowledge_points_map)

# Initialize the Q matrix
Q_matrix = np.zeros((num_exercises, num_knowledge_points))

# Fill in the Q matrix
for i, (exer_id, knowledge_list) in enumerate(exercises_data.items()):
    for kc in knowledge_list:
        if kc in knowledge_points_map:
            Q_matrix[i, knowledge_points_map[kc]] = 1

# Save the Q matrix and exercise ID mapping to files
np.save('./data/Q_matrix.npy', Q_matrix)
np.save('./data/exercises_data.npy', exercises_data)

# Save the exercise ID mapping as a JSON file
with open('./data/exer_id_mapping.json', 'w') as f:
    json.dump(exer_id_mapping, f)

# Save the KC mapping as a JSON file
with open('./data/knowledge_points_map.json', 'w') as f:
    json.dump(knowledge_points_map, f)
print("Q matrix and ID mappings have been saved.")
