# This file's purpose is to read the original student answer records CSV file and convert them into corresponding data, such as: incorrect is 0, partially correct is 1, fully correct is 2.
# The converted data will then be saved to disk.

# Read the data
import pandas as pd
import numpy as np

# Read the CSV file
file_path = "./data/pisa-science-2015.csv"
df = pd.read_csv(file_path)

# Exclude specified columns
excluded_columns = ['STU_ID']
question_ids = [col for col in df.columns if col not in excluded_columns]

# Convert answers to numerical values
def transform_to_number(ans):
    if isinstance(ans, str):  # Ensure the answer is a string type
        if 'No credit' in ans:
            return 0
        if 'Partial credit' in ans:
            return 1
        if 'Full credit' in ans:
            return 2
    return np.NaN  # Return NaN for other cases

# Apply the transformation
for q in question_ids:
    df[q] = df[q].map(transform_to_number)  # Map to numbers
    #     count_of_ones = (df[q] == 1).sum()  # Count the number of ones
    #     counts[q] = count_of_ones  # Store the result in the dictionary
    #
    # # Output results
    # # Calculate the total number of ones across all questions
    # total_count_of_ones = sum(counts.values())
    #
    # # Output the total
    # print(f"Total number of ones across all questions: {total_count_of_ones}")

# Save the transformed DataFrame to a CSV file
output_file_path = "./data/pisa-science-2015-transform_to_number-3-class.csv"
df.to_csv(output_file_path, index=False)
print("transform_to_number conversion and saving successful!")
