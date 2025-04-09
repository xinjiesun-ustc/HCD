import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Ensure using a backend suitable for environments without a display
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file containing the remapped STU_ID and average scores
file_path = './data/average_scores_with_stu_id.csv'
df = pd.read_csv(file_path)

# Calculate the mean and standard deviation
mean_score = df['Average Score'].mean()
std_dev = df['Average Score'].std()

# Define score ranges based on mean and standard deviation
bins = [mean_score - 2 * std_dev,
        mean_score - std_dev,
        mean_score,
        mean_score + std_dev,
        mean_score + 2 * std_dev]
bins = [0] + bins + [100]  # Ensure the score ranges are within the 0-100 range

# Create labels for the score ranges
bin_labels = ['< -2σ', '-2σ to -1σ', '-1σ to Mean', 'Mean to +1σ', '+1σ to +2σ', '> +2σ']

# Map the scores to the defined ranges
df['Score Range'] = pd.cut(df['Average Score'], bins=bins, labels=bin_labels, include_lowest=True)

# Calculate the number of students in each score range
score_range_counts = df['Score Range'].value_counts().sort_index()

# Plot the bar chart
plt.figure(figsize=(10, 6))
score_range_counts.plot(kind='bar', color='skyblue', edgecolor='black')

# Set chart title and labels
plt.title('Number of Students in Each Score Range Based on Mean and Standard Deviation')
plt.xlabel('Score Range')
plt.ylabel('Number of Students')
plt.xticks(rotation=45)

# Save the chart
plt.tight_layout()
plt.savefig('./data/score_distribution.png')
plt.close()

# One-Hot Encoding for score ranges
one_hot_mapping = {
    '< -2σ': 0,
    '-2σ to -1σ': 1,
    '-1σ to Mean': 2,
    'Mean to +1σ': 3,
    '+1σ to +2σ': 4,
    '> +2σ': 5
}

# Create a DataFrame for One-Hot encoding
one_hot_encoded = pd.DataFrame(0, index=df.index, columns=one_hot_mapping.values())

# Fill the One-Hot encoding
for score_range, code in one_hot_mapping.items():
    one_hot_encoded.loc[df['Score Range'] == score_range, code] = 1

# Add the One-Hot encoding back to the original DataFrame
df = pd.concat([df, one_hot_encoded], axis=1)

# Rename the One-Hot encoded columns
df.columns = list(df.columns[:-len(one_hot_mapping)]) + list(one_hot_mapping.keys())

# Save the DataFrame with One-Hot encoding to a CSV file
one_hot_output_path = './data/one_hot_encoded_scores.csv'
df.to_csv(one_hot_output_path, index=False)

# The DataFrame now contains One-Hot encoded score ranges and can be used for further processing or analysis
print(f"One-Hot encoding has been successfully saved to: {one_hot_output_path}")
