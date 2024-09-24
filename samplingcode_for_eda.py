import pandas as pd

# Read the dataset from the specified path
file_path =  r'C:\Users\AP\Documents\GitHub\DSCapstone\combined_data_split.csv'
df = pd.read_csv(file_path)

# Group by 'YEAR' and sample 5000 entries for each year
sampled_df = df.groupby('YEAR').apply(lambda x: x.sample(min(len(x), 5000))).reset_index(drop=True)

# Save the sampled dataset to a new CSV
output_path = r'C:\Users\AP\Documents\GitHub\DSCapstone\sampled_combined_data_split.csv'
sampled_df.to_csv(output_path, index=False)

# Display the first few rows of the sampled data
sampled_df.head()