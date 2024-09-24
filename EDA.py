import pandas as pd

# Load the dataset
file_path = r'C:\Users\AP\Documents\GitHub\DSCapstone\combined_data_split.csv'
data = pd.read_csv(file_path)

# Inspect the first few rows
data.head()

# Check the data structure
data.info()

import pandas as pd
import matplotlib.pyplot as plt


# Group data by YEAR and OFFENSE, then count incidents
crime_by_year_offense = data.groupby(['YEAR', 'OFFENSE']).size().unstack().fillna(0)

# Calculate the proportion of each offense type per year
crime_by_year_offense_prop = crime_by_year_offense.div(crime_by_year_offense.sum(axis=1), axis=0)

# Plot stacked bar graph
crime_by_year_offense_prop.plot(kind='bar', stacked=True, figsize=(12, 8), cmap='tab20')
plt.title('Proportion of Crime Types by Year')
plt.xlabel('Year')
plt.ylabel('Proportion')
plt.legend(title='Offense Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Group by WARD and OFFENSE, then calculate proportions
crime_by_ward_offense = data.groupby(['WARD', 'OFFENSE']).size().unstack().fillna(0)
crime_by_ward_offense_prop = crime_by_ward_offense.div(crime_by_ward_offense.sum(axis=1), axis=0)

# Plot stacked bar graph
crime_by_ward_offense_prop.plot(kind='bar', stacked=True, figsize=(12, 8), cmap='tab20')
plt.title('Proportion of Crime Types by Ward')
plt.xlabel('Ward')
plt.ylabel('Proportion')
plt.legend(title='Offense Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Group by SHIFT and count incidents
crime_by_shift = data['SHIFT'].value_counts()

# Calculate the proportion of crime per shift
crime_by_shift_prop = crime_by_shift / crime_by_shift.sum()

# Display the proportions
print(crime_by_shift_prop)

import matplotlib.pyplot as plt

# Plot a bar chart for proportion of crime by shift
plt.figure(figsize=(8, 6))
crime_by_shift_prop.plot(kind='bar', color='skyblue')
plt.title('Proportion of Crime by Shift (Time of Day)')
plt.xlabel('Shift')
plt.ylabel('Proportion of Total Crimes')
plt.grid(True)
plt.show()

# Group data by YEAR and SHIFT, then count incidents
crime_by_year_shift = data.groupby(['YEAR', 'SHIFT']).size().unstack().fillna(0)

# Calculate the proportion of each shift per year
crime_by_year_shift_prop = crime_by_year_shift.div(crime_by_year_shift.sum(axis=1), axis=0)

# Display the proportions
print(crime_by_year_shift_prop)

# Plot stacked bar graph
crime_by_year_shift_prop.plot(kind='bar', stacked=True, figsize=(12, 8), cmap='Set3')
plt.title('Proportion of Crime by Shift for Each Year')
plt.xlabel('Year')
plt.ylabel('Proportion of Crimes')
plt.legend(title='Shift', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

