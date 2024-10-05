import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv(r'C:\Users\AP\Documents\GitHub\DSCapstone\DataCleaningandProcessing\combined_data.csv')

# Prepare data for predicting crime type and ward
# Filtering the relevant columns and rows
df = df[['YEAR', 'MONTH', 'DAY', 'HOUR', 'WARD', 'OFFENSE']]  # Use the necessary columns

# Step 1: Train a model to predict crime type
X = df[['YEAR', 'MONTH', 'DAY', 'HOUR']]  # Features
y_offense = df['OFFENSE']  # Target for crime type

# Train-test split
X_train_offense, X_test_offense, y_train_offense, y_test_offense = train_test_split(X, y_offense, test_size=0.2, random_state=42)

# Random Forest Classifier for crime type
crime_type_model = RandomForestClassifier()
crime_type_model.fit(X_train_offense, y_train_offense)

# Step 2: Train a model to predict the ward
y_ward = df['WARD']  # Target for ward

# Train-test split
X_train_ward, X_test_ward, y_train_ward, y_test_ward = train_test_split(X, y_ward, test_size=0.2, random_state=42)

# Random Forest Classifier for ward
ward_model = RandomForestClassifier()
ward_model.fit(X_train_ward, y_train_ward)

# Now, let's generate data for the predicted crimes in 2024

# Predicted number of crimes for each month in 2024
predicted_crimes = {
    'January': 2306,
    'February': 2358,
    'March': 2410,
    'April': 2462
}

# Create a DataFrame for predictions (January to April 2024)
predicted_2024_data = pd.DataFrame({
    'YEAR': [2024] * sum(predicted_crimes.values()),
    'MONTH': [1] * predicted_crimes['January'] + 
             [2] * predicted_crimes['February'] + 
             [3] * predicted_crimes['March'] + 
             [4] * predicted_crimes['April'],
    'DAY': np.random.randint(1, 29, size=sum(predicted_crimes.values())),  # Random days of the month
    'HOUR': np.random.randint(0, 24, size=sum(predicted_crimes.values()))  # Random hours of the day
})

# Predict crime types for J-A 2024
predicted_crime_types = crime_type_model.predict(predicted_2024_data[['YEAR', 'MONTH', 'DAY', 'HOUR']])

# Predict wards for J-A 2024
predicted_wards = ward_model.predict(predicted_2024_data[['YEAR', 'MONTH', 'DAY', 'HOUR']])

predicted_2024_data['PREDICTED_CRIME_TYPE'] = predicted_crime_types
predicted_2024_data['PREDICTED_WARD'] = predicted_wards

print(predicted_2024_data.head())

crime_type_summary = predicted_2024_data.groupby(['MONTH', 'PREDICTED_CRIME_TYPE']).size().reset_index(name='COUNT')
ward_summary = predicted_2024_data.groupby(['MONTH', 'PREDICTED_WARD']).size().reset_index(name='COUNT')

#Summaries - crime type & ward predictions J-A 24
print("Crime Type Summary for Jan-Apr 2024")
print(crime_type_summary)
print("\nWard Summary for Jan-Apr 2024")
print(ward_summary)

#%%
# Plotting Bar Graphs with Labels 
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
month_map = {1: 'January', 2: 'February', 3: 'March', 4: 'April'}

predicted_2024_data['MONTH_NAME'] = predicted_2024_data['MONTH'].map(month_map)

def add_count_labels(ax):
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=10, color='black')

# Plot Crime Type Predictions
plt.figure(figsize=(12, 6))
ax = sns.countplot(x='MONTH_NAME', hue='PREDICTED_CRIME_TYPE', data=predicted_2024_data, palette='Set2')
add_count_labels(ax)

plt.title('Predicted Crime Types for Jan-Apr 2024')
plt.xlabel('Month')
plt.ylabel('Number of Crimes')
plt.legend(title='Crime Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Step 2: Plot Ward Predictions
plt.figure(figsize=(12, 6))
ax = sns.countplot(x='MONTH_NAME', hue='PREDICTED_WARD', data=predicted_2024_data, palette='Set1')
add_count_labels(ax)

plt.title('Predicted Wards for Crimes in Jan-Apr 2024')
plt.xlabel('Month')
plt.ylabel('Number of Crimes')
plt.legend(title='Ward', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
