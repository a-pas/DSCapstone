import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r'C:\Users\AP\Documents\GitHub\DSCapstone\DataCleaningandProcessing\combined_data.csv')

df = df[['YEAR', 'MONTH', 'DAY', 'HOUR', 'WARD', 'OFFENSE']]  # Use the necessary columns

X = df[['YEAR', 'MONTH', 'DAY', 'HOUR']] 
y_offense = df['OFFENSE'] 
X_train_offense, X_test_offense, y_train_offense, y_test_offense = train_test_split(X, y_offense, test_size=0.2, random_state=42)

# Random Forest Classifier 
crime_type_model = RandomForestClassifier()
crime_type_model.fit(X_train_offense, y_train_offense)

#%%
# RCF - Ward
y_ward = df['WARD'] 

X_train_ward, X_test_ward, y_train_ward, y_test_ward = train_test_split(X, y_ward, test_size=0.2, random_state=42)

ward_model = RandomForestClassifier()
ward_model.fit(X_train_ward, y_train_ward)

# Generating Data Based on Previous Predictions for Count/Month
predicted_crimes = {
    'January': 2306,
    'February': 2358,
    'March': 2410,
    'April': 2462
}

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

#%%
#Pie Chart - Wards
import matplotlib.pyplot as plt

months = ['January', 'February', 'March', 'April']

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten() 

for i, month in enumerate(months):
    month_data = predicted_2024_data[predicted_2024_data['MONTH_NAME'] == month]
    
    ward_counts = month_data.groupby('PREDICTED_WARD')['PREDICTED_WARD'].count()
    
    def autopct_format(values):
        def inner_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return f'{val} ({pct:.1f}%)'
        return inner_autopct
    
    axes[i].pie(ward_counts, labels=ward_counts.index, autopct=autopct_format(ward_counts), startangle=90, colors=sns.color_palette('Set1'))
    axes[i].set_title(f'Predicted Ward Distribution for Crimes in {month} 2024')
    axes[i].axis('equal') 

plt.tight_layout()
plt.show()

# %%
