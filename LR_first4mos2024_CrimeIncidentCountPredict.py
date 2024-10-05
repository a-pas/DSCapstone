import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'C:\Users\AP\Documents\GitHub\DSCapstone\DataCleaningandProcessing\combined_data.csv')

# Aggregate the data to get the total number of crimes per month
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
df['MONTH_YEAR'] = df['DATE'].dt.to_period('M')  # Create a column for year-month
monthly_crime_counts = df.groupby(['YEAR', 'MONTH']).size().reset_index(name='CRIME_COUNT')

# Prepare the data for time series prediction
X = monthly_crime_counts[['YEAR', 'MONTH']]  # Features: Year and Month
y = monthly_crime_counts['CRIME_COUNT']  # Target: Monthly crime count

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X, y)

# Predict the number of crimes for January to April 2024
months_2024 = pd.DataFrame({
    'YEAR': [2024, 2024, 2024, 2024],
    'MONTH': [1, 2, 3, 4]
})

predicted_crime_counts = lr_model.predict(months_2024)

# Create a DataFrame with the predicted results
predicted_crime_df = pd.DataFrame({
    'MONTH': ['January', 'February', 'March', 'April'],
    'PREDICTED_CRIME_COUNT': predicted_crime_counts.astype(int)
})

# Display the predicted counts
print(predicted_crime_df)

# Plot the historical data and predictions
plt.figure(figsize=(10, 6))
plt.plot(monthly_crime_counts['YEAR'] + (monthly_crime_counts['MONTH'] - 1) / 12, monthly_crime_counts['CRIME_COUNT'], label='Historical Data')
plt.scatter([2024 + (i - 1) / 12 for i in range(1, 5)], predicted_crime_counts, color='red', label='Predictions (2024)')
plt.title('Crime Count Predictions for Jan-Apr 2024')
plt.xlabel('Year')
plt.ylabel('Crime Count')
plt.legend()
plt.grid(True)
plt.show()