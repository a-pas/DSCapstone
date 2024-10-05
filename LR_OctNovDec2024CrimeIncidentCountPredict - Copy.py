import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r'C:\Users\AP\Documents\GitHub\DSCapstone\DataCleaningandProcessing\combined_data_2024.csv')

df = df.dropna()


df = pd.get_dummies(df, columns=['SHIFT', 'METHOD', 'OFFENSE', 'WARD', 'ANC', 'DISTRICT', 'PSA', 'VOTING_PRECINCT'], drop_first=True)

df['COUNT'] = 1 

df_grouped = df.groupby(['YEAR', 'MONTH']).agg({'COUNT': 'sum'}).reset_index()

df_grouped['LAST_MONTH_COUNT'] = df_grouped['COUNT'].shift(1)  # Last month's count
df_grouped['LAST_YEAR_COUNT'] = df_grouped['COUNT'].shift(12)  # Same month last year

df_grouped.fillna(0, inplace=True)

X = df_grouped.drop(columns=['COUNT'])
y = df_grouped['COUNT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

predictions = []

for month in [10, 11, 12]:
    last_month_count = df_grouped.loc[(df_grouped['YEAR'] == 2024) & (df_grouped['MONTH'] == month - 1), 'COUNT']
    last_month_count = last_month_count.values[0] if not last_month_count.empty else 0  # Set to 0 if empty

    last_year_count = df_grouped.loc[(df_grouped['YEAR'] == 2023) & (df_grouped['MONTH'] == month), 'COUNT']
    last_year_count = last_year_count.values[0] if not last_year_count.empty else 0  # Set to 0 if empty

    future_row = {
        'YEAR': 2024,
        'MONTH': month,
        'LAST_MONTH_COUNT': last_month_count,
        'LAST_YEAR_COUNT': last_year_count,
    }


    for col in X.columns:
        if col not in future_row:
            future_row[col] = 0


    future_month_data = pd.DataFrame(future_row, index=[0])
    future_month_data = future_month_data[X.columns] 


    month_prediction = model.predict(future_month_data)
    future_month_data['PREDICTED_COUNT'] = month_prediction

    predictions.append(future_month_data)

predicted_counts = pd.concat(predictions)

print(predicted_counts[['YEAR', 'MONTH', 'PREDICTED_COUNT']])

feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("Feature Importance:\n", feature_importance.sort_values(by='Coefficient', ascending=False))

#%%
