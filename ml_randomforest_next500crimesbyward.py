import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv(r'C:\Users\AP\Documents\GitHub\DSCapstone\sampled_combined_data_split.csv')

df = pd.get_dummies(df, columns=['SHIFT', 'OFFENSE'])  # Use dummies for categorical variables

X = df[['YEAR', 'MONTH', 'DAY', 'HOUR', 'LATITUDE', 'LONGITUDE', 'SHIFT_DAY', 'SHIFT_EVENING', 'SHIFT_MIDNIGHT']]  # Plus other necessary features
y = df['WARD'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

importances = rf.feature_importances_
feature_names = X.columns
sorted_indices = importances.argsort()[::-1]

print("Feature importance ranking:")
for i in sorted_indices:
    print(f"{feature_names[i]}: {importances[i]}")

import numpy as np

next_500_crimes = pd.DataFrame({
    'YEAR': [2024]*500,  # 
    'MONTH': np.random.randint(1, 13, size=500),  
    'DAY': np.random.randint(1, 31, size=500),  
    'HOUR': np.random.randint(0, 24, size=500), 
    'LATITUDE': np.random.uniform(df['LATITUDE'].min(), df['LATITUDE'].max(), size=500),  # Random latitude within historical bounds
    'LONGITUDE': np.random.uniform(df['LONGITUDE'].min(), df['LONGITUDE'].max(), size=500)  # Random longitude within historical bounds
})

next_500_crimes['SHIFT'] = np.random.choice(['DAY', 'EVENING', 'MIDNIGHT'], size=500)

next_500_crimes = pd.get_dummies(next_500_crimes, columns=['SHIFT'])

ward_predictions = rf.predict(next_500_crimes)

print("Predicted wards for the next 500 crimes:")
print(ward_predictions)

#Count crimes by ward
ward_counts = pd.Series(ward_predictions).value_counts()
print("Predicted ward counts:")
print(ward_counts)
