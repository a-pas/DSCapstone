# %% lightGBM predicting crime count, no covid years 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt


df = pd.read_csv(r'C:\Users\AP\Documents\GitHub\DSCapstone\DataCleaningandProcessing\combined_data_2024.csv')
df = df.dropna()
df = df[(df['YEAR'] < 2020) | (df['YEAR'] > 2022)]


if 'SHIFT' in df.columns:
    df['SHIFT'] = df['SHIFT'].astype('category').cat.codes  # Encode SHIFT as numeric values
else:
    df['SHIFT'] = 0


monthly_crime_counts = df.groupby(['YEAR', 'MONTH']).size().reset_index(name='CRIME_COUNT')

monthly_crime_counts['LAST_MONTH_COUNT'] = monthly_crime_counts['CRIME_COUNT'].shift(1)
monthly_crime_counts['2_MONTHS_AGO_COUNT'] = monthly_crime_counts['CRIME_COUNT'].shift(2)
monthly_crime_counts['3_MONTHS_AGO_COUNT'] = monthly_crime_counts['CRIME_COUNT'].shift(3)
monthly_crime_counts['LAST_YEAR_COUNT'] = monthly_crime_counts['CRIME_COUNT'].shift(12)

monthly_crime_counts['ROLLING_MEAN_1'] = monthly_crime_counts['CRIME_COUNT'].rolling(window=1).mean().fillna(0)
monthly_crime_counts['ROLLING_MEAN_3'] = monthly_crime_counts['CRIME_COUNT'].rolling(window=3).mean().fillna(0)
monthly_crime_counts['ROLLING_MEAN_9'] = monthly_crime_counts['CRIME_COUNT'].rolling(window=9).mean().fillna(0)
monthly_crime_counts['ROLLING_MEAN_12'] = monthly_crime_counts['CRIME_COUNT'].rolling(window=12).mean().fillna(0)
monthly_crime_counts['ROLLING_STD_3'] = monthly_crime_counts['CRIME_COUNT'].rolling(window=3).std().fillna(0)
monthly_crime_counts['ROLLING_STD_9'] = monthly_crime_counts['CRIME_COUNT'].rolling(window=9).std().fillna(0)
monthly_crime_counts['ROLLING_STD_12'] = monthly_crime_counts['CRIME_COUNT'].rolling(window=12).std().fillna(0)


monthly_crime_counts.fillna(0, inplace=True)


month_mapping = {
    'January': 0, 'February': 1, 'March': 2, 'April': 3, 'May': 4, 'June': 5,
    'July': 6, 'August': 7, 'September': 8, 'October': 9, 'November': 10, 'December': 11
}
monthly_crime_counts['MONTH'] = monthly_crime_counts['MONTH'].map(month_mapping)

year_month_count_cols = [
    'LAST_MONTH_COUNT', '3_MONTHS_AGO_COUNT',
    'LAST_YEAR_COUNT', 'ROLLING_MEAN_1', 'ROLLING_MEAN_3', 'ROLLING_MEAN_12', 'ROLLING_STD_9', 'ROLLING_STD_12'
]
monthly_crime_counts[year_month_count_cols] = monthly_crime_counts[year_month_count_cols].apply(pd.to_numeric, errors='coerce')

X = monthly_crime_counts[year_month_count_cols]
y = monthly_crime_counts['CRIME_COUNT']

X = X.fillna(0)
y = y.fillna(0)

from sklearn.model_selection import train_test_split

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1765, random_state=42)  # This makes validation set 15% of the total dataset

lgb_model = lgb.LGBMRegressor(objective='regression', random_state=42)
lgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'num_leaves': [31, 50, 100]
}

lgb_grid_search = GridSearchCV(estimator=lgb_model, param_grid=lgb_param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
lgb_grid_search.fit(X_train, y_train)

best_lgb_model = lgb_grid_search.best_estimator_

# Feature importance - Crime count
import seaborn as sns

feature_importances = best_lgb_model.feature_importances_
features = X_train.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importances from LightGBM Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
#%%

# df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
# df['YEAR'] = df['DATE'].dt.year

# yearly_totals = df.groupby('YEAR').size().reset_index(name='CRIME_COUNT')

# yearly_2024_total = yearly_totals[yearly_totals['YEAR'] == 2024]['CRIME_COUNT'].sum() + future_predictions.sum()
# yearly_totals = yearly_totals[yearly_totals['YEAR'] != 2024]
# yearly_totals = yearly_totals.append({'YEAR': 2024, 'CRIME_COUNT': yearly_2024_total}, ignore_index=True)

# plt.figure(figsize=(10, 5), dpi=100)
# plt.plot(yearly_totals['YEAR'], yearly_totals['CRIME_COUNT'], color='navy', label='Historical Data', marker='o')
# plt.plot(yearly_totals[yearly_totals['YEAR'] == 2024]['YEAR'], yearly_totals[yearly_totals['YEAR'] == 2024]['CRIME_COUNT'], color='red', label='2024 Predicted', marker='o')

# plt.plot([2019, 2023], [yearly_totals.loc[yearly_totals['YEAR'] == 2019, 'CRIME_COUNT'].values[0], yearly_totals.loc[yearly_totals['YEAR'] == 2023, 'CRIME_COUNT'].values[0]], color='lightgray', linestyle='--', linewidth=2)

# for i, row in yearly_totals.iterrows():
#     plt.text(row['YEAR'], row['CRIME_COUNT'] + 300, f"{int(row['CRIME_COUNT'])}", fontsize=10, ha='center', va='bottom')


# plt.xticks(ticks=yearly_totals['YEAR'], labels=yearly_totals['YEAR'].astype(int), rotation=45)
# plt.xlabel('Year')
# plt.ylabel('Total Crime Count')
# plt.ylim(10000, 40000)
# plt.title('Total Crime Count per Year (2008 - 2024) - Not Including Covid Years 2020-2022')
# plt.legend()
# plt.grid(True)
# plt.show()

# Evaluate the test set
y_test_pred = best_lgb_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = best_lgb_model.score(X_test, y_test)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
epsilon = np.finfo(np.float64).eps
y_test_non_zero = np.where(y_test == 0, epsilon, y_test)
test_mape = np.mean(np.abs((y_test_pred - y_test) / y_test_non_zero)) * 100

print("\nTest Mean Squared Error:", test_mse)
print("Test R-squared:", test_r2)
print("Test Root Mean Squared Error:", test_rmse)
print("Test Mean Absolute Error:", test_mae)
print("Test Mean Absolute Percentage Error:", test_mape, "%")

#%%

# Predicting crime counts for October, November, and December 2024
months_to_predict = ['October', 'November', 'December']
encoded_months = [month_mapping[month] for month in months_to_predict]

# Lag Features
future_data = pd.DataFrame({
    'LAST_MONTH_COUNT': monthly_crime_counts.tail(3)['CRIME_COUNT'].values,  # Using last 3 months' data as a simple example
    '3_MONTHS_AGO_COUNT': monthly_crime_counts.tail(3)['CRIME_COUNT'].shift(2).fillna(0).values,
    'LAST_YEAR_COUNT': monthly_crime_counts[monthly_crime_counts['YEAR'] == 2023].tail(3)['CRIME_COUNT'].values,
    'ROLLING_MEAN_1': monthly_crime_counts.tail(3)['ROLLING_MEAN_1'].values,
    'ROLLING_MEAN_3': monthly_crime_counts.tail(3)['ROLLING_MEAN_3'].values,
    'ROLLING_MEAN_12': monthly_crime_counts.tail(3)['ROLLING_MEAN_12'].values,
    'ROLLING_STD_9': monthly_crime_counts.tail(3)['ROLLING_STD_9'].values,
    'ROLLING_STD_12': monthly_crime_counts.tail(3)['ROLLING_STD_12'].values
})

future_data = future_data.fillna(0)

future_predictions = best_lgb_model.predict(future_data)
#%%
print("Predicted Crime Counts for October, November, December 2024 (LightGBM):")
for month, prediction in zip(months_to_predict, future_predictions):
    print(f"{month}: {round(prediction)}")

# %% Ward Prediction
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    make_scorer,
)
import matplotlib.pyplot as plt
import seaborn as sns  

df = pd.read_csv(
    r"C:\Users\AP\Documents\GitHub\DSCapstone\DataCleaningandProcessing\combined_data_2024.csv"
)

df = df.dropna()

df = df[(df["YEAR"] < 2020) | (df["YEAR"] > 2022)]


if (
    "MONTH" not in df.columns
    or "YEAR" not in df.columns
    or "WARD" not in df.columns
):
    print("Ensure that 'MONTH', 'YEAR', and 'WARD' columns exist in your DataFrame.")
else:
    pass

ward_counts = (
    df.groupby(["YEAR", "MONTH", "WARD"])
    .size()
    .reset_index(name="CRIME_COUNT")
)

# Creating lag features
ward_counts["LAST_MONTH_COUNT"] = ward_counts.groupby("WARD")["CRIME_COUNT"].shift(1)
ward_counts["2_MONTHS_AGO_COUNT"] = ward_counts.groupby("WARD")["CRIME_COUNT"].shift(2)
ward_counts["3_MONTHS_AGO_COUNT"] = ward_counts.groupby("WARD")["CRIME_COUNT"].shift(3)
ward_counts["6_MONTHS_AGO_COUNT"] = ward_counts.groupby("WARD")["CRIME_COUNT"].shift(6)
ward_counts["12_MONTHS_AGO_COUNT"] = ward_counts.groupby("WARD")["CRIME_COUNT"].shift(12)

# Creating rolling mean and std features per ward with different windows
ward_counts["ROLLING_MEAN_3"] = (
    ward_counts.groupby("WARD")["CRIME_COUNT"]
    .rolling(window=3)
    .mean()
    .reset_index(level=0, drop=True)
)
ward_counts["ROLLING_STD_3"] = (
    ward_counts.groupby("WARD")["CRIME_COUNT"]
    .rolling(window=3)
    .std()
    .reset_index(level=0, drop=True)
)
ward_counts["ROLLING_MEAN_6"] = (
    ward_counts.groupby("WARD")["CRIME_COUNT"]
    .rolling(window=6)
    .mean()
    .reset_index(level=0, drop=True)
)
ward_counts["ROLLING_STD_6"] = (
    ward_counts.groupby("WARD")["CRIME_COUNT"]
    .rolling(window=6)
    .std()
    .reset_index(level=0, drop=True)
)
ward_counts["ROLLING_MEAN_12"] = (
    ward_counts.groupby("WARD")["CRIME_COUNT"]
    .rolling(window=12)
    .mean()
    .reset_index(level=0, drop=True)
)
ward_counts["ROLLING_STD_12"] = (
    ward_counts.groupby("WARD")["CRIME_COUNT"]
    .rolling(window=12)
    .std()
    .reset_index(level=0, drop=True)
)


ward_counts.fillna(0, inplace=True)

X = ward_counts[
    [
        "YEAR",
        "MONTH",
        "WARD",
        "LAST_MONTH_COUNT",
        "2_MONTHS_AGO_COUNT",
        "3_MONTHS_AGO_COUNT",
        "6_MONTHS_AGO_COUNT",
        "12_MONTHS_AGO_COUNT",
        "ROLLING_MEAN_3",
        "ROLLING_STD_3",
        "ROLLING_MEAN_6",
        "ROLLING_STD_6",
        "ROLLING_MEAN_12",
        "ROLLING_STD_12",
    ]
]
y = ward_counts["CRIME_COUNT"]

y_log = np.log1p(y)

X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.15, random_state=42
)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

param_grid = {
    "num_leaves": [31, 50, 70],
    "max_depth": [5, 10, 15],
    "learning_rate": [0.03, 0.05, 0.01],
    "n_estimators": [500, 1000, 1500],
}

gbm = lgb.LGBMRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=gbm,
    param_grid=param_grid,
    cv=5,
    scoring=rmse_scorer,
    n_jobs=-1,
)

grid_search.fit(
    X_train,
    y_train_log,
    categorical_feature=["WARD"], 
)

best_params = grid_search.best_params_
print("\nBest parameters found:", best_params)

model = lgb.LGBMRegressor(**best_params, random_state=42)
model.fit(
    X_train,
    y_train_log,
    eval_set=[(X_test, y_test_log)],
    eval_metric="rmse",
    categorical_feature=["WARD"],
    callbacks=[early_stopping(50), log_evaluation(100)],
)

# Feature Importance 
feature_importances = model.feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importances from LightGBM Model')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test = np.expm1(y_test_log)


test_mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

y_test_non_zero = y_test.replace(0, np.finfo(float).eps)
test_mape = np.mean(np.abs((y_test_non_zero - y_pred) / y_test_non_zero)) * 100


print("\nEvaluation Metrics on Test Set:")
print(f"RMSE: {test_rmse:.4f}")
print(f"MSE: {test_mse:.4f}")
print(f"MAE: {test_mae:.4f}")
print(f"R-squared: {test_r2:.4f}")
print(f"MAPE: {test_mape:.2f}%")

# Predicting crime counts per ward for October and November 2024
months_to_predict = [10, 11]  
wards = ward_counts["WARD"].unique()

future_data = pd.DataFrame()
for ward in wards:
    for month in months_to_predict:
        recent_data = ward_counts[
            (ward_counts["WARD"] == ward) & (ward_counts["YEAR"] == 2023)
        ].tail(12)
        if not recent_data.empty:
            last_month_count = recent_data["CRIME_COUNT"].iloc[-1]
            two_months_ago_count = (
                recent_data["CRIME_COUNT"].shift(1).fillna(0).iloc[-1]
            )
            three_months_ago_count = (
                recent_data["CRIME_COUNT"].shift(2).fillna(0).iloc[-1]
            )
            six_months_ago_count = (
                recent_data["CRIME_COUNT"].shift(5).fillna(0).iloc[-1]
            )
            twelve_months_ago_count = (
                ward_counts[
                    (ward_counts["WARD"] == ward)
                    & (ward_counts["YEAR"] == 2023)
                    & (ward_counts["MONTH"] == month)
                ]["CRIME_COUNT"].sum()
            )
            rolling_mean_3 = (
                recent_data["CRIME_COUNT"].rolling(window=3).mean().iloc[-1]
            )
            rolling_std_3 = recent_data["CRIME_COUNT"].rolling(window=3).std().iloc[-1]
            rolling_mean_6 = (
                recent_data["CRIME_COUNT"].rolling(window=6).mean().iloc[-1]
            )
            rolling_std_6 = recent_data["CRIME_COUNT"].rolling(window=6).std().iloc[-1]
            rolling_mean_12 = (
                recent_data["CRIME_COUNT"].rolling(window=12).mean().iloc[-1]
            )
            rolling_std_12 = recent_data["CRIME_COUNT"].rolling(window=12).std().iloc[-1]
        else:
            last_month_count = 0
            two_months_ago_count = 0
            three_months_ago_count = 0
            six_months_ago_count = 0
            twelve_months_ago_count = 0
            rolling_mean_3 = 0
            rolling_std_3 = 0
            rolling_mean_6 = 0
            rolling_std_6 = 0
            rolling_mean_12 = 0
            rolling_std_12 = 0

        future_data = future_data.append(
            {
                "YEAR": 2024,
                "MONTH": month,
                "WARD": ward,
                "LAST_MONTH_COUNT": last_month_count,
                "2_MONTHS_AGO_COUNT": two_months_ago_count,
                "3_MONTHS_AGO_COUNT": three_months_ago_count,
                "6_MONTHS_AGO_COUNT": six_months_ago_count,
                "12_MONTHS_AGO_COUNT": twelve_months_ago_count,
                "ROLLING_MEAN_3": rolling_mean_3,
                "ROLLING_STD_3": rolling_std_3,
                "ROLLING_MEAN_6": rolling_mean_6,
                "ROLLING_STD_6": rolling_std_6,
                "ROLLING_MEAN_12": rolling_mean_12,
                "ROLLING_STD_12": rolling_std_12,
            },
            ignore_index=True,
        )


future_predictions_log = model.predict(future_data)
future_predictions = np.expm1(future_predictions_log)

# Add predictions to the future_data DataFrame
future_data["PREDICTED_CRIME_COUNT"] = future_predictions.round().astype(int)


month_mapping = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}
future_data["MONTH_NAME"] = future_data["MONTH"].map(month_mapping)


future_data['WARD'] = future_data['WARD'].astype(int)

print("\nPredicted Crime Counts per Ward for October and November 2024:")
print(future_data[["YEAR", "MONTH_NAME", "WARD", "PREDICTED_CRIME_COUNT"]])

#%%
# Plot Predicted Ward Data


predicted_data_pivot = future_data.pivot(index='WARD', columns='MONTH_NAME', values='PREDICTED_CRIME_COUNT').fillna(0)

predicted_data_pivot.index = predicted_data_pivot.index.astype(int)

predicted_data_pivot = predicted_data_pivot[['October', 'November']]

ax = predicted_data_pivot.plot(kind='bar', figsize=(12, 8), color=['orange', 'lightgreen'])
plt.title('Predicted Crime Counts per Ward for October and November 2024')
plt.xlabel('Ward')
plt.ylabel('Predicted Crime Count')
plt.legend(title='Month')
plt.tight_layout()

plt.ylim(0, 600)

for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', fontsize=9)


plt.show()


#%% PLOTTING ACTUAL OCT & NOV 2024 DATA FROM OPEN DATA DC
import pandas as pd
import matplotlib.pyplot as plt

file_path = r"C:\Users\AP\Documents\GitHub\DSCapstone\DataCleaningandProcessing\Crime_Incidents_in_2024_OctNov.csv"

actual_data = pd.read_csv(file_path)

actual_data['REPORT_DAT'] = pd.to_datetime(actual_data['REPORT_DAT'], errors='coerce')
actual_data['MONTH'] = actual_data['REPORT_DAT'].dt.month

actual_oct_nov_data = actual_data[actual_data['MONTH'].isin([10, 11])]

actual_ward_counts = (
    actual_oct_nov_data.groupby(["MONTH", "WARD"])
    .size()
    .reset_index(name="ACTUAL_CRIME_COUNT")
)

month_mapping = {10: "October", 11: "November"}
actual_ward_counts["MONTH_NAME"] = actual_ward_counts["MONTH"].map(month_mapping)


actual_data_pivot = actual_ward_counts.pivot(index='WARD', columns='MONTH_NAME', values='ACTUAL_CRIME_COUNT').fillna(0)

actual_data_pivot.index = actual_data_pivot.index.astype(int)

actual_data_pivot = actual_data_pivot[['October', 'November']]

ax = actual_data_pivot.plot(kind='bar', figsize=(12, 8), color=['orange', 'lightgreen'])
plt.title('Actual Crime Counts per Ward for October and November 2024')
plt.xlabel('Ward')
plt.ylabel('Actual Crime Count')
plt.legend(title='Month')
plt.tight_layout()

plt.ylim(0, 600)

for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', fontsize=9)

plt.show()

#%%

# # Visualize ward-level predictions for all months (including December if needed)
# wards = future_data["WARD"].unique().astype(int)
# months = ["October", "November"]

# data = pd.DataFrame(0, index=months, columns=wards)

# for _, row in future_data.iterrows():
#     data.at[row["MONTH_NAME"], row["WARD"]] = row["PREDICTED_CRIME_COUNT"]

# # Plot predictions
# bottom = np.zeros(len(months))
# num_wards = len(wards)
# colors = plt.cm.tab20(np.linspace(0, 1, num_wards))

# fig, ax = plt.subplots(figsize=(12, 8))

# for i, ward in enumerate(wards):
#     incidents = data[ward].values
#     bars = ax.bar(months, incidents, bottom=bottom, label=f'Ward {ward}', color=colors[i])

#     # Annotate bars
#     for bar, count in zip(bars, incidents):
#         if count > 0:
#             ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
#                     f'{int(count)}', ha='center', va='center', fontsize=9)

#     bottom += incidents

# # Set y-axis limit to 600
# plt.ylim(0, 600)
# # Finalize plot
# ax.set_xlabel('Month (2024)', fontsize=12)
# ax.set_ylabel('Predicted Incidents', fontsize=12)
# ax.set_title('Predicted Crime Incidents per Ward for October and November 2024', fontsize=14)
# ax.legend(title='Ward', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# %% Crime Count by Crime Type
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    make_scorer,
)
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"C:\Users\AP\Documents\GitHub\DSCapstone\DataCleaningandProcessing\combined_data_2024.csv"
)

df = df.dropna()


df = df[(df["YEAR"] < 2020) | (df["YEAR"] > 2022)]


if (
    "MONTH" not in df.columns
    or "YEAR" not in df.columns
    or "OFFENSE" not in df.columns
):
    print("Ensure that 'MONTH', 'YEAR', and 'OFFENSE' columns exist in your DataFrame.")
else:
    df["CRIME_TYPE"] = df["OFFENSE"]
#%%
crime_counts = (
    df.groupby(["YEAR", "MONTH", "CRIME_TYPE"])
    .size()
    .reset_index(name="CRIME_COUNT")
)

crime_counts["LAST_MONTH_COUNT"] = crime_counts.groupby("CRIME_TYPE")[
    "CRIME_COUNT"
].shift(1)
crime_counts["2_MONTHS_AGO_COUNT"] = crime_counts.groupby("CRIME_TYPE")[
    "CRIME_COUNT"
].shift(2)
crime_counts["3_MONTHS_AGO_COUNT"] = crime_counts.groupby("CRIME_TYPE")[
    "CRIME_COUNT"
].shift(3)
crime_counts["6_MONTHS_AGO_COUNT"] = crime_counts.groupby("CRIME_TYPE")[
    "CRIME_COUNT"
].shift(6)
crime_counts["12_MONTHS_AGO_COUNT"] = crime_counts.groupby("CRIME_TYPE")[
    "CRIME_COUNT"
].shift(12)


crime_counts["ROLLING_MEAN_3"] = (
    crime_counts.groupby("CRIME_TYPE")["CRIME_COUNT"]
    .rolling(window=3)
    .mean()
    .reset_index(level=0, drop=True)
)
crime_counts["ROLLING_STD_3"] = (
    crime_counts.groupby("CRIME_TYPE")["CRIME_COUNT"]
    .rolling(window=3)
    .std()
    .reset_index(level=0, drop=True)
)
crime_counts["ROLLING_MEAN_6"] = (
    crime_counts.groupby("CRIME_TYPE")["CRIME_COUNT"]
    .rolling(window=6)
    .mean()
    .reset_index(level=0, drop=True)
)
crime_counts["ROLLING_STD_6"] = (
    crime_counts.groupby("CRIME_TYPE")["CRIME_COUNT"]
    .rolling(window=6)
    .std()
    .reset_index(level=0, drop=True)
)
crime_counts["ROLLING_MEAN_12"] = (
    crime_counts.groupby("CRIME_TYPE")["CRIME_COUNT"]
    .rolling(window=12)
    .mean()
    .reset_index(level=0, drop=True)
)
crime_counts["ROLLING_STD_12"] = (
    crime_counts.groupby("CRIME_TYPE")["CRIME_COUNT"]
    .rolling(window=12)
    .std()
    .reset_index(level=0, drop=True)
)


crime_counts.fillna(0, inplace=True)


crime_counts["CRIME_TYPE"] = crime_counts["CRIME_TYPE"].astype("category")


X = crime_counts[
    [
        "YEAR",
        "CRIME_TYPE",
        "LAST_MONTH_COUNT",
        "2_MONTHS_AGO_COUNT",
        "ROLLING_MEAN_3"
    ]
]
y = crime_counts["CRIME_COUNT"]

y_log = np.log1p(y)


X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.15, random_state=42
)

categorical_features = ["CRIME_TYPE"]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

param_grid = {
    "num_leaves": [31, 50, 70],
    "max_depth": [5, 10, 15],
    "learning_rate": [0.01, 0.02, 0.05],
    "n_estimators": [200, 300, 500, 1000],
}

gbm = lgb.LGBMRegressor(random_state=42)


grid_search = GridSearchCV(
    estimator=gbm,
    param_grid=param_grid,
    cv=5,
    scoring=rmse_scorer,
    n_jobs=-1,
)

grid_search.fit(
    X_train,
    y_train_log,
    categorical_feature=categorical_features,
)

best_params = grid_search.best_params_
print("\nBest parameters found:", best_params)

model = lgb.LGBMRegressor(**best_params, random_state=42)
model.fit(
    X_train,
    y_train_log,
    eval_set=[(X_test, y_test_log)],
    eval_metric="rmse",
    categorical_feature=categorical_features,
    callbacks=[
        early_stopping(50),
        log_evaluation(100)
    ]
)


y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log) 
y_test = np.expm1(y_test_log)
#%%
test_mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

y_test_non_zero = y_test.replace(0, np.finfo(float).eps)
test_mape = np.mean(np.abs((y_test_non_zero - y_pred) / y_test_non_zero)) * 100
#%%
print("\nEvaluation Metrics on Test Set:")
print(f"RMSE: {test_rmse:.4f}")
print(f"MSE: {test_mse:.4f}")
print(f"MAE: {test_mae:.4f}")
print(f"R-squared: {test_r2:.4f}")
print(f"MAPE: {test_mape:.2f}%")

residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
#%%
# Feature Importance
feature_importance = model.feature_importances_
feature_names = X_train.columns

fi_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
fi_df = fi_df.sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(fi_df)


plt.figure(figsize=(10, 6))
sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importances from LightGBM Model')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

#%%
# Predicting crime counts per crime type for October, November, December 2024
months_to_predict = [10, 11, 12] 
crime_types = crime_counts["CRIME_TYPE"].unique()

future_data_records = []

for crime_type in crime_types:
    recent_data = crime_counts[crime_counts["CRIME_TYPE"] == crime_type].copy()

    for month in months_to_predict:
        if not recent_data.empty:
            last_month_count = recent_data["CRIME_COUNT"].iloc[-1]
            two_months_ago_count = recent_data["CRIME_COUNT"].shift(1).fillna(0).iloc[-1]
            rolling_mean_3 = recent_data["CRIME_COUNT"].rolling(window=3).mean().iloc[-1]
        else:
            last_month_count = 0
            two_months_ago_count = 0
            rolling_mean_3 = 0

        input_row = {
            "YEAR": 2024,
            "MONTH": month,
            "CRIME_TYPE": crime_type,
            "LAST_MONTH_COUNT": last_month_count,
            "2_MONTHS_AGO_COUNT": two_months_ago_count,
            "ROLLING_MEAN_3": rolling_mean_3,
        }

        future_data_records.append(input_row)

        input_df = pd.DataFrame([input_row])[feature_columns]
        input_df["CRIME_TYPE"] = input_df["CRIME_TYPE"].astype("category") 

        predicted_crime_count = np.expm1(model.predict(input_df)[0])

        predicted_row = pd.DataFrame([{
            "YEAR": 2024,
            "MONTH": month,
            "CRIME_TYPE": crime_type,
            "CRIME_COUNT": predicted_crime_count
        }])
        recent_data = pd.concat([recent_data, predicted_row], ignore_index=True)

future_data = pd.DataFrame(future_data_records)

future_data["CRIME_TYPE"] = future_data["CRIME_TYPE"].astype("category")


future_data["MONTH_NAME"] = future_data["MONTH"].map(month_mapping)


feature_columns = ["YEAR", "CRIME_TYPE", "LAST_MONTH_COUNT", "2_MONTHS_AGO_COUNT", "ROLLING_MEAN_3"]


future_data_for_prediction = future_data[feature_columns]


future_predictions_log = model.predict(
    future_data_for_prediction,
    categorical_feature=["CRIME_TYPE"]
)
future_predictions = np.expm1(future_predictions_log)


future_data["PREDICTED_CRIME_COUNT"] = future_predictions.round().astype(int)

print(
    "\nPredicted Crime Counts per Crime Type for October, November, December 2024:"
)
print(
    future_data[
        [
            "YEAR",
            "MONTH_NAME",
            "CRIME_TYPE",
            "PREDICTED_CRIME_COUNT",
        ]
    ]
)


#%%
# Plot Predicted Crime Count for October and November


predicted_crime_types = future_data["CRIME_TYPE"].unique()
predicted_crime_types = sorted(predicted_crime_types) 
bar_width = 0.35
predicted_month_colors = {"October": "skyblue", "November": "lightcoral"}

x_predicted = np.arange(len(predicted_crime_types))

fig, ax = plt.subplots(figsize=(12, 8))

for i, month in enumerate(["October", "November"]):
    month_data = future_data[future_data["MONTH_NAME"] == month]
    month_data = month_data.sort_values(by="CRIME_TYPE") 
    bars = ax.bar(
        x_predicted + i * bar_width,
        month_data["PREDICTED_CRIME_COUNT"],
        width=bar_width,
        label=f"{month} 2024",
        color=predicted_month_colors[month]
    )

    for bar, count in zip(bars, month_data["PREDICTED_CRIME_COUNT"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{int(count)}',
            ha='center',
            va='bottom',
            fontsize=9
        )

ax.set_xlabel("Crime Type", fontsize=12)
ax.set_ylabel("Predicted Crime Count", fontsize=12)
ax.set_title("Predicted Crime Counts per Crime Type for October and November 2024", fontsize=14)

ax.set_xticks(x_predicted + bar_width / 2)
ax.set_xticklabels(predicted_crime_types, rotation=45, ha='right')

plt.tight_layout()


ax.legend(title="Month")


plt.show()
#%%
# Predicted Crime Counts: Plot for October and November in similar style

predicted_crime_types = future_data["CRIME_TYPE"].unique()
predicted_crime_types = sorted(predicted_crime_types)
bar_width = 0.35
predicted_month_colors = {"October": "skyblue", "November": "lightcoral", "December": "orange"}

x_predicted = np.arange(len(predicted_crime_types))

fig, ax = plt.subplots(figsize=(12, 8))

for i, month in enumerate(["October", "November", "December"]):
    month_data = future_data[future_data["MONTH_NAME"] == month]
    month_data = month_data.sort_values(by="CRIME_TYPE") 
    bars = ax.bar(
        x_predicted + i * bar_width,
        month_data["PREDICTED_CRIME_COUNT"],
        width=bar_width,
        label=f"{month} 2024",
        color=predicted_month_colors[month]
    )


    for bar, count in zip(bars, month_data["PREDICTED_CRIME_COUNT"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{int(count)}',
            ha='center',
            va='bottom',
            fontsize=9
        )

ax.set_xlabel("Crime Type", fontsize=12)
ax.set_ylabel("Predicted Crime Count", fontsize=12)
ax.set_title("Predicted Crime Counts per Crime Type for October, November, and December 2024", fontsize=14)
ax.set_ylim(0, 2000) 

ax.set_xticks(x_predicted + bar_width / 2)
ax.set_xticklabels(predicted_crime_types, rotation=45, ha='right')

plt.tight_layout()

ax.legend(title="Month")

plt.show()

#%%
# Plotting Actual Crime Type Count
actual_data = pd.read_csv(
    r"C:\Users\AP\Documents\GitHub\DSCapstone\DataCleaningandProcessing\Crime_Incidents_in_2024_OctNov.csv"
)


actual_data['REPORT_DAT'] = pd.to_datetime(actual_data['REPORT_DAT'])
actual_data['YEAR'] = actual_data['REPORT_DAT'].dt.year
actual_data['MONTH'] = actual_data['REPORT_DAT'].dt.month

actual_data["OFFENSE"] = actual_data["OFFENSE"].replace(
    {"THEFT/OTHER": "THEFT", "THEFT F/AUTO": "THEFT"}
)

actual_data["CRIME_TYPE"] = actual_data["OFFENSE"]
actual_data_grouped = (
    actual_data.groupby(["YEAR", "MONTH", "CRIME_TYPE"])
    .size()
    .reset_index(name="ACTUAL_CRIME_COUNT")
)

actual_data_oct_nov = actual_data_grouped[
    (actual_data_grouped["YEAR"] == 2024) & (actual_data_grouped["MONTH"].isin([10, 11]))
]

all_crime_types = set(actual_data["CRIME_TYPE"].unique())
all_crime_types.add("ARSON") 


expanded_data = []
for month in [10, 11]: 
    for crime_type in all_crime_types:
        existing_data = actual_data_oct_nov[
            (actual_data_oct_nov["MONTH"] == month) & (actual_data_oct_nov["CRIME_TYPE"] == crime_type)
        ]
        if not existing_data.empty:
            expanded_data.append(existing_data.iloc[0].to_dict())  # Convert row to dict
        else:
            expanded_data.append({"YEAR": 2024, "MONTH": month, "CRIME_TYPE": crime_type, "ACTUAL_CRIME_COUNT": 0})

actual_data_oct_nov = pd.DataFrame(expanded_data)

actual_data_oct_nov = actual_data_oct_nov.sort_values(by="CRIME_TYPE", ascending=True)

actual_data_oct_nov["MONTH_NAME"] = actual_data_oct_nov["MONTH"].map({
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
})

#Plot actual crime counts for October and November
import matplotlib.pyplot as plt
import numpy as np

crime_types = actual_data_oct_nov["CRIME_TYPE"].unique()
bar_width = 0.35
month_colors = {"October": "skyblue", "November": "lightcoral"}

x = np.arange(len(crime_types))
fig, ax = plt.subplots(figsize=(12, 8))

for i, month in enumerate(["October", "November"]):
    month_data = actual_data_oct_nov[actual_data_oct_nov["MONTH_NAME"] == month]
    month_data = month_data.sort_values(by="CRIME_TYPE") 
    bars = ax.bar(
        x + i * bar_width,
        month_data["ACTUAL_CRIME_COUNT"],
        width=bar_width,
        label=f"{month} 2024",
        color=month_colors[month]
    )

    for bar, count in zip(bars, month_data["ACTUAL_CRIME_COUNT"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{int(count)}',
            ha='center',
            va='bottom',
            fontsize=9
        )

ax.set_xlabel("Crime Type", fontsize=12)
ax.set_ylabel("Actual Crime Count", fontsize=12)
ax.set_title("Actual Crime Counts per Crime Type for October and November 2024", fontsize=14)


ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(crime_types, rotation=45, ha='right')


plt.tight_layout()


ax.legend(title="Month")

plt.show()
