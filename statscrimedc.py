#%%

import pandas as pd

crimedc = pd.read_csv('combined_data_split.csv')
# %%
crimedc.head()

# %%
print(crimedc.columns)

print(crimedc.dtypes)

#%% 
print(crimedc.describe)

#%%
print(crimedc.describe(include='object'))

# %%
## Basic View of the amount of crime according to different columns

offense_by_year = crimedc['YEAR'].value_counts(ascending=False)
print(f"These are the offenses ranked according to year: {offense_by_year}")

offense_by_ward = crimedc['WARD'].value_counts()
print(f"These are the offenses ranked according to ward: {offense_by_ward}")

offense_by_shift = crimedc['SHIFT'].value_counts()
print(f"These are the offenses ranked according to shift: {offense_by_shift}")

offense_by_district = crimedc['DISTRICT'].value_counts()
print(f"These are the offenses ranked according to district: {offense_by_district}")

offense_by_offense = crimedc['OFFENSE'].value_counts()
print(f"These are the offenses ranked according to type of offense: {offense_by_offense}")

offenses_per_hour = crimedc['HOUR'].value_counts()
print(f"These are the offenses ranked according to hour:{offenses_per_hour}")

offenses_by_method = crimedc['METHOD'].value_counts()
print(f"These are the offenses ranked according to method: {offenses_by_method}")

offenses_by_month = crimedc['MONTH'].value_counts()
print(f"These are the offenses according to month:{offenses_by_month}")
# %%
# Group by offense and method
method_offense_counts = crimedc.groupby(['OFFENSE','METHOD']).size().unstack(fill_value=0)
print(method_offense_counts)

percentage_method_offense = method_offense_counts.copy()

for offense in method_offense_counts.index:
    total_count = method_offense_counts.loc[offense].sum()
    if total_count > 0 :
        percentage_method_offense.loc[offense]=(method_offense_counts.loc[offense]/total_count)*100

print (percentage_method_offense.round(1))
#%%
# Group by WARD and OFFENSE to count crimes by ward and offense type
ward_offense_counts = crimedc.groupby(['WARD', 'OFFENSE']).size().unstack(fill_value=0)
print(f"These are the offense types by ward{ward_offense_counts}")

percentage_ward_offense = ward_offense_counts.copy()

for ward in ward_offense_counts.index:
    total_count = ward_offense_counts.loc[ward].sum()  
    if total_count > 0: 
        percentage_ward_offense.loc[ward] = (ward_offense_counts.loc[ward] / total_count) * 100

percentage_ward_offense = percentage_ward_offense.round(2)

print(f"These are their percentages by ward{percentage_ward_offense}")
# %%
import statistics

# Group by 'YEAR' and count the number of offenses in each year
offenses_per_year_count = crimedc.groupby(['OFFENSE','YEAR']).size().unstack(fill_value=0)

print(offenses_per_year_count)


# %%

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


encoded_crimedc = crimedc.copy()


categorical_columns = ['SHIFT', 'METHOD', 'OFFENSE', 'BLOCK', 'WARD', 'ANC', 'DISTRICT', 'PSA', 'VOTING_PRECINCT']


encoder = LabelEncoder()


for col in categorical_columns:
    encoded_crimedc[col] = encoder.fit_transform(crimedc[col].astype(str)) 


correlation_matrix = encoded_crimedc.corr()


plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# %%
