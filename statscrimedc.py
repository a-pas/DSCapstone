#%%

import pandas as pd

crimedc = pd.read_csv('combined_data_split.csv')
# %%

crimedc.columns
# %%
## Basic View of the Data

years_present = crimedc['YEAR'].unique()
print(f"These are the years in question: {years_present}")

wards_present = crimedc['WARD'].unique()
print(f"These are the wards in question: {wards_present}")

shift_types = crimedc['SHIFT'].unique()
print(f"These are the types of shifts: {shift_types}")

districts = crimedc['DISTRICT'].unique()
print(f"These are the districts: {districts}")

possible_offenses = crimedc['OFFENSE'].unique()
print(f"These are the possible offenses: {possible_offenses}")

types_of_methods = crimedc['METHOD'].unique()
print(f"These are the possible methods: {possible_offenses}")

time_span = crimedc['YEAR'].unique()
print(f"This data spans the following years: {time_span}")

# %%
# Frequency counts for SHIFT
shift_counts = crimedc['SHIFT'].value_counts()
print(f"The amount of offenses per shift are: {shift_counts}")

# Frequency counts for METHOD (e.g., gun, knife, etc.)
method_counts = crimedc['METHOD'].value_counts()
print(f"The amount of offenses according to each method are: {method_counts}")

# Frequency counts for different types of OFFENSE
offense_counts = crimedc['OFFENSE'].value_counts()
print(f"The amoount of offenses according to the type of offense is: {offense_counts}")

#%%
# Frequency counts for the different years of crime
years_counts = crimedc['YEAR'].value_counts()
print(f"The amount of crime by the year are: {years_counts}")

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
print(ward_offense_counts)

percentage_ward_offense = ward_offense_counts.copy()

for ward in ward_offense_counts.index:
    total_count = ward_offense_counts.loc[ward].sum()  # Total count for this ward
    if total_count > 0:  # Avoid division by zero
        # Update only the row for the current ward with percentage values
        percentage_ward_offense.loc[ward] = (ward_offense_counts.loc[ward] / total_count) * 100

percentage_ward_offense = percentage_ward_offense.round(2)

print(percentage_ward_offense)
# %%
# Group by HOUR and OFFENSE to see crimes by time of day
hour_counts = crimedc.groupby(['YEAR'],['HOUR']).value_counts().sort_index()
print(hour_counts)

# %%
# Total NaN values in the entire dataset
nan_total = crimedc.isna().sum().sum()
print(f"Total NaN values in the dataset: {nan_total}")

# %%
# Check data types of all columns
print(crimedc.dtypes)

# %%
