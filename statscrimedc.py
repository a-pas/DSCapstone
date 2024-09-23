#%%

import pandas as pd

crimedc = pd.read_csv('combined_data_split.csv')
# %%

crimedc.columns
# %%
## Basic View of the Data
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
print(f"The amount of offenses per fhist are: {shift_counts}")

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
