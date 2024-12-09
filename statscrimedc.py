#%%

import pandas as pd

crimedc = pd.read_csv('DataCleaningandProcessing/combined_data.csv')
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
#Percentage of each offense in 15 year data

offense_counts = crimedc['OFFENSE'].value_counts()
total_offenses = offense_counts.sum()
offense_percentages = ((offense_counts/total_offenses)*100).round(1)

print(f"The rate of crime according to type is {offense_percentages}")

# %%
#Percentage of each method including and excluding "other"

##including
method_counts = crimedc['METHOD'].value_counts()
total_method = method_counts.sum()
method_percentages = ((method_counts/total_method)*100).round(1)
print(f"The rate of the method of crime including others is {method_percentages}")

##excluding others
crimedc_no_others_method = crimedc[crimedc['METHOD']!= 'OTHERS']

method_filtered_counts = crimedc_no_others_method['METHOD'].value_counts()

total_method_filtered = method_filtered_counts.sum()

method_filtered_percentages = ((method_filtered_counts/total_method_filtered)*100).round(1)

print(f"The rate of the method of crime excluding others is {method_filtered_percentages}")

# %%

#Let's see the breakdown of crime per shift

shift_counts = crimedc['SHIFT'].value_counts()

total_shifts = len(crimedc)

shift_percentages = ((shift_counts/total_shifts)*100).round(1)

print(f"The breakdown of crime according to which shift they occur during is {shift_percentages}")
# %%

#Let's see the breakdown of crime by ward

ward_counts = crimedc['WARD'].value_counts()
total_ward = len(crimedc)
ward_percentages = ((ward_counts/total_ward)*100).round(1)

print(f"The breakdown of crime according to ward is {ward_percentages}")

# %%
#Let's see the breakdown of crime by Police District

district_counts = crimedc['DISTRICT'].value_counts()
district_total = len(crimedc)
district_percentages = ((district_counts/district_total)*100).round(1)

print(f"The rate of crime according to police districts is {district_percentages}")

#%%

#Let's see the breakdown of crime by Police Service Area
psa_counts = crimedc['PSA'].value_counts()
psa_total = len(crimedc)
psa_percentages = ((psa_counts/psa_total)*100).round(1)

print(f"The rate of crime according to police service areas is {psa_percentages}")

#%%
#Let's see the breakdown of crime according to the blocks

block_counts = crimedc['BLOCK'].value_counts()
block_total = len(crimedc)
block_percentages = ((block_counts/block_total)*100).round(1)
print(f"The rate of crime one each block is {block_percentages}")
# %%
#Let's now see how much crime has evolved from 2008 to 2023

crime_2008 = crimedc[crimedc['YEAR'] == 2008]['OFFENSE'].count()
crime_2023 = crimedc[crimedc['YEAR'] == 2023]['OFFENSE'].count()

percentage_change = ((crime_2023 - crime_2008) / crime_2008) * 100

print(f"The percentage change in crime from 2008 to 2023 is {percentage_change:.2f}%")

# %%
#Let's see how much crime has evolved since the year with the most crime (2014)

crime_2014 = crimedc[crimedc['YEAR'] == 2014]['OFFENSE'].count()
crime_2023 = crimedc[crimedc['YEAR'] == 2023]['OFFENSE'].count()

percentage_change = ((crime_2023 - crime_2014) / crime_2014) * 100

print(f"The percentage change in crime from 2014 to 2023 is {percentage_change:.2f}%")

# %%
#Let's see how much crime has evolved since the first pandemic year (2020)


crime_2020 = crimedc[crimedc['YEAR'] == 2020]['OFFENSE'].count()
crime_2023 = crimedc[crimedc['YEAR'] == 2023]['OFFENSE'].count()

percentage_change = ((crime_2023 - crime_2020) / crime_2020) * 100

print(f"The percentage change in crime from 2020 to 2023 is {percentage_change:.2f}%")
# %%
#Seeing as the summer and early fall see the most crime, let's see how much of those months make up crime overall
summer_fall_crimes = crimedc[crimedc['MONTH'].isin([6, 7, 8, 9, 10])]
total_crimes = len(crimedc)
summer_fall_count = len(summer_fall_crimes)
percentage_summer_fall = (summer_fall_count / total_crimes) * 100

print(f"The percentage of crimes occurring in June to October is {percentage_summer_fall:.2f}%")


# %%
#Let's now be a bit more detailed and create a temporal heatmap

import seaborn as sns

crimedc['TIME'] = crimedc['HOUR'].astype(str) + ':' + crimedc['MINUTE'].astype(str)

heatmap_data = crimedc.groupby('HOUR').size().reset_index(name='count')

heatmap_data = heatmap_data.pivot_table(index='HOUR', values='count', fill_value=0)

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, cbar=True)

plt.title('Crime Distribution by Hour of the Day')
plt.xlabel('Count of Crimes')
plt.ylabel('Hour of the Day')

plt.show()


# %%
#Let's visualize the crime according to the time using a histogram now
plt.figure(figsize=(10, 6))
plt.hist(crimedc['HOUR'], bins=24, edgecolor='black')

plt.title('Crime Distribution by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Crimes')
plt.xticks(range(0, 24))

plt.show()

# %%
# A Quick Bar Chart of the Crimes per year
offenses_per_year = crimedc['YEAR'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.bar(offenses_per_year.index, offenses_per_year.values, color='grey', edgecolor='black')

plt.title('Offenses per Year')
plt.xlabel('Year')
plt.ylabel('Number of Offenses')
plt.xticks(offenses_per_year.index, rotation=45) 

plt.show()

# %%
#Let's see where homicides mainly occur
homicides = crimedc[crimedc['OFFENSE'] == 'HOMICIDE']

homicide_counts = homicides.groupby('WARD').size()

homicide_counts = homicide_counts.sort_values(ascending=False)

print(homicide_counts)
# %%

import matplotlib.pyplot as plt


homicides = crimedc[crimedc['OFFENSE'] == 'HOMICIDE']

homicide_counts_by_year = homicides.groupby('YEAR').size()

plt.figure(figsize=(10, 6))
plt.plot(homicide_counts_by_year.index, homicide_counts_by_year.values, marker='o', linestyle='-', color='b')

plt.title('Evolution of Homicides (2008 to 2023)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Homicides', fontsize=12)

plt.grid(True)

plt.tight_layout()
plt.show()


# %%
# How much has the rate of homicides changed since 2008
homicides = crimedc[crimedc['OFFENSE'] == 'HOMICIDE']

homicides_by_year = homicides.groupby('YEAR').size()

homicides_2008 = homicides_by_year.get(2008, 0)
homicides_2023 = homicides_by_year.get(2023, 0)

percent_change = ((homicides_2023 - homicides_2008) / homicides_2008) * 100
print(f"The percent change in homicides from 2008 to 2023 is: {percent_change:.2f}%")

# %%
#How many homocides occur in Ward 8 (% wise)
homicides = crimedc[crimedc['OFFENSE'] == 'HOMICIDE']

homicides_ward_8 = homicides[homicides['WARD'] == 8]


total_homicides = homicides.shape[0]  
homicides_in_ward_8 = homicides_ward_8.shape[0]  



percent_in_ward_8 = (homicides_in_ward_8 / total_homicides) * 100
print(f"The percentage of homicides occurring in Ward 8 is: {percent_in_ward_8:.2f}%")

# %%

#How many homicides occur in Ward 7 (% wise)
homicides = crimedc[crimedc['OFFENSE'] == 'HOMICIDE']

homicides_ward_8 = homicides[homicides['WARD'] == 7]


total_homicides = homicides.shape[0]  
homicides_in_ward_8 = homicides_ward_8.shape[0]  



percent_in_ward_8 = (homicides_in_ward_8 / total_homicides) * 100
print(f"The percentage of homicides occurring in Ward 7 is: {percent_in_ward_8:.2f}%")
