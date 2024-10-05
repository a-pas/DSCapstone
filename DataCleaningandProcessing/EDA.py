import pandas as pd


file_path = r'C:\Users\AP\Documents\GitHub\DSCapstone\DataCleaningandProcessing\combined_data_2024.csv'
data = pd.read_csv(file_path)

data.head()
data.info()


print(data['YEAR'].unique()) 


import pandas as pd
import matplotlib.pyplot as plt


# Offenses over the years
crime_by_year_offense = data.groupby(['YEAR', 'OFFENSE']).size().unstack().fillna(0)
crime_by_year_offense_prop = crime_by_year_offense.div(crime_by_year_offense.sum(axis=1), axis=0)
crime_by_year_offense_prop.plot(kind='bar', stacked=True, figsize=(12, 8), cmap='tab20')
plt.title('Proportion of Crime Types by Year')
plt.xlabel('Year')
plt.ylabel('Proportion')
plt.xticks(range(len(crime_by_year_offense.index)), crime_by_year_offense.index, rotation=45)
plt.legend(title='Offense Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

#Ward and Offense, Per Year 
crime_by_ward_offense = data.groupby(['WARD', 'OFFENSE']).size().unstack().fillna(0)
crime_by_ward_offense_prop = crime_by_ward_offense.div(crime_by_ward_offense.sum(axis=1), axis=0)


crime_by_ward_offense_prop.plot(kind='bar', stacked=True, figsize=(12, 8), cmap='tab20')
plt.title('Proportion of Crime Types by Ward')
plt.xlabel('Ward')
plt.ylabel('Proportion')
plt.legend(title='Offense Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()


# Crime by Shift, Per Year
crime_by_year_shift = data.groupby(['YEAR', 'SHIFT']).size().unstack().fillna(0)

crime_by_year_shift_prop = crime_by_year_shift.div(crime_by_year_shift.sum(axis=1), axis=0)

print(crime_by_year_shift_prop)


crime_by_year_shift_prop.plot(kind='bar', stacked=True, figsize=(12, 8), cmap='Set3')
plt.title('Proportion of Crime by Shift for Each Year')
plt.xlabel('Year')
plt.ylabel('Proportion of Crimes')
plt.legend(title='Shift', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Total Crime per Year
crime_by_year = data.groupby('YEAR').size()

print(crime_by_year)

plt.figure(figsize=(10, 6))
crime_by_year.plot(kind='line', marker='o', color='b')
plt.title('Total Crime Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Crime Incidents')
plt.xticks(crime_by_year.index, rotation=45)
plt.grid(True)
plt.ylim(0, max(crime_by_year) + 100)  
plt.show()

# Violent Crime vs. Non Violent Crime


violent_crimes = ['HOMICIDE', 'SEX ABUSE', 'ASSAULT W/DANGEROUS WEAPON', 'ROBBERY']

violent_crime_data = data[data['OFFENSE'].isin(violent_crimes)]
non_violent_crime_data = data[~data['OFFENSE'].isin(violent_crimes)]

violent_crime_by_year = violent_crime_data.groupby('YEAR').size()
non_violent_crime_by_year = non_violent_crime_data.groupby('YEAR').size()


plt.figure(figsize=(10, 6))

plt.plot(violent_crime_by_year.index, violent_crime_by_year.values, marker='o', label='Violent Crime', color='r')
plt.plot(non_violent_crime_by_year.index, non_violent_crime_by_year.values, marker='o', label='Non-Violent Crime', color='b')
plt.title('Violent vs Non-Violent Crime Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Crime Incidents')
plt.xticks(non_violent_crime_by_year.index, rotation=45)
plt.legend()
plt.grid(True)
plt.show()

# Types of violent crime over the years

violent_crimes = ['HOMICIDE', 'SEX ABUSE', 'ASSAULT W/DANGEROUS WEAPON', 'ROBBERY']

violent_crime_data = data[data['OFFENSE'].isin(violent_crimes)]
violent_crime_by_year_offense = violent_crime_data.groupby(['YEAR', 'OFFENSE']).size().unstack().fillna(0)

print(violent_crime_by_year_offense)

plt.figure(figsize=(12, 8))

for crime_type in violent_crime_by_year_offense.columns:
    plt.plot(violent_crime_by_year_offense.index, violent_crime_by_year_offense[crime_type], 
             marker='o', label=crime_type)

plt.title('Violent Crimes Over the Years (By Type)')
plt.xlabel('Year')
plt.ylabel('Number of Violent Crime Incidents')

plt.xticks(violent_crime_by_year_offense.index)

plt.legend(title='Violent Crime Types')
plt.grid(True)
plt.show()


# Non-Violent crime over the years

nonviolent_crimes = ['ARSON', 'BURGLARY', 'MOTOR VEHICLE THEFT', 'THEFT']

nonviolent_crime_data = data[data['OFFENSE'].isin(nonviolent_crimes)]
nonviolent_crime_by_year_offense = nonviolent_crime_data.groupby(['YEAR', 'OFFENSE']).size().unstack().fillna(0)

print(nonviolent_crime_by_year_offense)

plt.figure(figsize=(12, 8))

for crime_type in nonviolent_crime_by_year_offense.columns:
    plt.plot(nonviolent_crime_by_year_offense.index, nonviolent_crime_by_year_offense[crime_type], 
             marker='o', label=crime_type)

plt.title('Non-Violent Crimes Over the Years (By Type)')
plt.xlabel('Year')
plt.ylabel('Number of Non-Violent Crime Incidents')

plt.xticks(nonviolent_crime_by_year_offense.index)

plt.legend(title='Non-Violent Crime Types')
plt.grid(True)
plt.show()

crime_by_ward = data.groupby('WARD').size()
violent_crimes = ['HOMICIDE', 'SEX ABUSE', 'ASSAULT W/DANGEROUS WEAPON', 'ROBBERY']
non_violent_crimes = ['THEFT', 'AUTO THEFT', 'BURGLARY', 'VANDALISM', 'ARSON']
data['CRIME_TYPE'] = data['OFFENSE'].apply(lambda x: 'Violent' if x in violent_crimes else 'Non-Violent')
crime_by_ward_type = data.groupby(['WARD', 'CRIME_TYPE']).size().unstack().fillna(0)

print(crime_by_ward_type)

plt.figure(figsize=(12, 6))
crime_by_ward_type.plot(kind='bar', stacked=True, color=['lightcoral', 'skyblue'])
plt.title('Total Crime Incidents by Ward (Violent vs Non-Violent)')
plt.xlabel('Ward')
plt.ylabel('Number of Crime Incidents')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
#%%
