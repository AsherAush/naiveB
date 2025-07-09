import pandas as pd
table = pd.read_csv('data for NB buys computer.csv')
table = table.drop(columns=['id'])

# some yes and no
count_yes = table['Buy_Computer'].value_counts()['yes']
count_no = table['Buy_Computer'].value_counts()['no']

# Calculate the prior probabilities
percent_yes = count_yes / len(table)
percent_no = count_no / len(table)

# Initialize dictionaries for conditional probabilities
yes_dict = {}
no_dict = {}

for col in table.columns[:-1]:
    yes_dict[col] = {}
    no_dict[col] = {}
for index, row in table.iterrows():
    label = row['Buy_Computer']
    for col in table.columns[:-1]:
        val = row[col]
        if label == 'yes':
            yes_dict[col][val] = yes_dict[col].get(val, 0) + 1
        else:
            no_dict[col][val] = no_dict[col].get(val, 0) + 1

for col in yes_dict:
    for val in yes_dict[col]:
        yes_dict[col][val] /= count_yes

for col in no_dict:
    for val in no_dict[col]:
        no_dict[col][val] /= count_no

print(yes_dict)
print(no_dict)
