from pprint import pprint

import pandas as pd

# new file and delete the id column
table = pd.read_csv('data for NB buys computer.csv')
table = table.drop(columns=['id'])

# teh target column
target = table.columns[-1]

# the unique values in the target column
labels = table[target].unique()

# Calculating the basic probability by target
label_counts = table[target].value_counts()
priors = {label: label_counts[label] / len(table) for label in labels}

# A dictionary of all probabilities
conditional_probs = {label: {} for label in labels}

# Goes through all columns except the target column
features = table.columns.drop(target)
for feature in features:
    for label in labels:
        # Filtering the table for the current label
        filtered = table[table[target] == label]
        value_counts = filtered[feature].value_counts()
        total = len(filtered)

        # Calculating the conditional probabilities
        probs = {val: count / total for val, count in value_counts.items()}
        conditional_probs[label][feature] = probs
pprint(conditional_probs)

