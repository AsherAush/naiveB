import  pandas as pd
table = pd.read_csv('data for NB buys computer.csv')
table = table.drop(columns=['id'])


features = table.drop(columns=['Buy_Computer'])
labels = table['Buy_Computer']

priors = labels.value_counts(normalize=True)
subset_yes = table[table['Buy_Computer'] == 'yes']
prob_age_youth_given_yes = len(subset_yes[subset_yes['age'] == 'youth']) / len(subset_yes)
conditional_probs = {}

target_values = labels.unique()
features_cols = features.columns

for feature in features_cols:
    conditional_probs[feature] = {}
    for feature_val in table[feature].unique():
        conditional_probs[feature][feature_val] = {}
        for target_val in target_values:
            subset = table[table['Buy_Computer'] == target_val]
            prob = (len(subset[subset[feature] == feature_val]) + 1) / (len(subset) + len(table[feature].unique()))  # Laplace smoothing
            conditional_probs[feature][feature_val][target_val] = prob

def predict_row(row, priors, conditional_probs):
    probs = {}
    for target_val in priors.index:
        prob = priors[target_val]
        for feature in row.index:
            feature_val = row[feature]
            prob *= conditional_probs[feature].get(feature_val, {}).get(target_val, 1e-6)  # אם הערך לא קיים - ערך קטן מאוד
        probs[target_val] = prob
    return probs

# דוגמה על השורה הראשונה:
example_row = features.iloc[0]
result = predict_row(example_row, priors, conditional_probs)
predictions = []
for i, row in features.iterrows():
    probs = predict_row(row, priors, conditional_probs)
    predicted_class = max(probs, key=probs.get)
    predictions.append({'index': i, 'probabilities': probs, 'predicted': predicted_class})

# אפשר להוסיף את זה לטבלה המקורית
table['Predicted'] = [p['predicted'] for p in predictions]
for class_val in priors.index:
    table[f'P({class_val})'] = [p['probabilities'][class_val] for p in predictions]

print(table)








