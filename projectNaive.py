from pprint import pprint

import pandas as pd


class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load(self):
        self.df = pd.read_csv(self.filepath)
        return self.df

    def drop_columns(self, columns):
        if self.df is not None:
            self.df = self.df.drop(columns=columns)
        return self.df

    def get_data(self):
        return self.df


class NaiveBayesClassifier:
    def __init__(self):
        self.target = None
        self.labels = None
        self.priors = {}
        self.conditional_probs = {}

    def fit(self, table: pd.DataFrame):
        # Step 1: detect the target column (last column)
        self.target = table.columns[-1]
        self.labels = table[self.target].unique()

        # Step 2: calculate priors (P(label))
        label_counts = table[self.target].value_counts()
        self.priors = {label: label_counts[label] / len(table) for label in self.labels}

        # Step 3: calculate conditional probabilities
        self.conditional_probs = {label: {} for label in self.labels}
        features = table.columns.drop(self.target)

        for feature in features:
            for label in self.labels:
                filtered = table[table[self.target] == label]
                value_counts = filtered[feature].value_counts()
                total = len(filtered)
                probs = {val: count / total for val, count in value_counts.items()}
                self.conditional_probs[label][feature] = probs


loader = DataLoader("data for NB buys computer.csv")
loader.load()
loader.drop_columns(['id'])
clean_df = loader.get_data()

model = NaiveBayesClassifier()
model.fit(clean_df)
pprint(model.conditional_probs)