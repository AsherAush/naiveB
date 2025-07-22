import pandas as pd

class NaiveBayesClassifier:
    def __init__(self):
        # Variable that holds the column on which I perform the test
        self.target = None
        # Variable that holds the values from the tested column
        self.labels = None
        # Variable that holds the percentage of each value from the tested column
        self.priors = {}
        # Variable that holds the probabilities of all values from all the data
        self.conditional_probs = {}
        self.columns = None

    def fit(self, table: pd.DataFrame):
        # Gets the last column (the tested one)
        self.target = table.columns[-1]
        # Gets the values
        self.labels = table[self.target].unique()

        # Saves how many times each value appears in the tested column
        label_counts = table[self.target].value_counts()
        for label in self.labels:
            # Calculates the probability of each value in the tested column and puts it in dictionary
            self.priors[label] = label_counts[label] / len(table)

        for label in self.labels:
            # Puts in the dictionary of all probabilities a key with the name of value from the tested and dictionary for the following probabilities
            self.conditional_probs[label] = {}
            # Saves in array all columns from the data not including the tested column
        features = table.columns.drop(self.target)
        self.columns = features

        # For each column it checks how many times it appears with each value
        for feature in features:
            for label in self.labels:
                # Saves every row from the data according to the tested value from the tested column
                filtered = table[table[self.target] == label]
                # Counts how many times each value appears in the tested column
                value_counts = filtered[feature].value_counts()
                # Counts how many rows there are in the tested column to know by how much to divide
                total = len(filtered)
                # Creates dictionary for probabilities of each value in the tested column
                probs = {}
                # For each value in the tested column it calculates its probability
                for val, count in value_counts.items():

                    probs[val] = count / total
                self.conditional_probs[label][feature] = probs