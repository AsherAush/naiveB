import pandas as pd
from naive_bayes import NaiveBayesClassifier
from predictor import NaiveBayesPredictor

# Load data
df = pd.read_csv("phishing.csv")

# Shuffle data
mixing_df = df.sample(frac=1).reset_index(drop=True)

# Split data: 70% train, 30% test
distribution_location = int(0.7 * len(mixing_df))
train_df = mixing_df[:distribution_location]
test_df = mixing_df[distribution_location:]

# Train the model
model = NaiveBayesClassifier()
model_dict = model.fit(train_df)
predictor = NaiveBayesPredictor(model_dict)

# Test model accuracy
true = 0
false = 0
for _, series in test_df.iterrows():
    observation = series.iloc[:-1].to_dict()
    prediction = predictor.predict(observation)
    if prediction == series.iloc[-1]:
        true += 1
    else:
        false += 1

# Print accuracy percentage
print("The model accuracy percentage is:", (true / len(test_df)) * 100)