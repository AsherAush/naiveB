import  pandas as pd
from naive_bayes import NaiveBayesClassifier
from predictor import NaiveBayesPredictor

df = pd.read_csv("phishing.csv")

mixing_df = df.sample(frac=1).reset_index(drop=True)

distribution_location = int(0.7 * len(mixing_df))

train_df = mixing_df[:distribution_location]
test_df = mixing_df[distribution_location:]

model = NaiveBayesClassifier()
model.fit(train_df)
predictor = NaiveBayesPredictor(model)

true = 0
false = 0
for _, series in test_df.iterrows():
    obsebservation = series.iloc[:-1].to_dict()
    prediction = predictor.predict(obsebservation)
    if prediction == series.iloc[-1]:
        true += 1
    else:
        false += 1

print("The model accuracy percentage is:", (true / len(test_df)) * 100)

