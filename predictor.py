import pandas as pd

class NaiveBayesPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, user_data : dict):
        result = {}
        for label in self.model.labels:
            probability = self.model.priors[label]
            for  key in user_data:
                probability *= self.model.conditional_probs[label][key].get(user_data[key], 1e-6)
            result[label] = probability
        predicted_label = max(result, key=result.get)
        print("Predicted probabilities:", result[predicted_label], "for label:", predicted_label)
        return predicted_label