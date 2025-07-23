from nt import startfile
import __main__
import pandas as pd

class NaiveBayesPredictor:
    def __init__(self, model):
        # Initialize predictor with trained Naive Bayes model
        self.model = model

    def predict(self, user_data : dict):
        # Dictionary to store probability calculations for each label
        result = {}
        # Iterate through each possible label/class
        for label in self.model.labels:
            # Start with prior probability of this label
            probability = self.model.priors[label]
            # Multiply by conditional probability for each feature
            for  key in user_data:
                # Get conditional probability or use small value if feature value not seen in training
                probability *= self.model.conditional_probs[label][key].get(user_data[key], 1e-6)
            # Store final probability for this label
            result[label] = probability
        # Find the label with highest probability
        predicted_label = max(result, key=result.get)
        # Print the prediction results
        if __main__.__file__.endswith("main.py"):
           print("Predicted probabilities:", result[predicted_label], "for label:", predicted_label)
        # print("Predicted probabilities:", result[predicted_label], "for label:", predicted_label)
        # Return the predicted label
        return predicted_label