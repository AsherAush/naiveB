from data_loader import DataLoader
from naive_bayes import NaiveBayesClassifier
from predictor import NaiveBayesPredictor

# This is the main control center for everything
class Main:
    def __init__(self):
        self.filepath = "data for NB buys computer.csv"
        self.model = None
        self.predictor = None
        self.df = None
    # This function runs the main logic of the program
    def run(self):
        # Load data
        loader = DataLoader(self.filepath)
        loader.load()
        columns_to_drop = ["id"]
        if columns_to_drop:
            loader.drop_columns(columns_to_drop)
        else:
            print("No columns entered to delete. Continuing with all columns.")

        self.df = loader.get_data()

        # Train the model
        self.model = NaiveBayesClassifier()
        model_dict = self.model.fit(self.df)

        # Create the predictor
        self.predictor = NaiveBayesPredictor(model_dict)

        # Test observation for checking
        test_observation = {futer : None for futer in self.model.columns }
        for feature in test_observation:
            value = input(f"Enter value for {feature}: ").strip()
            test_observation[feature] = value


        self.predictor.predict(test_observation)

if __name__ == "__main__":
    main = Main()
    main.run()