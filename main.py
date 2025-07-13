from data_loader import DataLoader
from naive_bayes import NaiveBayesClassifier
from predictor import NaiveBayesPredictor


class Main:
    def __init__(self):
        self.filepath = input("Enter the name of the CSV file (including the extension): ").strip()
        self.model = None
        self.predictor = None
        self.df = None

    def run(self):
        # טוען נתונים
        loader = DataLoader(self.filepath)
        loader.load()
        columns_input = input("Enter column names to delete (comma separated): ")
        columns_to_drop = [col.strip() for col in columns_input.split(',') if col.strip()]
        if columns_to_drop:
            loader.drop_columns(columns_to_drop)
        else:
            print("No columns entered to delete. Continuing with all columns.")

        self.df = loader.get_data()

        # מאמן את המודל
        self.model = NaiveBayesClassifier()
        self.model.fit(self.df)

        # יוצר את החוזה
        self.predictor = NaiveBayesPredictor(self.model)

        # תצפית לבדיקה
        test_observation = {futer : None for futer in self.model.columns }
        for feature in test_observation:
            value = input(f"Enter value for {feature}: ").strip()
            test_observation[feature] = value


        self.predictor.predict(test_observation)

if __name__ == "__main__":
    main = Main()
    main.run()
