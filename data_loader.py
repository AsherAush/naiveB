import pandas as pd



class DataLoader:
    # Function that receives the file and also defines the new data
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    # Function that assigns from the file to data
    def load(self):
        self.df = pd.read_csv(self.filepath)
        return self.df

    # Function that deletes irrelevant columns
    def drop_columns(self, columns):
        if self.df is not None:
            self.df = self.df.drop(columns=columns)
        return self.df

    # Function that returns the data
    def get_data(self):
        return self.df