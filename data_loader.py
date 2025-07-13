import pandas as pd



class DataLoader:
    # פןנקציה שמקבלת את הקובץ וגם מגדירה את ה דאטה החדשה
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    # פונקציה שעושה השמה מהקובץ לדאטה
    def load(self):
        self.df = pd.read_csv(self.filepath)
        return self.df

    #פונקציה שמוחקת עמודות לא רלוונטיות
    def drop_columns(self, columns):
        if self.df is not None:
            self.df = self.df.drop(columns=columns)
        return self.df

    # פונקציה שמחזירה את הדאטה
    def get_data(self):
        return self.df