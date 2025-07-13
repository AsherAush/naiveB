import pandas as pd

class NaiveBayesClassifier:
    def __init__(self):
        # משתנה שמחזיקה את העמודה שעליה אני עושה את הבדיקה
        self.target = None
        #משתנה שמחזיק את העכים מהעמודה הנבדקת
        self.labels = None
        # משתנה שמחזיק את האחוז של כל ערך מהעמודה של הנבדקת
        self.priors = {}
        # משתנה שמחזיק את ההסתברויות של כל הערכים מכל הדאטה
        self.conditional_probs = {}
        self.columns = None

    def fit(self, table: pd.DataFrame):
        # מקבל את העמודה אחרונה (הנבדקת)
        self.target = table.columns[-1]
        # מקבל את ההערכים
        self.labels = table[self.target].unique()

        # שומר כמה פעמים כל ערך מופיע בהעמודה הנבדקת
        label_counts = table[self.target].value_counts()
        for label in self.labels:
            # מחשב כמה ההסתברות של כל ערך בעמודה הנבדקת ומכניסה למילון
            self.priors[label] = label_counts[label] / len(table)

        for label in self.labels:
            # שם במילון של כל ההסתברויות מפתח בשם של ערך מהנבדקת ומילון להסתברויות הבאות
            self.conditional_probs[label] = {}
            # שומר במערך את כל העמודות מהדאטה לא כולל העמודה הנבדקת
        features = table.columns.drop(self.target)
        self.columns = features

        # עבור כל עמודה הוא עושה בדיקה כמה פעמים הוא מופיע עם כל ערך
        for feature in features:
            for label in self.labels:
                # שומר כל שורה מהדאטה לפי הערך הנבדק מהעמודה הנבדקת
                filtered = table[table[self.target] == label]
                # סופר כמה פעמים כל ערך מופיע בעמודה הנבדקת
                value_counts = filtered[feature].value_counts()
                #   סופר כמה שורות יש בעמודה הנבדקת כדי לדעת בכמה לחלק
                total = len(filtered)
                # יוצר מילון להסתברויות של כל ערך בעמודה הנבדקת
                probs = {}
                # עבור כל ערך בעמודה הנבדקת הוא מחשב את ההסתברות שלו
                for val, count in value_counts.items():

                    probs[val] = count / total
                self.conditional_probs[label][feature] = probs

