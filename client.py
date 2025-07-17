import requests

# נתונים
CSV_FILE = "data for NB buys computer.csv"  # שם הקובץ
COLUMNS_TO_DROP = "id"  # אפשר גם ריק אם לא רוצים למחוק
PREDICTION_INPUT = {
     "age": "youth",
        "income": "medium",
        "student": "no",
        "credit_rating": "fair"
}

# כתובת השרת
BASE_URL = "http://127.0.0.1:8000"

# שליחת קובץ ואימון המודל
with open(CSV_FILE, 'rb') as f:
    response = requests.post(
        f"{BASE_URL}/train/",
        files={"file": f},
        data={"columns_to_drop": COLUMNS_TO_DROP}
    )
    print(response.json())

# שליחת תצפית לניבוי
response = requests.post(
    f"{BASE_URL}/predict/",
    json={"features": PREDICTION_INPUT}
)
print(response.json())
