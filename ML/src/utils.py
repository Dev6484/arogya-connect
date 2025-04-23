import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(path):
    data = pd.read_csv(path)

    # Encode the disease column
    encoder = LabelEncoder()
    data["disease"] = encoder.fit_transform(data["disease"])

    X = data.iloc[:, :-1]
    y = data["disease"]

    # Index symptoms
    symptom_index = {symptom: idx for idx, symptom in enumerate(X.columns)}

    return X, y, encoder, symptom_index
