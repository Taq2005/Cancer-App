import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def get_clean_data():
    data = pd.read_csv("data.csv")
    data = data.drop(["Unnamed: 32",'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    return data

def create_model(data):
    x = data.drop(["diagnosis"], axis=1)
    Y = data['diagnosis']
    scaler = StandardScaler()
    X = scaler.fit_transform(x)
    model = LogisticRegression()
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=42, test_size=0.2)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy*100,"%")

    return model, scaler

def main():
    data = get_clean_data()
    model, scaler = create_model(data)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

if __name__ == "__main__":
    main()