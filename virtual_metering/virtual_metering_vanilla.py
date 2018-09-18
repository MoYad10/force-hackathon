import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(df):
    df = df[df["SKAP_18HV3806/BCH/10sSAMP|stepinterpolation"] > 0]

    df = df.dropna()

    output_tags = [
        "SKAP_18FI381-VFlLGas/Y/10sSAMP|average",
        "SKAP_18FI381-VFlLH2O/Y/10sSAMP|average",
        "SKAP_18FI381-VFlLOil/Y/10sSAMP|average",
    ]

    y = df[output_tags]
    y.columns = ["gas", "water", "oil"]

    X = df.drop(output_tags + ["SKAP_18HV3806/BCH/10sSAMP|stepinterpolation", "Unnamed: 0", "timestamp"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    with open("../data/y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)

    with open("../data/X_scaler.pkl", "wb") as f:
        pickle.dump(X_scaler, f)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled


def train(X, y):
    lr = LinearRegression()
    model = lr.fit(X, y)
    with open("../data/model.pkl", "wb") as f:
        pickle.dump(model, f)


def test(X, y_true):
    with open("../data/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("../data/y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)

    y_pred = model.predict(X)

    print(score(y_scaler.inverse_transform(y_pred), y_scaler.inverse_transform(y_true)))


def score(Y_pred, Y_true):
    E = np.sqrt(np.sum((Y_pred - Y_true) ** 2))
    L2pred = np.sqrt(np.sum(Y_pred ** 2))
    L2true = np.sqrt(np.sum(Y_true ** 2))

    return 2 * E / (L2pred + L2true)


def main():
    df_d02 = pd.read_csv("../data/d2_train.csv")
    X_train, X_test, y_train, y_test = preprocess(df_d02)

    train(X_train, y_train)
    test(X_test, y_test)


if __name__ == "__main__":
    main()
