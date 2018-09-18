import pickle

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    X_scaler = StandardScaler()

    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled


def train(X, y):
    lr = LinearRegression()
    model = lr.fit(X, y)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)


def test(X, y_true):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    # res = model.score(X, y)
    y_pred = model.predict(X)
    l1_res = mean_absolute_error(y_true, y_pred)
    l2_res = mean_squared_error(y_true, y_pred)

    print(l1_res)
    print(l2_res)


def main():
    df_d02 = pd.read_csv("./d02.csv")
    X_train, X_test, y_train, y_test = preprocess(df_d02)

    train(X_train, y_train)
    test(X_test, y_test)


if __name__ == "__main__":
    main()
