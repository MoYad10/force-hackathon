import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as Predictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# from xgboost import XGBRegressor as Predictor


def preprocess(df):
    df = df.dropna()
    df = df.dropna(axis=1, how="all")
    df = df.drop(["Unnamed: 0", "timestamp", "SKAP_18HV3806/BCH/10sSAMP|stepinterpolation"], axis=1)

    output_tags = [
        "SKAP_18FI381-VFlLGas/Y/10sSAMP|average",
        "SKAP_18FI381-VFlLH2O/Y/10sSAMP|average",
        "SKAP_18FI381-VFlLOil/Y/10sSAMP|average",
    ]

    y_tags = ["SKAP_18FI381-VFlLGas/Y/10sSAMP|average"]

    df_train, df_test = train_test_split(df, test_size=0.1, shuffle=False)

    X_train = df_train.drop(output_tags, axis=1)
    y_train = df_train[y_tags]
    y_train.columns = ["gas"]

    X_test = df_test.drop(output_tags, axis=1)
    y_test = df_test[y_tags]
    y_test.columns = ["gas"]

    # Scale
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
    lr = Predictor()
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
    plot(y_scaler.inverse_transform(y_pred), y_scaler.inverse_transform(y_true))


def score(Y_pred, Y_true):
    E = np.sqrt(np.sum((Y_pred - Y_true) ** 2))
    L2pred = np.sqrt(np.sum(Y_pred ** 2))
    L2true = np.sqrt(np.sum(Y_true ** 2))

    return 2 * E / (L2pred + L2true)


def plot(y_pred, y_true):
    df_y_pred = pd.DataFrame(y_pred)
    df_y_true = pd.DataFrame(y_true)

    df = pd.concat((df_y_pred, df_y_true), axis=1)
    df.columns = ["gas_pred", "gas_true"]
    df.plot()

    import matplotlib.pyplot as plt

    plt.show()


def main():
    df_d02 = pd.read_csv("../data/d2.csv")
    X_train, X_test, y_train, y_test = preprocess(df_d02)

    train(X_train, y_train)
    test(X_test, y_test)


if __name__ == "__main__":
    main()
