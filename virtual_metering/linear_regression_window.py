import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

WINDOW_SIZE = 1000 * 60 * 60 * 24 * 14
HISTORY_LENGTH = 1000 * 60 * 60 * 24 * 30 * 6
MIN_HISTORY_POINTS = 5000

def X_y_t(data):
    data = data.drop(['SKAP_18SCSSV3205/BCH/10sSAMP|average', 'SKAP_18HPB320/BCH/10sSAMP|average', ], axis=1)
    time_cond = (data.timestamp > int(datetime(2014, 3, 1).timestamp() * 1000))
    data = data[time_cond]

    output_columns = [
        'SKAP_18FI381-VFlLGas/Y/10sSAMP|average',
        # 'SKAP_18FI381-VFlLH2O/Y/10sSAMP|average',
        # 'SKAP_18FI381-VFlLOil/Y/10sSAMP|average',
    ]

    y = data[output_columns]
    X = data.drop(output_columns, axis=1)
    X = X.drop(['timestamp', 'Unnamed: 0'], axis=1)

    t = data['timestamp'].values

    return X, y, t

def preprocess(X, y, t):

    now = t[-1]

    history_split = now - HISTORY_LENGTH
    if len(X[t > history_split]) < MIN_HISTORY_POINTS:
        X = X[-MIN_HISTORY_POINTS:]
        y = y[-MIN_HISTORY_POINTS:]
        t = t[-MIN_HISTORY_POINTS:]
    else:
        X = X[t > history_split]
        y = y[t > history_split]
        t = t[t > history_split]

    train_val_split = now - WINDOW_SIZE
    X_train = X[t < train_val_split]
    X_test = X[t >= train_val_split]

    y_train = y[t < train_val_split]
    y_test = y[t >= train_val_split]

    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    X_train_scaled = X_train_scaled.clip(-5, 5)
    X_test_scaled = X_test_scaled.clip(-5, 5)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scaler, y_scaler


def train(X, y):
    lr = LinearRegression()
    model = lr.fit(X, y, sample_weight=np.linspace(0, 1, len(X)))
    return model


def test(model, X, y_true, y_scaler):
    y_pred = model.predict(X)
    y_true = y_true

    return y_scaler.inverse_transform(y_pred), y_scaler.inverse_transform(y_true)



def score(Y_pred, Y_true):
    E = np.sqrt(np.sum((Y_pred - Y_true) ** 2))
    L2pred = np.sqrt(np.sum(Y_pred ** 2))
    L2true = np.sqrt(np.sum(Y_true ** 2))

    return 2 * E / (L2pred + L2true)


def main():
    data = pd.read_csv("../data/d2_train.csv")

    X, y, t = X_y_t(data)
    last_t = t[20000]
    y_trues, y_preds = None, None
    xs = None
    for i in range(len(X)):
        if t[i] > last_t + WINDOW_SIZE:
            last_t = t[i]

            X_train, X_test, y_train, y_test, X_scaler, y_scaler = preprocess(X[:i], y[:i], t[:i])

            model = train(X_train, y_train)
            y_pred, y_true = test(model, X_test, y_test, y_scaler)

            y_trues = np.concatenate((y_trues, y_true), axis=0) if y_trues is not None else y_true
            y_preds = np.concatenate((y_preds, y_pred), axis=0) if y_preds is not None else y_pred

            xs = np.concatenate((xs, X_test), axis=0) if xs is not None else X_test

    print(score(y_preds, y_trues))

    # plt.plot(y_preds, label='pred')
    # plt.plot(y_trues, label='true')
    # plt.plot(xs, label='x')
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()

