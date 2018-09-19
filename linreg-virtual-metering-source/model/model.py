import json
import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


class Model:
    def __init__(self, model, X_scaler, y_scaler):
        self._model = model
        self._X_scaler = X_scaler
        self._y_scaler = y_scaler

    @staticmethod
    def train(file_io, well, **kwargs):
        """The method to train your model.
        Should produce at least a serialized model. Can also produce other
        serialized artefacts that you will need when predicting.

        Args:
            file_io:    A callable which allows you to write to model hosting storage.
                        Works just like the built-in function open
        Keyword Args:
            Any user defined arguments

        Returns:
            None
        """
        with file_io("d" + well + ".csv", "rb") as f:
            df = pd.read_csv(f)

        X, y = Model.train_preprocess(df, well, file_io)

        model = LinearRegression()
        model.fit(X, y, sample_weight=np.linspace(0, 1, len(X)))

        with file_io("model.pkl", "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def train_preprocess(df, well, file_io):
        D2_switch = "SKAP_18HV3806/BCH/10sSAMP|stepinterpolation"
        D3_switch = "SKAP_18HV3821/BCH/10sSAMP|stepinterpolation"
        switch = D2_switch if well == '2' else D3_switch

        condition = df[switch] > 0.9

        data = df[condition]
        data = data.drop([switch], axis=1)

        output_columns = [
            "SKAP_18FI381-VFlLGas/Y/10sSAMP|average",
            "SKAP_18FI381-VFlLH2O/Y/10sSAMP|average",
            "SKAP_18FI381-VFlLOil/Y/10sSAMP|average",
        ]
        y_columns = [
            "SKAP_18FI381-VFlLGas/Y/10sSAMP|average",
        ]

        y = data[y_columns]
        X = data.drop(output_columns, axis=1)
        X = X.drop(["timestamp"], axis=1)

        X = X.fillna(method="bfill")
        y = y.fillna(method="bfill")

        X_scaler = StandardScaler()
        X = X_scaler.fit_transform(X)

        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(y)

        with file_io("X_scaler.pkl", "wb") as f:
            pickle.dump(X_scaler, f)

        with file_io("y_scaler.pkl", "wb") as f:
            pickle.dump(y_scaler, f)

        X = X.clip(-5, 5)
        X = X.clip(-5, 5)

        return X, y

    def predict(self, data, **kwargs):
        """Method to perform predictions on your model.

        Args:
            data:       The input to your model.
        Keyword Args:
            Any user defined arguments
        Returns:
            Json serializable output from your model.
        """

        data = np.array(data)
        data = self._X_scaler.transform(data)

        res = self._model.predict(data)
        res = self._y_scaler.inverse_transform(res)
        return res.tolist()

    @staticmethod
    def load(file_io):
        """Method to load your serialzed model into memory

        Can also load other artifacts such as preprocessors
        Args:
            file_io:    A callable which allows you to write to model hosting storage.
                        Works just like the built-in function open
        Returns:
            An instance of Model
        """
        with file_io("model.pkl", "rb") as f:
            model = pickle.load(f)
        with file_io("X_scaler.pkl", "rb") as f:
            X_scaler = pickle.load(f)
        with file_io("y_scaler.pkl", "rb") as f:
            y_scaler = pickle.load(f)
        return Model(model, X_scaler, y_scaler)


if __name__ == "__main__":
    Model.train(open, well="2")
    model = Model.load(open)
    res = model.predict([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
    json.dumps(res)
    print(res)
