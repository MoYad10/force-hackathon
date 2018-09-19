import json
import pickle

import numpy as np
import pandas as pd

import lightgbm as gbm


class Model:
    def __init__(self, model):
        self._model = model

    @staticmethod
    def train(file_io, **kwargs):
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
        with file_io("d2.csv", "rb") as f:
            df_d02 = pd.read_csv(f)

        X, y = Model.preprocess(df_d02)

        X = X.rolling(5).median()
        X = X.fillna(method="bfill")
        y = y.rolling(5).median()
        y = y.fillna(method="bfill")

        estimator = gbm.LGBMRegressor(n_estimators=100)

        estimator.fit(X, y)

        with file_io("model.pkl", "wb") as f:
            pickle.dump(estimator, f)

    @staticmethod
    def preprocess(df):
        switch = "SKAP_18HV3806/BCH/10sSAMP|stepinterpolation"

        condition = df[switch] > 0.9

        data = df[condition]

        output_columns = [
            "SKAP_18FI381-VFlLGas/Y/10sSAMP|average",
            "SKAP_18FI381-VFlLH2O/Y/10sSAMP|average",
            "SKAP_18FI381-VFlLOil/Y/10sSAMP|average",
        ]

        y_columns = ["SKAP_18FI381-VFlLGas/Y/10sSAMP|average"]

        output_data = data[y_columns]
        input_data = data.drop(output_columns, axis=1)
        input_data = input_data.drop(["timestamp"], axis=1)
        return input_data, output_data

    def predict(self, data, **kwargs):
        """Method to perform predictions on your model.

        Args:
            data:       The input to your model.
        Keyword Args:
            Any user defined arguments
        Returns:
            Json serializable output from your model.
        """
        res = self._model.predict(data)
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
        return Model(model)


if __name__ == "__main__":
    Model.train(open)
    model = Model.load(open)
    res = model.predict(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]).reshape(1, -1))
    json.dumps(res)
    print(res)
