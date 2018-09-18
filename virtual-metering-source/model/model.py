import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from model.preprocessing import preprocess


class Model:
    def __init__(self, model, X_scaler, y_scaler):
        self._model = model
        self._X_scaler = X_scaler
        self._y_scaler = y_scaler

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
        with file_io("d2_train.csv", "rb") as f:
            df_d02 = pd.read_csv(f)
        X_train, X_test, y_train, y_test = preprocess(df_d02, file_io)

        lr = LinearRegression()
        model = lr.fit(X_train, y_train)

        with file_io("model.pkl", "wb") as f:
            pickle.dump(model, f)

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
        with file_io("y_scaler.pkl", "rb") as f:
            y_scaler = pickle.load(f)
        with file_io("X_scaler.pkl", "rb") as f:
            X_scaler = pickle.load(f)
        return Model(model, X_scaler, y_scaler)


if __name__ == "__main__":
    Model.train(open)
    model = Model.load(open)
    res = model.predict(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]).reshape(1, -1))
    json.dumps(res)
    print(res)
