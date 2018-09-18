
import pandas as pd
import numpy as np
from datetime import datetime


def test_train_split():
    d2 = pd.read_csv("d02_on_MF.csv")
    d3 = pd.read_csv("d03_on_MF.csv")

    split_time = int(datetime(2017, 1, 1).timestamp() * 1000)
    d2_condition = d2.timestamp >= split_time
    d3_condition = d3.timestamp >= split_time

    d2_train = d2[~d2_condition]
    d2_test = d2[d2_condition]

    d3_train = d3[~d3_condition]
    d3_test = d3[d3_condition]

    d2_train.to_csv("d2_train.csv", index=False)
    d2_test.to_csv("d2_test.csv", index=False)
    d3_train.to_csv("d3_train.csv", index=False)
    d3_test.to_csv("d3_test.csv", index=False)


if __name__ == "__main__":
    test_train_split()

