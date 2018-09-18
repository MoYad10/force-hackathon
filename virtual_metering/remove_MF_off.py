
import pandas as pd
import numpy as np
import datetime

D3_switch = "SKAP_18HV3821/BCH/10sSAMP|stepinterpolation"
D2_switch = "SKAP_18HV3806/BCH/10sSAMP|stepinterpolation"


def remove_MF_off():
    d2 = pd.read_csv("d02.csv")
    d3 = pd.read_csv("d03.csv")

    d2_condition = d2[D2_switch] > 0.9
    d3_condition = d3[D3_switch] > 0.9

    d2_data = d2[d2_condition]
    d3_data = d3[d3_condition]

    d2_data.to_csv("d02_on_MF.csv")
    d3_data.to_csv("d03_on_MF.csv")


if __name__ == "__main__":
    remove_MF_off()

