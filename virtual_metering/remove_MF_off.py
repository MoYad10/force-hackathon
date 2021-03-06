
import datetime

import numpy as np
import pandas as pd

D3_switch = "SKAP_18HV3821/BCH/10sSAMP|stepinterpolation"
D2_switch = "SKAP_18HV3806/BCH/10sSAMP|stepinterpolation"


def remove_MF_off():
    d2 = pd.read_csv("../data/d2.csv")
    d3 = pd.read_csv("../data/d3.csv")

    d2_condition = d2[D2_switch] > 0.9
    d3_condition = d3[D3_switch] > 0.9

    d2_int = d2_condition.values.astype(int)
    sw = np.zeros(len(d2))
    K = 6
    val = np.ones(len(sw[K:-K]))
    for i in range(0, 2 * K):
        ixm = i
        ixp = -2 * K + i
        val *= d2_int[ixm:ixp]

    sw[K:-K] = val
    d2_condition = sw.astype(bool)

    d3_int = d3_condition.values.astype(int)
    sw = np.zeros(len(d3))
    K = 6
    val = np.ones(len(sw[K:-K]))
    for i in range(0, 2 * K):
        ixm = i
        ixp = -2 * K + i
        val *= d3_int[ixm:ixp]

    sw[K:-K] = val
    d3_condition = sw.astype(bool)

    d2_data = d2[d2_condition]
    d3_data = d3[d3_condition]

    d2_data.to_csv("../data/d2_on_MF.csv", index=False)
    d3_data.to_csv("../data/d3_on_MF.csv", index=False)


if __name__ == "__main__":
    remove_MF_off()
