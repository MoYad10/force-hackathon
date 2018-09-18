
import pandas as pd
import numpy as np
import datetime

D3_switch = "SKAP_18HV3821/BCH/10sSAMP"
D2_switch = "SKAP_18HV3806/BCH/10sSAMP"


def split_into_wells(data):
    d2_condition = data[D2_switch] > 0.9
    d3_condition = data[D3_switch] > 0.9

    d2_data = data[d2_condition]
    d3_data = data[d3_condition]

    well_tags = pd.read_csv("../tags/well_tags.csv")
    D2_tags = well_tags[well_tags["placement"] == "WellD02"]
    D3_tags = well_tags[well_tags["placement"] == "WellD03"]

    D2_lst = [col for col in data.columns if col.split("|")[0] in D2_tags.tag.values]
    D3_lst = [col for col in data.columns if col.split("|")[0] in D3_tags.tag.values]

    d2_data = data.drop(D3_lst, axis=1)
    d3_data = data.drop(D2_lst, axis=1)

    return d2_data, d3_data


def train_verification_split(data):
    split = int(datetime.datetime(2017, 6, 1).timestamp() * 1000)
    cond = data.timestamp > split
    training_data = data[~cond]
    verification_data = data[cond]

    return training_data, verification_data

