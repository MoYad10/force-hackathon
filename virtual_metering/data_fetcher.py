import os
from datetime import datetime

import pandas as pd

from cognite.config import configure_session
from cognite.data_transfer_service import DataSpec, DataTransferService, TimeSeries, TimeSeriesDataSpec

EXCLUDE_TAGS_D2 = [
    "SKAP_18TI3202/Y/10sSAMP",
    "SKAP_18ESV3211/BCH/10sSAMP",
    "SKAP_18ESV3220/BCH/10sSamp",
    "SKAP_18ESV3218/BCH/10sSamp",
    "SKAP_18ESV3215/BCH/10sSamp",
    "SKAP_18ESV3214/BCH/10sSAMP",
    "SKAP_18ESV3213/BCH/10sSamp",
    "SKAP_18ESV3207/BCH/10sSamp",
    "SKAP_18ESV3206/BCH/10sSamp",
    "SKAP_18ESV3219/BCH/10sSamp",
    "SKAP_18PI3230/MeasA/10sSAMP",
    "SKAP_18SCSSV3205/BCH/10sSAMP",
    "SKAP_18HPB320/BCH/10sSAMP",
    "SKAP_13TI6147/Y/10sSAMP",
    "SKAP_18PI3201/Y/10sSAMP",
    "SKAP_18PI3203/Y/10sSAMP",
    "SKAP_18TI3204/Y/10sSAMP",
]

EXCLUDE_TAGS_D3 = [
    "SKAP_18TI3302/Y/10sSAMP",
    "SKAP_18ESV3311/BCH/10sSAMP",
    "SKAP_18ESV3320/BCH/10sSamp",
    "SKAP_18ESV3318/BCH/10sSamp",
    "SKAP_18ESV3315/BCH/10sSamp",
    "SKAP_18ESV3314/BCH/10sSAMP",
    "SKAP_18ESV3313/BCH/10sSamp",
    "SKAP_18ESV3307/BCH/10sSamp",
    "SKAP_18ESV3306/BCH/10sSamp",
    "SKAP_18ESV3319/BCH/10sSamp",
    "SKAP_18PI3330/MeasA/10sSAMP",
    "SKAP_18SCSSV3305/BCH/10sSAMP",
    "SKAP_18HPB330/BCH/10sSAMP",
]

EXCLUDE_TAGS = EXCLUDE_TAGS_D2 + EXCLUDE_TAGS_D3


def main():
    configure_session(api_key=os.getenv("COGNITE_API_KEY"), project="akerbp", debug=True)
    tags_d03 = []
    tags_d02 = []

    for root, subdirs, files in os.walk("../tags"):
        for file in files:
            if file in ("well_tags.csv", "routing.csv", "output.csv", "riser_tags.csv", "template_tags.csv"):
                with open(os.path.join(root, file)) as f:
                    df = pd.read_csv(f)

                    placements = ["T3 WGM", "Template", "Riser"]
                    placements_d03 = ["WellD03"] + placements
                    placements_d02 = ["WellD02"] + placements

                    df = df[~df["tag"].isin(EXCLUDE_TAGS)]

                    tags_d03.append(df[df["placement"].isin(placements_d03)])
                    tags_d02.append(df[df["placement"].isin(placements_d02)])

    tags_d02_concat = pd.concat(tags_d02, ignore_index=True)
    tags_d03_concat = pd.concat(tags_d03, ignore_index=True)

    tags_d02_concat = tags_d02_concat.drop_duplicates(subset="tag")
    tags_d03_concat = tags_d03_concat.drop_duplicates(subset="tag")

    d02_input_time_series = []
    d03_input_time_series = []

    for tag in tags_d02_concat["tag"]:
        aggregate = "step" if ("ESV" in tag or "18HV" in tag) else "avg"
        missing_data_strategy = "ffill" if ("ESV" in tag or "18HV" in tag) else "linearInterpolation"
        ts = TimeSeries(name=tag, missing_data_strategy=missing_data_strategy, aggregates=[aggregate])
        d02_input_time_series.append(ts)

    for tag in tags_d03_concat["tag"]:
        aggregate = "step" if ("ESV" in tag or "18HV" in tag) else "avg"
        missing_data_strategy = "ffill" if ("ESV" in tag or "18HV" in tag) else "linearInterpolation"
        ts = TimeSeries(name=tag, missing_data_strategy=missing_data_strategy, aggregates=[aggregate])
        d03_input_time_series.append(ts)

    d02_tsds = TimeSeriesDataSpec(
        time_series=d02_input_time_series,
        aggregates=["avg"],
        granularity="10s",
        start=int(datetime(2017, 3, 1).timestamp() * 1e3),
        label="d2",
    )
    d03_tsds = TimeSeriesDataSpec(
        time_series=d03_input_time_series,
        aggregates=["avg"],
        granularity="10s",
        start=int(datetime(2017, 3, 1).timestamp() * 1e3),
        label="d3",
    )

    data_spec = DataSpec(time_series_data_specs=[d02_tsds, d03_tsds])

    dts = DataTransferService(data_spec, num_of_processes=10)

    print(data_spec.to_JSON())

    df_dict = dts.get_dataframes()

    for label, df in df_dict.items():
        df.to_csv(f"../data/{label}.csv")
        print(df.shape)


if __name__ == "__main__":
    main()
