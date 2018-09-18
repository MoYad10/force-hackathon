import os
from datetime import datetime

import pandas as pd
from cognite.config import configure_session
from cognite.data_transfer_service import DataSpec, DataTransferService, TimeSeries, TimeSeriesDataSpec


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
        granularity="10m",
        start=datetime(2014, 3, 1),
        end="now",
        label="d02",
    )
    d03_tsds = TimeSeriesDataSpec(
        time_series=d03_input_time_series,
        aggregates=["avg"],
        granularity="10m",
        start=datetime(2014, 3, 1),
        end="now",
        label="d03",
    )

    data_spec = DataSpec(time_series_data_specs=[d02_tsds, d03_tsds])

    dts = DataTransferService(data_spec, num_of_processes=5)

    df_dict = dts.get_dataframes()

    for label, df in df_dict.items():
        df.to_csv(f"../data/{label}.csv")


if __name__ == "__main__":
    main()
