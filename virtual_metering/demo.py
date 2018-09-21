import os
import pickle
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from cognite._utils import APIError
from cognite.config import configure_session
from cognite.data_transfer_service import DataSpec, DataTransferService, TimeSeries, TimeSeriesDataSpec
from cognite.v05.dto import Datapoint
from cognite.v05.dto import TimeSeries as TimeSeriesDTO
from cognite.v05.timeseries import post_datapoints, post_time_series
from virtual_metering.data_fetcher import EXCLUDE_TAGS

configure_session(api_key=os.getenv("COGNITE_API_KEY"), project="akerbp", debug=True)

try:
    ts = TimeSeriesDTO(name="SKAP_18FI381-VFlLGas/Y/10sSAMP_calc_D02_2", asset_id=8129784932439587)
    res = post_time_series([ts])
    print(res)
except APIError as e:
    print(e)
    print("øladfhjaoøsidfhjapsoidfjaspodifhjaspoid")
    pass


def main():
    output_columns = [
        "SKAP_18FI381-VFlLGas/Y/10sSAMP|average",
        "SKAP_18FI381-VFlLH2O/Y/10sSAMP|average",
        "SKAP_18FI381-VFlLOil/Y/10sSAMP|average",
    ]
    router = "SKAP_18HV3806/BCH/10sSAMP|stepinterpolation"
    last_processed_timestamp = int(datetime(2018, 9, 19, 8).timestamp() * 1e3)

    is_first = True

    while True:
        d2_inputs = pd.DataFrame([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
        d2_inputs.columns = ["hoho", "blaa", "hgi"] + output_columns
        while d2_inputs.drop(output_columns, axis=1).isna().any().any():
            ds = generate_data_spec(last_processed_timestamp)

            dts = DataTransferService(data_spec=ds)
            failed = True
            while failed:
                try:
                    d2_inputs = dts.get_dataframes()["d2"]
                    failed = False
                except:
                    time.sleep(2)
            any_nans_per_column = d2_inputs.drop(output_columns, axis=1).isna().any()
            all_nans_per_column = d2_inputs.drop(output_columns, axis=1).isna().all()

            print(any_nans_per_column)
            print(all_nans_per_column)

            if any_nans_per_column.any() and not all_nans_per_column.any():
                last_processed_timestamp -= 10000

            print(datetime.fromtimestamp(last_processed_timestamp * 1e-3))
            time.sleep(2)

        last_ts = d2_inputs["timestamp"].iloc[-1]

        print(d2_inputs[output_columns[0]].values.tolist())
        d2_inputs_formatted = (
            d2_inputs.drop("timestamp", axis=1).drop(router, axis=1).drop(output_columns, axis=1).values.tolist()
        )
        timestamps = d2_inputs["timestamp"]
        # res = models.online_predict(
        #     model_id=3885574571413770, version_id=2537754531443017, instances=[d2_inputs_formatted]
        # )

        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("X_scaler.pkl", "rb") as f:
            X_scaler = pickle.load(f)
        with open("y_scaler.pkl", "rb") as f:
            y_scaler = pickle.load(f)

        d2_inputs_formatted = X_scaler.transform(d2_inputs_formatted)

        print(d2_inputs_formatted.tolist()[0])
        res = model.predict(d2_inputs_formatted)
        res = y_scaler.inverse_transform(res)
        res = [int(r[0]) for r in res]
        last_processed_timestamp = int(last_ts)

        dps = [Datapoint(ts, value) for ts, value in zip(timestamps.values.tolist(), res)]
        print([dp.value for dp in dps])
        if is_first:
            failed = True
            while failed:
                try:
                    post_datapoints(name="SKAP_18FI381-VFlLGas/Y/10sSAMP_calc_D02_2", datapoints=dps)
                    failed = False
                except:
                    time.sleep(2)
            is_first = False
        else:
            for dp in dps:
                failed = True
                while failed:
                    try:
                        post_datapoints(name="SKAP_18FI381-VFlLGas/Y/10sSAMP_calc_D02_2", datapoints=[dp])
                        failed = False
                    except:
                        time.sleep(2)
                time.sleep(5)


def generate_data_spec(last_processed_timestamp, granularity="10s"):
    tags_d03 = []
    tags_d02 = []

    for root, subdirs, files in os.walk("../tags"):
        for file in files:
            if file in ("well_tags.csv", "routing.csv", "riser_tags.csv", "output.csv", "template_tags.csv"):
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
        granularity=granularity,
        start=last_processed_timestamp,
        end=int(datetime.now().timestamp() * 1e3),
        label="d2",
        missing_data_strategy="ffill",
    )
    d03_tsds = TimeSeriesDataSpec(
        time_series=d03_input_time_series,
        aggregates=["avg"],
        granularity=granularity,
        start=last_processed_timestamp,
        end=int(datetime.now().timestamp() * 1e3),
        label="d3",
        missing_data_strategy="ffill",
    )

    return DataSpec(time_series_data_specs=[d02_tsds, d03_tsds])


if __name__ == "__main__":
    main()
