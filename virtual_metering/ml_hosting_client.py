import os

import requests

COGNITE_API_KEY = os.getenv("COGNITE_API_KEY")
PROJECT = "akerbp"
BASE_URL = f"/api/0.6/projects/{PROJECT}/models"


def start_train(source_id, model_id):
    url = BASE_URL + f"/{model_id}/versions/train"
    data = {
        "source_package_id": source_id,
        "name": "string",
        "description": "string",
        "training_details": {
            "source_package_id": source_id,
            "data_spec": {
                "time_series_data_specs": [
                    {
                        "time_series": [
                            {
                                "name": "SKAP_41FSI0102/MeasA/PRIMnames",
                                "aggregates": ["avg"],
                                "missing_data_strategy": "linearInterpolation",
                            }
                        ],
                        "aggregates": ["avg"],
                        "granularity": "1s",
                        "missing_data_strategy": "linearInterpolation",
                        "start": 1357147740000,
                        "end": 1357147780000,
                        "label": "testdata",
                    }
                ],
                "files_data_spec": {"file_ids": []},
            },
            "args": {"event_type": "rapture_disc_state"},
        },
        "meta_data": {},
    }

    headers = {"api-key": COGNITE_API_KEY, "Content-Type": "application/json"}

    ret = requests.post(url, json=data, headers=headers)
    return ret
