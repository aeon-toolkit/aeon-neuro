"""experimental code.

some functions to find out things about the data.
"""

import time

__author__ = ["TonyBagnall"]

datasets = [
    "Blink",
    "EyesOpenShut",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "MindReading",
    "MotorImagery",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
]


def time_all_dataloading(return_type="nested_univ"):
    times = {}
    for d in datasets:
        start = int(round(time.time() * 1000))
        X, y = load_UCR_UEA_dataset(
            name=d,
            extract_path="C:\\Code\aeon-neuro\aeon-neuro" r"\data",
            return_type=return_type,
        )
        load_time = int(round(time.time() * 1000)) - start
        print(
            d,
            " shape  = ",
            X.shape,
            " loaded into",
            return_type,
            " ie. ",
            type(X),
            "took ",
            load_time,
            " to load",
        )
        times[d] = load_time

    return times


if __name__ == "__main__":
    times = time_all_dataloading()
    times2 = time_all_dataloading(return_type="numpy3d")
    for d in datasets:
        print(
            d, " pandas load time = ", times[d] / 1000, " numpy load time = ", times2[d]
        )
