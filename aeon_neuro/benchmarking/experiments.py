# -*- coding: utf-8 -*-
"""experimental code.

some functions to find out things about the data.
"""
import time

from sktime.datasets import load_UCR_UEA_dataset

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
            extract_path="C:\Code\sktime-neuro\sktime-neuro" "\data",
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


times = time_all_dataloading()
times2 = time_all_dataloading(return_type="numpy3d")
for d in datasets:
    print(d, " pandas load time = ", times[d] / 1000, " numpy load time = ", times2[d])
