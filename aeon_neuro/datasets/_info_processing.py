"""Execute bounded EEG operations while preserving source and current layouts."""

from copy import deepcopy
from fractions import Fraction
from importlib.metadata import version
from numbers import Real

import numpy as np

from aeon_neuro import __version__
from aeon_neuro.datasets._info_view import _positive, _recording_view


def _parameters(operation, parameters):
    fields = {
        "channels": ({"names"}, set()),
        "scale": ({"factor", "units"}, set()),
        "crop": (set(), {"tmin", "tmax", "start_time"}),
        "resample": ({"sfreq"}, set()),
        "filter": ({"l_freq", "h_freq"}, {"method"}),
        "reference": ({"channels"}, set()),
    }
    if operation not in fields:
        raise ValueError(f"Unsupported EEG operation: {operation!r}")
    required, optional = fields[operation]
    if not required <= parameters.keys() or parameters.keys() - (required | optional):
        raise ValueError(
            f"Invalid parameters for {operation}; required: {sorted(required)}"
        )


def _names(value, names):
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(name, str) for name in value)
        or len(set(value)) != len(value)
        or any(name not in names for name in value)
    ):
        raise ValueError("Select unique, known current channel names")
    return [names.index(name) for name in value]


def _apply(array, layout, operation, parameters):
    names, rate = layout["channel_names"], layout["sampling_frequency"]
    if operation == "channels":
        picks = _names(parameters["names"], names)
        layout["channel_names"] = [names[i] for i in picks]
        layout["units"] = [layout["units"][i] for i in picks]
        if "source_channel_names" in layout:
            layout["source_channel_names"] = [
                layout["source_channel_names"][i] for i in picks
            ]
        return array[picks].copy()
    if operation == "scale":
        factor = parameters["factor"]
        if (
            isinstance(factor, bool)
            or not isinstance(factor, Real)
            or not np.isfinite(factor)
            or factor == 0
        ):
            raise ValueError("Scale factor must be finite and nonzero")
        units = parameters["units"]
        if isinstance(units, str):
            units = [units] * len(names)
        if (
            not isinstance(units, list)
            or len(units) != len(names)
            or any(not isinstance(unit, str) or not unit for unit in units)
        ):
            raise ValueError("Output units must explicitly name every current channel")
        layout["units"] = list(units)
        return array * float(factor)
    if operation == "crop":
        from aeon_neuro.datasets._classification import _samples
        from aeon_neuro.transformations.collection import TimeCrop

        crop = TimeCrop(sfreq=rate, **parameters)
        result = crop.fit_transform([array])[0]
        start = (
            0
            if crop.tmin is None
            else _samples(crop.tmin - crop.start_time, rate, "crop")
        )
        layout["source_origin_sample"] += start * layout["source_sample_step"]
        return result
    if operation == "resample":
        from scipy.signal import resample_poly

        target = _positive(parameters["sfreq"], "resampling frequency")
        # Reflection needs a neighbor. SciPy can terminate the interpreter for
        # a singleton input, so reject it before entering the native routine.
        if array.shape[-1] < 2 and target != rate:
            raise ValueError("Resampling to a new rate requires at least two samples")
        # Preserve the rate ratio, not the rounded output-count ratio used by
        # MNE's polyphase wrapper. Decimal rates avoid binary-float denominators.
        ratio = Fraction(str(target)) / Fraction(str(rate))
        # SciPy's default filter length grows with the largest reduced factor.
        # Bound that work instead of approximating the requested sample grid.
        if max(ratio.numerator, ratio.denominator) > 100_000:
            raise ValueError(
                "Resampling rate ratio requires factors above 100000; "
                "choose rates with smaller reduced integer factors"
            )
        result = resample_poly(
            array.astype(float),
            ratio.numerator,
            ratio.denominator,
            axis=-1,
            padtype="reflect",
        )
        layout["sampling_frequency"] = target
        layout["source_sample_step"] *= rate / target
        return result
    if operation == "filter":
        from aeon_neuro.transformations.series import BandPassSeriesTransformer

        return BandPassSeriesTransformer(sfreq=rate, **parameters).fit_transform(
            array.astype(float)
        )
    reference = parameters["channels"]
    picks = (
        list(range(len(names))) if reference == "average" else _names(reference, names)
    )
    # Subtraction is only meaningful between known commensurate channels.
    if None in layout["units"] or len(set(layout["units"])) != 1:
        raise ValueError("Rereferencing requires equal known current channel units")
    layout["reference"] = {"method": "mean", "channels": [names[i] for i in picks]}
    return array - array[picks].mean(axis=0, keepdims=True)


def _process(X, info, operation, parameters):
    from aeon_neuro.datasets._dataset_info import EEGDatasetInfo, _json_history_value

    _parameters(operation, parameters)
    # Validate portability before any work, so operation records cannot fail only
    # after a costly transform has already run.
    _json_history_value(parameters)
    parameters = deepcopy(parameters)
    info.validate_layout(X)
    if not len(info):
        raise ValueError("Cannot process an empty information view")
    report = info.to_dict()
    view = deepcopy(info._view) if info._view is not None else _recording_view(info)
    before = deepcopy(view["layouts"])
    arrays = []
    for array, layout in zip(X, view["layouts"], strict=True):
        result = _apply(array, layout, operation, parameters)
        if operation in ("filter", "resample"):
            # Later cases can depend on neighboring samples of this input row.
            layout["temporal_context"] = "whole_row"
        if not np.isfinite(result).all() or result.shape[-1] == 0:
            raise ValueError("Processing produced empty or nonfinite samples")
        layout["n_samples"] = int(result.shape[-1])
        arrays.append(result)
    # Keep support conservative: earlier filtering/resampling may have used the
    # full input row even when a later crop displays only a smaller interval.
    report["view"] = view
    processed = EEGDatasetInfo(report)._with_history(
        operation,
        {
            "execution": "performed",
            "parameters": parameters,
            "case_ids": [row["case_id"] for row in view["rows"]],
            "input_layouts": before,
            "output_layouts": view["layouts"],
            "support_policy": "conservative original input interval per row",
            "backend_versions": {
                name: version(name)
                for name in ("numpy", "scipy" if operation == "resample" else "mne")
            },
        },
        implementation="aeon_neuro.datasets.EEGDatasetInfo.process",
        version=__version__,
        fit_scope="none; independently per row",
    )
    output = np.stack(arrays) if isinstance(X, np.ndarray) else arrays
    processed.validate_layout(output)
    return output, processed
