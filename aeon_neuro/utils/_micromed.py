"""Adapt continuous Micromed TRC recordings to MNE Raw."""

import os
import struct

import mne
import numpy as np

_TRIGGER = np.dtype([("sample", "<u4"), ("code", "<u2")])
_VOLTS_PER_UNIT = {-1: 1e-9, 0: 1e-6, 1: 1e-3, 2: 1.0}
_HEADER_SIZE = 176 + 15 * 16


def _trc_layout(path):
    """Check source facts that Neo can default or omit during decoding."""
    with path.open("rb") as stream:
        header = stream.read(_HEADER_SIZE)
        if len(header) != _HEADER_SIZE:
            raise ValueError("Incomplete Micromed TRC header.")
        if header[175] != 4:
            raise ValueError("Micromed TRC requires a type-4 header.")
        data_start, n_channels, _, base_rate, sample_width = struct.unpack_from(
            "<IHHHH", header, 138
        )
        payload = os.fstat(stream.fileno()).st_size - data_start
        if (
            not n_channels
            or not base_rate
            or sample_width not in (1, 2, 4)
            or data_start < _HEADER_SIZE
            or payload <= 0
            or payload % (n_channels * sample_width)
        ):
            raise ValueError(
                "Invalid Micromed TRC signal layout or incomplete samples."
            )
        n_samples = payload // (n_channels * sample_width)
        zones = {
            name.rstrip(b" "): (start, size)
            for name, start, size in struct.iter_unpack("<8sII", header[176:])
        }
        for name in (b"ORDER", b"LABCOD", b"TRIGGER"):
            start, size = zones.get(name, (0, 0))
            if (name != b"TRIGGER" or size) and (
                start < _HEADER_SIZE or start + size > data_start
            ):
                raise ValueError("Invalid Micromed TRC channel or trigger area.")

        order_start, order_size = zones[b"ORDER"]
        channel_start, channel_size = zones[b"LABCOD"]
        if order_size < 2 * n_channels:
            raise ValueError("Incomplete Micromed TRC channel order.")
        stream.seek(order_start)
        order = np.fromfile(stream, dtype="<u2", count=n_channels)
        scales = []
        for index in order:
            electrode = int(index) * 128
            if electrode + 128 > channel_size:
                raise ValueError(
                    "Micromed TRC channel order references a missing electrode."
                )
            # Neo assumes microvolts for unknown codes. Check the source code first.
            stream.seek(channel_start + electrode + 34)
            (unit,) = struct.unpack("<h", stream.read(2))
            if unit not in _VOLTS_PER_UNIT:
                raise ValueError("Micromed TRC channels must declare voltage units.")
            scales.append(_VOLTS_PER_UNIT[unit])

        # Read trigger positions on the stored sample grid. Neo drops every
        # sample-zero entry, including real codes next to zeroed padding rows.
        trigger_start, trigger_size = zones.get(b"TRIGGER", (0, 0))
        if trigger_size % _TRIGGER.itemsize:
            raise ValueError("Incomplete Micromed TRC trigger entry.")
        stream.seek(trigger_start)
        triggers = np.fromfile(
            stream, dtype=_TRIGGER, count=trigger_size // _TRIGGER.itemsize
        )
        triggers = triggers[(triggers["sample"] != 0) | (triggers["code"] != 0)]
        if np.any(triggers["sample"] >= n_samples):
            raise ValueError("A Micromed TRC trigger lies outside the recording.")
    return n_samples, np.asarray(scales), triggers


def _read_micromed(path, *, preload=False, verbose=None):
    """Read one type-4 TRC segment as calibrated voltage and numeric triggers."""
    if not isinstance(preload, bool):
        raise ValueError(
            "Micromed TRC always loads into memory; preload must be a bool."
        )
    try:
        from neo.core import NeoReadWriteError
        from neo.rawio import MicromedRawIO
    except ModuleNotFoundError as error:
        if error.name != "neo":
            raise
        raise ImportError(
            "Micromed TRC requires Neo. Install 'aeon-neuro[eeg_io]'."
        ) from error

    n_samples, scales, triggers = _trc_layout(path)
    reader = MicromedRawIO(filename=str(path))
    try:
        reader.parse_header()
    except (
        NeoReadWriteError,
        ValueError,
        IndexError,
        struct.error,
        ZeroDivisionError,
    ) as error:
        raise ValueError(
            f"Micromed TRC header could not be decoded: {error}"
        ) from error
    if (
        reader.segment_count(0) != 1
        or reader.get_signal_size(0, 0, stream_index=0) != n_samples
    ):
        # The released Neo reader can skip samples before the first segment.
        raise ValueError("Micromed TRC requires one continuous, complete segment.")
    sfreq = reader.get_signal_sampling_rate(stream_index=0)
    channels = reader.header["signal_channels"]
    names = [name.strip() for name in channels["name"]]
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Micromed TRC requires a positive sampling rate.")
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("Micromed TRC requires unique, nonempty channel names.")
    if np.any(channels["gain"] <= 0):
        raise ValueError("Micromed TRC requires positive channel calibration gains.")

    data = np.empty((len(names), n_samples), dtype=np.float64)
    # Scale in bounded chunks so the calibrated array is the only full-size copy.
    chunk_size = max(1, (1 << 20) // len(names))
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        codes = reader.get_analogsignal_chunk(
            i_start=start, i_stop=stop, stream_index=0
        )
        values = reader.rescale_signal_raw_to_float(
            codes, dtype="float64", stream_index=0
        )
        np.multiply(values.T, scales[:, None], out=data[:, start:stop])

    raw = mne.io.RawArray(data, mne.create_info(names, sfreq, "eeg"), verbose=verbose)
    raw.set_annotations(
        mne.Annotations(triggers["sample"] / sfreq, 0, triggers["code"].astype(str))
    )
    return raw
