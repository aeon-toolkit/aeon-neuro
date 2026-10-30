"""Preprocessing utilities for EEG recordings."""

from copy import deepcopy

import mne
import numpy as np


def preprocess_eeg(
    X,
    y,
    metadata,
    include_channels=None,
    exclude_channels=None,
    l_freq=1.0,
    h_freq=40.0,
    epoch_duration=None,
    epoch_overlap=0.0,
):
    """Preprocess EEG recordings and optionally split them into epochs.

    Parameters
    ----------
    X : np.ndarray or list of np.ndarray
        EEG recordings, with each recording shaped
        (n_channels, n_times).
    y : array-like
        One label per input recording.
    metadata : list of dict
        Recording metadata containing 'channel_names' and
        'sampling_frequency' (or 'sfreq').
    include_channels : list of str or None
        Channel names to retain.
    exclude_channels : list of str or None
        Channel names to exclude.
    l_freq : float or None, default=1.0
        High-pass cutoff frequency in Hz. None disables high-pass filtering.
    h_freq : float or None, default=40.0
        Low-pass cutoff frequency in Hz. None disables low-pass filtering.
    epoch_duration : float or None, default=None
        Epoch duration in seconds. None returns continuous recordings.
    epoch_overlap : float, default=0.0
        Overlap between epochs in seconds.

    Returns
    -------
    X_processed : np.ndarray or list of np.ndarray
        Preprocessed recordings or epochs. When epoching is enabled,
        the output has shape (n_epochs, n_channels, n_samples).
    y_processed : np.ndarray
        One label per output recording or epoch.
    metadata_processed : list of dict
        Metadata corresponding to each output recording or epoch.
        Epoch metadata includes 'source_recording' and 'epoch_index'.
    """
    if isinstance(X, np.ndarray):
        if X.ndim != 3:
            raise ValueError("X must have shape (n_recordings, n_channels, n_times).")
        recordings = [X[i] for i in range(X.shape[0])]
    else:
        recordings = list(X)

    y = np.asarray(y, dtype=object)
    metadata = deepcopy(metadata)

    if not (len(recordings) == len(y) == len(metadata)):
        raise ValueError("X, y, and metadata must have equal lengths.")

    if not recordings:
        raise ValueError("X contains no recordings.")

    if l_freq is not None and l_freq <= 0:
        raise ValueError("l_freq must be positive or None.")

    if h_freq is not None and h_freq <= 0:
        raise ValueError("h_freq must be positive or None.")

    if l_freq is not None and h_freq is not None and l_freq >= h_freq:
        raise ValueError("l_freq must be lower than h_freq.")

    if epoch_duration is not None and epoch_duration <= 0:
        raise ValueError("epoch_duration must be positive or None.")

    if epoch_overlap < 0:
        raise ValueError("epoch_overlap cannot be negative.")

    if epoch_duration is not None and epoch_overlap >= epoch_duration:
        raise ValueError("epoch_overlap must be smaller than epoch_duration.")

    processed = []
    processed_y = []
    processed_metadata = []

    for index, (data, record_metadata, label) in enumerate(
        zip(recordings, metadata, y)
    ):
        data = np.asarray(data, dtype=float)

        if data.ndim != 2:
            raise ValueError(
                f"Recording {index} must have shape (n_channels, n_times)."
            )

        channel_names = record_metadata.get("channel_names")
        sfreq = record_metadata.get(
            "sampling_frequency",
            record_metadata.get("sfreq"),
        )

        if channel_names is None:
            raise ValueError(f"Recording {index} metadata has no channel_names.")

        if sfreq is None or float(sfreq) <= 0:
            raise ValueError(f"Recording {index} has an invalid sampling frequency.")

        channel_names = list(channel_names)

        if len(channel_names) != data.shape[0]:
            raise ValueError(
                f"Recording {index}: channel count does not match metadata."
            )

        if len(set(channel_names)) != len(channel_names):
            raise ValueError(f"Recording {index} contains duplicate channel names.")

        selected = channel_names.copy()

        if include_channels is not None:
            missing = set(include_channels) - set(channel_names)
            if missing:
                raise ValueError(
                    f"Recording {index} is missing requested channels: "
                    f"{sorted(missing)}"
                )
            selected = [name for name in selected if name in include_channels]

        if exclude_channels is not None:
            selected = [name for name in selected if name not in exclude_channels]

        if not selected:
            raise ValueError(f"No channels remain in recording {index}.")

        channel_indices = [channel_names.index(name) for name in selected]
        data = data[channel_indices, :]

        raw = mne.io.RawArray(
            data,
            mne.create_info(
                ch_names=selected,
                sfreq=float(sfreq),
                ch_types=["eeg"] * len(selected),
            ),
            verbose=False,
        )

        nyquist = float(sfreq) / 2

        if h_freq is not None and h_freq >= nyquist:
            raise ValueError(
                f"h_freq must be below the Nyquist frequency ({nyquist} Hz) "
                f"for recording {index}."
            )

        if l_freq is not None and l_freq >= nyquist:
            raise ValueError(
                f"l_freq must be below the Nyquist frequency ({nyquist} Hz) "
                f"for recording {index}."
            )

        if l_freq is not None or h_freq is not None:
            raw.filter(
                l_freq=l_freq,
                h_freq=h_freq,
                picks="eeg",
                verbose=False,
            )

        updated_metadata = deepcopy(record_metadata)
        updated_metadata["channel_names"] = list(raw.ch_names)
        updated_metadata["sampling_frequency"] = float(raw.info["sfreq"])

        if epoch_duration is None:
            processed.append(raw.get_data())
            processed_y.append(label)
            processed_metadata.append(updated_metadata)
            continue

        sfreq = float(raw.info["sfreq"])
        epoch_samples = int(round(epoch_duration * sfreq))
        overlap_samples = int(round(epoch_overlap * sfreq))
        step_samples = epoch_samples - overlap_samples

        if epoch_samples < 1 or step_samples < 1:
            raise ValueError("Epoch duration and overlap are too small.")

        data = raw.get_data()
        epoch_index = 0

        for start in range(0, data.shape[1] - epoch_samples + 1, step_samples):
            stop = start + epoch_samples
            epoch = data[:, start:stop]

            processed.append(epoch)
            processed_y.append(label)

            epoch_metadata = deepcopy(updated_metadata)
            epoch_metadata["source_recording"] = record_metadata.get(
                "filepath",
                record_metadata.get("filename", str(index)),
            )
            epoch_metadata["epoch_index"] = epoch_index
            epoch_metadata["start_time"] = start / sfreq
            epoch_metadata["end_time"] = stop / sfreq

            processed_metadata.append(epoch_metadata)
            epoch_index += 1

    if not processed:
        raise ValueError(
            "Preprocessing produced no output. Check recording lengths "
            "and epoch_duration."
        )

    compatible = all(
        data.shape == processed[0].shape
        and info["channel_names"] == processed_metadata[0]["channel_names"]
        and np.isclose(
            info["sampling_frequency"],
            processed_metadata[0]["sampling_frequency"],
        )
        for data, info in zip(processed, processed_metadata)
    )

    X_processed = np.stack(processed, axis=0) if compatible else processed

    return (
        X_processed,
        np.asarray(processed_y, dtype=object),
        processed_metadata,
    )
