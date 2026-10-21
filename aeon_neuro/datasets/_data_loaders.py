"""Function to load EEG Datasets from Zenodo."""

import os

from aeon.datasets._data_loaders import _load_collection
from aeon.datasets._single_problem_loaders import _load_saved_dataset
from aeon.datasets.dataset_collections import get_downloaded_tsc_tsr_datasets

import aeon_neuro
from aeon_neuro.datasets.classification_datasets import dataset_map

DIRNAME = "data"
MODULE = os.path.join(os.path.dirname(aeon_neuro.__file__), "datasets")


def load_eeg_classification(
    name,
    split=None,
    extract_path=None,
    return_metadata=False,
):
    """Load an EEG classification dataset.

    This function loads EEG TSC problems into memory, attempting to load from the
    specified local path ``extract_path`` or trying to download from
    https://zenodo.org/ with ``aeon``'s collection loader if the data is not
    in the local path. To download from
    Zenodo, the dataset must have a record ID in ``dataset_map`` in
    ``aeon_neuro.datasets.classification_datasets``. Datasets with the placeholder
    ID 49 are not yet available on Zenodo and can only be loaded from a local path.
    This function assumes the data is stored in format
    ``<extract_path>/<name>/<name>_TRAIN.ts`` and
    ``<extract_path>/<name>/<name>_TEST.ts``. If you want to load a file directly
    from a full path that is in ``aeon`` ts format, use the function
    ``load_from_ts_file`` in ``aeon`` directly. If you do not specify
    ``extract_path``, it looks in ``aeon_neuro/datasets/data`` (datasets shipped
    with the package) and otherwise downloads to the ``aeon`` data directory: the
    ``AEON_DATA`` environment variable if set, else ``~/aeon_data/``.

    Data is assumed to be in the standard ``aeon`` .ts format: each row is a (possibly
    multivariate) time series. Each channel is separated by a colon, each value in
    a series is comma separated. For examples see aeon_neuro.datasets.data.

    Parameters
    ----------
    name : str
        Name of data set. If a dataset that is listed in ``dataset_map`` is given,
        this function will look in the extract_path first, and if it is not present,
        attempt to download the data from Zenodo, saving it to the extract_path
        (or the ``aeon`` data directory if no extract_path is given).
    split : None or str{"train", "test"}, default=None
        Whether to load the train or test partition of the problem. By default it
        loads both into a single dataset, otherwise it looks only for files of the
        format <name>_TRAIN.ts or <name>_TEST.ts.
    extract_path : str, default=None
        the path to look for the data. If no path is provided, the function
        looks in ``aeon_neuro/datasets/data/`` and then the ``aeon`` data
        directory (``AEON_DATA`` or ``~/aeon_data/``). If a path is given, it can
        be absolute, e.g. C:/Temp/ or relative, e.g. Temp/ or ./Temp/.
    return_metadata : boolean, default = False
        If True, returns a tuple (X, y, metadata)

    Returns
    -------
    X: np.ndarray or list of np.ndarray
    y: np.ndarray
        The class labels for each case in X
    metadata: dict, optional
        returns the following metadata
        'problemname',timestamps, missing,univariate,equallength, class_values
        targetlabel should be false, and classlabel true

    Raises
    ------
    URLError or HTTPError
        If the website is not accessible.
    ValueError
        If a dataset name that does not exist on the repo is given, if the
        dataset is not yet available on Zenodo, or if a webpage is requested that
        does not exist.

    Examples
    --------
    >>> from aeon_neuro.datasets import load_eeg_classification
    >>> X, y = load_eeg_classification("SelfRegulationSCP1")  # doctest: +SKIP
    """
    if extract_path is not None:
        local_module = extract_path
        local_dirname = None
    else:
        local_module = MODULE
        local_dirname = "data"
    if local_dirname is None:
        path = local_module
    else:
        path = os.path.join(local_module, local_dirname)
    if not os.path.exists(path):
        os.makedirs(path)
    if name not in get_downloaded_tsc_tsr_datasets(path):
        if name not in dataset_map:
            raise ValueError(
                f"File name {name} is not in the list of valid files to download, "
                f"see aeon_neuro.datasets.classification_datasets for the current "
                f"list of maintained datasets."
            )
        if dataset_map[name] == 49:
            raise ValueError(
                f"Dataset {name} is not yet available to download from Zenodo. "
                f"Place {name}_TRAIN.ts and {name}_TEST.ts in <extract_path>/{name}/ "
                f"and pass extract_path to load it locally."
            )
        X, y, meta = _load_collection(
            name=name,
            problem_dict=dataset_map,
            split=split,
            extract_path=extract_path,
            return_metadata=True,
        )
    else:
        X, y, meta = _load_saved_dataset(
            name=name,
            dir_name=name,
            split=split,
            local_module=local_module,
            local_dirname=local_dirname,
            return_meta=True,
        )
    # Check this is a classification problem
    if "classlabel" not in meta or not meta["classlabel"]:
        raise ValueError(
            f"You have tried to load a regression problem called {name} with "
            f"load_classifier. This will cause unintended consequences for any "
            f"classifier you build. If you want to load a regression problem, "
            f"use load_regression in ``aeon`` "
        )
    if return_metadata:
        return X, y, meta
    return X, y
