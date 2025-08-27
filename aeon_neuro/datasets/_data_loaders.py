"""Function to load EEG Datasets from Zenodo."""

import os
from urllib.request import urlretrieve

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
    specified local path `extract_path`` or trying to download from
    https://zenodo.org// if the data is not in the local path. To download from
    zenodo, the dataset must be in the list ``dataset_map`` in data._data_loaders.py.
    This function assumes the data is stored in format
    ``<extract_path>/<name>/<name>_TRAIN.ts`` and
    ``<extract_path>/<name>/<name>_TEST.ts.`` If you want to load a file directly
    from a full path that is in ``aeon`` ts format, use the function
    `load_from_ts_file`` in ``aeon`` directly. If
    you do not specify ``extract_path``, it will set the path to
    ``aeon_neuro/datasets/local_data``.

    Data is assumed to be in the standard ``aeon`` .ts format: each row is a (possibly
    multivariate) time series. Each channel is separated by a colon, each value in
    a series is comma separated. For examples see aeon_neuro.datasets.data.

    Parameters
    ----------
    name : str
        Name of data set. If a dataset that is listed in tsc_datasets is given,
        this function will look in the extract_path first, and if it is not present,
        attempt to download the data from www.timeseriesclassification.com, saving it to
        the extract_path.
    split : None or str{"train", "test"}, default=None
        Whether to load the train or test partition of the problem. By default it
        loads both into a single dataset, otherwise it looks only for files of the
        format <name>_TRAIN.ts or <name>_TEST.ts.
    extract_path : str, default=None
        the path to look for the data. If no path is provided, the function
        looks in `aeon/datasets/local_data/`. If a path is given, it can be absolute,
        e.g. C:/Temp/ or relative, e.g. Temp/ or ./Temp/.
    return_metadata : boolean, default = True
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
        If a dataset name that does not exist on the repo is given or if a
        webpage is requested that does not exist.

    Examples
    --------
    >>> from aeon.datasets import load_classification
    >>> X, y = load_classification(name="ArrowHead")  # doctest: +SKIP
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
        if extract_path is None:
            local_dirname = "local_data"
            path = os.path.join(local_module, local_dirname)
        else:
            path = extract_path
        if not os.path.exists(path):
            os.makedirs(path)
        error_str = (
            f"File name {name} is not in the list of valid files to download,"
            f"see aeon_neuro.datasets.classification for the current list of "
            f"maintained datasets."
        )

        if name not in get_downloaded_tsc_tsr_datasets(path):
            # Check if in the zenodo list
            if name in dataset_map.keys():
                id = dataset_map[name]
                if id == 49:
                    raise ValueError(error_str)
                url_train = f"https://zenodo.org/record/{id}/files/{name}_TRAIN.ts"
                url_test = f"https://zenodo.org/record/{id}/files/{name}_TEST.ts"
                full_path = os.path.join(path, name)
                if not os.path.exists(full_path):
                    os.makedirs(full_path)
                train_save = f"{full_path}/{name}_TRAIN.ts"
                test_save = f"{full_path}/{name}_TEST.ts"
                try:
                    urlretrieve(url_train, train_save)
                    urlretrieve(url_test, test_save)
                except Exception:
                    raise ValueError(error_str)
            else:
                raise ValueError(error_str)
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
