"""Run channel selection algorithms on the local EEG datasets."""

import sys
import warnings
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from os import cpu_count
from pathlib import Path
from time import perf_counter

from aeon.classification.convolution_based import MiniRocketClassifier
from aeon.datasets import load_from_ts_file, save_to_ts_file
from aeon.transformations.collection.channel_selection import (
    ChannelScorer,
    ElbowClassPairwise,
    ElbowClassSum,
    RandomChannelSelector,
    TSelect,
)

from aeon_neuro.transformations.collection.channel_creation import (
    CommonSpacialPatterns,
)
from aeon_neuro.transformations.collection.channel_selection import (
    DetachRocketChannelSelector,
    Riemannian,
)

try:
    from experiments.channel_selection._datasets import (
        eeg_channel_selection_datasets,
    )
except ModuleNotFoundError:
    from _datasets import eeg_channel_selection_datasets

# Algorithms are run in this order. Remove entries here to disable them.
channel_selectors = [
    # "ECS",
    # "ECP",
    # "Random",
    # "Riemannian",
    # "ChannelScorer",
    "DetachRocket",
    # "TSelect",
    # "CSP",
]

SEED = 0
CHANNEL_PROPORTION = 0.25
SELECTOR_FACTORIES = {
    "ECS": ElbowClassSum,
    "ECP": ElbowClassPairwise,
    "TSelect": TSelect,
    "Random": lambda: RandomChannelSelector(p=CHANNEL_PROPORTION, random_state=SEED),
    "Riemannian": lambda: Riemannian(
        proportion=CHANNEL_PROPORTION,
        regularization=1e-6,
    ),
    "ChannelScorer": lambda: ChannelScorer(
        estimator=MiniRocketClassifier(
            n_kernels=2000,
            # max_dilations_per_kernel=32,
            random_state=SEED,
        ),
        scoring_function=None,
        score_sign=None,
        proportion=CHANNEL_PROPORTION,
    ),
    "DetachRocket": lambda: DetachRocketChannelSelector(
        proportion=CHANNEL_PROPORTION,
        n_kernels=2000,
        n_jobs=1,
        random_state=SEED,
    ),
}

DEFAULT_DATA_ROOT = Path(r"D:\Data\EEG")
DEFAULT_OUTPUT_ROOT = Path(r"D:\Data\ChannelSelection")
DEFAULT_WORKERS = cpu_count() or 1
SUMMARY_FILE_NAME = "selected_channels.txt"


def _input_paths(dataset_name, data_root):
    source_dir = Path(data_root) / dataset_name
    return (
        source_dir / f"{dataset_name}_TRAIN.ts",
        source_dir / f"{dataset_name}_TEST.ts",
    )


def _output_paths(dataset_name, output_root, selector_name):
    output_dir = Path(output_root) / selector_name / dataset_name
    return (
        output_dir / f"{dataset_name}_TRAIN.ts",
        output_dir / f"{dataset_name}_TEST.ts",
    )


def _load_summary(summary_path):
    results = {}
    if summary_path.is_file():
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            dataset_name, separator, _ = line.partition(":")
            if separator:
                results[dataset_name] = line
    return results


def _save_summary(summary_path, results):
    summary_path.write_text(
        "\n".join(results.values()) + "\n",
        encoding="utf-8",
    )


def _make_transformer(selector_name, n_channels):
    """Construct a selector or channel creator for one dataset."""
    if selector_name == "CSP":
        return CommonSpacialPatterns(
            n_components=ceil(CHANNEL_PROPORTION * n_channels),
            log=None,
            transform_into="csp_space",
            random_state=SEED,
        )
    return SELECTOR_FACTORIES[selector_name]()


def _pending_datasets(datasets, selector_name, output_root, results):
    """Return datasets without complete output files."""
    pending = []
    for dataset_name in datasets:
        train_output, test_output = _output_paths(
            dataset_name, output_root, selector_name
        )
        if train_output.is_file() and test_output.is_file():
            detail = results.get(dataset_name, "channel details not recorded")
            print(  # noqa: T201
                f"{selector_name} - {dataset_name}: already present, "
                f"skipping ({detail})",
                flush=True,
            )
        else:
            pending.append(dataset_name)
    return pending


def run_channel_selector(
    dataset_name,
    selector_name,
    data_root=DEFAULT_DATA_ROOT,
    output_root=DEFAULT_OUTPUT_ROOT,
):
    """Fit a selector on TRAIN, transform both splits, and save the results."""
    train_path, test_path = _input_paths(dataset_name, data_root)
    missing_files = [path for path in (train_path, test_path) if not path.is_file()]
    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing input file(s): {missing}")

    X_train, y_train = load_from_ts_file(train_path)
    X_test, y_test = load_from_ts_file(test_path)
    selector = _make_transformer(selector_name, X_train.shape[1])

    total_start = perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_start = perf_counter()
        selector.fit(X_train, y_train)
        fit_seconds = perf_counter() - fit_start

        train_start = perf_counter()
        X_train_transformed = selector.transform(X_train)
        train_transform_seconds = perf_counter() - train_start

        test_start = perf_counter()
        X_test_transformed = selector.transform(X_test)
        test_transform_seconds = perf_counter() - test_start

    total_seconds = perf_counter() - total_start
    if hasattr(selector, "channels_selected_"):
        selected = [int(channel) for channel in selector.channels_selected_]
        output_description = (
            f"{len(selected)} of {X_train.shape[1]} channels: {selected}"
        )
    else:
        n_components = X_train_transformed.shape[1]
        output_description = (
            f"{n_components} components from {X_train.shape[1]} channels"
        )

    output_dir = Path(output_root) / selector_name / dataset_name
    save_to_ts_file(
        X_train_transformed,
        y_train,
        label_type="classification",
        path=output_dir,
        problem_name=dataset_name,
        file_suffix="_TRAIN",
    )
    save_to_ts_file(
        X_test_transformed,
        y_test,
        label_type="classification",
        path=output_dir,
        problem_name=dataset_name,
        file_suffix="_TEST",
    )

    result = (
        f"{dataset_name}: {output_description}; "
        f"fit={fit_seconds:.6f}s; train_transform={train_transform_seconds:.6f}s; "
        f"test_transform={test_transform_seconds:.6f}s; total={total_seconds:.6f}s"
    )
    print(f"{selector_name} - {result}", flush=True)  # noqa: T201
    return result


def main():
    """Run the configured channel selectors for the requested EEG datasets."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        default=eeg_channel_selection_datasets,
        help="Dataset names to transform (default: all configured datasets).",
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        choices=[*SELECTOR_FACTORIES, "CSP"],
        default=channel_selectors,
        help="Selectors to run (default: the channel_selectors list).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=rf"EEG dataset root (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=rf"Transformed dataset root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Dataset worker threads (default: {DEFAULT_WORKERS}).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    for selector_name in args.selectors:
        summary_path = args.output_root / selector_name / SUMMARY_FILE_NAME
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        results = _load_summary(summary_path)

        print(  # noqa: T201
            f"\nStarting {selector_name} with {args.workers} worker threads",
            flush=True,
        )
        pending = _pending_datasets(
            args.datasets, selector_name, args.output_root, results
        )
        for dataset_name in pending:
            print(  # noqa: T201
                f"{selector_name} - running {dataset_name}...", flush=True
            )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_channel_selector,
                    dataset_name,
                    selector_name,
                    args.data_root,
                    args.output_root,
                ): dataset_name
                for dataset_name in pending
            }
            for future in as_completed(futures):
                dataset_name = futures[future]
                try:
                    result = future.result()
                except FileNotFoundError as error:
                    print(  # noqa: T201
                        f"{selector_name} - {dataset_name}: missing, "
                        f"skipping: {error}",
                        flush=True,
                    )
                    continue
                except Exception as error:
                    print(  # noqa: T201
                        f"{selector_name} - {dataset_name}: failed: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    for pending_future in futures:
                        pending_future.cancel()
                    raise

                results[dataset_name] = result
                _save_summary(summary_path, results)

        print(f"{selector_name} summary: {summary_path}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
