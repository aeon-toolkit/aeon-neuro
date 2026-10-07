"""Summarise channel-selection classification results."""

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

try:
    from experiments.channel_selection._datasets import (
        eeg_channel_selection_datasets,
    )
except ModuleNotFoundError:
    from _datasets import eeg_channel_selection_datasets


DEFAULT_RESULTS_ROOT = Path(r"D:\Results\ChannelSelection")


def find_result_sets(results_root):
    """Return discovered classifier result directories grouped by classifier."""
    result_sets = defaultdict(list)

    for selector_dir in sorted(results_root.iterdir()):
        if not selector_dir.is_dir():
            continue

        for classifier_dir in sorted(selector_dir.iterdir()):
            predictions_dir = classifier_dir / "Predictions"
            if predictions_dir.is_dir():
                result_sets[classifier_dir.name].append(
                    (selector_dir.name, predictions_dir)
                )

    return result_sets


def summarise_predictions(predictions_dir, expected_datasets, resample):
    """Find present, missing, incomplete, and unexpected dataset results."""
    expected = set(expected_datasets)
    present = []
    incomplete = []

    for dataset in expected_datasets:
        dataset_dir = predictions_dir / dataset
        test_file = dataset_dir / f"testResample{resample}.csv"
        if test_file.is_file() and test_file.stat().st_size > 0:
            present.append(dataset)
        elif dataset_dir.is_dir():
            incomplete.append(dataset)

    missing = [
        dataset
        for dataset in expected_datasets
        if dataset not in present and dataset not in incomplete
    ]
    unexpected = sorted(
        path.name
        for path in predictions_dir.iterdir()
        if path.is_dir() and path.name not in expected
    )
    return present, missing, incomplete, unexpected


def format_names(names):
    """Format a dataset list for terminal output."""
    return ", ".join(names) if names else "none"


def format_python_list(names):
    """Format dataset names as a Python list of strings."""
    return "[" + ", ".join(repr(name) for name in names) + "]"


def main():
    """Print result completeness by classifier and channel selector."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=rf"Results root (default: {DEFAULT_RESULTS_ROOT}).",
    )
    parser.add_argument(
        "--resample",
        type=int,
        default=0,
        help="Resample number to check (default: 0).",
    )
    parser.add_argument(
        "--classifier",
        help="Only report this classifier.",
    )
    parser.add_argument(
        "--python-lists",
        action="store_true",
        help="Print missing datasets as Python lists keyed by selector.",
    )
    args = parser.parse_args()

    if not args.results_root.is_dir():
        raise FileNotFoundError(f"Results directory not found: {args.results_root}")

    result_sets = find_result_sets(args.results_root)
    if not result_sets:
        print(  # noqa: T201
            f"No classifier prediction directories found in {args.results_root}"
        )
        return

    expected_count = len(eeg_channel_selection_datasets)
    print(  # noqa: T201
        f"Expected datasets: {expected_count}; checking testResample"
        f"{args.resample}.csv under {args.results_root}"
    )

    classifiers = sorted(result_sets, key=str.casefold)
    if args.classifier:
        classifiers = [
            classifier
            for classifier in classifiers
            if classifier.casefold() == args.classifier.casefold()
        ]
        if not classifiers:
            raise ValueError(f"Classifier not found: {args.classifier}")

    for classifier in classifiers:
        print(f"\nClassifier: {classifier}")  # noqa: T201
        for selector, predictions_dir in sorted(
            result_sets[classifier], key=lambda item: item[0].casefold()
        ):
            present, missing, incomplete, unexpected = summarise_predictions(
                predictions_dir,
                eeg_channel_selection_datasets,
                args.resample,
            )
            if args.python_lists:
                missing_results = [
                    dataset
                    for dataset in eeg_channel_selection_datasets
                    if dataset in missing or dataset in incomplete
                ]
                print(  # noqa: T201
                    f"{selector} = {format_python_list(missing_results)}"
                )
                continue

            print(  # noqa: T201
                f"  {selector}: {len(present)}/{expected_count} present, "
                f"{len(missing)} missing, {len(incomplete)} incomplete"
            )
            print(f"    Missing: {format_names(missing)}")  # noqa: T201
            if incomplete:
                print(f"    Incomplete: {format_names(incomplete)}")  # noqa: T201
            if unexpected:
                print(f"    Unexpected: {format_names(unexpected)}")  # noqa: T201


if __name__ == "__main__":
    main()
