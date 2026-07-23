"""Run one channel-selection algorithm on one dataset.

This entry point is intended for cluster jobs. Each invocation is independent and
single-process, so concurrency should be managed by the cluster scheduler.
"""

import os
import warnings
from argparse import ArgumentParser
from math import ceil
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
    BPSO,
    UMAP,
    CaseTimeReducer,
    CLeVerCluster,
    CLeVerHybrid,
    CLeVerRank,
    DetachRocketChannelSelector,
    Riemannian,
)

# Keep native numerical libraries within a one-CPU cluster allocation.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_variable] = "1"

SEED = 0
CHANNEL_PROPORTION = 0.25

# Set these to the cluster paths, or override them with the CLI options.
DEFAULT_DATA_ROOT = Path(r"D:\Data\EEG")
DEFAULT_OUTPUT_ROOT = Path(r"D:\Data\ChannelSelection")

ALGORITHMS = (
    "ECS",
    "ECP",
    "Random",
    "Riemannian",
    "BPSO",
    "ChannelScorer",
    "DetachRocket",
    "TSelect",
    "CSP",
    "UMAP",
    "CaseTimeReducer",
    "CLeVerRank",
    "CLeVerCluster",
    "CLeVerHybrid",
)
SUMMARY_FILE_NAME = "selection_summary.txt"


def _make_transformer(algorithm, n_channels):
    """Construct a single-threaded transformer."""
    n_components = ceil(CHANNEL_PROPORTION * n_channels)
    if algorithm == "ECS":
        return ElbowClassSum()
    if algorithm == "ECP":
        return ElbowClassPairwise()
    if algorithm == "Random":
        return RandomChannelSelector(
            p=CHANNEL_PROPORTION,
            random_state=SEED,
        )
    if algorithm == "Riemannian":
        return Riemannian(
            proportion=CHANNEL_PROPORTION,
            regularization=1e-6,
            n_jobs=1,
        )
    if algorithm == "BPSO":
        return BPSO(
            proportion=CHANNEL_PROPORTION,
            estimator=MiniRocketClassifier(
                n_kernels=2000,
                n_jobs=1,
                random_state=SEED,
            ),
            random_state=SEED,
        )
    if algorithm == "ChannelScorer":
        return ChannelScorer(
            estimator=MiniRocketClassifier(
                n_kernels=2000,
                n_jobs=1,
                random_state=SEED,
            ),
            scoring_function=None,
            score_sign=None,
            proportion=CHANNEL_PROPORTION,
        )
    if algorithm == "DetachRocket":
        return DetachRocketChannelSelector(
            proportion=CHANNEL_PROPORTION,
            n_kernels=2000,
            n_jobs=1,
            random_state=SEED,
        )
    if algorithm == "TSelect":
        return TSelect(random_state=SEED)
    if algorithm == "CSP":
        return CommonSpacialPatterns(
            n_components=n_components,
            log=None,
            transform_into="csp_space",
            random_state=SEED,
        )
    if algorithm == "UMAP":
        return UMAP(
            n_components=n_components,
            random_state=SEED,
        )
    if algorithm == "CaseTimeReducer":
        return CaseTimeReducer(
            strategy="auto",
            random_state=SEED,
            n_jobs=1,
        )
    if algorithm == "CLeVerRank":
        return CLeVerRank(n_channels=n_components)
    if algorithm == "CLeVerCluster":
        return CLeVerCluster(
            n_channels=n_components,
            random_state=SEED,
        )
    if algorithm == "CLeVerHybrid":
        return CLeVerHybrid(
            n_channels=n_components,
            random_state=SEED,
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _input_paths(dataset, data_root):
    source_dir = data_root / dataset
    return (
        source_dir / f"{dataset}_TRAIN.ts",
        source_dir / f"{dataset}_TEST.ts",
    )


def _output_paths(dataset, output_dir):
    return (
        output_dir / f"{dataset}_TRAIN.ts",
        output_dir / f"{dataset}_TEST.ts",
    )


def run_job(algorithm, dataset, data_root, output_root, output_name=None):
    """Fit on TRAIN, transform TRAIN and TEST, and save one cluster job."""
    output_name = algorithm if output_name is None else output_name
    train_path, test_path = _input_paths(dataset, data_root)
    missing_files = [path for path in (train_path, test_path) if not path.is_file()]
    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing input file(s): {missing}")

    X_train, y_train = load_from_ts_file(train_path)
    X_test, y_test = load_from_ts_file(test_path)
    transformer = _make_transformer(algorithm, X_train.shape[1])

    total_start = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        fit_start = perf_counter()
        transformer.fit(X_train, y_train)
        fit_seconds = perf_counter() - fit_start

        train_start = perf_counter()
        if isinstance(transformer, CaseTimeReducer):
            X_train_transformed, y_train_transformed = transformer.resample_train(
                X_train, y_train
            )
        else:
            X_train_transformed = transformer.transform(X_train)
            y_train_transformed = y_train
        train_transform_seconds = perf_counter() - train_start

        test_start = perf_counter()
        X_test_transformed = transformer.transform(X_test)
        test_transform_seconds = perf_counter() - test_start

    total_seconds = perf_counter() - total_start
    if isinstance(transformer, CaseTimeReducer):
        output_description = (
            f"{X_train_transformed.shape[0]} of {X_train.shape[0]} train cases; "
            f"{X_train_transformed.shape[2]} of {X_train.shape[2]} time points; "
            f"candidate={transformer.selected_candidate_['candidate']}; "
            f"tuning_score={transformer.selection_score_:.6f}"
        )
    elif hasattr(transformer, "channels_selected_"):
        selected = [int(channel) for channel in transformer.channels_selected_]
        output_description = (
            f"{len(selected)} of {X_train.shape[1]} channels: {selected}"
        )
    else:
        output_description = (
            f"{X_train_transformed.shape[1]} components from "
            f"{X_train.shape[1]} channels"
        )

    output_dir = output_root / output_name / dataset
    save_to_ts_file(
        X_train_transformed,
        y_train_transformed,
        label_type="classification",
        path=output_dir,
        problem_name=dataset,
        file_suffix="_TRAIN",
    )
    save_to_ts_file(
        X_test_transformed,
        y_test,
        label_type="classification",
        path=output_dir,
        problem_name=dataset,
        file_suffix="_TEST",
    )

    result = (
        f"{dataset}: {output_description}; "
        f"fit={fit_seconds:.6f}s; "
        f"train_transform={train_transform_seconds:.6f}s; "
        f"test_transform={test_transform_seconds:.6f}s; "
        f"total={total_seconds:.6f}s"
    )
    (output_dir / SUMMARY_FILE_NAME).write_text(result + "\n", encoding="utf-8")
    print(f"{algorithm} - {result}", flush=True)  # noqa: T201
    return result


def main():
    """Parse one algorithm and dataset, then run the cluster job."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("algorithm", choices=ALGORITHMS, help="Algorithm to run.")
    parser.add_argument("dataset", help="Dataset directory and problem name.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Input dataset root (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--output-name",
        help="Output subdirectory name (default: algorithm name).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Run even when both transformed output files already exist.",
    )
    args = parser.parse_args()

    output_name = args.algorithm if args.output_name is None else args.output_name
    output_dir = args.output_root / output_name / args.dataset
    train_output, test_output = _output_paths(args.dataset, output_dir)
    if not args.overwrite and train_output.is_file() and test_output.is_file():
        print(  # noqa: T201
            f"{args.algorithm} - {args.dataset}: outputs already exist; skipping",
            flush=True,
        )
        return

    print(f"Starting {args.algorithm} on {args.dataset}", flush=True)  # noqa: T201
    run_job(
        args.algorithm,
        args.dataset,
        args.data_root,
        args.output_root,
        args.output_name,
    )


if __name__ == "__main__":
    main()
