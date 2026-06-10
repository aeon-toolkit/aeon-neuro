"""Collate channel-selection timing, channel count, and overlap summaries."""

import ast
import csv
import re
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

DEFAULT_DATA_ROOT = Path(r"D:\Data\ChannelSelection")
SUMMARY_FILE_NAME = "selected_channels.txt"
LINE_PATTERN = re.compile(
    r"^(?P<dataset>[^:]+): "
    r"(?P<n_selected>\d+) of (?P<n_channels>\d+) channels: "
    r"(?P<channels>\[[^\]]*\]); "
    r"fit=(?P<fit>[\d.]+)s; "
    r"train_transform=(?P<train_transform>[\d.]+)s; "
    r"test_transform=(?P<test_transform>[\d.]+)s; "
    r"total=(?P<total>[\d.]+)s$"
)
COMPONENT_LINE_PATTERN = re.compile(
    r"^(?P<dataset>[^:]+): "
    r"(?P<n_selected>\d+) components from (?P<n_channels>\d+) channels; "
    r"fit=(?P<fit>[\d.]+)s; "
    r"train_transform=(?P<train_transform>[\d.]+)s; "
    r"test_transform=(?P<test_transform>[\d.]+)s; "
    r"total=(?P<total>[\d.]+)s$"
)


def parse_summary_file(path, selector):
    """Parse one selected_channels.txt file."""
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        match = LINE_PATTERN.fullmatch(line.strip())
        output_type = "selected_channels"
        if match is None:
            match = COMPONENT_LINE_PATTERN.fullmatch(line.strip())
            output_type = "created_components"
        if match is None:
            raise ValueError(f"Cannot parse {path}:{line_number}: {line}")

        values = match.groupdict()
        n_selected = int(values["n_selected"])
        channels = (
            [int(channel) for channel in ast.literal_eval(values["channels"])]
            if output_type == "selected_channels"
            else []
        )
        if output_type == "selected_channels" and len(channels) != n_selected:
            raise ValueError(
                f"{path}:{line_number} reports {n_selected} selected channels "
                f"but lists {len(channels)}"
            )

        train_transform = float(values["train_transform"])
        test_transform = float(values["test_transform"])
        records.append(
            {
                "selector": selector,
                "dataset": values["dataset"],
                "output_type": output_type,
                "n_selected": n_selected,
                "n_channels": int(values["n_channels"]),
                "selected_fraction": (
                    n_selected / int(values["n_channels"])
                    if output_type == "selected_channels"
                    else ""
                ),
                "channels": channels,
                "fit_seconds": float(values["fit"]),
                "train_transform_seconds": train_transform,
                "test_transform_seconds": test_transform,
                "transform_seconds": train_transform + test_transform,
                "total_seconds": float(values["total"]),
            }
        )
    return records


def load_records(data_root):
    """Load every selector summary below the data root."""
    records = []
    for summary_path in sorted(data_root.glob(f"*/{SUMMARY_FILE_NAME}")):
        records.extend(parse_summary_file(summary_path, summary_path.parent.name))
    return records


def write_csv(path, fieldnames, rows):
    """Write dictionaries to a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_selectors(records):
    """Aggregate channel counts and timings by selector."""
    grouped = defaultdict(list)
    for record in records:
        grouped[record["selector"]].append(record)

    rows = []
    for selector in sorted(grouped, key=str.casefold):
        selector_records = grouped[selector]
        count = len(selector_records)
        channel_records = [
            record
            for record in selector_records
            if record["output_type"] == "selected_channels"
        ]
        channel_count = len(channel_records)
        rows.append(
            {
                "selector": selector,
                "datasets": count,
                "total_channels_selected": (
                    sum(record["n_selected"] for record in channel_records)
                    if channel_records
                    else ""
                ),
                "mean_channels_selected": (
                    sum(record["n_selected"] for record in channel_records)
                    / channel_count
                    if channel_records
                    else ""
                ),
                "mean_selected_fraction": (
                    sum(record["selected_fraction"] for record in channel_records)
                    / channel_count
                    if channel_records
                    else ""
                ),
                "total_fit_seconds": sum(
                    record["fit_seconds"] for record in selector_records
                ),
                "total_transform_seconds": sum(
                    record["transform_seconds"] for record in selector_records
                ),
                "total_fit_and_transform_seconds": sum(
                    record["total_seconds"] for record in selector_records
                ),
                "mean_fit_and_transform_seconds": sum(
                    record["total_seconds"] for record in selector_records
                )
                / count,
            }
        )
    return rows


def calculate_overlaps(records):
    """Calculate pairwise selected-channel intersections for each dataset."""
    by_dataset = defaultdict(dict)
    for record in records:
        if record["output_type"] == "selected_channels":
            by_dataset[record["dataset"]][record["selector"]] = record

    rows = []
    for dataset in sorted(by_dataset, key=str.casefold):
        dataset_records = by_dataset[dataset]
        selectors = sorted(dataset_records, key=str.casefold)
        for index, selector_a in enumerate(selectors):
            channels_a = set(dataset_records[selector_a]["channels"])
            for selector_b in selectors[index + 1 :]:
                channels_b = set(dataset_records[selector_b]["channels"])
                intersection = channels_a & channels_b
                union = channels_a | channels_b
                rows.append(
                    {
                        "dataset": dataset,
                        "selector_a": selector_a,
                        "selector_b": selector_b,
                        "channels_a": len(channels_a),
                        "channels_b": len(channels_b),
                        "channels_in_common": len(intersection),
                        "common_channels": repr(sorted(intersection)),
                        "jaccard": len(intersection) / len(union) if union else 1.0,
                    }
                )
    return rows


def overlap_matrices(records, overlap_rows):
    """Build pairwise count and proportional overlap cross-tabs."""
    selectors = sorted(
        {
            record["selector"]
            for record in records
            if record["output_type"] == "selected_channels"
        },
        key=str.casefold,
    )
    selected_totals = defaultdict(int)
    for record in records:
        if record["output_type"] == "selected_channels":
            selected_totals[record["selector"]] += record["n_selected"]

    overlap_values = defaultdict(list)
    jaccard_values = defaultdict(list)
    for row in overlap_rows:
        key = frozenset((row["selector_a"], row["selector_b"]))
        overlap_values[key].append(row["channels_in_common"])
        jaccard_values[key].append(row["jaccard"])

    total_rows = []
    mean_rows = []
    proportion_rows = []
    dataset_count_rows = []
    for selector_a in selectors:
        total_row = {"selector": selector_a}
        mean_row = {"selector": selector_a}
        proportion_row = {"selector": selector_a}
        count_row = {"selector": selector_a}
        for selector_b in selectors:
            if selector_a == selector_b:
                total_row[selector_b] = selected_totals[selector_a]
                mean_row[selector_b] = ""
                proportion_row[selector_b] = 1.0
                count_row[selector_b] = ""
                continue

            key = frozenset((selector_a, selector_b))
            values = overlap_values[key]
            proportions = jaccard_values[key]
            total_row[selector_b] = sum(values)
            mean_row[selector_b] = sum(values) / len(values) if values else ""
            proportion_row[selector_b] = (
                sum(proportions) / len(proportions) if proportions else ""
            )
            count_row[selector_b] = len(values)
        total_rows.append(total_row)
        mean_rows.append(mean_row)
        proportion_rows.append(proportion_row)
        dataset_count_rows.append(count_row)

    return (
        selectors,
        total_rows,
        mean_rows,
        proportion_rows,
        dataset_count_rows,
    )


def main():
    """Collate summaries and write comparison CSV files."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=rf"Channel-selection data root (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <data-root>/Summary).",
    )
    args = parser.parse_args()

    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {args.data_root}")

    output_dir = args.output_dir or args.data_root / "Summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.data_root)
    if not records:
        raise FileNotFoundError(
            f"No */{SUMMARY_FILE_NAME} files found under {args.data_root}"
        )

    detail_rows = []
    for record in records:
        row = dict(record)
        row["channels"] = repr(record["channels"])
        detail_rows.append(row)
    write_csv(
        output_dir / "dataset_selector_summary.csv",
        list(detail_rows[0]),
        detail_rows,
    )

    aggregate_rows = aggregate_selectors(records)
    write_csv(
        output_dir / "selector_summary.csv",
        list(aggregate_rows[0]),
        aggregate_rows,
    )

    overlap_rows = calculate_overlaps(records)
    write_csv(
        output_dir / "channel_overlap_by_dataset.csv",
        list(overlap_rows[0]),
        overlap_rows,
    )

    (
        selectors,
        total_rows,
        mean_rows,
        proportion_rows,
        dataset_count_rows,
    ) = overlap_matrices(records, overlap_rows)
    matrix_fields = ["selector", *selectors]
    write_csv(output_dir / "channel_overlap_totals.csv", matrix_fields, total_rows)
    write_csv(output_dir / "channel_overlap_means.csv", matrix_fields, mean_rows)
    write_csv(
        output_dir / "channel_overlap_proportions.csv",
        matrix_fields,
        proportion_rows,
    )
    write_csv(
        output_dir / "channel_overlap_dataset_counts.csv",
        matrix_fields,
        dataset_count_rows,
    )

    print(f"Loaded {len(records)} dataset-selector records.")  # noqa: T201
    print(f"Selectors: {', '.join(selectors)}")  # noqa: T201
    print(f"Output: {output_dir}")  # noqa: T201
    print("\nSelector totals:")  # noqa: T201
    for row in aggregate_rows:
        channel_text = (
            f"{row['total_channels_selected']} channels, "
            if row["total_channels_selected"] != ""
            else ""
        )
        print(  # noqa: T201
            f"  {row['selector']}: {row['datasets']} datasets, "
            f"{channel_text}"
            f"fit={row['total_fit_seconds']:.3f}s, "
            f"transform={row['total_transform_seconds']:.3f}s, "
            f"fit+transform={row['total_fit_and_transform_seconds']:.3f}s"
        )


if __name__ == "__main__":
    main()
