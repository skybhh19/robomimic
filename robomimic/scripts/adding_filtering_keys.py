import argparse
import csv
import math

import h5py
import numpy as np

from robomimic.utils.file_utils import create_hdf5_filter_key


def get_demo_keys(hdf5_path, input_filter_key):
    with h5py.File(hdf5_path, "r") as f:
        if input_filter_key is None:
            return sorted(list(f["data"].keys()))
        print("using filter key: {}".format(input_filter_key))
        return sorted([elem.decode("utf-8") for elem in np.array(f["mask/{}".format(input_filter_key)])])


def default_random_key_name(pct):
    pct_int = round(100 * pct)
    assert np.isclose(100 * pct, pct_int), "--pct must map to an integer percent, got {}".format(pct)
    return "random{}pct".format(pct_int)


def default_label_random_key_name(pct):
    pct_int = round(100 * pct)
    assert np.isclose(100 * pct, pct_int), "--pct must map to an integer percent, got {}".format(pct)
    return "observability{}pct".format(pct_int)


def demo_sort_key(demo_key):
    prefix, index = demo_key.rsplit("_", 1)
    return prefix, index.zfill(12)


def read_label_rows(label_csv, label_column, label_demo_column):
    with open(label_csv, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None, "{} has no header".format(label_csv)
        assert label_column in reader.fieldnames, "{} missing from {}".format(
            label_column, reader.fieldnames
        )
        assert label_demo_column is None or label_demo_column in reader.fieldnames, (
            "{} missing from {}".format(label_demo_column, reader.fieldnames)
        )
        return list(reader)


def get_demo_labels(demos, label_csv, label_column, label_demo_column):
    rows = read_label_rows(label_csv, label_column, label_demo_column)
    if label_demo_column is None:
        ordered_demos = sorted(demos, key=demo_sort_key)
        assert len(rows) == len(ordered_demos), (
            "label csv has {} rows but dataset has {} demos".format(len(rows), len(ordered_demos))
        )
        return {demo: row[label_column] for demo, row in zip(ordered_demos, rows)}
    labels = {row[label_demo_column]: row[label_column] for row in rows}
    missing = [demo for demo in demos if demo not in labels]
    assert len(missing) == 0, "missing labels for demos: {}".format(missing)
    return labels


def label_prioritized_subset(demos, num_keep, rng, label_csv, label_column, label_demo_column, label_order):
    labels = get_demo_labels(demos, label_csv, label_column, label_demo_column)
    label_to_rank = {label: i for i, label in enumerate(label_order)}
    unknown = sorted(set(labels.values()) - set(label_to_rank.keys()))
    assert len(unknown) == 0, "labels {} are not in --label_order {}".format(unknown, label_order)
    ranked_demos = []
    for label in label_order:
        label_demos = [demo for demo in demos if labels[demo] == label]
        inds = rng.permutation(len(label_demos))
        ranked_demos.extend([label_demos[i] for i in inds])
    subset_keys = ranked_demos[:num_keep]
    label_counts = {label: sum(labels[demo] == label for demo in subset_keys) for label in label_order}
    print("Label counts kept: {}".format(label_counts))
    return subset_keys


def add_random_filter_key(
    hdf5_path,
    pct,
    seed,
    input_filter_key=None,
    output_filter_key=None,
    label_csv=None,
    label_column="label",
    label_demo_column=None,
    label_order=None,
):
    assert 0.0 < pct <= 1.0, "--pct must be in (0, 1], got {}".format(pct)
    demos = get_demo_keys(hdf5_path, input_filter_key)
    total_num_demos = len(demos)
    num_keep = math.floor(pct * total_num_demos)
    assert num_keep > 0, "--pct={} keeps 0 demos out of {}".format(pct, total_num_demos)

    rng = np.random.default_rng(seed)
    if label_csv is None:
        subset_inds = rng.choice(total_num_demos, size=num_keep, replace=False)
        subset_keys = [demos[i] for i in subset_inds]
    else:
        if label_order is None:
            label_order = ["full", "partial"]
        subset_keys = label_prioritized_subset(
            demos=demos,
            num_keep=num_keep,
            rng=rng,
            label_csv=label_csv,
            label_column=label_column,
            label_demo_column=label_demo_column,
            label_order=label_order,
        )

    if output_filter_key is None:
        if label_csv is None:
            name = default_random_key_name(pct)
        else:
            name = default_label_random_key_name(pct)
    else:
        name = output_filter_key
    if input_filter_key is not None:
        name = "{}_{}".format(input_filter_key, name)

    subset_lengths = create_hdf5_filter_key(hdf5_path=hdf5_path, demo_keys=subset_keys, key_name=name)
    print("Created filter key: {}".format(name))
    print("Number of demos kept: {} / {}".format(len(subset_keys), total_num_demos))
    print("Total number of subset samples: {}".format(np.sum(subset_lengths)))
    print("Average number of subset samples: {}".format(np.mean(subset_lengths)))
    print("Demo keys: {}".format(subset_keys))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="path to hdf5 dataset")
    parser.add_argument("--mode", type=str, choices=["random"], required=True)
    parser.add_argument("--pct", type=float, required=True, help="fraction of demos to keep, in (0, 1]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input_filter_key", type=str, default=None)
    parser.add_argument("--output_filter_key", type=str, default=None)
    parser.add_argument("--label_csv", type=str, default=None)
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--label_demo_column", type=str, default=None)
    parser.add_argument("--label_order", type=str, nargs="+", default=["full", "partial"])
    args = parser.parse_args()

    if args.mode == "random":
        add_random_filter_key(
            hdf5_path=args.dataset,
            pct=args.pct,
            seed=args.seed,
            input_filter_key=args.input_filter_key,
            output_filter_key=args.output_filter_key,
            label_csv=args.label_csv,
            label_column=args.label_column,
            label_demo_column=args.label_demo_column,
            label_order=args.label_order,
        )
