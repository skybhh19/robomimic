"""
Create filtered training splits for square HDF5 datasets.

The script writes new HDF5 mask keys. It filters only the input training
split and copies the input validation split unchanged, so validation metrics
remain comparable across filtered training sets.

Examples:
    # Drop the bottom 25% of square MH train demos by quality.
    python robomimic/scripts/filter_square_dataset_by_metric.py \
        --dataset robomimic/datasets/square/mh/demo.hdf5 \
        --metric quality \
        --discard_fraction 0.25

    # Drop the bottom 25% by observability, treating partial=1 and full=2.
    python robomimic/scripts/filter_square_dataset_by_metric.py \
        --dataset robomimic/datasets/square/ph/demo.hdf5 \
        --metric observability \
        --discard_fraction 0.25
"""

import argparse
import csv
import math
import os
import re
from collections import Counter

import h5py
import numpy as np


QUALITY_MASK_SCORES = {
    "worse": 1.0,
    "okay": 2.0,
    "better": 3.0,
}

OBSERVABILITY_LABEL_SCORES = {
    "partial": 1.0,
    "full": 2.0,
}


def create_hdf5_filter_key(hdf5_path, demo_keys, key_name):
    """
    Write demo keys under mask/<key_name> and return their episode lengths.
    This mirrors robomimic.utils.file_utils.create_hdf5_filter_key while keeping
    this script runnable directly from the repo checkout.
    """
    with h5py.File(hdf5_path, "a") as f:
        demos = sorted(list(f["data"].keys()))

        ep_lengths = []
        demo_key_set = set(demo_keys)
        for ep in demos:
            if ep in demo_key_set:
                ep_lengths.append(f["data/{}".format(ep)].attrs["num_samples"])

        key_path = "mask/{}".format(key_name)
        if key_path in f:
            del f[key_path]
        f[key_path] = np.array(demo_keys, dtype="S")

    return ep_lengths


def demo_index(demo_key):
    match = re.search(r"(\d+)$", demo_key)
    if match is None:
        return demo_key
    return int(match.group(1))


def sorted_demo_keys(demo_keys):
    return sorted(demo_keys, key=lambda k: (not isinstance(demo_index(k), int), demo_index(k)))


def read_filter_key(hdf5_file, filter_key):
    mask_path = "mask/{}".format(filter_key)
    if mask_path not in hdf5_file:
        raise KeyError("Missing HDF5 filter key: {}".format(mask_path))
    return [elem.decode("utf-8") for elem in np.asarray(hdf5_file[mask_path][:])]


def read_all_demo_keys(hdf5_file):
    return sorted_demo_keys(list(hdf5_file["data"].keys()))


def infer_annotation_dataset(dataset_path):
    parts = os.path.normpath(dataset_path).split(os.sep)
    if "square" not in parts:
        return None

    square_index = parts.index("square")
    if square_index + 1 >= len(parts):
        return None

    split = parts[square_index + 1]
    if split == "mh":
        return "square_mh"
    if split == "ph":
        return "square_ph"
    if split == "random_post":
        return "expert200"
    return None


def default_output_key(train_filter_key, metric, discard_fraction):
    discard_pct = int(round(100 * discard_fraction))
    return "{}_{}_drop{:02d}pct".format(train_filter_key, metric, discard_pct)


def load_quality_scores(hdf5_file):
    if "mask" not in hdf5_file:
        raise KeyError("Dataset has no mask group; cannot read quality masks.")

    scores = {}
    for mask_name, score in QUALITY_MASK_SCORES.items():
        mask_path = "mask/{}".format(mask_name)
        if mask_path not in hdf5_file:
            raise KeyError(
                "Missing quality mask {}. Expected masks: {}".format(
                    mask_path, ", ".join(sorted(QUALITY_MASK_SCORES))
                )
            )
        for demo_key in read_filter_key(hdf5_file, mask_name):
            if demo_key in scores:
                raise ValueError(
                    "Demo {} appears in multiple quality masks.".format(demo_key)
                )
            scores[demo_key] = score
    return scores


def load_observability_scores(csv_path, annotation_dataset, ep_idx_offset):
    if annotation_dataset is None:
        raise ValueError(
            "Could not infer annotation dataset from path. Pass --annotation_dataset."
        )

    scores = {}
    labels = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"dataset", "ep_idx", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "{} is missing required columns: {}".format(
                    csv_path, ", ".join(sorted(missing))
                )
            )
        for row in reader:
            if row["dataset"] != annotation_dataset:
                continue
            ep_idx = int(row["ep_idx"])
            demo_key = "demo_{}".format(ep_idx - ep_idx_offset)
            label = row["label"].strip().lower()
            if label in OBSERVABILITY_LABEL_SCORES:
                scores[demo_key] = OBSERVABILITY_LABEL_SCORES[label]
            labels[demo_key] = label
    if not labels:
        raise ValueError(
            "No rows for annotation dataset {!r} in {}".format(
                annotation_dataset, csv_path
            )
        )
    return scores, labels


def apply_missing_score_policy(scores, train_keys, missing_score_policy):
    missing = [key for key in train_keys if key not in scores]
    if not missing:
        return scores

    if missing_score_policy == "error":
        raise ValueError(
            "Missing metric scores for {} train demos, including: {}".format(
                len(missing), ", ".join(missing[:10])
            )
        )

    filled_scores = dict(scores)
    fill_value = math.inf if missing_score_policy == "keep" else -math.inf
    for key in missing:
        filled_scores[key] = fill_value
    return filled_scores


def choose_demos_to_drop(train_keys, scores, discard_fraction, seed, tie_break):
    if not (0.0 <= discard_fraction < 1.0):
        raise ValueError("--discard_fraction must be in [0, 1).")

    num_drop = int(math.floor(discard_fraction * len(train_keys)))
    if num_drop == 0:
        return []

    if tie_break == "random":
        rng = np.random.default_rng(seed)
        random_order = {key: order for order, key in enumerate(rng.permutation(train_keys))}
        ranked = sorted(train_keys, key=lambda key: (scores[key], random_order[key]))
    else:
        ranked = sorted(train_keys, key=lambda key: (scores[key], demo_index(key)))
    return ranked[:num_drop]


def summarize_scores(keys, scores):
    finite_scores = []
    special_scores = Counter()
    for key in keys:
        score = scores[key]
        if math.isinf(score):
            special_scores[str(score)] += 1
        else:
            finite_scores.append(score)
    counts = Counter(finite_scores)
    return ", ".join(
        ["{}:{}".format(score, counts[score]) for score in sorted(counts)]
        + ["{}:{}".format(score, count) for score, count in sorted(special_scores.items())]
    )


def filter_square_dataset_by_metric(args):
    dataset_path = os.path.expanduser(args.dataset)
    annotation_csv = os.path.expanduser(args.annotation_csv)
    output_train_key = args.output_train_filter_key
    if output_train_key is None:
        input_key_name = "all" if args.use_all_demos else args.train_filter_key
        output_train_key = default_output_key(
            input_key_name, args.metric, args.discard_fraction
        )

    output_valid_key = args.output_valid_filter_key
    if output_valid_key is None:
        if args.output_train_filter_key is None:
            output_valid_key = default_output_key(
                args.valid_filter_key, args.metric, args.discard_fraction
            )
        elif args.train_filter_key in output_train_key:
            output_valid_key = output_train_key.replace(
                args.train_filter_key, args.valid_filter_key, 1
            )
        else:
            output_valid_key = "{}_{}".format(output_train_key, args.valid_filter_key)

    if not args.skip_valid_copy and output_train_key == output_valid_key:
        raise ValueError(
            "Output train and valid filter keys are both {!r}.".format(output_train_key)
        )

    annotation_dataset = args.annotation_dataset
    if annotation_dataset is None and args.metric == "observability":
        annotation_dataset = infer_annotation_dataset(dataset_path)

    ep_idx_offset = args.ep_idx_offset
    if ep_idx_offset is None:
        ep_idx_offset = 1 if annotation_dataset == "expert200" else 0

    with h5py.File(dataset_path, "r") as f:
        if args.use_all_demos:
            train_keys = read_all_demo_keys(f)
            valid_keys = []
        else:
            train_keys = read_filter_key(f, args.train_filter_key)
            valid_keys = read_filter_key(f, args.valid_filter_key)

        if args.metric == "quality":
            scores = load_quality_scores(f)
            labels = None
        else:
            scores, labels = load_observability_scores(
                annotation_csv, annotation_dataset, ep_idx_offset
            )

    if args.metric == "observability" and args.unsure_score is not None:
        for demo_key, label in labels.items():
            if label == "unsure":
                scores[demo_key] = float(args.unsure_score)

    scores = apply_missing_score_policy(
        scores=scores,
        train_keys=train_keys,
        missing_score_policy=args.missing_score_policy,
    )

    drop_keys = set(
        choose_demos_to_drop(
            train_keys=train_keys,
            scores=scores,
            discard_fraction=args.discard_fraction,
            seed=args.seed,
            tie_break=args.tie_break,
        )
    )
    keep_train_keys = [key for key in train_keys if key not in drop_keys]
    dropped_keys = [key for key in train_keys if key in drop_keys]

    print("Dataset: {}".format(dataset_path))
    print("Metric: {}".format(args.metric))
    if args.metric == "observability":
        print("Annotation dataset: {}".format(annotation_dataset))
        print("Annotation ep_idx offset: {}".format(ep_idx_offset))
    if args.use_all_demos:
        print("Input demo set: all data demos ({} demos)".format(len(train_keys)))
    else:
        print("Input train key: {} ({} demos)".format(args.train_filter_key, len(train_keys)))
        print("Input valid key: {} ({} demos)".format(args.valid_filter_key, len(valid_keys)))
    print(
        "Dropping {} / {} train demos ({:.1f}%).".format(
            len(dropped_keys),
            len(train_keys),
            100.0 * len(dropped_keys) / max(1, len(train_keys)),
        )
    )
    print("Train score counts before: {}".format(summarize_scores(train_keys, scores)))
    print("Dropped score counts: {}".format(summarize_scores(dropped_keys, scores)))
    print("Output train key: {} ({} demos)".format(output_train_key, len(keep_train_keys)))
    if not args.skip_valid_copy:
        print("Output valid key: {} ({} demos, unchanged)".format(output_valid_key, len(valid_keys)))

    if args.print_dropped:
        print("Dropped demos: {}".format(", ".join(sorted_demo_keys(dropped_keys))))

    if args.dry_run:
        print("Dry run: no HDF5 masks were written.")
        return

    train_lengths = create_hdf5_filter_key(
        hdf5_path=dataset_path,
        demo_keys=keep_train_keys,
        key_name=output_train_key,
    )
    print("Total output train samples: {}".format(int(np.sum(train_lengths))))
    if len(train_lengths) > 0:
        print("Average output train samples: {:.1f}".format(float(np.mean(train_lengths))))

    if not args.skip_valid_copy:
        valid_lengths = create_hdf5_filter_key(
            hdf5_path=dataset_path,
            demo_keys=valid_keys,
            key_name=output_valid_key,
        )
        print("Total output valid samples: {}".format(int(np.sum(valid_lengths))))
        if len(valid_lengths) > 0:
            print("Average output valid samples: {:.1f}".format(float(np.mean(valid_lengths))))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to square HDF5 dataset.")
    parser.add_argument(
        "--metric",
        required=True,
        choices=("quality", "observability"),
        help="Metric used to rank training trajectories from low to high.",
    )
    parser.add_argument(
        "--discard_fraction",
        type=float,
        default=0.25,
        help="Fraction of train trajectories to discard from the bottom of the ranking.",
    )
    parser.add_argument("--train_filter_key", default="train")
    parser.add_argument("--valid_filter_key", default="valid")
    parser.add_argument(
        "--use_all_demos",
        action="store_true",
        help="Rank and filter all data/demo_* trajectories instead of the train mask.",
    )
    parser.add_argument(
        "--output_train_filter_key",
        default=None,
        help="Name for the new filtered train mask.",
    )
    parser.add_argument(
        "--output_valid_filter_key",
        default=None,
        help="Name for the copied validation mask.",
    )
    parser.add_argument(
        "--skip_valid_copy",
        action="store_true",
        help="Only write the filtered train mask; do not copy the validation mask.",
    )
    parser.add_argument(
        "--annotation_csv",
        default="robomimic/datasets/square/observability_annotations.csv",
        help="CSV with observability labels.",
    )
    parser.add_argument(
        "--annotation_dataset",
        default=None,
        help="Dataset name in the observability CSV, e.g. square_mh or square_ph.",
    )
    parser.add_argument(
        "--ep_idx_offset",
        type=int,
        default=None,
        help="Map demo_i to CSV ep_idx=i+offset. Defaults to 1 for expert200, else 0.",
    )
    parser.add_argument(
        "--unsure_score",
        type=float,
        default=None,
        help="Score to assign observability label 'unsure'. If omitted, policy below applies.",
    )
    parser.add_argument(
        "--missing_score_policy",
        choices=("error", "keep", "drop"),
        default="keep",
        help="How to handle train demos without a metric score.",
    )
    parser.add_argument(
        "--tie_break",
        choices=("demo_index", "random"),
        default="demo_index",
        help="Tie-breaker for demos with equal scores.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for random tie-breaking.")
    parser.add_argument("--dry_run", action="store_true", help="Print summary without writing masks.")
    parser.add_argument("--print_dropped", action="store_true", help="Print dropped demo keys.")
    return parser


if __name__ == "__main__":
    filter_square_dataset_by_metric(get_parser().parse_args())
