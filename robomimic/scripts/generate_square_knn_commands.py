"""
Generate extract_policy_latent_knn.py commands for Square sweep checkpoints.

The helper pairs each checkpoint with its corresponding dataset by reading the
training config.json next to the checkpoint. This avoids relying on directory
name conventions when the dataset path is already recorded in the run config.

Example:
    python robomimic/scripts/generate_square_knn_commands.py

This writes:
    robomimic/scripts/square_knn_commands.txt
"""

import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def natural_key(name):
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", str(name))]


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def repo_relative(path, repo_root):
    path = Path(path)
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_repo_path(path, repo_root):
    path = Path(os.path.expanduser(str(path)))
    if path.is_absolute():
        return path
    return repo_root / path


def load_dataset_path(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    data = config.get("train", {}).get("data")
    if not data:
        raise ValueError("No train.data found in {}".format(config_path))
    if len(data) != 1:
        raise ValueError(
            "Expected exactly one dataset in train.data for {}; got {}".format(
                config_path,
                len(data),
            )
        )
    dataset_path = data[0].get("path")
    if dataset_path is None:
        raise ValueError("No train.data[0].path found in {}".format(config_path))

    experiment_name = config.get("experiment", {}).get("name", config_path.parent.name)
    return dataset_path, experiment_name


def validation_checkpoint_sort_key(path):
    validation_match = re.search(r"_best_validation_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", path.name)
    if validation_match is None:
        return (float("inf"), -1, natural_key(path.name))
    validation_loss = float(validation_match.group(1))
    epoch_match = re.search(r"model_epoch_(\d+)", path.name)
    epoch = int(epoch_match.group(1)) if epoch_match is not None else -1
    return (validation_loss, epoch, natural_key(path.name))


def checkpoint_sort_key(path):
    name = path.name
    epoch_match = re.search(r"model_epoch_(\d+)", name)
    epoch = int(epoch_match.group(1)) if epoch_match is not None else -1
    mtime = path.stat().st_mtime
    return (epoch, mtime, natural_key(name))


def success_checkpoint_sort_key(path):
    success_match = re.search(r"_success_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", path.name)
    success = float(success_match.group(1)) if success_match is not None else -float("inf")
    epoch_match = re.search(r"model_epoch_(\d+)", path.name)
    epoch = int(epoch_match.group(1)) if epoch_match is not None else -1
    return (success, epoch, natural_key(path.name))


def select_checkpoints(run_dir, mode, checkpoint_pattern=None):
    if mode == "last":
        checkpoint = run_dir / "last.pth"
        return [checkpoint] if checkpoint.exists() else []

    model_paths = sorted((run_dir / "models").glob("*.pth"), key=checkpoint_sort_key)
    if mode == "all":
        return model_paths
    if mode in ("best-validation", "best"):
        best_paths = [p for p in model_paths if "best_validation" in p.name]
        if not best_paths:
            return []
        return [min(best_paths, key=validation_checkpoint_sort_key)]
    if mode == "best-success":
        success_paths = [p for p in model_paths if "_success_" in p.name]
        if not success_paths:
            return []
        return [max(success_paths, key=success_checkpoint_sort_key)]
    if mode == "latest-model":
        return model_paths[-1:] if model_paths else []
    if mode == "pattern":
        if checkpoint_pattern is None:
            raise ValueError("--checkpoint-pattern is required when --checkpoint-mode pattern")
        matches = sorted((run_dir / "models").glob(checkpoint_pattern), key=checkpoint_sort_key)
        if not matches and checkpoint_pattern in {"last.pth", "last"}:
            checkpoint = run_dir / "last.pth"
            return [checkpoint] if checkpoint.exists() else []
        return matches
    raise ValueError("Unknown checkpoint mode: {}".format(mode))


def infer_sweep_parts(config_path, models_root):
    rel_parts = config_path.relative_to(models_root).parts
    if len(rel_parts) < 6:
        return None, None, None
    dataset_group, policy, weight_decay = rel_parts[0], rel_parts[1], rel_parts[2]
    return dataset_group, policy, weight_decay


def discover_config_paths(models_root):
    return sorted(models_root.rglob("config.json"), key=natural_key)


def validate_pair(repo_root, config_path, dataset_path, checkpoint_path):
    dataset_abs = resolve_repo_path(dataset_path, repo_root)
    if not dataset_abs.exists():
        raise ValueError(
            "dataset from config does not exist: {}".format(
                repo_relative(dataset_abs, repo_root)
            )
        )
    if not checkpoint_path.exists():
        raise ValueError(
            "checkpoint does not exist: {}".format(
                repo_relative(checkpoint_path, repo_root)
            )
        )
    if config_path.parent not in checkpoint_path.parents and checkpoint_path.parent != config_path.parent:
        raise ValueError(
            "checkpoint is not under config run directory: {}".format(
                repo_relative(checkpoint_path, repo_root)
            )
        )


def command_for_run(
    repo_root,
    config_path,
    checkpoint_path,
    dataset_path,
    output_dir,
    output_prefix,
    k,
    include_action,
    entropy_num_samples,
    knn_baseline_percentile,
    extra_args,
    models_root,
    output_suffix,
):
    dataset_group, policy, weight_decay = infer_sweep_parts(config_path, models_root)
    experiment_name = config_path.parent.parent.name
    timestamp = config_path.parent.name
    checkpoint_stem = checkpoint_path.stem
    output_name_parts = [
        output_prefix,
        dataset_group,
        policy,
        weight_decay,
        experiment_name,
        timestamp,
        checkpoint_stem,
    ]
    output_name = safe_name("_".join(str(part) for part in output_name_parts if part)) + output_suffix
    output_path = output_dir / output_name

    cmd = [
        "python",
        "robomimic/scripts/extract_policy_latent_knn.py",
        "--dataset",
        dataset_path,
        "--checkpoint",
        repo_relative(checkpoint_path, repo_root),
        "--output",
        repo_relative(output_path, repo_root),
        "--k",
        str(k),
        "--entropy-num-samples",
        str(entropy_num_samples),
        "--knn-baseline-percentile",
        str(knn_baseline_percentile),
    ]
    if include_action:
        cmd.append("--include-action")
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    return " ".join(shlex.quote(part) for part in cmd)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Square latent-KNN extraction commands paired with the correct training dataset."
    )
    parser.add_argument(
        "--models-root",
        default="robomimic/trained_models/square/sweep",
        help="Root containing Square sweep run directories.",
    )
    parser.add_argument(
        "--output-file",
        default="robomimic/scripts/square_knn_commands.txt",
        help="Path to write generated commands.",
    )
    parser.add_argument(
        "--knn-output-dir",
        default="vis/knn_json/square",
        help="Directory used in each command's --output JSON path.",
    )
    parser.add_argument(
        "--checkpoint-mode",
        choices=["last", "best-validation", "best", "best-success", "latest-model", "all", "pattern"],
        default="last",
        help=(
            "Which checkpoint(s) to use from each run. 'last' uses last.pth, "
            "'best-validation' / 'best' uses the lowest *_best_validation_* loss, "
            "'best-success' uses the highest *_success_* checkpoint, 'latest-model' "
            "uses the largest model epoch, 'all' emits every model checkpoint, and "
            "'pattern' uses --checkpoint-pattern."
        ),
    )
    parser.add_argument(
        "--checkpoint-pattern",
        default=None,
        help=(
            "Glob pattern under each run's models/ directory when --checkpoint-mode pattern, "
            "for example 'model_epoch_200*.pth' or '*success_0.4.pth'."
        ),
    )
    parser.add_argument("--k", type=int, default=200, help="KNN neighbors for generated commands.")
    parser.add_argument(
        "--include-action",
        action="store_true",
        help="Add --include-action to generated commands.",
    )
    parser.add_argument(
        "--entropy-num-samples",
        type=int,
        default=64,
        help="Entropy sample count for generated commands.",
    )
    parser.add_argument(
        "--knn-baseline-percentile",
        type=float,
        default=1.0,
        help="KNN baseline percentile for generated commands.",
    )
    parser.add_argument(
        "--output-prefix",
        default="knn",
        help="Prefix for generated output JSON filenames.",
    )
    parser.add_argument(
        "--output-suffix",
        default=".json.gz",
        help="Suffix for generated KNN output filenames. Use .json.gz for compressed JSON.",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help="Extra arguments appended to every generated extraction command.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    models_root = (repo_root / args.models_root).resolve()
    output_file = repo_root / args.output_file
    knn_output_dir = repo_root / args.knn_output_dir

    config_paths = discover_config_paths(models_root)
    commands = []
    skipped = []
    dataset_counts = {}

    for config_path in config_paths:
        try:
            dataset_path, _ = load_dataset_path(config_path)
            run_dir = config_path.parent
            checkpoints = select_checkpoints(
                run_dir,
                args.checkpoint_mode,
                checkpoint_pattern=args.checkpoint_pattern,
            )
            if not checkpoints:
                skipped.append("{}: no checkpoint for mode {}".format(config_path, args.checkpoint_mode))
                continue
            for checkpoint_path in checkpoints:
                validate_pair(
                    repo_root=repo_root,
                    config_path=config_path,
                    dataset_path=dataset_path,
                    checkpoint_path=checkpoint_path,
                )
                dataset_counts[dataset_path] = dataset_counts.get(dataset_path, 0) + 1
                commands.append(
                    command_for_run(
                        repo_root=repo_root,
                        config_path=config_path,
                        checkpoint_path=checkpoint_path,
                        dataset_path=dataset_path,
                        output_dir=knn_output_dir,
                        output_prefix=args.output_prefix,
                        k=args.k,
                        include_action=args.include_action,
                        entropy_num_samples=args.entropy_num_samples,
                        knn_baseline_percentile=args.knn_baseline_percentile,
                        extra_args=args.extra_args,
                        models_root=models_root,
                        output_suffix=args.output_suffix,
                    )
                )
        except Exception as exc:
            skipped.append("{}: {}".format(config_path, exc))

    lines = [
        "# Generated by robomimic/scripts/generate_square_knn_commands.py",
        "# checkpoint_mode: {}".format(args.checkpoint_mode),
        "# checkpoint_pattern: {}".format(args.checkpoint_pattern),
        "# configs discovered: {}".format(len(config_paths)),
        "# commands: {}".format(len(commands)),
        "# output json dir: {}".format(repo_relative(knn_output_dir, repo_root)),
        "# output suffix: {}".format(args.output_suffix),
        "# dataset command counts:",
    ]
    for dataset_path in sorted(dataset_counts, key=natural_key):
        lines.append("#   {}: {}".format(dataset_path, dataset_counts[dataset_path]))
    lines.extend([
        "",
        "mkdir -p {}".format(shlex.quote(repo_relative(knn_output_dir, repo_root))),
        "",
    ])
    lines.extend(commands)
    if skipped:
        lines.extend(["", "# Skipped:"])
        lines.extend("# {}".format(item) for item in skipped)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")

    print("Wrote {} commands to {}".format(len(commands), repo_relative(output_file, repo_root)))
    if skipped:
        print("Skipped {} runs; see comments at end of command file.".format(len(skipped)))


if __name__ == "__main__":
    main()
