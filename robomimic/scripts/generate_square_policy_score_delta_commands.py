"""
Generate image+robot vs robot-only policy score delta commands for Square.

The generator pairs checkpoints from:
    robomimic/trained_models/square/sweep
with robot-only checkpoints from:
    robomimic/trained_models/square/sweep_robot_only

Pairs are matched by dataset group, policy type, and weight-decay bucket.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


ROBOT_POLICY_TO_VISUAL_POLICY = {
    "bc_gmm": "gmm",
    "bc_discrete": "discrete",
    "bc_discrete_gaussian": "discrete_gaussian",
}

VISUAL_DATASET_TO_ROBOT_DATASET = {
    "ph_left_low_close": "ph",
    "ph_left_close_low": "ph",
    "mh_left_low_close": "mh",
    "mh_left_close_low": "mh",
}


DEFAULT_CHECKPOINT_MODES = "best-validation,last,early-after-best-validation"


def natural_key(name):
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", str(name))]


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def checkpoint_output_label(checkpoint):
    if checkpoint.name == "last.pth":
        return ""
    epoch = checkpoint_epoch(checkpoint)
    epoch_part = "e{}".format(epoch) if epoch is not None else safe_name(checkpoint.stem)
    return epoch_part


def checkpoint_mode_output_label(mode):
    return {
        "best-validation": "bestval",
        "best": "bestval",
        "early-after-best-validation": "early",
        "best-success": "success",
        "latest-model": "latest",
    }.get(mode, mode)


def checkpoint_pair_output_label(mode, checkpoint):
    mode_label = checkpoint_mode_output_label(mode)
    checkpoint_label = checkpoint_output_label(checkpoint)
    return "{}_{}".format(mode_label, checkpoint_label) if checkpoint_label else mode_label


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


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def load_dataset_path(config_path):
    config = load_config(config_path)
    data = config.get("train", {}).get("data")
    if not data:
        raise ValueError("No train.data found in {}".format(config_path))
    if isinstance(data, str):
        return data
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
    return dataset_path


def policy_type_from_config(config):
    algo = config.get("algo", {})
    if algo.get("gmm", {}).get("enabled", False):
        return "gmm"
    if algo.get("discrete", {}).get("enabled", False):
        target_type = algo.get("discrete", {}).get("target_type", "one_hot")
        return "discrete_gaussian" if target_type == "gaussian" else "discrete"
    if algo.get("gaussian", {}).get("enabled", False):
        return "gaussian"
    return "deterministic"


def weight_decay_from_config(config):
    return float(
        config.get("algo", {})
        .get("optim_params", {})
        .get("policy", {})
        .get("regularization", {})
        .get("L2", 0.0)
    )


def validation_checkpoint_sort_key(path):
    validation_match = re.search(r"_best_validation_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", path.name)
    if validation_match is None:
        return (float("inf"), -1, natural_key(path.name))
    validation_loss = float(validation_match.group(1))
    epoch_match = re.search(r"model_epoch_(\d+)", path.name)
    epoch = int(epoch_match.group(1)) if epoch_match is not None else -1
    return (validation_loss, epoch, natural_key(path.name))


def checkpoint_sort_key(path):
    epoch_match = re.search(r"model_epoch_(\d+)", path.name)
    epoch = int(epoch_match.group(1)) if epoch_match is not None else -1
    return (epoch, path.stat().st_mtime, natural_key(path.name))


def success_checkpoint_sort_key(path):
    success_match = re.search(r"_success_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", path.name)
    success = float(success_match.group(1)) if success_match is not None else -float("inf")
    epoch_match = re.search(r"model_epoch_(\d+)", path.name)
    epoch = int(epoch_match.group(1)) if epoch_match is not None else -1
    return (success, epoch, natural_key(path.name))


def checkpoint_epoch(path):
    epoch_match = re.search(r"model_epoch_(\d+)", path.name)
    return int(epoch_match.group(1)) if epoch_match is not None else None


def select_best_validation_checkpoint(model_paths):
    best_paths = [p for p in model_paths if "best_validation" in p.name]
    if not best_paths:
        return None
    return min(best_paths, key=validation_checkpoint_sort_key)


def select_early_checkpoint_after_best_validation(model_paths):
    best_checkpoint = select_best_validation_checkpoint(model_paths)
    if best_checkpoint is None:
        return None
    best_epoch = checkpoint_epoch(best_checkpoint)
    if best_epoch is None:
        return None

    regular_paths = [
        p for p in model_paths
        if checkpoint_epoch(p) is not None
        and checkpoint_epoch(p) > best_epoch
        and "best_validation" not in p.name
        and "_success_" not in p.name
    ]
    if not regular_paths:
        return None
    return min(regular_paths, key=lambda p: (checkpoint_epoch(p), natural_key(p.name)))


def select_checkpoints(run_dir, mode, checkpoint_pattern=None, expected_num_epochs=None):
    model_paths = sorted((run_dir / "models").glob("*.pth"), key=checkpoint_sort_key)
    if mode == "last":
        checkpoint = run_dir / "last.pth"
        if expected_num_epochs is not None and model_paths:
            final_paths = [
                p for p in model_paths
                if checkpoint_epoch(p) == int(expected_num_epochs)
            ]
            if not final_paths:
                return model_paths[-1:]
        if checkpoint.exists():
            return [checkpoint]
        return model_paths[-1:] if model_paths else []

    if mode == "all":
        return model_paths
    if mode in ("best-validation", "best"):
        checkpoint = select_best_validation_checkpoint(model_paths)
        return [checkpoint] if checkpoint is not None else []
    if mode == "early-after-best-validation":
        checkpoint = select_early_checkpoint_after_best_validation(model_paths)
        return [checkpoint] if checkpoint is not None else []
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


def discover_config_paths(models_root):
    return sorted(models_root.rglob("config.json"), key=natural_key)


def visual_key(config_path, models_root):
    rel_parts = config_path.relative_to(models_root).parts
    if len(rel_parts) < 6:
        return None
    dataset_group, policy, weight_decay = rel_parts[0], rel_parts[1], rel_parts[2]
    return dataset_group, policy, weight_decay


def robot_key(config_path, models_root):
    rel_parts = config_path.relative_to(models_root).parts
    if len(rel_parts) < 6:
        return None
    dataset_group, robot_policy, weight_decay = rel_parts[0], rel_parts[1], rel_parts[2]
    policy = ROBOT_POLICY_TO_VISUAL_POLICY.get(robot_policy)
    if policy is None:
        return None
    return dataset_group, policy, weight_decay


def robot_key_for_visual_key(key):
    dataset_group, policy, weight_decay = key
    return VISUAL_DATASET_TO_ROBOT_DATASET.get(dataset_group, dataset_group), policy, weight_decay


def validate_pair(repo_root, visual_config_path, robot_config_path, visual_checkpoint, robot_checkpoint):
    visual_config = load_config(visual_config_path)
    robot_config = load_config(robot_config_path)
    visual_dataset = load_dataset_path(visual_config_path)
    robot_dataset = load_dataset_path(robot_config_path)

    for dataset_path in (visual_dataset, robot_dataset):
        dataset_abs = resolve_repo_path(dataset_path, repo_root)
        if not dataset_abs.exists():
            raise ValueError("dataset does not exist: {}".format(repo_relative(dataset_abs, repo_root)))
    for checkpoint_path in (visual_checkpoint, robot_checkpoint):
        if not checkpoint_path.exists():
            raise ValueError("checkpoint does not exist: {}".format(repo_relative(checkpoint_path, repo_root)))

    visual_policy_type = policy_type_from_config(visual_config)
    robot_policy_type = policy_type_from_config(robot_config)
    if visual_policy_type != robot_policy_type:
        raise ValueError(
            "policy type mismatch: visual {} vs robot {}".format(
                visual_policy_type,
                robot_policy_type,
            )
        )

    visual_wd = weight_decay_from_config(visual_config)
    robot_wd = weight_decay_from_config(robot_config)
    if abs(visual_wd - robot_wd) > 1e-12:
        raise ValueError("weight decay mismatch: visual {} vs robot {}".format(visual_wd, robot_wd))

    return visual_dataset, robot_dataset, visual_policy_type, visual_wd


def validate_single_policy(repo_root, config_path, checkpoint):
    config = load_config(config_path)
    dataset = load_dataset_path(config_path)
    dataset_abs = resolve_repo_path(dataset, repo_root)
    if not dataset_abs.exists():
        raise ValueError("dataset does not exist: {}".format(repo_relative(dataset_abs, repo_root)))
    if not checkpoint.exists():
        raise ValueError("checkpoint does not exist: {}".format(repo_relative(checkpoint, repo_root)))
    return dataset, policy_type_from_config(config), weight_decay_from_config(config)


def command_for_pair(
    repo_root,
    visual_key_parts,
    visual_config_path,
    robot_config_path,
    visual_checkpoint,
    robot_checkpoint,
    visual_dataset,
    robot_dataset,
    output_dir,
    output_prefix,
    entropy_num_samples,
    batch_size,
    extra_args,
    output_suffix,
    visual_checkpoint_mode,
    robot_checkpoint_mode,
):
    dataset_group, policy, weight_decay = visual_key_parts
    visual_experiment = visual_config_path.parent.parent.name
    visual_timestamp = visual_config_path.parent.name
    robot_experiment = robot_config_path.parent.parent.name
    robot_timestamp = robot_config_path.parent.name
    output_identity = "|".join(
        [
            dataset_group,
            policy,
            weight_decay,
            str(visual_config_path),
            str(robot_config_path),
            str(visual_checkpoint),
            str(robot_checkpoint),
            visual_checkpoint_mode,
            robot_checkpoint_mode,
        ]
    )
    output_hash = hashlib.sha1(output_identity.encode("utf-8")).hexdigest()[:10]
    output_name = safe_name(
        "_".join(
            [
                output_prefix,
                dataset_group,
                policy,
                weight_decay,
                "v{}".format(checkpoint_pair_output_label(visual_checkpoint_mode, visual_checkpoint)),
                "r{}".format(checkpoint_pair_output_label(robot_checkpoint_mode, robot_checkpoint)),
                output_hash,
            ]
        )
    )
    output_path = output_dir / "{}{}".format(output_name, output_suffix)

    cmd = [
        "python",
        "robomimic/scripts/extract_policy_score_deltas.py",
        "--image-dataset",
        visual_dataset,
        "--robot-dataset",
        robot_dataset,
        "--image-checkpoint",
        repo_relative(visual_checkpoint, repo_root),
        "--robot-checkpoint",
        repo_relative(robot_checkpoint, repo_root),
        "--output",
        repo_relative(output_path, repo_root),
        "--entropy-num-samples",
        str(entropy_num_samples),
        "--batch-size",
        str(batch_size),
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    return " ".join(shlex.quote(part) for part in cmd)


def command_for_single_policy(
    repo_root,
    key_parts,
    config_path,
    checkpoint,
    dataset,
    output_dir,
    output_prefix,
    entropy_num_samples,
    batch_size,
    action_gmm_dir,
    extra_args,
    output_suffix,
    checkpoint_mode,
):
    dataset_group, policy, hp_tag = key_parts
    output_identity = "|".join(
        [
            dataset_group,
            policy,
            hp_tag,
            str(config_path),
            str(checkpoint),
            checkpoint_mode,
        ]
    )
    output_hash = hashlib.sha1(output_identity.encode("utf-8")).hexdigest()[:10]
    output_name = safe_name(
        "_".join(
            [
                output_prefix,
                dataset_group,
                policy,
                hp_tag,
                checkpoint_pair_output_label(checkpoint_mode, checkpoint),
                output_hash,
            ]
        )
    )
    output_path = output_dir / "{}{}".format(output_name, output_suffix)
    cmd = [
        "python",
        "robomimic/scripts/extract_policy_score_deltas.py",
        "--policy-dataset",
        dataset,
        "--policy-checkpoint",
        repo_relative(checkpoint, repo_root),
        "--output",
        repo_relative(output_path, repo_root),
        "--entropy-num-samples",
        str(entropy_num_samples),
        "--batch-size",
        str(batch_size),
        "--action-gmm-dir",
        repo_relative(action_gmm_dir, repo_root),
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    return " ".join(shlex.quote(part) for part in cmd)


def generate_prob_sweep_commands(args, repo_root):
    prob_root = (repo_root / args.prob_models_root).resolve()
    action_gmm_dir = (repo_root / args.action_gmm_dir).resolve()
    default_delta_output_file = "robomimic/scripts/square_policy_score_delta_commands.txt"
    default_delta_score_dir = "vis/policy_score_deltas/square"
    output_file_arg = (
        "robomimic/scripts/square_prob_sweep_policy_score_commands.txt"
        if args.output_file == default_delta_output_file
        else args.output_file
    )
    score_output_dir_arg = (
        "vis/policy_scores_prob_sweep_jsons"
        if args.score_output_dir == default_delta_score_dir
        else args.score_output_dir
    )
    output_file = repo_root / output_file_arg
    score_output_dir = repo_root / score_output_dir_arg

    default_delta_dataset_groups = "ph,mh,ph_left_low_close,mh_left_low_close,random_post"
    dataset_group_arg = (
        "ph,mh,ph_left_low_close,mh_left_low_close,random_post,random_post_left_low_close"
        if args.dataset_groups == default_delta_dataset_groups
        else args.dataset_groups
    )
    dataset_groups = parse_dataset_groups(dataset_group_arg)
    checkpoint_modes = (
        [args.checkpoint_mode]
        if args.checkpoint_mode is not None
        else (
            ["last"]
            if args.checkpoint_modes == DEFAULT_CHECKPOINT_MODES
            else parse_checkpoint_modes(args.checkpoint_modes)
        )
    )

    commands = []
    skipped = []
    matched = []
    for config_path in discover_config_paths(prob_root):
        key = visual_key(config_path, prob_root)
        if key is None or key[0] not in dataset_groups:
            continue
        expected_num_epochs = load_config(config_path).get("train", {}).get("num_epochs", None)
        for checkpoint_mode in checkpoint_modes:
            selected_checkpoints = select_checkpoints(
                config_path.parent,
                checkpoint_mode,
                checkpoint_pattern=args.checkpoint_pattern,
                expected_num_epochs=expected_num_epochs,
            )
            if not selected_checkpoints:
                skipped.append("{}: no checkpoint for mode {}".format(config_path, checkpoint_mode))
                continue
            for checkpoint in selected_checkpoints:
                try:
                    dataset, policy_type, weight_decay = validate_single_policy(
                        repo_root=repo_root,
                        config_path=config_path,
                        checkpoint=checkpoint,
                    )
                    matched.append((key, policy_type, weight_decay, dataset))
                    commands.append(
                        command_for_single_policy(
                            repo_root=repo_root,
                            key_parts=key,
                            config_path=config_path,
                            checkpoint=checkpoint,
                            dataset=dataset,
                            output_dir=score_output_dir,
                            output_prefix=args.output_prefix,
                            entropy_num_samples=args.entropy_num_samples,
                            batch_size=args.batch_size,
                            action_gmm_dir=action_gmm_dir,
                            extra_args=args.extra_args,
                            output_suffix=args.output_suffix,
                            checkpoint_mode=checkpoint_mode,
                        )
                    )
                except Exception as exc:
                    skipped.append("{}: {}".format(config_path, exc))

    lines = [
        "# Generated by robomimic/scripts/generate_square_policy_score_delta_commands.py --mode prob-sweep",
        "# checkpoint_modes: {}".format(",".join(checkpoint_modes)),
        "# checkpoint_pattern: {}".format(args.checkpoint_pattern),
        "# dataset_groups: {}".format(",".join(sorted(dataset_groups))),
        "# commands: {}".format(len(commands)),
        "# output json dir: {}".format(repo_relative(score_output_dir, repo_root)),
        "# conditional p(a|s) policy root: {}".format(repo_relative(prob_root, repo_root)),
        "# non-conditional p(a) baseline dir: {}".format(repo_relative(action_gmm_dir, repo_root)),
        "# single-policy p(a|s) NLL / entropy scoring for sweep_cond_prob checkpoints",
        "# action_gmm is only used for non-conditional baseline scores and conditional-minus-nonconditional variants",
        "",
        "mkdir -p {}".format(shlex.quote(repo_relative(score_output_dir, repo_root))),
        "",
    ]
    lines.extend(commands)
    if skipped:
        lines.extend(["", "# Skipped:"])
        lines.extend("# {}".format(item) for item in skipped)

    if not args.verify_only:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write("\n".join(lines))
            f.write("\n")

    keys = sorted({item[0] for item in matched}, key=natural_key)
    print("Verified {} scoring commands for {} keys.".format(len(commands), len(keys)))
    for key in keys:
        print("  {} / {} / {}".format(key[0], key[1], key[2]))
    if args.verify_only:
        print("Verify-only mode; command file was not written.")
    else:
        print("Wrote {} commands to {}".format(len(commands), repo_relative(output_file, repo_root)))
    if skipped:
        print("Skipped {} configs or checkpoint selections; see command file comments.".format(len(skipped)))


def parse_dataset_groups(value):
    return {part.strip() for part in value.split(",") if part.strip()}


def parse_checkpoint_modes(value):
    modes = [part.strip() for part in value.split(",") if part.strip()]
    valid_modes = {
        "last",
        "best-validation",
        "best",
        "best-success",
        "latest-model",
        "all",
        "pattern",
        "early-after-best-validation",
    }
    invalid = [mode for mode in modes if mode not in valid_modes]
    if invalid:
        raise ValueError("Invalid checkpoint mode(s): {}".format(", ".join(invalid)))
    return modes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Square policy NLL/entropy delta commands for visual and robot-only policy pairs."
    )
    parser.add_argument(
        "--mode",
        choices=["delta", "prob-sweep"],
        default="delta",
        help=(
            "delta emits existing image+robot vs robot-only commands. prob-sweep emits "
            "single-policy p(a|s) score commands for sweep_cond_prob checkpoints."
        ),
    )
    parser.add_argument("--visual-models-root", default="robomimic/trained_models/square/sweep")
    parser.add_argument("--robot-models-root", default="robomimic/trained_models/square/sweep_robot_only")
    parser.add_argument("--prob-models-root", default="robomimic/trained_models/square/sweep_cond_prob")
    parser.add_argument(
        "--action-gmm-dir",
        default="robomimic/trained_models/square/action_gmm",
        help="Non-conditional p(a) action GMM directory used only as the baseline in prob-sweep mode.",
    )
    parser.add_argument(
        "--dataset-groups",
        default="ph,mh,ph_left_low_close,mh_left_low_close,random_post",
        help=(
            "Comma-separated visual dataset groups to pair. Left-low-close visual groups "
            "are paired to the matching base robot-only group."
        ),
    )
    parser.add_argument(
        "--output-file",
        default="robomimic/scripts/square_policy_score_delta_commands.txt",
        help="Path to write generated commands.",
    )
    parser.add_argument(
        "--score-output-dir",
        default="vis/policy_score_deltas/square",
        help="Directory used in each generated command's --output path.",
    )
    parser.add_argument(
        "--checkpoint-mode",
        choices=[
            "last",
            "best-validation",
            "best",
            "best-success",
            "latest-model",
            "all",
            "pattern",
            "early-after-best-validation",
        ],
        default=None,
        help="Single checkpoint mode. If omitted, --checkpoint-modes is used.",
    )
    parser.add_argument(
        "--checkpoint-modes",
        default=DEFAULT_CHECKPOINT_MODES,
        help=(
            "Comma-separated checkpoint modes emitted for both visual and robot-only policies. "
            "Default: {}. The early-after-best-validation mode selects the first regular "
            "model_epoch_N.pth with N greater than the best-validation epoch."
        ).format(DEFAULT_CHECKPOINT_MODES),
    )
    parser.add_argument("--checkpoint-pattern", default=None)
    parser.add_argument(
        "--checkpoint-pairing",
        choices=["cross-product", "same-label"],
        default="same-label",
        help=(
            "How to pair selected visual and robot-only checkpoints. 'cross-product' "
            "emits every visual-mode x robot-mode command; 'same-label' emits only "
            "best-validation vs best-validation, last vs last, etc."
        ),
    )
    parser.add_argument("--entropy-num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-prefix", default="policy_scores")
    parser.add_argument("--output-suffix", default=".json.gz")
    parser.add_argument("--extra-args", default="", help="Extra args appended to every scoring command.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate pair discovery without writing the command file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if args.mode == "prob-sweep":
        generate_prob_sweep_commands(args, repo_root)
        return

    visual_root = (repo_root / args.visual_models_root).resolve()
    robot_root = (repo_root / args.robot_models_root).resolve()
    output_file = repo_root / args.output_file
    score_output_dir = repo_root / args.score_output_dir
    dataset_groups = parse_dataset_groups(args.dataset_groups)
    robot_dataset_groups = {
        VISUAL_DATASET_TO_ROBOT_DATASET.get(dataset_group, dataset_group)
        for dataset_group in dataset_groups
    }
    checkpoint_modes = [args.checkpoint_mode] if args.checkpoint_mode is not None else parse_checkpoint_modes(args.checkpoint_modes)

    robot_configs_by_key = {}
    for config_path in discover_config_paths(robot_root):
        key = robot_key(config_path, robot_root)
        if key is None or key[0] not in robot_dataset_groups:
            continue
        robot_configs_by_key.setdefault(key, []).append(config_path)

    commands = []
    skipped = []
    matched_pairs = []
    visual_configs = discover_config_paths(visual_root)
    for visual_config_path in visual_configs:
        key = visual_key(visual_config_path, visual_root)
        if key is None or key[0] not in dataset_groups:
            continue
        robot_match_key = robot_key_for_visual_key(key)
        robot_configs = robot_configs_by_key.get(robot_match_key, [])
        if not robot_configs:
            skipped.append("{}: no robot-only config for key {} (visual key {})".format(
                visual_config_path,
                robot_match_key,
                key,
            ))
            continue
        if len(robot_configs) != 1:
            skipped.append("{}: expected one robot-only config for key {} (visual key {}), got {}".format(
                visual_config_path,
                robot_match_key,
                key,
                len(robot_configs),
            ))
            continue

        robot_config_path = robot_configs[0]
        visual_expected_num_epochs = load_config(visual_config_path).get("train", {}).get("num_epochs", None)
        robot_expected_num_epochs = load_config(robot_config_path).get("train", {}).get("num_epochs", None)
        visual_checkpoints_by_mode = {}
        robot_checkpoints_by_mode = {}
        for checkpoint_mode in checkpoint_modes:
            selected_visual_checkpoints = select_checkpoints(
                visual_config_path.parent,
                checkpoint_mode,
                checkpoint_pattern=args.checkpoint_pattern,
                expected_num_epochs=visual_expected_num_epochs,
            )
            selected_robot_checkpoints = select_checkpoints(
                robot_config_path.parent,
                checkpoint_mode,
                checkpoint_pattern=args.checkpoint_pattern,
                expected_num_epochs=robot_expected_num_epochs,
            )
            if selected_visual_checkpoints:
                visual_checkpoints_by_mode[checkpoint_mode] = selected_visual_checkpoints
            else:
                skipped.append("{}: no visual checkpoint for mode {}".format(visual_config_path, checkpoint_mode))
            if selected_robot_checkpoints:
                robot_checkpoints_by_mode[checkpoint_mode] = selected_robot_checkpoints
            else:
                skipped.append("{}: no robot checkpoint for mode {}".format(robot_config_path, checkpoint_mode))

        checkpoint_mode_pairs = []
        if args.checkpoint_pairing == "same-label":
            checkpoint_mode_pairs = [
                (mode, mode)
                for mode in checkpoint_modes
                if mode in visual_checkpoints_by_mode and mode in robot_checkpoints_by_mode
            ]
        else:
            checkpoint_mode_pairs = [
                (visual_mode, robot_mode)
                for visual_mode in checkpoint_modes
                for robot_mode in checkpoint_modes
                if visual_mode in visual_checkpoints_by_mode and robot_mode in robot_checkpoints_by_mode
            ]

        for visual_checkpoint_mode, robot_checkpoint_mode in checkpoint_mode_pairs:
            for visual_checkpoint in visual_checkpoints_by_mode[visual_checkpoint_mode]:
                for robot_checkpoint in robot_checkpoints_by_mode[robot_checkpoint_mode]:
                    try:
                        visual_dataset, robot_dataset, policy_type, weight_decay = validate_pair(
                            repo_root=repo_root,
                            visual_config_path=visual_config_path,
                            robot_config_path=robot_config_path,
                            visual_checkpoint=visual_checkpoint,
                            robot_checkpoint=robot_checkpoint,
                        )
                        matched_pairs.append((key, policy_type, weight_decay, visual_dataset, robot_dataset))
                        commands.append(
                            command_for_pair(
                                repo_root=repo_root,
                                visual_key_parts=key,
                                visual_config_path=visual_config_path,
                                robot_config_path=robot_config_path,
                                visual_checkpoint=visual_checkpoint,
                                robot_checkpoint=robot_checkpoint,
                                visual_dataset=visual_dataset,
                                robot_dataset=robot_dataset,
                                output_dir=score_output_dir,
                                output_prefix=args.output_prefix,
                                entropy_num_samples=args.entropy_num_samples,
                                batch_size=args.batch_size,
                                extra_args=args.extra_args,
                                output_suffix=args.output_suffix,
                                visual_checkpoint_mode=visual_checkpoint_mode,
                                robot_checkpoint_mode=robot_checkpoint_mode,
                            )
                        )
                    except Exception as exc:
                        skipped.append("{} vs {}: {}".format(visual_config_path, robot_config_path, exc))

    lines = [
        "# Generated by robomimic/scripts/generate_square_policy_score_delta_commands.py",
        "# checkpoint_modes: {}".format(",".join(checkpoint_modes)),
        "# checkpoint_pairing: {}".format(args.checkpoint_pairing),
        "# checkpoint_pattern: {}".format(args.checkpoint_pattern),
        "# dataset_groups: {}".format(",".join(sorted(dataset_groups))),
        "# commands: {}".format(len(commands)),
        "# output json dir: {}".format(repo_relative(score_output_dir, repo_root)),
        "# verified pairs match on compatible dataset group, policy type, and weight decay",
        "# left-low-close visual datasets are paired with base ph/mh robot-only datasets",
        "",
        "mkdir -p {}".format(shlex.quote(repo_relative(score_output_dir, repo_root))),
        "",
    ]
    lines.extend(commands)
    if skipped:
        lines.extend(["", "# Skipped:"])
        lines.extend("# {}".format(item) for item in skipped)

    if not args.verify_only:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write("\n".join(lines))
            f.write("\n")

    pair_keys = sorted({item[0] for item in matched_pairs}, key=natural_key)
    print("Verified {} matching policy pairs for {} keys.".format(len(commands), len(pair_keys)))
    for key in pair_keys:
        print("  {} / {} / {}".format(key[0], key[1], key[2]))
    if args.verify_only:
        print("Verify-only mode; command file was not written.")
    else:
        print("Wrote {} commands to {}".format(len(commands), repo_relative(output_file, repo_root)))
    if skipped:
        print("Skipped {} configs or checkpoint pairs; see command file comments.".format(len(skipped)))


if __name__ == "__main__":
    main()
