"""
Generate Square BC-GMM configs for conditional action density p(a | s).

The existing sweep_2 GMM configs use a single default GMM setting:

    num_modes = 5
    min_std = 1e-4

That is a fine baseline, but it does not sweep the main conditional density
hyperparameters. This generator writes a separate sweep_cond_prob grid over the
number of mixture modes and the diagonal covariance floor used by robomimic's
GMM policy head.

Example:
    python robomimic/scripts/generate_square_bc_sweep_cond_prob_configs.py
"""

import argparse
import json
from pathlib import Path

from generate_square_bc_sweep_configs import (
    DATASETS,
    make_config,
    load_templates,
    weight_decay_tag,
)


DEFAULT_OUTPUT_DIR = "robomimic/exps/square/sweep_cond_prob"
DEFAULT_TRAIN_OUTPUT_ROOT = "trained_models/square/sweep_cond_prob"


def float_tag(value):
    if value == 0:
        return "0"
    return "{:.0e}".format(value).replace("-", "m")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_dir", default="robomimic/exps/square/ph")
    parser.add_argument("--dataset_dir", default="robomimic/datasets/square")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--train_output_root",
        default=DEFAULT_TRAIN_OUTPUT_ROOT,
        help="Root directory used in config train.output_dir.",
    )
    parser.add_argument(
        "--config_paths_file",
        default=None,
        help=(
            "Path to write a newline-delimited list of generated config paths. "
            "Defaults to <output_dir>/config_paths.txt."
        ),
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--weight_decays", nargs="+", type=float, default=[0.0, 1e-4])
    parser.add_argument("--num_modes", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--min_stds", nargs="+", type=float, default=[1e-4])
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help=(
            "Number of epochs for the probability-model sweep. With the default "
            "--epoch_every_n_steps, this is 20k optimizer steps per config."
        ),
    )
    parser.add_argument("--epoch_every_n_steps", type=int, default=100)
    parser.add_argument("--save_every_n_epochs", type=int, default=50)
    parser.add_argument(
        "--validation_epoch_every_n_steps",
        type=int,
        default=None,
        help=(
            "Number of validation minibatches used for checkpoint selection. "
            "Default None runs a full validation pass for a lower-noise NLL estimate."
        ),
    )
    parser.add_argument(
        "--use_left_close_low_obs",
        action="store_true",
        help="Include left_close_low_image as a training RGB observation.",
    )
    parser.add_argument(
        "--camera_images_only",
        action="store_true",
        help="Train from camera image observations only, with no low-dimensional robot state inputs.",
    )
    return parser.parse_args()


def apply_cond_prob_overrides(
    config,
    dataset,
    weight_decay,
    num_modes,
    min_std,
    args,
):
    tag = "k{}_std{}_{}".format(
        num_modes,
        float_tag(min_std),
        weight_decay_tag(weight_decay),
    )
    obs_tag = "camera_image_only" if args.camera_images_only else "image"

    config["experiment"]["name"] = "bc_gmm_square_{}_{}_{}".format(
        dataset,
        obs_tag,
        tag,
    )
    config["experiment"]["logging"]["wandb_proj_name"] = "pomdp_square_cond_prob"
    config["experiment"]["epoch_every_n_steps"] = args.epoch_every_n_steps
    config["experiment"]["validate"] = True
    config["experiment"]["validation_epoch_every_n_steps"] = (
        args.validation_epoch_every_n_steps
    )
    config["experiment"]["rollout"]["enabled"] = False
    config["experiment"]["save"]["every_n_epochs"] = args.save_every_n_epochs
    config["experiment"]["save"]["on_best_validation"] = True
    config["experiment"]["save"]["on_best_rollout_success_rate"] = False
    config["experiment"]["save"]["on_best_rollout_return"] = False

    config["train"]["output_dir"] = str(
        Path(args.train_output_root) / dataset / "gmm" / tag
    )
    config["train"]["batch_size"] = args.batch_size
    config["train"]["num_epochs"] = args.num_epochs
    config["train"]["hdf5_cache_mode"] = "all"
    config["train"]["hdf5_filter_key"] = "train"
    config["train"]["hdf5_validation_filter_key"] = "valid"

    policy_optim = config["algo"]["optim_params"]["policy"]
    policy_optim["learning_rate"]["initial"] = args.learning_rate
    policy_optim["regularization"]["L2"] = weight_decay

    config["algo"]["gmm"]["enabled"] = True
    config["algo"]["gmm"]["num_modes"] = num_modes
    config["algo"]["gmm"]["min_std"] = min_std
    config["algo"]["gmm"]["std_activation"] = "softplus"
    config["algo"]["gmm"]["low_noise_eval"] = True
    config["algo"]["gaussian"]["enabled"] = False
    config["algo"]["discrete"]["enabled"] = False
    config["algo"]["vae"]["enabled"] = False
    config["algo"]["rnn"]["enabled"] = False
    config["algo"]["transformer"]["enabled"] = False
    return config, tag


def main():
    args = parse_args()
    templates = load_templates(args.template_dir)
    template = templates["gmm"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for dataset in args.datasets:
        for num_modes in args.num_modes:
            for min_std in args.min_stds:
                for weight_decay in args.weight_decays:
                    config = make_config(
                        template=template,
                        dataset=dataset,
                        policy="gmm",
                        weight_decay=weight_decay,
                        dataset_dir=args.dataset_dir,
                        use_left_close_low_obs=args.use_left_close_low_obs,
                        camera_images_only=args.camera_images_only,
                    )
                    config, tag = apply_cond_prob_overrides(
                        config=config,
                        dataset=dataset,
                        weight_decay=weight_decay,
                        num_modes=num_modes,
                        min_std=min_std,
                        args=args,
                    )

                    path = output_dir / dataset / "gmm" / "{}.json".format(tag)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(config, indent=4) + "\n")
                    written.append(path)

    config_paths_file = (
        Path(args.config_paths_file)
        if args.config_paths_file
        else output_dir / "config_paths.txt"
    )
    config_paths_file.parent.mkdir(parents=True, exist_ok=True)
    config_paths_file.write_text("\n".join(str(path) for path in written) + "\n")

    print("Wrote {} configs:".format(len(written)))
    for path in written:
        print(path)
    print("Wrote config paths to {}".format(config_paths_file))


if __name__ == "__main__":
    main()
