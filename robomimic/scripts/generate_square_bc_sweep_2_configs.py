"""
Generate Square visual BC sweep_2 configs.

This mirrors generate_square_bc_sweep_configs.py, but writes configs under
robomimic/exps/square/sweep_2 and uses the longer training schedule:

    batch_size = 128
    num_epochs = 2000
    epoch_every_n_steps = 200
    hdf5_cache_mode = "all"

All other config fields are inherited from the regular sweep generator.

Example:
    python robomimic/scripts/generate_square_bc_sweep_2_configs.py
"""

import argparse
import json
from pathlib import Path

from generate_square_bc_sweep_configs import (
    DATASETS,
    POLICIES,
    make_config,
    load_templates,
    weight_decay_tag,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_dir", default="robomimic/exps/square/ph")
    parser.add_argument("--dataset_dir", default="robomimic/datasets/square")
    parser.add_argument("--output_dir", default="robomimic/exps/square/sweep_2")
    parser.add_argument(
        "--config_paths_file",
        default=None,
        help=(
            "Path to write a newline-delimited list of generated config paths. "
            "Defaults to <output_dir>/config_paths.txt."
        ),
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    parser.add_argument("--weight_decays", nargs="+", type=float, default=[0.0, 1e-4])
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


def apply_sweep_2_overrides(config):
    config["train"]["output_dir"] = config["train"]["output_dir"].replace(
        "trained_models/square/sweep/",
        "trained_models/square/sweep_2/",
        1,
    )
    config["train"]["batch_size"] = 128
    config["train"]["num_epochs"] = 2000
    config["train"]["hdf5_cache_mode"] = "all"
    config["experiment"]["epoch_every_n_steps"] = 200
    config["experiment"]["validation_epoch_every_n_steps"] = 20
    config["experiment"]["logging"]["wandb_proj_name"] = "pomdp_square_2"
    config["experiment"]["save"]["every_n_epochs"] = 200
    config["experiment"]["save"]["on_best_validation"] = False
    config["experiment"]["rollout"]["enabled"] = False
    config["experiment"]["rollout"]["rate"] = 200
    config["experiment"]["rollout"]["n"] = 50
    config["experiment"]["rollout"]["warmstart"] = 600
    config["experiment"]["save"]["on_best_rollout_success_rate"] = False
    config["experiment"]["save"]["on_best_rollout_return"] = False
    return config


def main():
    args = parse_args()
    templates = load_templates(args.template_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for dataset in args.datasets:
        for policy in args.policies:
            for weight_decay in args.weight_decays:
                config = make_config(
                    template=templates[policy],
                    dataset=dataset,
                    policy=policy,
                    weight_decay=weight_decay,
                    dataset_dir=args.dataset_dir,
                    use_left_close_low_obs=args.use_left_close_low_obs,
                    camera_images_only=args.camera_images_only,
                )
                config = apply_sweep_2_overrides(config)

                path = output_dir / dataset / policy / "{}.json".format(
                    weight_decay_tag(weight_decay)
                )
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
