"""
Generate Square image+robot GMM configs for filtered HDF5 train masks.

Defaults generate no-weight-decay GMM policy configs for:
    - PH and PH left_close_low observability drops: 10, 20, 30, 40 percent
    - MH and MH left_close_low quality drops: 25, 50, 75 percent

The generated configs point at the copied HDF5 files with extra mask keys, e.g.
image_filter_keys.hdf5, so locked source datasets do not need to be modified.

Example:
    python robomimic/scripts/generate_square_filter_key_configs.py
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path


FILTER_SPECS = {
    "ph": {
        "template": Path("robomimic/exps/square/sweep/ph/gmm/wd0.json"),
        "dataset": "robomimic/datasets/square/ph/image_filter_keys.hdf5",
        "metric": "observability",
        "drop_pcts": [10, 20, 30, 40],
    },
    "ph_left_low_close": {
        "template": Path("robomimic/exps/square/sweep/ph_left_low_close/gmm/wd0.json"),
        "dataset": "robomimic/datasets/square/ph/square_ph_left_close_low_wrist_image_filter_keys.hdf5",
        "metric": "observability",
        "drop_pcts": [10, 20, 30, 40],
    },
    "mh": {
        "template": Path("robomimic/exps/square/sweep/mh/gmm/wd0.json"),
        "dataset": "robomimic/datasets/square/mh/image_filter_keys.hdf5",
        "metric": "quality",
        "drop_pcts": [25, 50, 75],
    },
    "mh_left_low_close": {
        "template": Path("robomimic/exps/square/sweep/mh_left_low_close/gmm/wd0.json"),
        "dataset": "robomimic/datasets/square/mh/image_left_close_low_wrist_filter_keys.hdf5",
        "metric": "quality",
        "drop_pcts": [25, 50, 75],
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="robomimic/exps/square/sweep_filter_keys",
    )
    parser.add_argument(
        "--config_paths_file",
        default=None,
        help=(
            "Path to write generated config paths. Defaults to "
            "<output_dir>/config_paths.txt."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(FILTER_SPECS),
        choices=sorted(FILTER_SPECS),
    )
    return parser.parse_args()


def filter_key(metric, drop_pct, split):
    return "{}_{}_drop{:02d}pct".format(split, metric, drop_pct)


def make_config(template, dataset_name, dataset_path, metric, drop_pct):
    config = deepcopy(template)
    drop_tag = "{}_drop{:02d}pct".format(metric, drop_pct)
    run_name = "bc_gmm_square_{}_image_wd0_{}".format(
        dataset_name,
        drop_tag,
    )

    config["experiment"]["name"] = run_name
    config["experiment"]["validate"] = False
    config["experiment"]["logging"]["wandb_proj_name"] = "pomdp_square_filter"
    config["experiment"]["rollout"]["enabled"] = False
    config["experiment"]["rollout"]["rate"] = 200
    config["experiment"]["rollout"]["n"] = 50
    config["experiment"]["rollout"]["warmstart"] = 600
    config["experiment"]["save"]["on_best_validation"] = False
    config["experiment"]["save"]["on_best_rollout_success_rate"] = False
    config["experiment"]["save"]["on_best_rollout_return"] = False
    config["experiment"]["save"]["every_n_epochs"] = 200
    config["experiment"]["epoch_every_n_steps"] = 200
    config["experiment"]["validation_epoch_every_n_steps"] = 20
    config["train"]["data"] = dataset_path
    config["train"]["output_dir"] = (
        "trained_models/square/sweep_filter_keys/{}/gmm/wd0/{}".format(
            dataset_name,
            drop_tag,
        )
    )
    config["train"]["hdf5_filter_key"] = filter_key(metric, drop_pct, "all")
    config["train"]["hdf5_validation_filter_key"] = None
    config["train"]["batch_size"] = 128
    config["train"]["num_epochs"] = 2000
    config["train"]["hdf5_cache_mode"] = "all"
    config["algo"]["optim_params"]["policy"]["regularization"]["L2"] = 0.0
    config["algo"]["gmm"]["enabled"] = True
    config["algo"]["gaussian"]["enabled"] = False
    config["algo"]["vae"]["enabled"] = False
    config["algo"]["rnn"]["enabled"] = False
    config["algo"]["transformer"]["enabled"] = False
    if "discrete" in config["algo"]:
        config["algo"]["discrete"]["enabled"] = False
    return config


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for dataset_name in args.datasets:
        spec = FILTER_SPECS[dataset_name]
        template = json.loads(spec["template"].read_text())
        for drop_pct in spec["drop_pcts"]:
            config = make_config(
                template=template,
                dataset_name=dataset_name,
                dataset_path=spec["dataset"],
                metric=spec["metric"],
                drop_pct=drop_pct,
            )
            path = output_dir / dataset_name / "gmm" / "wd0" / "{}_drop{:02d}pct.json".format(
                spec["metric"],
                drop_pct,
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
