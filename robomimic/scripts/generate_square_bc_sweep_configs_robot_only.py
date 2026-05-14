"""
Generate Square robot-only BC sweep configs.

Example:
    python robomimic/scripts/generate_square_bc_sweep_configs_robot_only.py

    python robomimic/scripts/generate_square_bc_sweep_configs_robot_only.py \
        --weight_decays 0 1e-4

    python robomimic/scripts/generate_square_bc_sweep_configs_robot_only.py \
        --config_paths_file robomimic/exps/square/sweep_robot_only/config_paths.txt
"""
import argparse
import json
from copy import deepcopy
from pathlib import Path


DATASETS = ("ph", "mh", "random_post")
POLICIES = ("bc_gmm", "bc_discrete", "bc_discrete_gaussian")
ROBOT_OBS_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
]
DATASET_SPECS = {
    "ph": {
        "path": Path("ph") / "low_dim_v15.hdf5",
        "validate": True,
        "filter_key": "train",
        "validation_filter_key": "valid",
    },
    "mh": {
        "path": Path("mh") / "low_dim_v15.hdf5",
        "validate": True,
        "filter_key": "train",
        "validation_filter_key": "valid",
    },
    "random_post": {
        "path": Path("random_post") / "expert200" / "low_dim_v15.hdf5",
        "validate": False,
        "filter_key": None,
        "validation_filter_key": None,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="robomimic/exps/square/ph/bc_gmm_image.json")
    parser.add_argument("--dataset_dir", default="robomimic/datasets/square")
    parser.add_argument("--output_dir", default="robomimic/exps/square/sweep_robot_only")
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
    return parser.parse_args()


def weight_decay_tag(weight_decay):
    if weight_decay == 0:
        return "wd0"
    return "wd{}".format("{:.0e}".format(weight_decay).replace("-", "m"))


def load_template(template):
    return json.loads(Path(template).read_text())


def set_robot_only_obs(config):
    obs = config["observation"]["modalities"]["obs"]
    obs["low_dim"] = list(ROBOT_OBS_KEYS)
    obs["rgb"] = []
    obs["depth"] = []
    obs["scan"] = []

    rgb_encoder = config["observation"]["encoder"]["rgb"]
    rgb_encoder["core_kwargs"] = {}
    rgb_encoder["obs_randomizer_class"] = None
    rgb_encoder["obs_randomizer_kwargs"] = {}

    config["experiment"]["env_meta_update_dict"] = {}


def set_policy(config, policy):
    config["algo"]["gmm"]["enabled"] = (policy == "bc_gmm")
    config["algo"]["gaussian"]["enabled"] = False
    config["algo"]["vae"]["enabled"] = False
    config["algo"]["rnn"]["enabled"] = False
    config["algo"]["transformer"]["enabled"] = False

    discrete_enabled = policy in ["bc_discrete", "bc_discrete_gaussian"]
    config["algo"]["discrete"]["enabled"] = discrete_enabled
    if discrete_enabled:
        config["algo"]["discrete"]["target_type"] = (
            "gaussian" if policy == "bc_discrete_gaussian" else "one_hot"
        )


def set_dataset(config, dataset, dataset_dir):
    dataset_spec = DATASET_SPECS[dataset]
    config["experiment"]["validate"] = dataset_spec["validate"]
    if not dataset_spec["validate"]:
        config["experiment"]["save"]["on_best_validation"] = False

    config["train"]["data"] = str(Path(dataset_dir) / dataset_spec["path"])
    config["train"]["hdf5_filter_key"] = dataset_spec["filter_key"]
    config["train"]["hdf5_validation_filter_key"] = dataset_spec["validation_filter_key"]


def make_config(template, dataset, policy, weight_decay, dataset_dir):
    config = deepcopy(template)
    tag = weight_decay_tag(weight_decay)
    run_name = "{}_square_{}_robot_{}".format(policy, dataset, tag)

    config["experiment"]["name"] = run_name
    config["experiment"]["logging"]["wandb_proj_name"] = "square_robot_only"
    config["train"]["output_dir"] = "trained_models/square/sweep_robot_only/{}/{}/{}".format(
        dataset, policy, tag)
    config["train"]["num_data_workers"] = 0
    config["train"]["hdf5_cache_mode"] = "all"
    config["train"]["batch_size"] = 100
    config["train"]["num_epochs"] = 2000
    config["experiment"]["epoch_every_n_steps"] = 100
    config["experiment"]["validation_epoch_every_n_steps"] = 10
    config["algo"]["optim_params"]["policy"]["regularization"]["L2"] = weight_decay

    set_dataset(config, dataset, dataset_dir)
    set_policy(config, policy)
    set_robot_only_obs(config)
    return config


def main():
    args = parse_args()
    template = load_template(args.template)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for dataset in args.datasets:
        for policy in args.policies:
            for weight_decay in args.weight_decays:
                config = make_config(
                    template=template,
                    dataset=dataset,
                    policy=policy,
                    weight_decay=weight_decay,
                    dataset_dir=args.dataset_dir,
                )
                path = output_dir / dataset / policy / "{}.json".format(weight_decay_tag(weight_decay))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(config, indent=4) + "\n")
                written.append(path)

    print("Wrote {} configs:".format(len(written)))
    for path in written:
        print(path)

    config_paths_file = Path(args.config_paths_file) if args.config_paths_file else output_dir / "config_paths.txt"
    config_paths_file.parent.mkdir(parents=True, exist_ok=True)
    config_paths_file.write_text("\n".join(str(path) for path in written) + "\n")
    print("Wrote config paths to {}".format(config_paths_file))


if __name__ == "__main__":
    main()
