"""
Generate Square visual BC sweep configs.

Example:
    python robomimic/scripts/generate_square_bc_sweep_configs.py

    python robomimic/scripts/generate_square_bc_sweep_configs.py \
        --weight_decays 0 1e-4

    python robomimic/scripts/generate_square_bc_sweep_configs.py \
        --use_left_close_low_obs

    python robomimic/scripts/generate_square_bc_sweep_configs.py \
        --config_paths_file robomimic/exps/square/sweep/config_paths.txt
"""
import argparse
import json
from copy import deepcopy
from pathlib import Path


DATASETS = (
    "ph",
    "mh",
    "ph_left_low_close",
    "random_post",
    "random_post_left_low_close",
    "mh_left_low_close",
)
POLICIES = ("gmm", "discrete", "discrete_gaussian")
AGENTVIEW_CAMERA_NAME = "agentview"
AGENTVIEW_OBS_KEY = "agentview_image"
LEFT_CLOSE_LOW_CAMERA_NAME = "left_close_low"
LEFT_CLOSE_LOW_OBS_KEY = "left_close_low_image"
WRIST_CAMERA_NAME = "robot0_eye_in_hand"
WRIST_OBS_KEY = "robot0_eye_in_hand_image"
DATASET_SPECS = {
    "ph": {
        "path": Path("ph") / "image.hdf5",
        "camera_names": [AGENTVIEW_CAMERA_NAME, WRIST_CAMERA_NAME],
        "rgb_obs": [AGENTVIEW_OBS_KEY, WRIST_OBS_KEY],
    },
    "mh": {
        "path": Path("mh") / "image.hdf5",
        "camera_names": [AGENTVIEW_CAMERA_NAME, WRIST_CAMERA_NAME],
        "rgb_obs": [AGENTVIEW_OBS_KEY, WRIST_OBS_KEY],
    },
    "ph_left_low_close": {
        "path": Path("ph") / "square_ph_left_close_low_wrist_image_train_valid.hdf5",
        "camera_names": [LEFT_CLOSE_LOW_CAMERA_NAME, WRIST_CAMERA_NAME],
        "rgb_obs": [LEFT_CLOSE_LOW_OBS_KEY, WRIST_OBS_KEY],
    },
    "random_post": {
        "path": Path("random_post") / "expert200" / "image.hdf5",
        "camera_names": [AGENTVIEW_CAMERA_NAME, WRIST_CAMERA_NAME],
        "rgb_obs": [AGENTVIEW_OBS_KEY, WRIST_OBS_KEY],
    },
    "random_post_left_low_close": {
        "path": Path("random_post") / "expert200" / "image_left_close_low_wrist.hdf5",
        "camera_names": [LEFT_CLOSE_LOW_CAMERA_NAME, WRIST_CAMERA_NAME],
        "rgb_obs": [LEFT_CLOSE_LOW_OBS_KEY, WRIST_OBS_KEY],
    },
    "mh_left_low_close": {
        "path": Path("mh") / "image_left_close_low_wrist.hdf5",
        "camera_names": [LEFT_CLOSE_LOW_CAMERA_NAME, WRIST_CAMERA_NAME],
        "rgb_obs": [LEFT_CLOSE_LOW_OBS_KEY, WRIST_OBS_KEY],
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_dir", default="robomimic/exps/square/ph")
    parser.add_argument("--dataset_dir", default="robomimic/datasets/square")
    parser.add_argument("--output_dir", default="robomimic/exps/square/sweep")
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
        help="Include left_close_low_image as a training RGB observation. Only use with datasets that contain this key.",
    )
    return parser.parse_args()


def weight_decay_tag(weight_decay):
    if weight_decay == 0:
        return "wd0"
    return "wd{}".format("{:.0e}".format(weight_decay).replace("-", "m"))


def load_templates(template_dir):
    template_dir = Path(template_dir)
    return {
        "gmm": json.loads((template_dir / "bc_gmm_image.json").read_text()),
        "discrete": json.loads((template_dir / "bc_discrete_image.json").read_text()),
        "discrete_gaussian": json.loads((template_dir / "bc_discrete_image.json").read_text()),
    }


def set_dataset_camera_obs(config, dataset, use_left_close_low_obs):
    dataset_spec = DATASET_SPECS[dataset]

    rgb_obs = config["observation"]["modalities"]["obs"]["rgb"]
    rgb_obs[:] = list(dataset_spec["rgb_obs"])

    env_kwargs = config["experiment"]["env_meta_update_dict"].setdefault("env_kwargs", {})
    camera_names = env_kwargs.setdefault("camera_names", [])
    camera_names[:] = list(dataset_spec["camera_names"])

    if use_left_close_low_obs:
        if LEFT_CLOSE_LOW_OBS_KEY not in rgb_obs:
            rgb_obs.append(LEFT_CLOSE_LOW_OBS_KEY)
        if LEFT_CLOSE_LOW_CAMERA_NAME not in camera_names:
            camera_names.append(LEFT_CLOSE_LOW_CAMERA_NAME)


def make_config(template, dataset, policy, weight_decay, dataset_dir, use_left_close_low_obs):
    config = deepcopy(template)
    tag = weight_decay_tag(weight_decay)
    run_name = "bc_{}_square_{}_image_{}".format(policy, dataset, tag)

    config["experiment"]["name"] = run_name
    config["experiment"]["logging"]["wandb_proj_name"] = "pomdp_square"
    config["experiment"]["render_video"] = True
    config["experiment"]["rollout"]["n"] = 1
    config["train"]["data"] = str(Path(dataset_dir) / DATASET_SPECS[dataset]["path"])
    config["train"]["output_dir"] = "trained_models/square/sweep/{}/{}/{}".format(
        dataset, policy, tag)
    config["train"]["num_data_workers"] = 0
    config["algo"]["optim_params"]["policy"]["regularization"]["L2"] = weight_decay

    discrete_enabled = policy in ["discrete", "discrete_gaussian"]
    config["algo"]["gmm"]["enabled"] = (policy == "gmm")
    config["algo"]["discrete"]["enabled"] = discrete_enabled
    if discrete_enabled:
        config["algo"]["discrete"]["target_type"] = "gaussian" if policy == "discrete_gaussian" else "one_hot"
    config["algo"]["gaussian"]["enabled"] = False
    config["algo"]["vae"]["enabled"] = False
    config["algo"]["rnn"]["enabled"] = False
    config["algo"]["transformer"]["enabled"] = False
    set_dataset_camera_obs(config, dataset, use_left_close_low_obs)
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
