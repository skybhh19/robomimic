"""
Generate Square visual BC sweep configs.

Example:
    python robomimic/scripts/generate_square_bc_sweep_configs.py

    python robomimic/scripts/generate_square_bc_sweep_configs.py \
        --weight_decays 0 1e-4

    python robomimic/scripts/generate_square_bc_sweep_configs.py \
        --use_left_close_low_obs

    python robomimic/scripts/generate_square_bc_sweep_configs.py \
        --camera_images_only

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
DEFAULT_OUTPUT_DIR = "robomimic/exps/square/sweep"
IMAGE_ONLY_OUTPUT_DIR = "robomimic/exps/square/sweep_image_only"
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
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
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
    parser.add_argument(
        "--camera_images_only",
        action="store_true",
        help="Train from camera image observations only, with no low-dimensional robot state inputs.",
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


def set_camera_images_only_obs(config):
    config["observation"]["modalities"]["obs"]["low_dim"] = []


def make_config(
    template,
    dataset,
    policy,
    weight_decay,
    dataset_dir,
    use_left_close_low_obs,
    camera_images_only,
):
    config = deepcopy(template)
    tag = weight_decay_tag(weight_decay)
    obs_tag = "camera_image_only" if camera_images_only else "image"
    run_name = "bc_{}_square_{}_{}_{}".format(policy, dataset, obs_tag, tag)

    config["experiment"]["name"] = run_name
    config["experiment"]["logging"]["wandb_proj_name"] = "pomdp_square_1"
    config["experiment"]["render_video"] = True
    config["experiment"]["validate"] = False
    config["experiment"]["save"]["on_best_validation"] = False
    config["experiment"]["rollout"]["enabled"] = False
    config["experiment"]["rollout"]["n"] = 50
    config["experiment"]["rollout"]["warmstart"] = 100
    config["train"]["data"] = str(Path(dataset_dir) / DATASET_SPECS[dataset]["path"])
    config["train"]["hdf5_filter_key"] = None
    config["train"]["hdf5_validation_filter_key"] = None
    train_output_root = "sweep_image_only" if camera_images_only else "sweep"
    config["train"]["output_dir"] = "trained_models/square/{}/{}/{}/{}".format(
        train_output_root, dataset, policy, tag)
    config["train"]["num_data_workers"] = 0
    config["train"]["hdf5_cache_mode"] = "all"
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
    if camera_images_only:
        set_camera_images_only_obs(config)
    return config


def main():
    args = parse_args()
    templates = load_templates(args.template_dir)
    output_dir = Path(args.output_dir)
    if args.camera_images_only and args.output_dir == DEFAULT_OUTPUT_DIR:
        output_dir = Path(IMAGE_ONLY_OUTPUT_DIR)
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
