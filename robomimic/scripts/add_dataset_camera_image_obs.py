"""
Append additional robosuite camera image observations to an existing robomimic
image dataset.

This is intentionally narrower than dataset_states_to_obs.py: it preserves the
existing dataset and only writes missing camera image keys under obs/ and
next_obs/.
"""

import argparse
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy

import h5py
import numpy as np
from tqdm import tqdm

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robosuite.macros as macros
from robomimic.scripts.dataset_states_to_obs import get_camera_info, get_env_metadata_from_dataset
from robosuite.environments.manipulation.nut_assembly import SQUARE_CAMERA_SPECS
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING, array_to_string


EXTRA_CAMERA_SPECS = {
    "left_close_low": {
        "mode": "fixed",
        "pos": "0.42205740 -0.23999999 1.15230719",
        "quat": "0.81392215 0.36066498 0.18452251 0.41641680",
    },
    "shouldercamera0_randomview6": {
        "mode": "fixed",
        "pos": "0.53090359 -1.05271819 1.39209263",
        "quat": "0.83644916 0.49136112 0.12294401 0.20928887",
    },
    "shouldercamera1_randomview6": {
        "mode": "fixed",
        "pos": "-0.44163169 0.91721496 1.34016649",
        "quat": "-0.19105282 -0.10298015 0.46316663 0.85928493",
    },
    **{
        camera_name: {"mode": "fixed", "pos": array_to_string(pos), "quat": array_to_string(quat)}
        for camera_name, (pos, quat) in SQUARE_CAMERA_SPECS.items()
    },
}


def _decode_xml(xml):
    return xml.decode("utf-8") if isinstance(xml, bytes) else xml


def inject_camera_xml(xml, camera_names):
    """Add fixed cameras to stored episode XML when the old dataset lacks them."""
    xml = _decode_xml(xml)
    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    if worldbody is None:
        return xml

    for camera_name in camera_names:
        spec = EXTRA_CAMERA_SPECS.get(camera_name)
        if spec is None:
            continue
        if worldbody.find("./camera[@name='{}']".format(camera_name)) is None:
            worldbody.append(ET.Element("camera", attrib={"name": camera_name, **spec}))

    return ET.tostring(root, encoding="utf8").decode("utf8")


def inject_alias_camera_xml(xml, camera_sources):
    xml = _decode_xml(xml)
    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    assert worldbody is not None, xml[:200]
    for camera_name, source_camera in camera_sources.items():
        pos, quat = SQUARE_CAMERA_SPECS[source_camera]
        existing = worldbody.find("./camera[@name='{}']".format(camera_name))
        spec = {
            "mode": "fixed",
            "name": camera_name,
            "pos": array_to_string(pos),
            "quat": array_to_string(quat),
        }
        if existing is None:
            worldbody.append(ET.Element("camera", attrib=spec))
        else:
            for key, value in spec.items():
                existing.set(key, value)
    return ET.tostring(root, encoding="utf8").decode("utf8")


def sorted_demos(data_group):
    demos = [k for k in data_group.keys() if re.match(r"demo_\d+$", k)]
    return sorted(demos, key=lambda x: int(x[5:]))


def append_global_camera_metadata(dataset_path, camera_names, camera_height, camera_width):
    with h5py.File(dataset_path, "r+") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        env_camera_names = env_args.setdefault("env_kwargs", {}).get("camera_names", [])
        if isinstance(env_camera_names, str):
            env_camera_names = [env_camera_names]
        for camera_name in camera_names:
            if camera_name not in env_camera_names:
                env_camera_names.append(camera_name)
        env_args["env_kwargs"]["camera_names"] = env_camera_names
        env_args["env_kwargs"]["camera_heights"] = camera_height
        env_args["env_kwargs"]["camera_widths"] = camera_width
        f["data"].attrs["env_args"] = json.dumps(env_args, indent=4)


def camera_keys_present(traj_group, camera_names):
    for camera_name in camera_names:
        key = "{}_image".format(camera_name)
        if "obs/{}".format(key) not in traj_group or "next_obs/{}".format(key) not in traj_group:
            return False
    return True


def write_camera_dataset(group, key, data, overwrite=False):
    if key in group:
        if not overwrite:
            return
        del group[key]
    group.create_dataset(key, data=data)


def random_source_keys_present(traj_group, output_camera_names):
    for camera_name in output_camera_names:
        key = "{}_image".format(camera_name)
        if "obs/{}".format(key) not in traj_group or "next_obs/{}".format(key) not in traj_group:
            return False
        if "{}_source_camera".format(camera_name) not in traj_group.attrs:
            return False
    return True


def sample_camera_sources(rng, source_camera_names, output_camera_names):
    sampled = rng.choice(source_camera_names, size=len(output_camera_names), replace=False)
    return dict(zip(output_camera_names, sampled.tolist()))


def render_camera_image(env, camera_name, camera_height, camera_width):
    image = env.env.sim.render(camera_name=camera_name, width=camera_width, height=camera_height)
    return image[:: IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]]


def append_random_source_metadata(dataset_path, output_camera_names, source_camera_names, camera_height, camera_width, seed):
    with h5py.File(dataset_path, "r+") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        env_camera_names = env_args.setdefault("env_kwargs", {}).get("camera_names", [])
        if isinstance(env_camera_names, str):
            env_camera_names = [env_camera_names]
        for camera_name in output_camera_names:
            if camera_name not in env_camera_names:
                env_camera_names.append(camera_name)
        env_args["env_kwargs"]["camera_names"] = env_camera_names
        env_args["env_kwargs"]["camera_heights"] = camera_height
        env_args["env_kwargs"]["camera_widths"] = camera_width
        f["data"].attrs["env_args"] = json.dumps(env_args, indent=4)
        f["data"].attrs["thirdperson_random_camera_sampling"] = json.dumps(
            {
                "output_camera_names": output_camera_names,
                "source_camera_names": source_camera_names,
                "seed": seed,
            },
            indent=4,
        )


def augment_random_source_dataset(args):
    dataset_path = os.path.expanduser(args.dataset)
    output_path = os.path.expanduser(args.output) if args.output is not None else None
    if output_path is not None:
        if os.path.abspath(output_path) != os.path.abspath(dataset_path):
            if os.path.exists(output_path) and not args.overwrite_output:
                raise FileExistsError("output exists; pass --overwrite-output: {}".format(output_path))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(dataset_path, output_path)
            dataset_path = output_path

    output_camera_names = args.random_output_camera_names
    source_camera_names = args.random_source_camera_names
    assert len(set(output_camera_names)) == len(output_camera_names), output_camera_names
    assert len(set(source_camera_names)) == len(source_camera_names), source_camera_names
    assert len(source_camera_names) >= len(output_camera_names), (source_camera_names, output_camera_names)
    for camera_name in source_camera_names:
        assert camera_name in SQUARE_CAMERA_SPECS, camera_name

    env_meta = get_env_metadata_from_dataset(dataset_path=dataset_path)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=[],
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=False,
        render_offscreen=True,
        use_image_obs=False,
    )

    append_random_source_metadata(
        dataset_path=dataset_path,
        output_camera_names=output_camera_names,
        source_camera_names=source_camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    with h5py.File(dataset_path, "r+") as f:
        demos = sorted_demos(f["data"])
        if args.n is not None:
            demos = demos[: args.n]

        for demo in tqdm(demos):
            camera_sources = sample_camera_sources(rng, source_camera_names, output_camera_names)
            traj_group = f["data/{}".format(demo)]
            if random_source_keys_present(traj_group, output_camera_names) and not args.overwrite:
                continue
            present = [
                "{}/{}_image".format(group_name, camera_name)
                for camera_name in output_camera_names
                for group_name in ("obs", "next_obs")
                if "{}/{}_image".format(group_name, camera_name) in traj_group
            ]
            assert args.overwrite or len(present) == 0, present

            states = traj_group["states"][()]
            actions = traj_group["actions"][()]
            initial_state = {"states": states[0]}
            if EnvUtils.is_robosuite_env(env_meta):
                model_xml = inject_alias_camera_xml(traj_group.attrs["model_file"], camera_sources)
                initial_state["model"] = model_xml
                initial_state["ep_meta"] = traj_group.attrs.get("ep_meta", None)

            obs = env.reset_to(initial_state)
            if EnvUtils.is_robosuite_env(env_meta):
                traj_group.attrs["model_file"] = model_xml
            camera_info = get_camera_info(
                env=env,
                camera_names=output_camera_names,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
            )
            existing_info = {}
            if "camera_info" in traj_group.attrs:
                existing_info = json.loads(traj_group.attrs["camera_info"])
            existing_info.update(camera_info)
            traj_group.attrs["camera_info"] = json.dumps(existing_info, indent=4)
            for camera_name, source_camera in camera_sources.items():
                traj_group.attrs["{}_source_camera".format(camera_name)] = source_camera

            obs_images = {camera_name: [] for camera_name in output_camera_names}
            next_obs_images = {camera_name: [] for camera_name in output_camera_names}

            for t in range(states.shape[0]):
                for camera_name in output_camera_names:
                    obs_images[camera_name].append(
                        render_camera_image(env, camera_name, args.camera_height, args.camera_width)
                    )

                if t == states.shape[0] - 1:
                    next_obs, _, _, _ = env.step(actions[t])
                else:
                    next_obs = env.reset_to({"states": states[t + 1]})

                for camera_name in output_camera_names:
                    next_obs_images[camera_name].append(
                        render_camera_image(env, camera_name, args.camera_height, args.camera_width)
                    )
                obs = next_obs

            for camera_name in output_camera_names:
                key = "{}_image".format(camera_name)
                write_camera_dataset(
                    traj_group["obs"],
                    key,
                    np.asarray(obs_images[camera_name], dtype=np.uint8),
                    overwrite=args.overwrite,
                )
                write_camera_dataset(
                    traj_group["next_obs"],
                    key,
                    np.asarray(next_obs_images[camera_name], dtype=np.uint8),
                    overwrite=args.overwrite,
                )

    print("Augmented dataset: {}".format(dataset_path))


def augment_dataset(args):
    if args.random_source_camera_names is not None:
        augment_random_source_dataset(args)
        return

    dataset_path = os.path.expanduser(args.dataset)
    output_path = os.path.expanduser(args.output) if args.output is not None else None
    if output_path is not None:
        if os.path.abspath(output_path) != os.path.abspath(dataset_path):
            if os.path.exists(output_path) and not args.overwrite_output:
                raise FileExistsError("output exists; pass --overwrite-output: {}".format(output_path))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(dataset_path, output_path)
            dataset_path = output_path

    env_meta = get_env_metadata_from_dataset(dataset_path=dataset_path)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=False,
    )

    append_global_camera_metadata(
        dataset_path=dataset_path,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
    )

    with h5py.File(dataset_path, "r+") as f:
        demos = sorted_demos(f["data"])
        if args.n is not None:
            demos = demos[: args.n]

        for demo in tqdm(demos):
            traj_group = f["data/{}".format(demo)]
            if camera_keys_present(traj_group, args.camera_names) and not args.overwrite:
                continue

            states = traj_group["states"][()]
            actions = traj_group["actions"][()]
            initial_state = {"states": states[0]}
            if EnvUtils.is_robosuite_env(env_meta):
                model_xml = inject_camera_xml(traj_group.attrs["model_file"], args.camera_names)
                initial_state["model"] = model_xml
                initial_state["ep_meta"] = traj_group.attrs.get("ep_meta", None)

            obs = env.reset_to(initial_state)
            if EnvUtils.is_robosuite_env(env_meta):
                traj_group.attrs["model_file"] = model_xml
            camera_info = get_camera_info(
                env=env,
                camera_names=args.camera_names,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
            )
            existing_info = {}
            if "camera_info" in traj_group.attrs:
                existing_info = json.loads(traj_group.attrs["camera_info"])
            existing_info.update(camera_info)
            traj_group.attrs["camera_info"] = json.dumps(existing_info, indent=4)

            obs_images = {camera_name: [] for camera_name in args.camera_names}
            next_obs_images = {camera_name: [] for camera_name in args.camera_names}

            for t in range(states.shape[0]):
                for camera_name in args.camera_names:
                    obs_images[camera_name].append(deepcopy(obs["{}_image".format(camera_name)]))

                if t == states.shape[0] - 1:
                    next_obs, _, _, _ = env.step(actions[t])
                else:
                    next_obs = env.reset_to({"states": states[t + 1]})

                for camera_name in args.camera_names:
                    next_obs_images[camera_name].append(deepcopy(next_obs["{}_image".format(camera_name)]))
                obs = next_obs

            for camera_name in args.camera_names:
                key = "{}_image".format(camera_name)
                write_camera_dataset(
                    traj_group["obs"],
                    key,
                    np.asarray(obs_images[camera_name], dtype=np.uint8),
                    overwrite=args.overwrite,
                )
                write_camera_dataset(
                    traj_group["next_obs"],
                    key,
                    np.asarray(next_obs_images[camera_name], dtype=np.uint8),
                    overwrite=args.overwrite,
                )

    print("Augmented dataset: {}".format(dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Existing HDF5 dataset to augment.")
    parser.add_argument("--output", default=None, help="Optional copy to write before augmenting.")
    parser.add_argument("--overwrite-output", action="store_true", help="Overwrite --output if it exists.")
    parser.add_argument("--camera_names", nargs="+", default=["left_close_low", "right_high"])
    parser.add_argument("--camera_height", type=int, default=84)
    parser.add_argument("--camera_width", type=int, default=84)
    parser.add_argument("--n", type=int, default=None, help="Only process first n demos, for debugging.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite camera datasets if present.")
    parser.add_argument("--random_source_camera_names", nargs="+", default=None)
    parser.add_argument("--random_output_camera_names", nargs="+", default=["thirdperson_1", "thirdperson_2"])
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs={
            "obs": {
                "low_dim": [],
                "rgb": [
                    "{}_image".format(camera_name)
                    for camera_name in (
                        args.random_output_camera_names
                        if args.random_source_camera_names is not None
                        else args.camera_names
                    )
                ],
            }
        }
    )
    augment_dataset(args)
