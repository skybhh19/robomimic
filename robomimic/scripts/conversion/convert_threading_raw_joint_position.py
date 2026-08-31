import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from convert_threading_raw_osc_pose import (
    DEFAULT_ENV_VERSION,
    resolve_raw_dir,
    sanitize_model_xml,
    set_optional_demo_attrs,
    sorted_manifest_entries,
    source_environment_metadata,
    split_train_val_from_hdf5,
    write_label_masks,
)


ACTION_INFO_KEYS = (
    "actions",
    "absolute_joint_target",
    "actions_absolute_joint_position",
    "actions_joint_delta",
    "joint_delta",
    "joint_delta_exceeds_reference_scale",
    "joint_delta_reference_scaled",
    "joint_delta_scale",
    "joint_position",
    "robot0_joint_pos",
)

COLLECTION_METADATA_KEYS = (
    "control_mode",
    "action_representation",
    "trajectory_timing_profile",
    "joint_action_noise_scale_rad",
    "max_lift_height_m",
)


def validate_joint_env_info(env_info):
    assert env_info["control_mode"] == "joint_position", env_info
    assert env_info["action_representation"] == "absolute_joint_position", env_info
    body_parts = env_info["controller_configs"]["body_parts"]
    joint_parts = [part for part in body_parts.values() if part["type"] == "JOINT_POSITION"]
    assert len(joint_parts) == 1, body_parts
    assert joint_parts[0]["input_type"] == "absolute", joint_parts[0]


def action_arrays(action_infos):
    assert all(set(ACTION_INFO_KEYS).issubset(info) for info in action_infos)
    arrays = {key: np.asarray([info[key] for info in action_infos]) for key in ACTION_INFO_KEYS}
    actions = arrays["actions"]
    assert actions.ndim == 2 and actions.shape[1] == 8, actions.shape
    assert np.array_equal(actions, arrays["actions_absolute_joint_position"])
    assert np.allclose(actions[:, :7], arrays["absolute_joint_target"])
    return arrays


def load_raw_episode(ep_dir):
    npz_paths = sorted(ep_dir.glob("state_*.npz"))
    assert len(npz_paths) > 0, ep_dir
    state_chunks = []
    action_infos = []
    env_names = set()
    for npz_path in npz_paths:
        with np.load(npz_path, allow_pickle=True) as raw:
            state_chunks.append(np.asarray(raw["states"], dtype=np.float64))
            action_infos.extend(raw["action_infos"])
            env_names.add(str(raw["env"]))
    states_all = np.concatenate(state_chunks, axis=0)
    arrays = action_arrays(action_infos)
    assert states_all.shape[0] == arrays["actions"].shape[0] + 1, (
        ep_dir,
        states_all.shape,
        arrays["actions"].shape,
    )
    assert len(env_names) == 1, (ep_dir, env_names)

    stats_path = ep_dir / "policy_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    assert stats["control_mode"] == "joint_position", (ep_dir, stats)
    assert stats["action_representation"] == "absolute_joint_position", (ep_dir, stats)
    model_path = ep_dir / "model.xml"
    assert model_path.exists(), model_path
    ep_meta_path = ep_dir / "ep_meta.json"
    ep_meta = json.loads(ep_meta_path.read_text()) if ep_meta_path.exists() else {}
    return {
        "states": states_all[:-1],
        "action_arrays": arrays,
        "env_name": next(iter(env_names)),
        "model_xml": model_path.read_text(),
        "ep_meta": ep_meta,
        "stats": stats,
    }


def convert_threading_raw_joint_position(raw_dir, output_path, labels_csv=None, val_ratio=0.1, split_seed=0):
    assert 0.0 <= val_ratio < 1.0, val_ratio
    raw_dir = resolve_raw_dir(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted_manifest_entries(raw_dir, labels_csv=labels_csv)
    episodes = []
    for entry in entries:
        ep_dir = raw_dir / entry["merged_episode"]
        assert ep_dir.exists(), ep_dir
        episodes.append((entry, ep_dir, load_raw_episode(ep_dir)))

    env_names = {episode["env_name"] for _, _, episode in episodes}
    assert len(env_names) == 1, env_names
    env_name = next(iter(env_names))
    source_metadata = source_environment_metadata(raw_dir)
    assert source_metadata is not None, "joint-position conversion requires source environment metadata"
    assert source_metadata["env_name"] == env_name, (source_metadata["env_name"], env_name)
    source_env_info = source_metadata["env_info"]
    collection_metadata = {key: source_env_info[key] for key in COLLECTION_METADATA_KEYS}
    env_info = {key: value for key, value in source_env_info.items() if key not in COLLECTION_METADATA_KEYS}
    env_version = source_metadata["env_version"] or DEFAULT_ENV_VERSION
    validate_joint_env_info({**env_info, **collection_metadata})
    env_args = {
        "type": 1,
        "env_name": env_name,
        "env_kwargs": env_info,
        "env_version": env_version,
    }

    total = 0
    label_by_demo = {}
    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        data_grp.attrs["env_args"] = json.dumps(env_args, indent=4)
        data_grp.attrs["env"] = env_name
        data_grp.attrs["env_info"] = json.dumps(env_info)
        data_grp.attrs["repository_version"] = env_version
        data_grp.attrs["num_demos"] = len(episodes)
        data_grp.attrs["source_dataset_root"] = str(raw_dir)
        data_grp.attrs["action_representation"] = "absolute_joint_position"
        data_grp.attrs["collection_metadata"] = json.dumps(collection_metadata)

        for demo_idx, (entry, ep_dir, episode) in enumerate(episodes):
            demo_key = entry["demo_key"] if "demo_key" in entry else "demo_{}".format(demo_idx)
            ep_grp = data_grp.create_group(demo_key)
            for key, value in episode["action_arrays"].items():
                ep_grp.create_dataset(key, data=value)
            ep_grp.create_dataset("states", data=episode["states"])
            ep_grp.attrs["model_file"] = sanitize_model_xml(episode["model_xml"])
            ep_grp.attrs["num_samples"] = episode["action_arrays"]["actions"].shape[0]
            ep_grp.attrs["ep_meta"] = json.dumps(episode["ep_meta"], indent=4)
            ep_grp.attrs["source_episode"] = ep_dir.name
            ep_grp.attrs["source_metadata"] = json.dumps(entry, indent=4)
            ep_grp.attrs["observability"] = entry["label"]
            ep_grp.attrs["action_representation"] = "absolute_joint_position"
            set_optional_demo_attrs(ep_grp, entry, episode["stats"])
            total += ep_grp.attrs["num_samples"]
            label_by_demo[demo_key] = entry["label"]
        data_grp.attrs["total"] = total

    split_train_val_from_hdf5(output_path, val_ratio=val_ratio, seed=split_seed)
    write_label_masks(output_path, label_by_demo)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels_csv")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=0)
    args = parser.parse_args()
    output_path = convert_threading_raw_joint_position(
        raw_dir=args.raw_dir,
        output_path=args.output,
        labels_csv=args.labels_csv,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
    )
    print("Wrote {}".format(output_path))


if __name__ == "__main__":
    main()
