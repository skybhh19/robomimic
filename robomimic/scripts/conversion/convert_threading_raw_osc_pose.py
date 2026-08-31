import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_ENV_VERSION = "1.5.2"

COLLECTION_METADATA_KEYS = (
    "control_mode",
    "action_representation",
    "trajectory_timing_profile",
    "joint_action_noise_scale_rad",
    "max_lift_height_m",
)


def osc_controller_config(input_ref_frame):
    return {
        "type": "BASIC",
        "body_parts": {
            "right": {
                "type": "OSC_POSE",
                "input_max": 1,
                "input_min": -1,
                "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                "kp": 150,
                "damping_ratio": 1,
                "impedance_mode": "fixed",
                "kp_limits": [0, 300],
                "damping_ratio_limits": [0, 10],
                "position_limits": None,
                "orientation_limits": None,
                "uncouple_pos_ori": True,
                "input_type": "delta",
                "input_ref_frame": input_ref_frame,
                "interpolation": None,
                "ramp_ratio": 0.2,
                "gripper": {"type": "GRIP"},
            }
        },
    }


def fallback_env_info(env_name):
    input_ref_frame = "world" if env_name in ("Threading", "Threading_D1") else "base"
    return {
        "env_name": env_name,
        "robots": ["Panda"],
        "controller_configs": osc_controller_config(input_ref_frame),
    }


def validate_osc_env_info(env_info):
    controller_config = env_info["controller_configs"]
    body_parts = controller_config["body_parts"]
    osc_parts = [part for part in body_parts.values() if part["type"].startswith("OSC")]
    assert len(osc_parts) == 1, "expected one OSC arm controller, got {}".format(body_parts)
    arm = osc_parts[0]
    assert arm["type"] == "OSC_POSE", arm
    assert arm["input_type"] == "delta", arm
    assert arm["input_ref_frame"] in ("base", "world"), arm


def threading_env_args(env_name, env_info, env_version):
    assert env_info["env_name"] == env_name, (env_info["env_name"], env_name)
    validate_osc_env_info(env_info)
    return {
        "type": 1,
        "env_name": env_name,
        "env_kwargs": env_info,
        "env_version": env_version,
    }


def sanitize_model_xml(model_xml):
    robosuite_root = Path(__file__).resolve().parents[4] / "robosuite" / "robosuite"
    replacements = {
        "/Users/jasonyan/Desktop/robosuite-pomdp/robosuite": str(robosuite_root),
        "/home/soroushn/code/robosuite-dev/robosuite": str(robosuite_root),
        "/iris/u/jasonyan/repos/robosuite-pomdp/robosuite": str(robosuite_root),
    }
    for old, new in replacements.items():
        model_xml = model_xml.replace(old, new)
    return model_xml


def create_hdf5_filter_key(hdf5_path, demo_keys, key_name):
    with h5py.File(hdf5_path, "a") as f:
        key_path = "mask/{}".format(key_name)
        if key_path in f:
            del f[key_path]
        f[key_path] = np.array(demo_keys, dtype="S")


def split_train_val_from_hdf5(hdf5_path, val_ratio=0.1, seed=0):
    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"], key=lambda x: int(x.rsplit("_", 1)[1]))
    num_val = int(val_ratio * len(demos))
    mask = np.zeros(len(demos), dtype=np.int64)
    mask[:num_val] = 1
    rng = np.random.default_rng(seed)
    rng.shuffle(mask)
    valid_keys = [demos[i] for i in mask.nonzero()[0]]
    train_keys = [demos[i] for i in (1 - mask).nonzero()[0]]
    create_hdf5_filter_key(hdf5_path, train_keys, "train")
    create_hdf5_filter_key(hdf5_path, valid_keys, "valid")


def axis_angle_to_rot_6d(axis_angle):
    theta = np.linalg.norm(axis_angle, axis=1, keepdims=True)
    axis = np.divide(axis_angle, theta, out=np.zeros_like(axis_angle), where=theta > 0)
    x = axis[:, 0]
    y = axis[:, 1]
    z = axis[:, 2]
    cos = np.cos(theta[:, 0])
    sin = np.sin(theta[:, 0])
    one_minus_cos = 1 - cos
    rot = np.empty((axis_angle.shape[0], 3, 3), dtype=np.float32)
    rot[:, 0, 0] = cos + x * x * one_minus_cos
    rot[:, 0, 1] = x * y * one_minus_cos - z * sin
    rot[:, 0, 2] = x * z * one_minus_cos + y * sin
    rot[:, 1, 0] = y * x * one_minus_cos + z * sin
    rot[:, 1, 1] = cos + y * y * one_minus_cos
    rot[:, 1, 2] = y * z * one_minus_cos - x * sin
    rot[:, 2, 0] = z * x * one_minus_cos - y * sin
    rot[:, 2, 1] = z * y * one_minus_cos + x * sin
    rot[:, 2, 2] = cos + z * z * one_minus_cos
    return rot[:, :2, :].reshape(axis_angle.shape[0], 6)


def write_action_dict(ep_grp, actions):
    assert actions.ndim == 2 and actions.shape[1] == 7, actions.shape
    values = {
        "rel_pos": actions[:, :3].astype(np.float32),
        "rel_rot_axis_angle": actions[:, 3:6].astype(np.float32),
        "rel_rot_6d": axis_angle_to_rot_6d(actions[:, 3:6]).astype(np.float32),
        "gripper": actions[:, 6:7].astype(np.float32),
    }
    action_dict = ep_grp.require_group("action_dict")
    for key, value in values.items():
        action_dict.create_dataset(key, data=value)


def normalize_manifest_entry(row, idx):
    entry = dict(row)
    if "merged_idx" not in entry:
        if "dataset_index" in entry:
            entry["merged_idx"] = entry["dataset_index"]
        elif "demo_index" in entry:
            entry["merged_idx"] = entry["demo_index"]
        else:
            entry["merged_idx"] = idx
    if "merged_episode" not in entry:
        if "output_episode" in entry:
            entry["merged_episode"] = entry["output_episode"]
        else:
            entry["merged_episode"] = entry["episode"]
    if "label" not in entry:
        if "observability" in entry:
            entry["label"] = entry["observability"]
        elif "manual_label" in entry:
            entry["label"] = entry["manual_label"]
        else:
            entry["label"] = "unlabeled"
    return entry


def read_manifest(path):
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
    else:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    assert len(rows) > 0, path
    return [normalize_manifest_entry(row, idx) for idx, row in enumerate(rows)]


def find_labels_csv(raw_dir, labels_csv):
    if labels_csv is not None:
        path = Path(labels_csv)
        assert path.exists(), path
        return path
    candidates = (
        raw_dir / "observability_labels.csv",
        raw_dir.parent / "observability_labels.csv",
        raw_dir.parent / "dataset" / "observability_labels.csv",
    )
    return next((path for path in candidates if path.exists()), None)


def sorted_manifest_entries(raw_dir, labels_csv=None):
    manifest_names = (
        "merged_manifest.json",
        "selection_manifest.json",
        "merged_manifest.csv",
        "selection_manifest.csv",
        "annotations.csv",
    )
    manifest_path = next((raw_dir / name for name in manifest_names if (raw_dir / name).exists()), None)
    if manifest_path is not None:
        entries = read_manifest(manifest_path)
    else:
        labels_path = find_labels_csv(raw_dir, labels_csv)
        if labels_path is not None:
            entries = []
            with labels_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) > 0, labels_path
            for idx, row in enumerate(rows):
                assert "raw_episode" in row and "observability" in row, row
                entry = dict(row)
                entry["merged_episode"] = row["raw_episode"]
                entry["merged_idx"] = int(row["demo"].rsplit("_", 1)[1]) if "demo" in row else idx
                entry["demo_key"] = row["demo"] if "demo" in row else "demo_{}".format(idx)
                entry["label"] = row["observability"]
                entries.append(entry)
        else:
            episode_dirs = sorted(path for path in raw_dir.glob("ep_*") if path.is_dir())
            assert len(episode_dirs) > 0, "no episodes found in {}".format(raw_dir)
            entries = [
                {
                    "merged_idx": idx,
                    "merged_episode": path.name,
                    "label": "unlabeled",
                }
                for idx, path in enumerate(episode_dirs)
            ]
    entries = sorted(entries, key=lambda x: int(x["merged_idx"]))
    episodes = [entry["merged_episode"] for entry in entries]
    assert len(episodes) == len(set(episodes)), "duplicate episodes in manifest"
    return entries


def resolve_raw_dir(path):
    path = Path(path)
    if (path / "raw").is_dir():
        path = path / "raw"
    assert path.exists(), path
    assert any(candidate.is_dir() for candidate in path.glob("ep_*")), "no episode directories in {}".format(path)
    return path


def action_array(action_infos):
    actions = []
    for info in action_infos:
        assert "actions" in info, info
        actions.append(np.asarray(info["actions"], dtype=np.float64))
    actions = np.asarray(actions, dtype=np.float64)
    assert actions.ndim == 2 and actions.shape[1] == 7, "OSC_POSE actions must have shape (T, 7), got {}".format(
        actions.shape
    )
    return actions


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
    actions = action_array(action_infos)
    assert states_all.shape[0] == actions.shape[0] + 1, (ep_dir, states_all.shape, actions.shape)
    assert len(env_names) == 1, (ep_dir, env_names)

    stats_path = ep_dir / "policy_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    if "control_mode" in stats:
        assert stats["control_mode"] == "osc_pose", (ep_dir, stats["control_mode"])
    if "action_representation" in stats:
        assert stats["action_representation"] == "delta_eef_pose", (ep_dir, stats["action_representation"])

    model_path = ep_dir / "model.xml"
    assert model_path.exists(), model_path
    ep_meta_path = ep_dir / "ep_meta.json"
    ep_meta = json.loads(ep_meta_path.read_text()) if ep_meta_path.exists() else {}
    return {
        "states": states_all[:-1],
        "actions": actions,
        "env_name": next(iter(env_names)),
        "model_xml": model_path.read_text(),
        "ep_meta": ep_meta,
        "stats": stats,
    }


def source_environment_metadata(raw_dir):
    candidates = (
        raw_dir / "demo.hdf5",
        raw_dir.parent / "dataset" / "demo.hdf5",
    )
    source_hdf5 = next((path for path in candidates if path.exists()), None)
    if source_hdf5 is None:
        return None
    with h5py.File(source_hdf5, "r") as f:
        attrs = f["data"].attrs
        if "env_info" in attrs:
            return {
                "env_name": str(attrs["env"]),
                "env_info": json.loads(attrs["env_info"]),
                "env_version": str(attrs["repository_version"]),
            }
        if "env_args" in attrs:
            env_args = json.loads(attrs["env_args"])
            return {
                "env_name": env_args["env_name"],
                "env_info": env_args["env_kwargs"],
                "env_version": env_args["env_version"],
            }
    return None


def write_label_masks(output_path, label_by_demo):
    for label in sorted(set(label_by_demo.values())):
        demo_keys = [demo for demo, demo_label in label_by_demo.items() if demo_label == label]
        create_hdf5_filter_key(output_path, demo_keys, label)


def set_optional_demo_attrs(ep_grp, entry, stats):
    values = {
        "target_grasp_angle_deg": entry.get("target_grasp_angle_deg", stats.get("target_grasp_angle_deg")),
        "initial_state_id": entry.get("initial_state_id", stats.get("initial_state_sampling", {}).get("state_id")),
        "retry_index": entry.get("retry_index", stats.get("initial_state_sampling", {}).get("retry_index")),
    }
    if values["target_grasp_angle_deg"] not in (None, ""):
        ep_grp.attrs["target_grasp_angle_deg"] = float(values["target_grasp_angle_deg"])
    if values["initial_state_id"] not in (None, ""):
        ep_grp.attrs["initial_state_id"] = str(values["initial_state_id"])
    if values["retry_index"] not in (None, ""):
        ep_grp.attrs["retry_index"] = int(values["retry_index"])


def convert_threading_raw_osc_pose(raw_dir, output_path, labels_csv=None, val_ratio=0.1, split_seed=0):
    assert 0.0 <= val_ratio < 1.0, val_ratio
    raw_dir = resolve_raw_dir(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted_manifest_entries(raw_dir, labels_csv=labels_csv)

    episodes = []
    for entry in entries:
        ep_dir = raw_dir / entry["merged_episode"]
        if not ep_dir.exists():
            ep_dir = raw_dir / "episodes" / entry["merged_episode"]
        assert ep_dir.exists(), ep_dir
        episodes.append((entry, ep_dir, load_raw_episode(ep_dir)))

    env_names = {episode["env_name"] for _, _, episode in episodes}
    assert len(env_names) == 1, env_names
    env_name = next(iter(env_names))
    source_metadata = source_environment_metadata(raw_dir)
    if source_metadata is None:
        env_info = fallback_env_info(env_name)
        env_version = DEFAULT_ENV_VERSION
    else:
        assert source_metadata["env_name"] == env_name, (source_metadata["env_name"], env_name)
        source_env_info = source_metadata["env_info"]
        collection_metadata = {
            key: source_env_info[key] for key in COLLECTION_METADATA_KEYS if key in source_env_info
        }
        env_info = {key: value for key, value in source_env_info.items() if key not in COLLECTION_METADATA_KEYS}
        env_version = source_metadata["env_version"]
    env_args = threading_env_args(env_name, env_info, env_version)

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
        data_grp.attrs["action_representation"] = "delta_eef_pose"
        if source_metadata is not None:
            data_grp.attrs["collection_metadata"] = json.dumps(collection_metadata)

        used_demo_keys = set()
        for demo_idx, (entry, ep_dir, episode) in enumerate(episodes):
            demo_key = entry["demo_key"] if "demo_key" in entry else "demo_{}".format(demo_idx)
            assert demo_key not in used_demo_keys, demo_key
            used_demo_keys.add(demo_key)
            ep_grp = data_grp.create_group(demo_key)
            ep_grp.create_dataset("actions", data=episode["actions"])
            ep_grp.create_dataset("states", data=episode["states"])
            write_action_dict(ep_grp, episode["actions"])
            ep_grp.attrs["model_file"] = sanitize_model_xml(episode["model_xml"])
            ep_grp.attrs["num_samples"] = episode["actions"].shape[0]
            ep_grp.attrs["ep_meta"] = json.dumps(episode["ep_meta"], indent=4)
            ep_grp.attrs["source_episode"] = ep_dir.name
            ep_grp.attrs["source_metadata"] = json.dumps(entry, indent=4)
            ep_grp.attrs["observability"] = entry["label"]
            set_optional_demo_attrs(ep_grp, entry, episode["stats"])
            total += episode["actions"].shape[0]
            label_by_demo[demo_key] = entry["label"]

        data_grp.attrs["total"] = total

    split_train_val_from_hdf5(output_path, val_ratio=val_ratio, seed=split_seed)
    write_label_masks(output_path, label_by_demo)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="threading_raw_data")
    parser.add_argument("--output", type=str, default="datasets/threading/demo_osc_pose_v15.hdf5")
    parser.add_argument("--labels_csv", type=str)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=0)
    args = parser.parse_args()
    output_path = convert_threading_raw_osc_pose(
        raw_dir=args.raw_dir,
        output_path=args.output,
        labels_csv=args.labels_csv,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
    )
    print("Wrote {}".format(output_path))


if __name__ == "__main__":
    main()
