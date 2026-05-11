"""
Render per-episode Square dataset videos from image observations.

This writes the mp4 layout consumed by vis/score_vis/build_manifest.py:

    vis/dataset_videos/videos/square/<slug>/square_<slug>_<split>_ep0000_agent.mp4

Example:
    python robomimic/scripts/render_square_dataset_videos.py \
        --dataset robomimic/datasets/square/random_post/expert200/image_left_close_low_wrist.hdf5 \
        --slug random_post \
        --views agent wrist left_close_low

    python robomimic/scripts/render_square_dataset_videos.py \
        --dataset robomimic/datasets/square/mh/image_left_close_low_wrist.hdf5 \
        --slug mh \
        --views left_close_low
"""

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import h5py
import imageio
import numpy as np
from tqdm import tqdm


VIEW_OBS_KEYS = {
    "agent": ("agentview_image",),
    "wrist": ("robot0_eye_in_hand_image",),
    "left_close_low": ("left_close_low_image",),
    "wrist_agent": ("robot0_eye_in_hand_image", "agentview_image"),
    "wrist_left_close_low": ("robot0_eye_in_hand_image", "left_close_low_image"),
}


def natural_key(name):
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", name)]


def demo_index_from_key(demo_key):
    match = re.search(r"(\d+)$", demo_key)
    return int(match.group(1)) if match is not None else demo_key


def get_demo_keys(hdf5_file, filter_key):
    if filter_key is None:
        keys = list(hdf5_file["data"].keys())
    else:
        keys = [
            elem.decode("utf-8")
            for elem in np.asarray(hdf5_file["mask/{}".format(filter_key)][:])
        ]
    return sorted(keys, key=natural_key)


def split_for_demo(hdf5_file, demo_key, valid_keys):
    if demo_key in valid_keys:
        return "val"
    return "train"


def load_valid_keys(hdf5_file):
    if "mask/valid" not in hdf5_file:
        return set()
    return {
        elem.decode("utf-8")
        for elem in np.asarray(hdf5_file["mask/valid"][:])
    }


def render_episode(hdf5_file, demo_key, obs_keys, output_path, fps, overwrite):
    if output_path.exists() and not overwrite:
        return False

    frames = []
    for obs_key in obs_keys:
        dataset_path = "data/{}/obs/{}".format(demo_key, obs_key)
        if dataset_path not in hdf5_file:
            raise KeyError("{} not found in dataset".format(dataset_path))
        frames.append(hdf5_file[dataset_path])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps, macro_block_size=1) as writer:
        for frame_index in range(frames[0].shape[0]):
            frame = np.concatenate(
                [np.asarray(frames_i[frame_index], dtype=np.uint8) for frames_i in frames],
                axis=1,
            )
            writer.append_data(frame)
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Render Square dataset videos from HDF5 image observations.")
    parser.add_argument("--dataset", required=True, help="Input HDF5 dataset with image observations.")
    parser.add_argument("--slug", required=True, choices=["ph", "mh", "random_post"], help="Output dataset slug.")
    parser.add_argument(
        "--views",
        nargs="+",
        default=["agent", "wrist"],
        choices=sorted(VIEW_OBS_KEYS),
        help="Video views to render.",
    )
    parser.add_argument("--output-root", default="vis/dataset_videos/videos/square", help="Output root.")
    parser.add_argument("--filter-key", default=None, help="Optional HDF5 mask key to render.")
    parser.add_argument("--num-demos", type=int, default=None, help="Optional cap on rendered demos.")
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing mp4 files.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(os.path.expanduser(args.dataset))
    output_dir = Path(args.output_root) / args.slug
    rendered = 0
    skipped = 0

    with h5py.File(dataset_path, "r") as hdf5_file:
        demo_keys = get_demo_keys(hdf5_file, args.filter_key)
        if args.num_demos is not None:
            demo_keys = demo_keys[: args.num_demos]
        valid_keys = load_valid_keys(hdf5_file)

        for demo_key in tqdm(demo_keys, desc="Rendering dataset videos"):
            split = split_for_demo(hdf5_file, demo_key, valid_keys)
            episode = demo_index_from_key(demo_key)
            for view in args.views:
                output_path = output_dir / "square_{}_{}_ep{:04d}_{}.mp4".format(
                    args.slug,
                    split,
                    int(episode),
                    view,
                )
                did_render = render_episode(
                    hdf5_file=hdf5_file,
                    demo_key=demo_key,
                    obs_keys=VIEW_OBS_KEYS[view],
                    output_path=output_path,
                    fps=args.fps,
                    overwrite=args.overwrite,
                )
                rendered += int(did_render)
                skipped += int(not did_render)

    print("Rendered {} videos; skipped {} existing videos in {}".format(rendered, skipped, output_dir))


if __name__ == "__main__":
    main()
