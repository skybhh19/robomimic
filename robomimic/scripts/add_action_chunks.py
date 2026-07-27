"""
Add flattened future-action chunks to a robomimic HDF5 dataset.

This is useful for training density models for p(a_{t:t+H-1} | s_t) with
standard BC-GMM configs: set train.action_keys to the generated dataset key.

Example:
    python robomimic/scripts/add_action_chunks.py \
        --dataset datasets/square/ph/low_dim_v15.hdf5 \
        --horizons 10
"""

import argparse

import h5py
import numpy as np


def make_action_chunks(actions, horizon, pad_mode):
    if actions.ndim != 2:
        raise ValueError("Expected actions to have shape (T, D), got {}".format(actions.shape))
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    traj_len, action_dim = actions.shape
    chunks = np.zeros((traj_len, horizon, action_dim), dtype=np.float32)
    for t in range(traj_len):
        end = min(traj_len, t + horizon)
        valid_len = end - t
        chunks[t, :valid_len] = actions[t:end]
        if valid_len < horizon and pad_mode == "repeat_last":
            chunks[t, valid_len:] = actions[-1]
    return chunks.reshape(traj_len, horizon * action_dim)


def add_action_chunks(dataset, horizons, action_key, output_prefix, pad_mode, overwrite):
    with h5py.File(dataset, "a") as f:
        for demo_key in sorted(f["data"].keys()):
            demo = f["data"][demo_key]
            actions = demo[action_key][()].astype(np.float32)
            for horizon in horizons:
                chunk_key = "{}{}".format(output_prefix, horizon)
                if chunk_key in demo:
                    if not overwrite:
                        print("{} already has {}; leaving it in place".format(demo_key, chunk_key))
                        continue
                    del demo[chunk_key]
                chunks = make_action_chunks(
                    actions=actions,
                    horizon=horizon,
                    pad_mode=pad_mode,
                )
                demo.create_dataset(chunk_key, data=chunks, compression="gzip")
                print("{}: wrote {} with shape {}".format(demo_key, chunk_key, chunks.shape))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to the HDF5 dataset to update.")
    parser.add_argument("--horizons", nargs="+", type=int, default=[10])
    parser.add_argument("--action_key", default="actions")
    parser.add_argument("--output_prefix", default="action_chunks_h")
    parser.add_argument(
        "--pad_mode",
        choices=["repeat_last", "zero"],
        default="repeat_last",
        help="How to fill chunk elements that run past the end of a trajectory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    add_action_chunks(
        dataset=args.dataset,
        horizons=args.horizons,
        action_key=args.action_key,
        output_prefix=args.output_prefix,
        pad_mode=args.pad_mode,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
