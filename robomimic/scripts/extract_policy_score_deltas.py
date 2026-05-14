"""
Evaluate paired image+robot and robot-only policies on aligned Square episodes.

For every transition, this script computes:
    NLL_images+robot, Entropy_images+robot,
    NLL_images+robot - NLL_robot,
    Entropy_images+robot - Entropy_robot,
    NLL_images+robot - NLL_nonconditional,
    Entropy_images+robot - Entropy_nonconditional.

It also saves trajectory-level means and run-level means into a JSON or JSON.GZ
file. The image+robot NLL / entropy scores and raw image+robot-minus-robot
deltas are also globally normalized over all transitions using percentile
bounds.
"""

import argparse
import gzip
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import h5py
import numpy as np
import torch
from tqdm import tqdm

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.scripts.extract_policy_latent_knn import (
    demo_index_from_key,
    extract_latents_nll_and_entropy_for_batch,
    get_demo_keys,
    get_feedforward_policy_net,
    percentile_normalize_values,
    read_action_batch,
    resolve_dataset_path,
    resolve_discrete_target_type,
)


ACTION_GMM_FILENAMES = {
    "ph": "square_ph_action_gmm.npz",
    "mh": "square_mh_action_gmm.npz",
    "random_post": "square_random_post_action_gmm.npz",
}


def policy_type_from_config(config):
    algo = config.algo
    if algo.get("gmm", {}).get("enabled", False):
        return "gmm"
    if algo.get("discrete", {}).get("enabled", False):
        target_type = algo.discrete.get("target_type", "one_hot")
        return "discrete_gaussian" if target_type == "gaussian" else "discrete"
    if algo.get("gaussian", {}).get("enabled", False):
        return "gaussian"
    return "deterministic"


class ActionGMM:
    def __init__(self, weights, means, covariances, covariance_type, reg_covar, metadata, model_path):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.means = np.asarray(means, dtype=np.float64)
        self.covariances = np.asarray(covariances, dtype=np.float64)
        self.covariance_type = str(covariance_type)
        self.reg_covar = float(reg_covar)
        self.metadata = metadata
        self.model_path = model_path

    @property
    def action_dim(self):
        return int(self.means.shape[1])

    def _estimate_log_gaussian_prob(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(
                "Action GMM expected actions with shape (N, {}), got {}".format(
                    self.action_dim,
                    actions.shape,
                )
            )
        n, dim = actions.shape
        log_prob = np.empty((n, self.means.shape[0]), dtype=np.float64)
        log_2pi = dim * math.log(2.0 * math.pi)

        if self.covariance_type == "diag":
            cov = np.maximum(self.covariances, self.reg_covar)
            precisions = 1.0 / cov
            log_det = np.sum(np.log(cov), axis=1)
            diff = actions[:, None, :] - self.means[None, :, :]
            maha = np.sum(diff * diff * precisions[None, :, :], axis=2)
            return -0.5 * (log_2pi + log_det[None, :] + maha)

        if self.covariance_type == "spherical":
            cov = np.maximum(self.covariances, self.reg_covar)
            precisions = 1.0 / cov
            log_det = dim * np.log(cov)
            diff = actions[:, None, :] - self.means[None, :, :]
            maha = np.sum(diff * diff, axis=2) * precisions[None, :]
            return -0.5 * (log_2pi + log_det[None, :] + maha)

        if self.covariance_type == "tied":
            chol = np.linalg.cholesky(self.covariances)
            log_det = 2.0 * np.sum(np.log(np.diag(chol)))
            for k in range(self.means.shape[0]):
                diff = actions - self.means[k]
                sol = np.linalg.solve(chol, diff.T).T
                maha = np.sum(sol * sol, axis=1)
                log_prob[:, k] = -0.5 * (log_2pi + log_det + maha)
            return log_prob

        if self.covariance_type != "full":
            raise ValueError("Unsupported action GMM covariance_type {}".format(self.covariance_type))
        for k in range(self.means.shape[0]):
            chol = np.linalg.cholesky(self.covariances[k])
            log_det = 2.0 * np.sum(np.log(np.diag(chol)))
            diff = actions - self.means[k]
            sol = np.linalg.solve(chol, diff.T).T
            maha = np.sum(sol * sol, axis=1)
            log_prob[:, k] = -0.5 * (log_2pi + log_det + maha)
        return log_prob

    def score_samples(self, actions):
        weighted_log_prob = self._estimate_log_gaussian_prob(actions) + np.log(self.weights + 1e-300)
        return np.logaddexp.reduce(weighted_log_prob, axis=1)

    def negative_log_likelihood(self, actions):
        return -self.score_samples(actions)

    def sample(self, num_samples, seed):
        rng = np.random.default_rng(seed)
        num_samples = int(num_samples)
        component_ids = rng.choice(self.means.shape[0], size=num_samples, p=self.weights)
        samples = np.empty((num_samples, self.action_dim), dtype=np.float64)
        for component_id in range(self.means.shape[0]):
            rows = np.nonzero(component_ids == component_id)[0]
            if rows.size == 0:
                continue
            if self.covariance_type == "diag":
                std = np.sqrt(np.maximum(self.covariances[component_id], self.reg_covar))
                samples[rows] = rng.normal(self.means[component_id], std, size=(rows.size, self.action_dim))
            elif self.covariance_type == "spherical":
                std = math.sqrt(max(float(self.covariances[component_id]), self.reg_covar))
                samples[rows] = rng.normal(self.means[component_id], std, size=(rows.size, self.action_dim))
            elif self.covariance_type == "tied":
                samples[rows] = rng.multivariate_normal(self.means[component_id], self.covariances, size=rows.size)
            else:
                samples[rows] = rng.multivariate_normal(
                    self.means[component_id],
                    self.covariances[component_id],
                    size=rows.size,
                )
        return samples

    def entropy(self, num_samples, seed):
        samples = self.sample(num_samples=num_samples, seed=seed)
        return float(np.mean(self.negative_log_likelihood(samples)))


def load_action_gmm_model(model_path):
    expanded = os.path.expanduser(model_path)
    with np.load(expanded, allow_pickle=True) as data:
        metadata_json = str(data["metadata_json"]) if "metadata_json" in data else "{}"
        return ActionGMM(
            weights=data["weights"],
            means=data["means"],
            covariances=data["covariances"],
            covariance_type=str(data["covariance_type"]),
            reg_covar=float(data["reg_covar"]),
            metadata=json.loads(metadata_json),
            model_path=expanded,
        )


def square_dataset_group_from_path(path):
    parts = Path(str(path)).parts
    for idx, part in enumerate(parts[:-1]):
        if part == "square" and parts[idx + 1] in ACTION_GMM_FILENAMES:
            return parts[idx + 1]
    text = str(path)
    for dataset_group in ACTION_GMM_FILENAMES:
        if dataset_group in text:
            return dataset_group
    return None


def resolve_action_gmm_model_path(args, dataset_path):
    if args.action_gmm_model is not None:
        return os.path.expanduser(args.action_gmm_model)
    dataset_group = square_dataset_group_from_path(dataset_path)
    if dataset_group is None:
        raise ValueError(
            "Could not infer square dataset group from {}. Pass --action-gmm-model explicitly.".format(
                dataset_path
            )
        )
    return os.path.join(
        os.path.expanduser(args.action_gmm_dir),
        ACTION_GMM_FILENAMES[dataset_group],
    )


def load_policy(checkpoint, device):
    rollout_policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=checkpoint,
        device=device,
        verbose=False,
    )
    rollout_policy.start_episode()
    return rollout_policy, ckpt_dict


def extract_policy_scores(
    dataset,
    checkpoint,
    batch_size,
    entropy_num_samples,
    discrete_target_type,
    filter_key,
    num_demos,
    desc,
    device,
):
    rollout_policy, ckpt_dict = load_policy(checkpoint, device)
    config = rollout_policy.policy.global_config
    resolved_target_type = resolve_discrete_target_type(config, discrete_target_type)
    if resolved_target_type not in ["one_hot", "gaussian"]:
        raise ValueError(
            "--discrete-target-type must resolve to one_hot or gaussian; got {}".format(
                resolved_target_type
            )
        )

    dataset = resolve_dataset_path(argparse.Namespace(dataset=dataset), config)
    shape_meta = ckpt_dict["shape_metadata"]
    obs_keys = list(shape_meta["all_obs_keys"])
    action_keys = list(config.train.action_keys)
    action_stats = rollout_policy.action_normalization_stats
    policy_net = get_feedforward_policy_net(rollout_policy)

    nlls = []
    entropies = []
    transition_index = []
    trajectory_lengths = {}

    with h5py.File(os.path.expanduser(dataset), "r", swmr=True, libver="latest") as f:
        demo_keys = get_demo_keys(f, filter_key=filter_key)
        if num_demos is not None:
            demo_keys = demo_keys[:num_demos]

        for demo_key in tqdm(demo_keys, desc=desc):
            num_samples = int(f["data/{}".format(demo_key)].attrs["num_samples"])
            trajectory_lengths[demo_key] = num_samples
            ep_index = demo_index_from_key(demo_key)

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                obs_batch = {
                    key: f["data/{}/obs/{}".format(demo_key, key)][start:end]
                    for key in obs_keys
                }
                actions = read_action_batch(
                    hdf5_file=f,
                    demo_key=demo_key,
                    start=start,
                    end=end,
                    action_keys=action_keys,
                    action_normalization_stats=action_stats,
                )
                _, nll, entropy = extract_latents_nll_and_entropy_for_batch(
                    rollout_policy=rollout_policy,
                    policy_net=policy_net,
                    obs_batch=obs_batch,
                    actions=actions,
                    discrete_target_type=resolved_target_type,
                    entropy_num_samples=entropy_num_samples,
                )
                nlls.append(nll.detach().cpu().numpy().astype(np.float32))
                entropies.append(entropy.detach().cpu().numpy().astype(np.float32))
                transition_index.extend(
                    {
                        "episode": ep_index,
                        "episode_key": demo_key,
                        "step": step,
                    }
                    for step in range(start, end)
                )

    return {
        "dataset": dataset,
        "checkpoint": checkpoint,
        "policy_type": policy_type_from_config(config),
        "discrete_target_type": resolved_target_type,
        "nlls": np.concatenate(nlls, axis=0),
        "entropies": np.concatenate(entropies, axis=0),
        "transition_index": transition_index,
        "trajectory_lengths": trajectory_lengths,
        "action_keys": action_keys,
    }


def read_raw_action_batch(hdf5_file, demo_key, start, end, action_keys):
    actions = []
    for key in action_keys:
        action = hdf5_file["data/{}/{}".format(demo_key, key)][start:end].astype(np.float64)
        if len(action.shape) == 1:
            action = action.reshape(-1, 1)
        actions.append(action)
    return np.concatenate(actions, axis=1)


def extract_action_gmm_scores(
    dataset,
    action_keys,
    trajectory_lengths,
    batch_size,
    filter_key,
    num_demos,
    action_gmm,
    entropy_num_samples,
    entropy_seed,
):
    nlls = []
    transition_index = []
    with h5py.File(os.path.expanduser(dataset), "r", swmr=True, libver="latest") as f:
        demo_keys = get_demo_keys(f, filter_key=filter_key)
        if num_demos is not None:
            demo_keys = demo_keys[:num_demos]

        if list(demo_keys) != list(trajectory_lengths.keys()):
            raise ValueError(
                "Action GMM demo order does not match policy scores: {} vs {}".format(
                    list(demo_keys)[:5],
                    list(trajectory_lengths.keys())[:5],
                )
            )

        for demo_key in tqdm(demo_keys, desc="Scoring non-conditional action GMM"):
            num_samples = int(f["data/{}".format(demo_key)].attrs["num_samples"])
            if int(trajectory_lengths[demo_key]) != num_samples:
                raise ValueError(
                    "Action GMM length mismatch for {}: {} vs {}".format(
                        demo_key,
                        num_samples,
                        trajectory_lengths[demo_key],
                    )
                )
            ep_index = demo_index_from_key(demo_key)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                actions = read_raw_action_batch(f, demo_key, start, end, action_keys)
                nlls.append(action_gmm.negative_log_likelihood(actions).astype(np.float32))
                transition_index.extend(
                    {
                        "episode": ep_index,
                        "episode_key": demo_key,
                        "step": step,
                    }
                    for step in range(start, end)
                )

    return {
        "dataset": dataset,
        "model_path": action_gmm.model_path,
        "metadata": action_gmm.metadata,
        "nlls": np.concatenate(nlls, axis=0),
        "entropy": float(action_gmm.entropy(num_samples=entropy_num_samples, seed=entropy_seed)),
        "entropy_num_samples": int(entropy_num_samples),
        "entropy_seed": int(entropy_seed),
        "transition_index": transition_index,
    }


def check_transition_alignment(image_scores, robot_scores):
    image_index = image_scores["transition_index"]
    robot_index = robot_scores["transition_index"]
    if len(image_index) != len(robot_index):
        raise ValueError(
            "Transition count mismatch: image policy has {}, robot policy has {}".format(
                len(image_index),
                len(robot_index),
            )
        )
    for i, (image_meta, robot_meta) in enumerate(zip(image_index, robot_index)):
        if image_meta["episode_key"] != robot_meta["episode_key"] or image_meta["step"] != robot_meta["step"]:
            raise ValueError(
                "Transition alignment mismatch at row {}: image {} step {}, robot {} step {}".format(
                    i,
                    image_meta["episode_key"],
                    image_meta["step"],
                    robot_meta["episode_key"],
                    robot_meta["step"],
                )
            )


def check_optional_transition_alignment(image_scores, baseline_scores, baseline_name):
    if baseline_scores is None:
        return
    image_index = image_scores["transition_index"]
    baseline_index = baseline_scores["transition_index"]
    if len(image_index) != len(baseline_index):
        raise ValueError(
            "Transition count mismatch: image policy has {}, {} has {}".format(
                len(image_index),
                baseline_name,
                len(baseline_index),
            )
        )
    for i, (image_meta, baseline_meta) in enumerate(zip(image_index, baseline_index)):
        if image_meta["episode_key"] != baseline_meta["episode_key"] or image_meta["step"] != baseline_meta["step"]:
            raise ValueError(
                "Transition alignment mismatch at row {}: image {} step {}, {} {} step {}".format(
                    i,
                    image_meta["episode_key"],
                    image_meta["step"],
                    baseline_name,
                    baseline_meta["episode_key"],
                    baseline_meta["step"],
                )
            )


def summarize_values(
    nll_image,
    entropy_image,
    nll_robot,
    entropy_robot,
    normalized_nll_image=None,
    normalized_entropy_image=None,
    normalized_nll_delta=None,
    normalized_entropy_delta=None,
    nll_nonconditional=None,
    entropy_nonconditional=None,
):
    nll_delta = nll_image - nll_robot
    entropy_delta = entropy_image - entropy_robot
    summary = {
        "nll_images_robot": float(np.mean(nll_image)),
        "entropy_images_robot": float(np.mean(entropy_image)),
        "nll_robot": float(np.mean(nll_robot)),
        "entropy_robot": float(np.mean(entropy_robot)),
        "nll_images_robot_minus_robot": float(np.mean(nll_delta)),
        "entropy_images_robot_minus_robot": float(np.mean(entropy_delta)),
        "num_steps": int(len(nll_image)),
    }
    if normalized_nll_image is not None:
        summary.update(
            {
                "global_normalized_nll_images_robot": float(np.mean(normalized_nll_image)),
                "global_normalized_entropy_images_robot": float(np.mean(normalized_entropy_image)),
                "global_normalized_nll_images_robot_minus_robot": float(np.mean(normalized_nll_delta)),
                "global_normalized_entropy_images_robot_minus_robot": float(np.mean(normalized_entropy_delta)),
            }
        )
    if nll_nonconditional is not None:
        nll_nonconditional_delta = nll_image - nll_nonconditional
        entropy_nonconditional_delta = entropy_image - entropy_nonconditional
        summary.update(
            {
                "nll_nonconditional": float(np.mean(nll_nonconditional)),
                "entropy_nonconditional": float(np.mean(entropy_nonconditional)),
                "nll_images_robot_minus_nonconditional": float(np.mean(nll_nonconditional_delta)),
                "entropy_images_robot_minus_nonconditional": float(np.mean(entropy_nonconditional_delta)),
            }
        )
    return summary


def summarize_single_policy_values(
    nll_policy,
    entropy_policy,
    normalized_nll_policy=None,
    normalized_entropy_policy=None,
    nll_nonconditional=None,
    entropy_nonconditional=None,
    normalized_nll_nonconditional_delta=None,
    normalized_entropy_nonconditional_delta=None,
):
    summary = {
        "nll_conditional": float(np.mean(nll_policy)),
        "entropy_conditional": float(np.mean(entropy_policy)),
        "num_steps": int(len(nll_policy)),
    }
    if normalized_nll_policy is not None:
        summary.update(
            {
                "global_normalized_nll_conditional": float(np.mean(normalized_nll_policy)),
                "global_normalized_entropy_conditional": float(np.mean(normalized_entropy_policy)),
            }
        )
    if nll_nonconditional is not None:
        nll_nonconditional_delta = nll_policy - nll_nonconditional
        entropy_nonconditional_delta = entropy_policy - entropy_nonconditional
        summary.update(
            {
                "nll_nonconditional": float(np.mean(nll_nonconditional)),
                "entropy_nonconditional": float(np.mean(entropy_nonconditional)),
                "nll_conditional_minus_nonconditional": float(np.mean(nll_nonconditional_delta)),
                "entropy_conditional_minus_nonconditional": float(np.mean(entropy_nonconditional_delta)),
            }
        )
        if normalized_nll_nonconditional_delta is not None:
            summary.update(
                {
                    "global_normalized_nll_conditional_minus_nonconditional": float(
                        np.mean(normalized_nll_nonconditional_delta)
                    ),
                    "global_normalized_entropy_conditional_minus_nonconditional": float(
                        np.mean(normalized_entropy_nonconditional_delta)
                    ),
                }
            )
    return summary


def build_single_policy_output_payload(args, policy_scores, action_gmm_scores=None):
    check_optional_transition_alignment(policy_scores, action_gmm_scores, "non-conditional action GMM")

    nll_policy = policy_scores["nlls"].astype(np.float64)
    entropy_policy = policy_scores["entropies"].astype(np.float64)
    normalized_nll_policy = percentile_normalize_values(nll_policy)
    normalized_entropy_policy = percentile_normalize_values(entropy_policy)
    nll_nonconditional = None
    entropy_nonconditional = None
    nll_nonconditional_delta = None
    entropy_nonconditional_delta = None
    normalized_nll_nonconditional_delta = None
    normalized_entropy_nonconditional_delta = None
    if action_gmm_scores is not None:
        nll_nonconditional = action_gmm_scores["nlls"].astype(np.float64)
        entropy_nonconditional = np.full_like(nll_policy, float(action_gmm_scores["entropy"]))
        nll_nonconditional_delta = nll_policy - nll_nonconditional
        entropy_nonconditional_delta = entropy_policy - entropy_nonconditional
        normalized_nll_nonconditional_delta = percentile_normalize_values(nll_nonconditional_delta)
        normalized_entropy_nonconditional_delta = percentile_normalize_values(entropy_nonconditional_delta)

    trajectories = []
    transitions = []
    start = 0
    for demo_key, num_steps in policy_scores["trajectory_lengths"].items():
        end = start + int(num_steps)
        ep_meta = policy_scores["transition_index"][start]
        trajectory_summary = summarize_single_policy_values(
            nll_policy[start:end],
            entropy_policy[start:end],
            normalized_nll_policy[start:end],
            normalized_entropy_policy[start:end],
            nll_nonconditional[start:end] if nll_nonconditional is not None else None,
            entropy_nonconditional[start:end] if entropy_nonconditional is not None else None,
            normalized_nll_nonconditional_delta[start:end]
            if normalized_nll_nonconditional_delta is not None
            else None,
            normalized_entropy_nonconditional_delta[start:end]
            if normalized_entropy_nonconditional_delta is not None
            else None,
        )
        trajectory_summary.update(
            {
                "episode": ep_meta["episode"],
                "episode_key": demo_key,
            }
        )
        trajectories.append(trajectory_summary)

        for row in range(start, end):
            meta = policy_scores["transition_index"][row]
            transition = {
                "episode": meta["episode"],
                "episode_key": meta["episode_key"],
                "step": meta["step"],
                "nll_conditional": float(nll_policy[row]),
                "entropy_conditional": float(entropy_policy[row]),
                "global_normalized_nll_conditional": float(normalized_nll_policy[row]),
                "global_normalized_entropy_conditional": float(normalized_entropy_policy[row]),
            }
            if action_gmm_scores is not None:
                transition.update(
                    {
                        "nll_nonconditional": float(nll_nonconditional[row]),
                        "entropy_nonconditional": float(entropy_nonconditional[row]),
                        "nll_conditional_minus_nonconditional": float(nll_nonconditional_delta[row]),
                        "entropy_conditional_minus_nonconditional": float(entropy_nonconditional_delta[row]),
                        "global_normalized_nll_conditional_minus_nonconditional": float(
                            normalized_nll_nonconditional_delta[row]
                        ),
                        "global_normalized_entropy_conditional_minus_nonconditional": float(
                            normalized_entropy_nonconditional_delta[row]
                        ),
                    }
                )
            transitions.append(transition)
        start = end

    action_gmm_metadata = {}
    if action_gmm_scores is not None:
        action_gmm_metadata = {
            "action_gmm_model": action_gmm_scores["model_path"],
            "action_gmm_entropy_num_samples": int(action_gmm_scores["entropy_num_samples"]),
            "action_gmm_entropy_seed": int(action_gmm_scores["entropy_seed"]),
        }

    return {
        "metadata": {
            "policy_dataset": policy_scores["dataset"],
            "policy_checkpoint": policy_scores["checkpoint"],
            "policy_type": policy_scores["policy_type"],
            "discrete_target_type": policy_scores["discrete_target_type"],
            "entropy_num_samples": int(args.entropy_num_samples),
            "filter_key": args.filter_key,
            "num_demos": args.num_demos,
            **action_gmm_metadata,
            "score_definitions": {
                "nll_conditional": "-log p(a|s) from the conditional policy",
                "entropy_conditional": "H(pi(.|s)) from the conditional policy",
                "nll_nonconditional": "-log p(a) from the non-conditional action GMM",
                "entropy_nonconditional": "H(p(a)) from the non-conditional action GMM, estimated by Monte Carlo samples",
                "nll_conditional_minus_nonconditional": "conditional policy NLL minus non-conditional action GMM NLL on the aligned transition",
                "entropy_conditional_minus_nonconditional": "conditional policy entropy minus non-conditional action GMM entropy",
                "global_normalized_nll_conditional": "nll_conditional normalized over all transitions using 1st and 99th percentile bounds, clipped to [0, 1]",
                "global_normalized_entropy_conditional": "entropy_conditional normalized over all transitions using 1st and 99th percentile bounds, clipped to [0, 1]",
                "global_normalized_nll_conditional_minus_nonconditional": "raw nll_conditional_minus_nonconditional deltas normalized over all transitions using 1st and 99th percentile bounds, clipped to [0, 1]",
                "global_normalized_entropy_conditional_minus_nonconditional": "raw entropy_conditional_minus_nonconditional deltas normalized over all transitions using 1st and 99th percentile bounds, clipped to [0, 1]",
            },
            "entropy_estimator": (
                "Discrete policies use exact summed categorical entropy over action dimensions. "
                "Gaussian policies use torch distribution entropy. GMM policies use Monte Carlo "
                "entropy E[-log pi(a|s)] with actions sampled from the policy."
            ),
        },
        "run_scores": summarize_single_policy_values(
            nll_policy,
            entropy_policy,
            normalized_nll_policy,
            normalized_entropy_policy,
            nll_nonconditional,
            entropy_nonconditional,
            normalized_nll_nonconditional_delta,
            normalized_entropy_nonconditional_delta,
        ),
        "trajectories": trajectories,
        "transitions": transitions,
    }


def build_output_payload(args, image_scores, robot_scores, action_gmm_scores=None):
    check_transition_alignment(image_scores, robot_scores)
    check_optional_transition_alignment(image_scores, action_gmm_scores, "non-conditional action GMM")
    if image_scores["policy_type"] != robot_scores["policy_type"]:
        raise ValueError(
            "Policy type mismatch: image policy is {}, robot policy is {}".format(
                image_scores["policy_type"],
                robot_scores["policy_type"],
            )
        )
    if image_scores["discrete_target_type"] != robot_scores["discrete_target_type"]:
        raise ValueError(
            "Discrete target mismatch: image policy uses {}, robot policy uses {}".format(
                image_scores["discrete_target_type"],
                robot_scores["discrete_target_type"],
            )
        )

    nll_image = image_scores["nlls"].astype(np.float64)
    entropy_image = image_scores["entropies"].astype(np.float64)
    nll_robot = robot_scores["nlls"].astype(np.float64)
    entropy_robot = robot_scores["entropies"].astype(np.float64)
    nll_delta = nll_image - nll_robot
    entropy_delta = entropy_image - entropy_robot
    normalized_nll_image = percentile_normalize_values(nll_image)
    normalized_entropy_image = percentile_normalize_values(entropy_image)
    normalized_nll_delta = percentile_normalize_values(nll_delta)
    normalized_entropy_delta = percentile_normalize_values(entropy_delta)
    nll_nonconditional = None
    entropy_nonconditional = None
    nll_nonconditional_delta = None
    entropy_nonconditional_delta = None
    if action_gmm_scores is not None:
        nll_nonconditional = action_gmm_scores["nlls"].astype(np.float64)
        entropy_nonconditional = np.full_like(nll_image, float(action_gmm_scores["entropy"]))
        nll_nonconditional_delta = nll_image - nll_nonconditional
        entropy_nonconditional_delta = entropy_image - entropy_nonconditional

    trajectories = []
    transitions = []
    start = 0
    for demo_key, num_steps in image_scores["trajectory_lengths"].items():
        end = start + int(num_steps)
        ep_meta = image_scores["transition_index"][start]
        trajectory_summary = summarize_values(
            nll_image[start:end],
            entropy_image[start:end],
            nll_robot[start:end],
            entropy_robot[start:end],
            normalized_nll_image[start:end],
            normalized_entropy_image[start:end],
            normalized_nll_delta[start:end],
            normalized_entropy_delta[start:end],
            nll_nonconditional[start:end] if nll_nonconditional is not None else None,
            entropy_nonconditional[start:end] if entropy_nonconditional is not None else None,
        )
        trajectory_summary.update(
            {
                "episode": ep_meta["episode"],
                "episode_key": demo_key,
            }
        )
        trajectories.append(trajectory_summary)

        for row in range(start, end):
            meta = image_scores["transition_index"][row]
            transitions.append(
                {
                    "episode": meta["episode"],
                    "episode_key": meta["episode_key"],
                    "step": meta["step"],
                    "nll_images_robot": float(nll_image[row]),
                    "entropy_images_robot": float(entropy_image[row]),
                    "nll_robot": float(nll_robot[row]),
                    "entropy_robot": float(entropy_robot[row]),
                    "nll_images_robot_minus_robot": float(nll_delta[row]),
                    "entropy_images_robot_minus_robot": float(entropy_delta[row]),
                    "global_normalized_nll_images_robot": float(normalized_nll_image[row]),
                    "global_normalized_entropy_images_robot": float(normalized_entropy_image[row]),
                    "global_normalized_nll_images_robot_minus_robot": float(normalized_nll_delta[row]),
                    "global_normalized_entropy_images_robot_minus_robot": float(normalized_entropy_delta[row]),
                }
            )
            if action_gmm_scores is not None:
                transitions[-1].update(
                    {
                        "nll_nonconditional": float(nll_nonconditional[row]),
                        "entropy_nonconditional": float(entropy_nonconditional[row]),
                        "nll_images_robot_minus_nonconditional": float(nll_nonconditional_delta[row]),
                        "entropy_images_robot_minus_nonconditional": float(entropy_nonconditional_delta[row]),
                    }
                )
        start = end

    action_gmm_metadata = {}
    if action_gmm_scores is not None:
        action_gmm_metadata = {
            "action_gmm_model": action_gmm_scores["model_path"],
            "action_gmm_entropy_num_samples": int(action_gmm_scores["entropy_num_samples"]),
            "action_gmm_entropy_seed": int(action_gmm_scores["entropy_seed"]),
        }

    return {
        "metadata": {
            "image_dataset": image_scores["dataset"],
            "robot_dataset": robot_scores["dataset"],
            "image_checkpoint": image_scores["checkpoint"],
            "robot_checkpoint": robot_scores["checkpoint"],
            "policy_type": image_scores["policy_type"],
            "discrete_target_type": image_scores["discrete_target_type"],
            "entropy_num_samples": int(args.entropy_num_samples),
            "filter_key": args.filter_key,
            "num_demos": args.num_demos,
            **action_gmm_metadata,
            "score_definitions": {
                "nll_images_robot": "-log p(a|s) from the image+robot policy",
                "entropy_images_robot": "H(pi(.|s)) from the image+robot policy",
                "nll_nonconditional": "-log p(a) from the non-conditional action GMM",
                "entropy_nonconditional": "H(p(a)) from the non-conditional action GMM, estimated by Monte Carlo samples",
                "nll_images_robot_minus_robot": "image+robot NLL minus robot-only NLL on the aligned transition",
                "entropy_images_robot_minus_robot": "image+robot entropy minus robot-only entropy on the aligned transition",
                "nll_images_robot_minus_nonconditional": "image+robot NLL minus non-conditional action GMM NLL on the aligned transition",
                "entropy_images_robot_minus_nonconditional": "image+robot entropy minus non-conditional action GMM entropy",
                "global_normalized_nll_images_robot": "nll_images_robot normalized over all transitions using 1st and 99th percentile bounds, clipped to [0, 1]",
                "global_normalized_entropy_images_robot": "entropy_images_robot normalized over all transitions using 1st and 99th percentile bounds, clipped to [0, 1]",
                "global_normalized_nll_images_robot_minus_robot": "raw nll_images_robot_minus_robot deltas normalized over all transitions using 1st and 99th percentile bounds, clipped to [0, 1]",
                "global_normalized_entropy_images_robot_minus_robot": "raw entropy_images_robot_minus_robot deltas normalized over all transitions using 1st and 99th percentile bounds, clipped to [0, 1]",
            },
            "entropy_estimator": (
                "Discrete policies use exact summed categorical entropy over action dimensions. "
                "Gaussian policies use torch distribution entropy. GMM policies use Monte Carlo "
                "entropy E[-log pi(a|s)] with actions sampled from the policy."
            ),
        },
        "run_scores": summarize_values(
            nll_image,
            entropy_image,
            nll_robot,
            entropy_robot,
            normalized_nll_image,
            normalized_entropy_image,
            normalized_nll_delta,
            normalized_entropy_delta,
            nll_nonconditional,
            entropy_nonconditional,
        ),
        "trajectories": trajectories,
        "transitions": transitions,
    }


def write_json(output_path, payload):
    expanded = os.path.expanduser(output_path)
    output_dir = os.path.dirname(expanded)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    tmp_output = "{}.tmp".format(expanded)
    open_fn = gzip.open if expanded.endswith(".gz") else open
    with open_fn(tmp_output, "wt") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp_output, expanded)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute aligned image+robot vs robot-only NLL and entropy score deltas."
    )
    parser.add_argument("--policy-dataset", default=None, help="Dataset for single-policy p(a|s) scoring.")
    parser.add_argument("--policy-checkpoint", default=None, help="Single conditional policy checkpoint.")
    parser.add_argument("--image-dataset", default=None, help="Dataset for the image+robot policy.")
    parser.add_argument("--robot-dataset", default=None, help="Dataset for the robot-only policy.")
    parser.add_argument("--image-checkpoint", default=None, help="Image+robot policy checkpoint.")
    parser.add_argument("--robot-checkpoint", default=None, help="Robot-only policy checkpoint.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path. Use .json.gz for gzip-compressed output.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Scoring batch size.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for checkpoint loading and scoring. Use cpu for lightweight smoke tests.",
    )
    parser.add_argument(
        "--entropy-num-samples",
        type=int,
        default=128,
        help="Monte Carlo sample count for GMM entropy. Exact entropy policies ignore this.",
    )
    parser.add_argument(
        "--action-gmm-dir",
        default="robomimic/trained_models/square/action_gmm",
        help="Directory containing square_*_action_gmm.npz non-conditional action baselines.",
    )
    parser.add_argument(
        "--action-gmm-model",
        default=None,
        help="Optional explicit .npz action GMM baseline. Defaults to inferring from --image-dataset.",
    )
    parser.add_argument(
        "--action-gmm-entropy-num-samples",
        type=int,
        default=20000,
        help="Monte Carlo sample count for estimating non-conditional action GMM entropy.",
    )
    parser.add_argument(
        "--action-gmm-entropy-seed",
        type=int,
        default=0,
        help="Random seed for estimating non-conditional action GMM entropy.",
    )
    parser.add_argument(
        "--discrete-target-type",
        choices=["checkpoint", "one_hot", "gaussian"],
        default="checkpoint",
        help="NLL target for discrete policies; 'checkpoint' uses each policy config.",
    )
    parser.add_argument("--filter-key", default=None, help="Optional HDF5 mask key, e.g. train or valid.")
    parser.add_argument("--num-demos", type=int, default=None, help="Optional cap on demos.")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return TorchUtils.get_torch_device(try_to_use_cuda=True)
    return TorchUtils.get_torch_device(try_to_use_cuda=torch.cuda.is_available())


def main():
    args = parse_args()
    single_policy_mode = args.policy_checkpoint is not None
    paired_policy_mode = args.image_checkpoint is not None or args.robot_checkpoint is not None
    if single_policy_mode and paired_policy_mode:
        raise ValueError("Use either --policy-checkpoint or --image-checkpoint/--robot-checkpoint, not both.")
    if not single_policy_mode and (args.image_checkpoint is None or args.robot_checkpoint is None):
        raise ValueError(
            "Paired mode requires --image-checkpoint and --robot-checkpoint. "
            "For p(a|s) sweep scoring, use --policy-checkpoint."
        )
    if args.entropy_num_samples <= 0:
        raise ValueError("--entropy-num-samples must be positive.")
    if args.action_gmm_entropy_num_samples <= 0:
        raise ValueError("--action-gmm-entropy-num-samples must be positive.")

    device = resolve_device(args.device)
    if single_policy_mode:
        policy_scores = extract_policy_scores(
            dataset=args.policy_dataset,
            checkpoint=args.policy_checkpoint,
            batch_size=args.batch_size,
            entropy_num_samples=args.entropy_num_samples,
            discrete_target_type=args.discrete_target_type,
            filter_key=args.filter_key,
            num_demos=args.num_demos,
            desc="Scoring conditional policy",
            device=device,
        )
        action_gmm_path = resolve_action_gmm_model_path(args, policy_scores["dataset"])
        action_gmm = load_action_gmm_model(action_gmm_path)
        action_gmm_scores = extract_action_gmm_scores(
            dataset=policy_scores["dataset"],
            action_keys=policy_scores["action_keys"],
            trajectory_lengths=policy_scores["trajectory_lengths"],
            batch_size=args.batch_size,
            filter_key=args.filter_key,
            num_demos=args.num_demos,
            action_gmm=action_gmm,
            entropy_num_samples=args.action_gmm_entropy_num_samples,
            entropy_seed=args.action_gmm_entropy_seed,
        )
        payload = build_single_policy_output_payload(args, policy_scores, action_gmm_scores)
        write_json(args.output, payload)
        print(
            "Wrote {} transition scores and {} trajectory scores to {}".format(
                len(payload["transitions"]),
                len(payload["trajectories"]),
                args.output,
            )
        )
        return

    image_scores = extract_policy_scores(
        dataset=args.image_dataset,
        checkpoint=args.image_checkpoint,
        batch_size=args.batch_size,
        entropy_num_samples=args.entropy_num_samples,
        discrete_target_type=args.discrete_target_type,
        filter_key=args.filter_key,
        num_demos=args.num_demos,
        desc="Scoring image+robot policy",
        device=device,
    )
    robot_scores = extract_policy_scores(
        dataset=args.robot_dataset,
        checkpoint=args.robot_checkpoint,
        batch_size=args.batch_size,
        entropy_num_samples=args.entropy_num_samples,
        discrete_target_type=args.discrete_target_type,
        filter_key=args.filter_key,
        num_demos=args.num_demos,
        desc="Scoring robot-only policy",
        device=device,
    )
    action_gmm_path = resolve_action_gmm_model_path(args, image_scores["dataset"])
    action_gmm = load_action_gmm_model(action_gmm_path)
    action_gmm_scores = extract_action_gmm_scores(
        dataset=image_scores["dataset"],
        action_keys=image_scores["action_keys"],
        trajectory_lengths=image_scores["trajectory_lengths"],
        batch_size=args.batch_size,
        filter_key=args.filter_key,
        num_demos=args.num_demos,
        action_gmm=action_gmm,
        entropy_num_samples=args.action_gmm_entropy_num_samples,
        entropy_seed=args.action_gmm_entropy_seed,
    )
    payload = build_output_payload(args, image_scores, robot_scores, action_gmm_scores)
    write_json(args.output, payload)
    print(
        "Wrote {} transition scores and {} trajectory scores to {}".format(
            len(payload["transitions"]),
            len(payload["trajectories"]),
            args.output,
        )
    )


if __name__ == "__main__":
    main()
