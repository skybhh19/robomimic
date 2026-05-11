"""
Extract policy latents for every transition in a robomimic HDF5 dataset and
save exact nearest neighbors.

Example:
    python robomimic/scripts/extract_policy_latent_knn.py \
        --dataset robomimic/datasets/square/ph/image.hdf5 \
        --checkpoint checkpoints/square_ph_bc_checkpoints/square_ph_bc_gmm_agent_wrist_200_seed1_model_epoch_2000.pth \
        --output square_ph_latent_knn.json \
        --k 10 \
        --include-action \
        --discrete-target-type checkpoint

The latent used here is the policy trunk output immediately before the action
decoder / distribution-parameter head. This is a common representation choice
for policy-space similarity, but it is not canonical: depending on the question,
encoder features, value features, or a task-specific contrastive embedding can
also be reasonable.

When --include-action is set, KNN retrieval uses a weighted transition score:
state_weight * state latent cosine + action_weight * component action
similarity. For robosuite-style 7D actions, the action score compares
translation, rotation, and gripper components separately.
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import h5py
import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as F
from tqdm import tqdm

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.python_utils as PyUtils


def natural_key(name):
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", name)]


def demo_index_from_key(demo_key):
    match = re.search(r"(\d+)$", demo_key)
    return int(match.group(1)) if match is not None else demo_key


def get_demo_keys(hdf5_file, filter_key=None):
    if filter_key is not None:
        keys = [
            elem.decode("utf-8")
            for elem in np.asarray(hdf5_file["mask/{}".format(filter_key)][:])
        ]
    else:
        keys = list(hdf5_file["data"].keys())
    return sorted(keys, key=natural_key)


def get_feedforward_policy_net(rollout_policy):
    policy_net = rollout_policy.policy.nets["policy"]
    required = ("encoder", "mlp")
    if not hasattr(policy_net, "nets") or any(k not in policy_net.nets for k in required):
        raise ValueError(
            "This script currently supports feed-forward MIMO_MLP-style policies "
            "(for example BC / BC-GMM). RNN and Transformer policies need sequence "
            "context handling for a well-defined latent."
        )
    if "goal" in policy_net.input_obs_group_shapes:
        raise ValueError(
            "Goal-conditioned policies are not supported by this script yet because "
            "a per-transition goal observation must be specified."
        )
    return policy_net


def prepare_obs_batch(rollout_policy, obs_batch):
    return rollout_policy._prepare_observation(
        obs_batch,
        batched_ob=True,
        postprocess_visual_obs=True,
    )


@torch.no_grad()
def extract_latents_nll_and_entropy_for_batch(
    rollout_policy,
    policy_net,
    obs_batch,
    actions,
    discrete_target_type,
    entropy_num_samples,
):
    obs_batch = prepare_obs_batch(rollout_policy, obs_batch)
    enc = policy_net.nets["encoder"](obs=obs_batch)
    latent = policy_net.nets["mlp"](enc)

    if not hasattr(policy_net, "forward_train"):
        raise ValueError(
            "Negative log likelihood requires a stochastic policy with forward_train(). "
            "BC-Gaussian, BC-GMM, and BC-Discrete are supported; deterministic BC is not."
        )

    # Keep eval-time observation processing, but do not use low-noise eval
    # variances when evaluating the density of dataset actions.
    old_low_noise_eval = getattr(policy_net, "low_noise_eval", None)
    try:
        if old_low_noise_eval is not None:
            policy_net.low_noise_eval = False
        policy_out = policy_net.forward_train(obs_dict=obs_batch, goal_dict=None)
    finally:
        if old_low_noise_eval is not None:
            policy_net.low_noise_eval = old_low_noise_eval

    actions = torch.from_numpy(actions).to(rollout_policy.policy.device).float()
    if hasattr(policy_out, "log_prob"):
        nll = -policy_out.log_prob(actions)
        entropy = distribution_entropy(policy_out, entropy_num_samples)
    else:
        logits = policy_out
        nll = discrete_action_nll(
            logits=logits,
            actions=actions,
            config=rollout_policy.policy.global_config,
            target_type=discrete_target_type,
        )
        entropy = discrete_policy_entropy(logits)
    return latent, nll, entropy


def distribution_entropy(dist, num_samples):
    try:
        entropy = dist.entropy()
        if entropy.shape == dist.batch_shape:
            return entropy
    except NotImplementedError:
        pass

    samples = dist.sample(sample_shape=torch.Size([num_samples]))
    sampled_nlls = -dist.log_prob(samples)
    return sampled_nlls.mean(dim=0)


def discrete_policy_entropy(logits):
    categorical = D.Categorical(logits=logits)
    return categorical.entropy().sum(dim=-1)


def discrete_actions_to_bin_indices(actions, discrete_config, num_bins):
    action_min = float(discrete_config.action_min)
    action_max = float(discrete_config.action_max)
    actions = actions.clamp(action_min, action_max)
    scaled_actions = (actions - action_min) / (action_max - action_min)
    bin_indices = torch.round(scaled_actions * (num_bins - 1)).long()
    return bin_indices.clamp(0, num_bins - 1)


def discrete_actions_to_soft_bin_targets(actions, discrete_config, num_bins):
    action_min = float(discrete_config.action_min)
    action_max = float(discrete_config.action_max)
    sigma_bins = float(discrete_config.target_sigma_bins)
    if sigma_bins <= 0.0:
        raise ValueError("Gaussian discrete targets require target_sigma_bins > 0.")

    actions = actions.clamp(action_min, action_max)
    bin_centers = torch.linspace(
        action_min,
        action_max,
        num_bins,
        device=actions.device,
        dtype=actions.dtype,
    )
    bin_width = (action_max - action_min) / float(num_bins - 1)
    sigma = sigma_bins * bin_width
    distances = (bin_centers.view(*([1] * actions.ndim), num_bins) - actions.unsqueeze(-1)) / sigma
    return torch.softmax(-0.5 * distances.pow(2), dim=-1)


def discrete_action_nll(logits, actions, config, target_type):
    discrete_config = config.algo.discrete
    num_bins = logits.shape[-1]
    if int(discrete_config.num_bins) != num_bins:
        raise ValueError(
            "Checkpoint config discrete.num_bins ({}) does not match policy logits ({})".format(
                int(discrete_config.num_bins),
                num_bins,
            )
        )

    log_action_probs = F.log_softmax(logits, dim=-1)
    if target_type == "gaussian":
        action_soft_targets = discrete_actions_to_soft_bin_targets(
            actions=actions,
            discrete_config=discrete_config,
            num_bins=num_bins,
        )
        selected_log_probs = (action_soft_targets * log_action_probs).sum(dim=-1)
    elif target_type == "one_hot":
        target_bins = discrete_actions_to_bin_indices(
            actions=actions,
            discrete_config=discrete_config,
            num_bins=num_bins,
        )
        selected_log_probs = torch.gather(
            log_action_probs,
            dim=-1,
            index=target_bins.unsqueeze(-1),
        ).squeeze(-1)
    else:
        raise ValueError("Unknown discrete target type: {}".format(target_type))

    return -selected_log_probs.sum(dim=-1)


def normalize_action_dict(ac_dict, action_normalization_stats):
    if action_normalization_stats is None:
        return ac_dict
    return ObsUtils.normalize_dict(ac_dict, normalization_stats=action_normalization_stats)


def read_action_batch(hdf5_file, demo_key, start, end, action_keys, action_normalization_stats):
    ac_dict = OrderedDict()
    for key in action_keys:
        ac = hdf5_file["data/{}/{}".format(demo_key, key)][start:end].astype(np.float32)
        if len(ac.shape) == 1:
            ac = ac.reshape(-1, 1)
        ac_dict[key] = ac
    ac_dict = normalize_action_dict(ac_dict, action_normalization_stats)
    return PyUtils.action_dict_to_vector(ac_dict, action_keys=action_keys).astype(np.float32)


def action_config_uses_normalization(config, action_keys):
    action_config = config.train.action_config
    return any(action_config[key].get("normalization", None) is not None for key in action_keys)


def maybe_update_output_path_for_action(output_path, include_action):
    if not include_action:
        return output_path

    expanded = os.path.expanduser(output_path)
    dirname, basename = os.path.split(expanded)
    stem, ext = os.path.splitext(basename)
    if ext == "":
        ext = ".json"

    lower_stem = stem.lower()
    if "action" in lower_stem or "transition" in lower_stem:
        return expanded

    return os.path.join(dirname, "{}_state_action{}".format(stem, ext))


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def feature_cache_metadata(args, state_latent_dim=None, num_transitions=None):
    return {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "filter_key": args.filter_key,
        "num_demos": args.num_demos,
        "include_action": bool(args.include_action),
        "actions_normalized": bool(args.normalize_actions),
        "discrete_target_type": args.discrete_target_type,
        "entropy_num_samples": int(args.entropy_num_samples),
        "latent_choice": "policy MIMO_MLP trunk output immediately before action decoder/head",
        "state_latent_dim": state_latent_dim,
        "num_transitions": num_transitions,
    }


def default_feature_cache_path(args):
    metadata = feature_cache_metadata(args)
    key = json.dumps(metadata, sort_keys=True)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    ckpt_stem = safe_name(os.path.splitext(os.path.basename(os.path.expanduser(args.checkpoint)))[0])
    dataset_stem = safe_name(os.path.splitext(os.path.basename(os.path.expanduser(args.dataset)))[0])
    action_tag = "state_action" if args.include_action else "state"
    return os.path.join(
        "vis",
        "knn_cache",
        "{}_{}_{}_{}.npz".format(dataset_stem, ckpt_stem, action_tag, digest),
    )


def transition_index_to_arrays(transition_index):
    episodes = np.asarray([str(item["episode"]) for item in transition_index])
    episode_keys = np.asarray([item["episode_key"] for item in transition_index])
    steps = np.asarray([item["step"] for item in transition_index], dtype=np.int64)
    return episodes, episode_keys, steps


def arrays_to_transition_index(episodes, episode_keys, steps):
    return [
        {
            "episode": int(episode) if re.fullmatch(r"-?\d+", str(episode)) else str(episode),
            "episode_key": str(episode_key),
            "step": int(step),
        }
        for episode, episode_key, step in zip(episodes, episode_keys, steps)
    ]


def save_feature_cache(cache_path, args, state_latents, actions, nlls, entropies, transition_index):
    expanded = os.path.expanduser(cache_path)
    cache_dir = os.path.dirname(expanded)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    episodes, episode_keys, steps = transition_index_to_arrays(transition_index)
    metadata = feature_cache_metadata(
        args,
        state_latent_dim=int(state_latents.shape[1]),
        num_transitions=len(transition_index),
    )
    payload = {
        "state_latents": state_latents.astype(np.float32),
        "nlls": nlls.astype(np.float32),
        "entropies": entropies.astype(np.float32),
        "episodes": episodes,
        "episode_keys": episode_keys,
        "steps": steps,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if actions is not None:
        payload["actions"] = actions.astype(np.float32)
    np.savez_compressed(expanded, **payload)


def load_feature_cache(cache_path, args):
    expanded = os.path.expanduser(cache_path)
    with np.load(expanded, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        expected = feature_cache_metadata(args)
        mismatches = []
        for key, expected_value in expected.items():
            if key in {"state_latent_dim", "num_transitions"}:
                continue
            cached_value = metadata.get(key)
            if cached_value != expected_value:
                mismatches.append((key, cached_value, expected_value))
        if mismatches:
            details = ", ".join(
                "{} cached={!r} current={!r}".format(key, cached, current)
                for key, cached, current in mismatches
            )
            raise ValueError("Feature cache does not match current arguments: {}".format(details))
        if args.include_action and "actions" not in data:
            raise ValueError("Feature cache does not contain actions required by --include-action.")

        state_latents = data["state_latents"].astype(np.float32)
        actions = data["actions"].astype(np.float32) if "actions" in data else None
        nlls = data["nlls"].astype(np.float32)
        entropies = data["entropies"].astype(np.float32)
        transition_index = arrays_to_transition_index(
            episodes=data["episodes"],
            episode_keys=data["episode_keys"],
            steps=data["steps"],
        )
    return state_latents, actions, nlls, entropies, transition_index


def extract_features(args, rollout_policy, ckpt_dict):
    shape_meta = ckpt_dict["shape_metadata"]
    obs_keys = list(shape_meta["all_obs_keys"])
    action_keys = list(rollout_policy.policy.global_config.train.action_keys)
    nll_action_stats = rollout_policy.action_normalization_stats
    similarity_action_stats = rollout_policy.action_normalization_stats if args.normalize_actions else None
    policy_net = get_feedforward_policy_net(rollout_policy)

    latents = []
    actions = []
    nlls = []
    entropies = []
    transition_index = []

    with h5py.File(os.path.expanduser(args.dataset), "r", swmr=True, libver="latest") as f:
        demo_keys = get_demo_keys(f, filter_key=args.filter_key)
        if args.num_demos is not None:
            demo_keys = demo_keys[:args.num_demos]

        for demo_key in tqdm(demo_keys, desc="Extracting policy latents"):
            num_samples = int(f["data/{}".format(demo_key)].attrs["num_samples"])
            ep_index = demo_index_from_key(demo_key)

            for start in range(0, num_samples, args.batch_size):
                end = min(start + args.batch_size, num_samples)
                obs_batch = {
                    key: f["data/{}/obs/{}".format(demo_key, key)][start:end]
                    for key in obs_keys
                }
                nll_actions = read_action_batch(
                    hdf5_file=f,
                    demo_key=demo_key,
                    start=start,
                    end=end,
                    action_keys=action_keys,
                    action_normalization_stats=nll_action_stats,
                )
                latent, nll, entropy = extract_latents_nll_and_entropy_for_batch(
                    rollout_policy=rollout_policy,
                    policy_net=policy_net,
                    obs_batch=obs_batch,
                    actions=nll_actions,
                    discrete_target_type=args.discrete_target_type,
                    entropy_num_samples=args.entropy_num_samples,
                )
                latents.append(latent.detach().cpu().numpy().astype(np.float32))
                nlls.append(nll.detach().cpu().numpy().astype(np.float32))
                entropies.append(entropy.detach().cpu().numpy().astype(np.float32))

                if args.include_action:
                    actions.append(
                        read_action_batch(
                            hdf5_file=f,
                            demo_key=demo_key,
                            start=start,
                            end=end,
                            action_keys=action_keys,
                            action_normalization_stats=similarity_action_stats,
                        )
                    )

                transition_index.extend(
                    {
                        "episode": ep_index,
                        "episode_key": demo_key,
                        "step": step,
                    }
                    for step in range(start, end)
                )

    latents = np.concatenate(latents, axis=0)
    actions = np.concatenate(actions, axis=0) if args.include_action else None
    nlls = np.concatenate(nlls, axis=0)
    entropies = np.concatenate(entropies, axis=0)
    return latents, actions, nlls, entropies, transition_index


def exact_cosine_knn(features, k, query_block_size, index_block_size, include_self):
    features = torch.from_numpy(features)
    features = F.normalize(features, p=2, dim=1, eps=1e-12)
    num_items = features.shape[0]
    device = TorchUtils.get_torch_device(try_to_use_cuda=torch.cuda.is_available())
    search_k = min(k + (0 if include_self else 1), num_items)

    all_indices = []
    all_scores = []
    index_cpu = torch.arange(num_items)

    for q_start in tqdm(range(0, num_items, query_block_size), desc="Computing exact KNN"):
        q_end = min(q_start + query_block_size, num_items)
        q = features[q_start:q_end].to(device)
        block_best_scores = None
        block_best_indices = None

        for i_start in range(0, num_items, index_block_size):
            i_end = min(i_start + index_block_size, num_items)
            sims = q @ features[i_start:i_end].to(device).T

            if not include_self:
                diag_start = max(q_start, i_start)
                diag_end = min(q_end, i_end)
                if diag_start < diag_end:
                    q_diag = torch.arange(diag_start - q_start, diag_end - q_start, device=device)
                    i_diag = torch.arange(diag_start - i_start, diag_end - i_start, device=device)
                    sims[q_diag, i_diag] = -float("inf")

            local_k = min(search_k, sims.shape[1])
            scores, local_inds = torch.topk(sims, k=local_k, dim=1)
            global_inds = local_inds + i_start

            if block_best_scores is None:
                block_best_scores = scores
                block_best_indices = global_inds
            else:
                merged_scores = torch.cat([block_best_scores, scores], dim=1)
                merged_indices = torch.cat([block_best_indices, global_inds], dim=1)
                keep_scores, keep_pos = torch.topk(
                    merged_scores,
                    k=min(search_k, merged_scores.shape[1]),
                    dim=1,
                )
                block_best_scores = keep_scores
                block_best_indices = torch.gather(merged_indices, 1, keep_pos)

        block_best_scores = block_best_scores[:, :k].cpu()
        block_best_indices = block_best_indices[:, :k].cpu()
        all_scores.append(block_best_scores)
        all_indices.append(block_best_indices.to(index_cpu.dtype))

    return torch.cat(all_indices, dim=0).numpy(), torch.cat(all_scores, dim=0).numpy()


def action_component_similarity(query_actions, index_actions, action_range=None):
    action_dim = query_actions.shape[-1]
    if action_dim < 6:
        return F.cosine_similarity(query_actions[:, None, :], index_actions[None, :, :], dim=-1, eps=1e-12)

    pos_sim = F.cosine_similarity(
        query_actions[:, None, :3],
        index_actions[None, :, :3],
        dim=-1,
        eps=1e-12,
    )
    rot_sim = F.cosine_similarity(
        query_actions[:, None, 3:6],
        index_actions[None, :, 3:6],
        dim=-1,
        eps=1e-12,
    )

    if action_dim == 6:
        return 0.5 * pos_sim + 0.5 * rot_sim

    grip_delta = torch.abs(query_actions[:, None, 6:] - index_actions[None, :, 6:])
    if action_range is None:
        grip_range = torch.ones(action_dim - 6, device=query_actions.device) * 2.0
    else:
        grip_range = action_range[6:].to(query_actions.device).clamp(min=1e-6)
    grip_sim = 1.0 - 2.0 * torch.clamp((grip_delta / grip_range).mean(dim=-1), 0.0, 1.0)

    return 0.45 * pos_sim + 0.45 * rot_sim + 0.10 * grip_sim


def exact_transition_knn(
    state_features,
    actions,
    k,
    query_block_size,
    index_block_size,
    include_self,
    state_weight,
    action_weight,
):
    state_features = F.normalize(torch.from_numpy(state_features), p=2, dim=1, eps=1e-12)
    actions = torch.from_numpy(actions)
    action_range = actions.max(dim=0).values - actions.min(dim=0).values
    num_items = state_features.shape[0]
    device = TorchUtils.get_torch_device(try_to_use_cuda=torch.cuda.is_available())
    search_k = min(k + (0 if include_self else 1), num_items)

    all_indices = []
    all_scores = []
    all_state_scores = []
    all_action_scores = []
    index_cpu = torch.arange(num_items)

    for q_start in tqdm(range(0, num_items, query_block_size), desc="Computing transition KNN"):
        q_end = min(q_start + query_block_size, num_items)
        q_state = state_features[q_start:q_end].to(device)
        q_action = actions[q_start:q_end].to(device)
        block_best_scores = None
        block_best_indices = None
        block_best_state_scores = None
        block_best_action_scores = None

        for i_start in range(0, num_items, index_block_size):
            i_end = min(i_start + index_block_size, num_items)
            state_sims = q_state @ state_features[i_start:i_end].to(device).T
            action_sims = action_component_similarity(
                q_action,
                actions[i_start:i_end].to(device),
                action_range=action_range.to(device),
            )
            sims = state_weight * state_sims + action_weight * action_sims

            if not include_self:
                diag_start = max(q_start, i_start)
                diag_end = min(q_end, i_end)
                if diag_start < diag_end:
                    q_diag = torch.arange(diag_start - q_start, diag_end - q_start, device=device)
                    i_diag = torch.arange(diag_start - i_start, diag_end - i_start, device=device)
                    sims[q_diag, i_diag] = -float("inf")

            local_k = min(search_k, sims.shape[1])
            scores, local_inds = torch.topk(sims, k=local_k, dim=1)
            global_inds = local_inds + i_start
            state_scores = torch.gather(state_sims, 1, local_inds)
            action_scores = torch.gather(action_sims, 1, local_inds)

            if block_best_scores is None:
                block_best_scores = scores
                block_best_indices = global_inds
                block_best_state_scores = state_scores
                block_best_action_scores = action_scores
            else:
                merged_scores = torch.cat([block_best_scores, scores], dim=1)
                merged_indices = torch.cat([block_best_indices, global_inds], dim=1)
                merged_state_scores = torch.cat([block_best_state_scores, state_scores], dim=1)
                merged_action_scores = torch.cat([block_best_action_scores, action_scores], dim=1)
                keep_scores, keep_pos = torch.topk(
                    merged_scores,
                    k=min(search_k, merged_scores.shape[1]),
                    dim=1,
                )
                block_best_scores = keep_scores
                block_best_indices = torch.gather(merged_indices, 1, keep_pos)
                block_best_state_scores = torch.gather(merged_state_scores, 1, keep_pos)
                block_best_action_scores = torch.gather(merged_action_scores, 1, keep_pos)

        all_scores.append(block_best_scores[:, :k].cpu())
        all_indices.append(block_best_indices[:, :k].cpu().to(index_cpu.dtype))
        all_state_scores.append(block_best_state_scores[:, :k].cpu())
        all_action_scores.append(block_best_action_scores[:, :k].cpu())

    return (
        torch.cat(all_indices, dim=0).numpy(),
        torch.cat(all_scores, dim=0).numpy(),
        torch.cat(all_state_scores, dim=0).numpy(),
        torch.cat(all_action_scores, dim=0).numpy(),
    )


def percentile_normalize_values(values, lower_percentile=1.0, upper_percentile=99.0):
    values = np.asarray(values, dtype=np.float64)
    lower = float(np.percentile(values, lower_percentile))
    upper = float(np.percentile(values, upper_percentile))
    denom = upper - lower
    if denom <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - lower) / denom, 0.0, 1.0)


def query_knn_batch_baseline_adjusted_values(values, knn_indices, baseline_percentile=1.0):
    query_indices = np.arange(knn_indices.shape[0], dtype=np.int64)[:, None]
    batch_indices = np.concatenate((query_indices, knn_indices.astype(np.int64)), axis=1)
    batch_values = values[batch_indices].astype(np.float64)
    batch_baselines = np.percentile(batch_values, baseline_percentile, axis=1)
    return values.astype(np.float64) - batch_baselines


def write_knn_json(
    output_path,
    transition_index,
    nlls,
    entropies,
    knn_indices,
    knn_scores,
    state_scores,
    action_scores,
    metadata,
    baseline_percentile,
):
    global_normalized_nlls = percentile_normalize_values(nlls)
    global_normalized_entropies = percentile_normalize_values(entropies)
    query_batch_adjusted_nlls = query_knn_batch_baseline_adjusted_values(
        nlls,
        knn_indices,
        baseline_percentile=baseline_percentile,
    )
    query_batch_adjusted_entropies = query_knn_batch_baseline_adjusted_values(
        entropies,
        knn_indices,
        baseline_percentile=baseline_percentile,
    )
    query_batch_adjusted_normalized_nlls = percentile_normalize_values(query_batch_adjusted_nlls)
    query_batch_adjusted_normalized_entropies = percentile_normalize_values(query_batch_adjusted_entropies)

    records = []
    for query_i, query_meta in enumerate(transition_index):
        neighbor_indices = knn_indices[query_i]
        query_batch_adjusted_nll = float(query_batch_adjusted_nlls[query_i])
        query_batch_normalized_nll = float(query_batch_adjusted_normalized_nlls[query_i])
        query_batch_adjusted_entropy = float(query_batch_adjusted_entropies[query_i])
        query_batch_normalized_entropy = float(query_batch_adjusted_normalized_entropies[query_i])

        neighbors = []
        for neighbor_i, score in zip(neighbor_indices, knn_scores[query_i]):
            neighbor_meta = transition_index[int(neighbor_i)]
            neighbor = {
                "episode": neighbor_meta["episode"],
                "episode_key": neighbor_meta["episode_key"],
                "step": neighbor_meta["step"],
                "knn_score": float(score),
                "negative_log_likelihood": float(nlls[int(neighbor_i)]),
                "policy_sample_entropy": float(entropies[int(neighbor_i)]),
                "global_normalized_negative_log_likelihood": float(global_normalized_nlls[int(neighbor_i)]),
                "global_normalized_policy_sample_entropy": float(global_normalized_entropies[int(neighbor_i)]),
                "knn_batch_baseline_adjusted_negative_log_likelihood": float(
                    query_batch_adjusted_nlls[int(neighbor_i)]
                ),
                "knn_batch_normalized_negative_log_likelihood": float(
                    query_batch_adjusted_normalized_nlls[int(neighbor_i)]
                ),
                "knn_batch_baseline_adjusted_policy_sample_entropy": float(
                    query_batch_adjusted_entropies[int(neighbor_i)]
                ),
                "knn_batch_normalized_policy_sample_entropy": float(
                    query_batch_adjusted_normalized_entropies[int(neighbor_i)]
                ),
            }
            if state_scores is not None:
                neighbor["state_latent_cosine_similarity"] = float(state_scores[query_i, len(neighbors)])
            else:
                neighbor["state_latent_cosine_similarity"] = float(score)
            if action_scores is not None:
                neighbor["action_component_similarity"] = float(action_scores[query_i, len(neighbors)])
            neighbors.append(neighbor)
        records.append(
            {
                "episode": query_meta["episode"],
                "episode_key": query_meta["episode_key"],
                "step": query_meta["step"],
                "negative_log_likelihood": float(nlls[query_i]),
                "policy_sample_entropy": float(entropies[query_i]),
                "global_normalized_negative_log_likelihood": float(global_normalized_nlls[query_i]),
                "global_normalized_policy_sample_entropy": float(global_normalized_entropies[query_i]),
                "knn_batch_baseline_adjusted_negative_log_likelihood": query_batch_adjusted_nll,
                "knn_batch_normalized_negative_log_likelihood": query_batch_normalized_nll,
                "knn_batch_baseline_adjusted_policy_sample_entropy": query_batch_adjusted_entropy,
                "knn_batch_normalized_policy_sample_entropy": query_batch_normalized_entropy,
                "neighbors": neighbors,
            }
        )

    payload = {
        "metadata": metadata,
        "transitions": records,
    }
    expanded_output = os.path.expanduser(output_path)
    output_dir = os.path.dirname(expanded_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    tmp_output = "{}.tmp".format(expanded_output)
    open_fn = gzip.open if expanded_output.endswith(".gz") else open
    with open_fn(tmp_output, "wt") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp_output, expanded_output)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract policy last-hidden latents from a robomimic dataset and save transition KNNs."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Path to robomimic HDF5 dataset. If omitted, the script uses the single "
            "dataset path stored in the checkpoint training config."
        ),
    )
    parser.add_argument("--checkpoint", required=True, help="Path to trained policy checkpoint.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSON file. Use a .gz suffix for compact gzip-compressed JSON.",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors per transition.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for latent extraction.")
    parser.add_argument(
        "--entropy-num-samples",
        type=int,
        default=64,
        help=(
            "Number of policy action samples per state for Monte Carlo policy entropy "
            "estimation when exact entropy is unavailable, such as GMM policies. "
            "Plain Gaussian and discrete categorical policies use exact entropy."
        ),
    )
    parser.add_argument(
        "--knn-baseline-percentile",
        type=float,
        default=1.0,
        help=(
            "Lower percentile used as the query-local KNN baseline for NLL and entropy. "
            "The baseline set is the query transition plus its k nearest neighbors."
        ),
    )
    parser.add_argument("--query-block-size", type=int, default=1024, help="Query block size for exact KNN.")
    parser.add_argument("--index-block-size", type=int, default=8192, help="Index block size for exact KNN.")
    parser.add_argument(
        "--feature-cache",
        default="auto",
        help=(
            "Path to per-transition feature cache .npz. Use 'auto' to derive a cache "
            "path under vis/knn_cache, or 'none' to disable caching."
        ),
    )
    parser.add_argument(
        "--overwrite-feature-cache",
        action="store_true",
        help="Recompute per-transition features and overwrite --feature-cache if it exists.",
    )
    parser.add_argument("--filter-key", default=None, help="Optional HDF5 mask key, e.g. train or valid.")
    parser.add_argument("--num-demos", type=int, default=None, help="Optional cap on number of demos.")
    parser.add_argument(
        "--include-action",
        action="store_true",
        help="Use a weighted state-action transition score for KNN.",
    )
    parser.add_argument("--state-weight", type=float, default=0.7, help="State latent weight when --include-action is set.")
    parser.add_argument("--action-weight", type=float, default=0.3, help="Action similarity weight when --include-action is set.")
    parser.add_argument(
        "--raw-actions",
        action="store_true",
        help="When using --include-action, compare raw dataset actions instead of checkpoint-normalized actions.",
    )
    parser.add_argument(
        "--discrete-target-type",
        choices=["checkpoint", "one_hot", "gaussian"],
        default="checkpoint",
        help=(
            "Target type to use when computing NLL for discrete policies. "
            "'checkpoint' uses algo.discrete.target_type from the checkpoint config."
        ),
    )
    parser.add_argument("--include-self", action="store_true", help="Allow a transition to retrieve itself.")
    return parser.parse_args()


def resolve_discrete_target_type(config, requested_target_type):
    if requested_target_type != "checkpoint":
        return requested_target_type
    if "algo" not in config or "discrete" not in config.algo:
        return "one_hot"
    return config.algo.discrete.get("target_type", "one_hot")


def resolve_dataset_path(args, config):
    if args.dataset is not None:
        return args.dataset
    data = config.train.data
    if len(data) != 1:
        raise ValueError(
            "--dataset was omitted, but checkpoint config has {} train datasets. "
            "Please pass --dataset explicitly.".format(len(data))
        )
    dataset_path = data[0].get("path", None)
    if dataset_path is None:
        raise ValueError(
            "--dataset was omitted, but checkpoint config train.data[0].path is missing."
        )
    return dataset_path


def resolve_feature_cache_path(args):
    if args.feature_cache is None:
        return None
    if str(args.feature_cache).lower() in {"none", "false", "off", "disable", "disabled"}:
        return None
    if args.feature_cache == "auto":
        return default_feature_cache_path(args)
    return args.feature_cache


def main():
    args = parse_args()
    args.normalize_actions = not args.raw_actions
    original_output = args.output
    args.output = maybe_update_output_path_for_action(
        output_path=args.output,
        include_action=args.include_action,
    )
    if args.output != os.path.expanduser(original_output):
        print("--include-action enabled: writing to {} instead of {}".format(
            args.output,
            original_output,
        ))

    device = TorchUtils.get_torch_device(try_to_use_cuda=torch.cuda.is_available())
    rollout_policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.checkpoint,
        device=device,
        verbose=False,
    )
    rollout_policy.start_episode()
    args.dataset = resolve_dataset_path(args, rollout_policy.policy.global_config)
    args.discrete_target_type = resolve_discrete_target_type(
        config=rollout_policy.policy.global_config,
        requested_target_type=args.discrete_target_type,
    )
    if args.discrete_target_type not in ["one_hot", "gaussian"]:
        raise ValueError(
            "--discrete-target-type must resolve to one_hot or gaussian; got {}".format(
                args.discrete_target_type
            )
        )
    if args.entropy_num_samples <= 0:
        raise ValueError("--entropy-num-samples must be positive.")
    if args.knn_baseline_percentile < 0.0 or args.knn_baseline_percentile > 100.0:
        raise ValueError("--knn-baseline-percentile must be in [0, 100].")

    feature_cache_path = resolve_feature_cache_path(args)
    if (
        feature_cache_path is not None
        and os.path.exists(os.path.expanduser(feature_cache_path))
        and not args.overwrite_feature_cache
    ):
        print("Loading cached per-transition features from {}".format(feature_cache_path))
        state_latents, actions, nlls, entropies, transition_index = load_feature_cache(
            cache_path=feature_cache_path,
            args=args,
        )
    else:
        state_latents, actions, nlls, entropies, transition_index = extract_features(args, rollout_policy, ckpt_dict)
        if feature_cache_path is not None:
            print("Saving per-transition features to {}".format(feature_cache_path))
            save_feature_cache(
                cache_path=feature_cache_path,
                args=args,
                state_latents=state_latents,
                actions=actions,
                nlls=nlls,
                entropies=entropies,
                transition_index=transition_index,
            )

    action_keys = list(rollout_policy.policy.global_config.train.action_keys)
    action_similarity_normalized = (
        args.include_action
        and args.normalize_actions
        and action_config_uses_normalization(rollout_policy.policy.global_config, action_keys)
    )
    discrete_config = rollout_policy.policy.global_config.algo.get("discrete", None)
    if args.include_action:
        weight_sum = args.state_weight + args.action_weight
        if weight_sum <= 0.0:
            raise ValueError("--state-weight + --action-weight must be positive.")
        args.state_weight = args.state_weight / weight_sum
        args.action_weight = args.action_weight / weight_sum

    if args.k >= len(transition_index) and not args.include_self:
        raise ValueError("--k must be smaller than the number of transitions when --include-self is false.")

    state_scores = None
    action_scores = None
    if args.include_action:
        knn_indices, knn_scores, state_scores, action_scores = exact_transition_knn(
            state_features=state_latents,
            actions=actions,
            k=args.k,
            query_block_size=args.query_block_size,
            index_block_size=args.index_block_size,
            include_self=args.include_self,
            state_weight=args.state_weight,
            action_weight=args.action_weight,
        )
    else:
        knn_indices, knn_scores = exact_cosine_knn(
            features=state_latents,
            k=args.k,
            query_block_size=args.query_block_size,
            index_block_size=args.index_block_size,
            include_self=args.include_self,
        )
    write_knn_json(
        output_path=args.output,
        transition_index=transition_index,
        nlls=nlls,
        entropies=entropies,
        knn_indices=knn_indices,
        knn_scores=knn_scores,
        state_scores=state_scores,
        action_scores=action_scores,
        baseline_percentile=args.knn_baseline_percentile,
        metadata={
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "feature_cache": feature_cache_path,
            "k": args.k,
            "num_transitions": len(transition_index),
            "state_latent_dim": int(state_latents.shape[1]),
            "latent_choice": "policy MIMO_MLP trunk output immediately before action decoder/head",
            "include_action": bool(args.include_action),
            "actions_normalized": bool(action_similarity_normalized),
            "knn_metric": (
                "weighted_transition_similarity"
                if args.include_action
                else "state_latent_cosine_similarity"
            ),
            "state_weight": float(args.state_weight if args.include_action else 1.0),
            "action_weight": float(args.action_weight if args.include_action else 0.0),
            "action_metric": (
                "component_action_similarity(pos_cos=0.45, rot_cos=0.45, gripper_range_similarity=0.10)"
                if args.include_action
                else None
            ),
            "discrete_target_type": args.discrete_target_type,
            "discrete_target_sigma_bins": (
                float(discrete_config.target_sigma_bins)
                if discrete_config is not None and args.discrete_target_type == "gaussian"
                else None
            ),
            "nll": "negative log likelihood, -log p(a|s), computed in the policy training action space",
            "policy_sample_entropy": "Policy entropy at the state; plain Gaussian policies use exact distribution entropy, discrete categorical policies use exact sum of categorical entropies across action dimensions, and policies without exact entropy such as GMM use a Monte Carlo estimate computed as the mean negative log likelihood of sampled policy actions",
            "entropy_num_samples": int(args.entropy_num_samples),
            "knn_baseline_percentile": float(args.knn_baseline_percentile),
            "global_normalized_policy_sample_entropy": "policy_sample_entropy normalized over all transitions using 1st and 99th percentile bounds, clipped to [0, 1]",
            "global_normalized_negative_log_likelihood": "negative_log_likelihood normalized over all transitions using 1st and 99th percentile bounds, clipped to [0, 1]",
            "knn_batch_baseline_adjusted_negative_log_likelihood": "for each transition as query, negative_log_likelihood minus the knn_baseline_percentile of negative_log_likelihood over that query transition plus its k nearest neighbors, before global normalization; neighbor entries reuse the neighbor transition's query-adjusted value",
            "knn_batch_normalized_negative_log_likelihood": "for each transition as query, negative_log_likelihood minus the knn_baseline_percentile of negative_log_likelihood over that query transition plus its k nearest neighbors, then globally normalized using 1st and 99th percentile bounds over all query-adjusted values and clipped to [0, 1]; neighbor entries reuse the neighbor transition's query-adjusted normalized value",
            "knn_batch_baseline_adjusted_policy_sample_entropy": "for each transition as query, policy_sample_entropy minus the knn_baseline_percentile of policy_sample_entropy over that query transition plus its k nearest neighbors, before global normalization; neighbor entries reuse the neighbor transition's query-adjusted value",
            "knn_batch_normalized_policy_sample_entropy": "for each transition as query, policy_sample_entropy minus the knn_baseline_percentile of policy_sample_entropy over that query transition plus its k nearest neighbors, then globally normalized using 1st and 99th percentile bounds over all query-adjusted values and clipped to [0, 1]; neighbor entries reuse the neighbor transition's query-adjusted normalized value",
            "include_self": bool(args.include_self),
            "filter_key": args.filter_key,
        },
    )
    print("Wrote {} transition KNN records to {}".format(len(transition_index), args.output))


if __name__ == "__main__":
    main()
