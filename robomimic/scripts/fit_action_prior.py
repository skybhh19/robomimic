"""
Fit unconditional action prior densities to robomimic action distributions.

This script estimates p(a) from HDF5 demonstrations, selecting distribution
parameters with held-out validation NLL by default.

Example:
    python robomimic/scripts/fit_action_prior.py \
        --config robomimic/exps/square/action_prior/actions_k5_diag_fit_action_prior.json
"""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
from scipy.special import logsumexp


def parse_dataset_specs(specs: Optional[List[str]]) -> Dict[str, List[str]]:
    assert specs, "At least one --dataset name=path argument is required"
    datasets = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                "--dataset entries must use name=path, got {!r}".format(spec)
            )
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError("Invalid dataset spec {!r}".format(spec))
        paths = [elem.strip() for elem in path.split("+")]
        assert all(paths), "Invalid dataset path list {!r}".format(spec)
        datasets[name] = paths
    return datasets


def sorted_demo_keys(keys: Iterable[str]) -> List[str]:
    def demo_index(key):
        if key.startswith("demo_"):
            return (0, int(key[5:]))
        return (1, key)

    return sorted(keys, key=demo_index)


def demos_from_filter(f: h5py.File, filter_key: Optional[str]) -> List[str]:
    if filter_key is None:
        return sorted_demo_keys(f["data"].keys())
    mask_key = "mask/{}".format(filter_key)
    if mask_key not in f:
        raise KeyError("Filter key {!r} not found in {}".format(filter_key, f.filename))
    return sorted_demo_keys(elem.decode("utf-8") for elem in np.array(f[mask_key]))


def has_filter_key(dataset_paths: List[str], filter_key: Optional[str]) -> bool:
    if filter_key is None:
        return True
    for dataset_path in dataset_paths:
        with h5py.File(dataset_path, "r") as f:
            if "mask/{}".format(filter_key) not in f:
                return False
    return True


def load_single_dataset_actions(
    dataset_path: str,
    action_key: str,
    filter_key: Optional[str],
    dtype=np.float64,
) -> Tuple[np.ndarray, Dict[str, object]]:
    dataset_path = str(dataset_path)
    with h5py.File(dataset_path, "r") as f:
        demos = demos_from_filter(f, filter_key)
        if not demos:
            raise ValueError("No demonstrations found in {}".format(dataset_path))
        actions = []
        traj_lengths = []
        for ep in demos:
            key = "data/{}/{}".format(ep, action_key)
            if key not in f:
                raise KeyError("{} does not contain {}".format(dataset_path, key))
            arr = np.array(f[key], dtype=dtype)
            if arr.ndim != 2:
                raise ValueError("{} should be 2D, got shape {}".format(key, arr.shape))
            actions.append(arr)
            traj_lengths.append(arr.shape[0])
    actions = np.concatenate(actions, axis=0)
    meta = {
        "dataset_path": dataset_path,
        "action_key": action_key,
        "filter_key": filter_key,
        "num_demos": len(demos),
        "num_samples": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]),
        "traj_length_min": int(np.min(traj_lengths)),
        "traj_length_max": int(np.max(traj_lengths)),
        "traj_length_mean": float(np.mean(traj_lengths)),
        "action_min": np.min(actions, axis=0).tolist(),
        "action_max": np.max(actions, axis=0).tolist(),
        "action_mean": np.mean(actions, axis=0).tolist(),
        "action_std": np.std(actions, axis=0).tolist(),
    }
    return actions, meta


def load_actions(
    dataset_paths: List[str],
    action_key: str,
    filter_key: Optional[str],
    dtype=np.float64,
) -> Tuple[np.ndarray, Dict[str, object]]:
    actions = []
    sources = []
    for dataset_path in dataset_paths:
        source_actions, source_meta = load_single_dataset_actions(
            dataset_path=dataset_path,
            action_key=action_key,
            filter_key=filter_key,
            dtype=dtype,
        )
        if actions:
            assert source_actions.shape[1] == actions[0].shape[1], (
                source_actions.shape,
                actions[0].shape,
            )
        actions.append(source_actions)
        sources.append(source_meta)
    actions = np.concatenate(actions, axis=0)
    traj_lengths = [
        source["traj_length_mean"]
        for source in sources
        for _ in range(source["num_demos"])
    ]
    meta = {
        "dataset_path": dataset_paths[0] if len(dataset_paths) == 1 else dataset_paths,
        "action_key": action_key,
        "filter_key": filter_key,
        "num_demos": int(sum(source["num_demos"] for source in sources)),
        "num_samples": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]),
        "traj_length_min": int(min(source["traj_length_min"] for source in sources)),
        "traj_length_max": int(max(source["traj_length_max"] for source in sources)),
        "traj_length_mean": float(np.mean(traj_lengths)),
        "action_min": np.min(actions, axis=0).tolist(),
        "action_max": np.max(actions, axis=0).tolist(),
        "action_mean": np.mean(actions, axis=0).tolist(),
        "action_std": np.std(actions, axis=0).tolist(),
        "sources": sources,
    }
    return actions, meta


def rng_choice(rng: np.random.Generator, n: int, size: int, replace: bool) -> np.ndarray:
    return rng.choice(n, size=size, replace=replace)


def maybe_subsample(
    x: np.ndarray,
    max_samples: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    if max_samples is None or x.shape[0] <= max_samples:
        return x
    inds = rng_choice(rng, x.shape[0], max_samples, replace=False)
    return x[np.sort(inds)]


def kmeans_plus_plus_init(
    x: np.ndarray,
    n_components: int,
    rng: np.random.Generator,
    init_max_samples: int,
    lloyd_iters: int,
) -> np.ndarray:
    x_init = maybe_subsample(x, init_max_samples, rng)
    n, dim = x_init.shape
    means = np.empty((n_components, dim), dtype=np.float64)
    first = int(rng.integers(n))
    means[0] = x_init[first]

    closest_sq = np.sum((x_init - means[0]) ** 2, axis=1)
    for k in range(1, n_components):
        total = float(np.sum(closest_sq))
        if not np.isfinite(total) or total <= 0:
            means[k] = x_init[int(rng.integers(n))]
        else:
            probs = closest_sq / total
            means[k] = x_init[int(rng.choice(n, p=probs))]
        dist_sq = np.sum((x_init - means[k]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, dist_sq)

    for _ in range(lloyd_iters):
        dist_sq = np.sum((x_init[:, None, :] - means[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dist_sq, axis=1)
        for k in range(n_components):
            members = x_init[labels == k]
            if members.shape[0] == 0:
                means[k] = x_init[int(rng.integers(n))]
            else:
                means[k] = np.mean(members, axis=0)
    return means


class GaussianMixtureEM:
    distribution = "gmm"

    def __init__(
        self,
        n_components: int,
        covariance_type: str,
        min_std: float,
        tol: float,
        max_iter: int,
        random_state: int,
        init_max_samples: int,
        init_lloyd_iters: int,
    ):
        if covariance_type not in {"full", "diag", "tied", "spherical"}:
            raise ValueError("Unknown covariance_type {}".format(covariance_type))
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.min_std = float(min_std)
        self.reg_covar = self.min_std ** 2
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.init_max_samples = int(init_max_samples)
        self.init_lloyd_iters = int(init_lloyd_iters)
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.lower_bound_ = -np.inf
        self.n_iter_ = 0
        self.converged_ = False

    def _initialize(self, x: np.ndarray, rng: np.random.Generator) -> None:
        n, dim = x.shape
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)
        self.means_ = kmeans_plus_plus_init(
            x,
            self.n_components,
            rng,
            self.init_max_samples,
            self.init_lloyd_iters,
        )
        global_var = np.var(x, axis=0) + self.reg_covar
        if self.covariance_type == "full":
            self.covariances_ = np.tile(np.diag(global_var), (self.n_components, 1, 1))
        elif self.covariance_type == "diag":
            self.covariances_ = np.tile(global_var, (self.n_components, 1))
        elif self.covariance_type == "tied":
            self.covariances_ = np.diag(global_var)
        else:
            self.covariances_ = np.full(self.n_components, float(np.mean(global_var)))

    def _estimate_log_gaussian_prob(self, x: np.ndarray) -> np.ndarray:
        n, dim = x.shape
        log_prob = np.empty((n, self.n_components), dtype=np.float64)
        log_2pi = dim * math.log(2.0 * math.pi)

        if self.covariance_type == "diag":
            cov = np.maximum(self.covariances_, self.reg_covar)
            precisions = 1.0 / cov
            log_det = np.sum(np.log(cov), axis=1)
            diff = x[:, None, :] - self.means_[None, :, :]
            maha = np.sum(diff * diff * precisions[None, :, :], axis=2)
            return -0.5 * (log_2pi + log_det[None, :] + maha)

        if self.covariance_type == "spherical":
            cov = np.maximum(self.covariances_, self.reg_covar)
            precisions = 1.0 / cov
            log_det = dim * np.log(cov)
            diff = x[:, None, :] - self.means_[None, :, :]
            maha = np.sum(diff * diff, axis=2) * precisions[None, :]
            return -0.5 * (log_2pi + log_det[None, :] + maha)

        if self.covariance_type == "tied":
            chol = np.linalg.cholesky(self.covariances_)
            log_det = 2.0 * np.sum(np.log(np.diag(chol)))
            for k in range(self.n_components):
                diff = x - self.means_[k]
                sol = np.linalg.solve(chol, diff.T).T
                maha = np.sum(sol * sol, axis=1)
                log_prob[:, k] = -0.5 * (log_2pi + log_det + maha)
            return log_prob

        for k in range(self.n_components):
            chol = np.linalg.cholesky(self.covariances_[k])
            log_det = 2.0 * np.sum(np.log(np.diag(chol)))
            diff = x - self.means_[k]
            sol = np.linalg.solve(chol, diff.T).T
            maha = np.sum(sol * sol, axis=1)
            log_prob[:, k] = -0.5 * (log_2pi + log_det + maha)
        return log_prob

    def _e_step(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        weighted_log_prob = (
            self._estimate_log_gaussian_prob(x) + np.log(self.weights_ + 1e-300)
        )
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            log_resp = weighted_log_prob - log_prob_norm[:, None]
        return float(np.mean(log_prob_norm)), np.exp(log_resp)

    def _m_step(self, x: np.ndarray, resp: np.ndarray) -> None:
        n, dim = x.shape
        nk = resp.sum(axis=0) + 10.0 * np.finfo(np.float64).eps
        self.weights_ = nk / n
        self.means_ = (resp.T @ x) / nk[:, None]

        if self.covariance_type == "diag":
            avg_x2 = (resp.T @ (x * x)) / nk[:, None]
            cov = avg_x2 - self.means_ * self.means_
            self.covariances_ = np.maximum(cov, 0.0) + self.reg_covar
            return

        if self.covariance_type == "spherical":
            cov = np.empty(self.n_components, dtype=np.float64)
            for k in range(self.n_components):
                diff = x - self.means_[k]
                cov[k] = np.sum(resp[:, k] * np.sum(diff * diff, axis=1)) / (nk[k] * dim)
            self.covariances_ = np.maximum(cov, 0.0) + self.reg_covar
            return

        if self.covariance_type == "tied":
            cov = np.zeros((dim, dim), dtype=np.float64)
            for k in range(self.n_components):
                diff = x - self.means_[k]
                cov += (resp[:, k][:, None] * diff).T @ diff
            cov /= n
            cov.flat[:: dim + 1] += self.reg_covar
            self.covariances_ = cov
            return

        covariances = np.empty((self.n_components, dim, dim), dtype=np.float64)
        for k in range(self.n_components):
            diff = x - self.means_[k]
            cov = (resp[:, k][:, None] * diff).T @ diff / nk[k]
            cov.flat[:: dim + 1] += self.reg_covar
            covariances[k] = cov
        self.covariances_ = covariances

    def fit(self, x: np.ndarray) -> "GaussianMixtureEM":
        rng = np.random.default_rng(self.random_state)
        self._initialize(x, rng)
        previous_lower_bound = -np.inf
        for iteration in range(1, self.max_iter + 1):
            lower_bound, resp = self._e_step(x)
            self._m_step(x, resp)
            change = lower_bound - previous_lower_bound
            previous_lower_bound = lower_bound
            self.lower_bound_ = lower_bound
            self.n_iter_ = iteration
            if abs(change) < self.tol:
                self.converged_ = True
                break
        self.lower_bound_, _ = self._e_step(x)
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        weighted_log_prob = (
            self._estimate_log_gaussian_prob(x) + np.log(self.weights_ + 1e-300)
        )
        return logsumexp(weighted_log_prob, axis=1)

    def score(self, x: np.ndarray) -> float:
        return float(np.mean(self.score_samples(x)))

    def num_parameters(self, dim: int) -> int:
        cov_params = {
            "full": self.n_components * dim * (dim + 1) // 2,
            "diag": self.n_components * dim,
            "tied": dim * (dim + 1) // 2,
            "spherical": self.n_components,
        }[self.covariance_type]
        return (self.n_components - 1) + self.n_components * dim + cov_params

    def information_criteria(self, x: np.ndarray) -> Tuple[float, float, float]:
        log_likelihood = float(np.sum(self.score_samples(x)))
        n, dim = x.shape
        n_params = self.num_parameters(dim)
        bic = -2.0 * log_likelihood + n_params * math.log(n)
        aic = -2.0 * log_likelihood + 2.0 * n_params
        return log_likelihood, bic, aic


class GaussianActionPrior:
    distribution = "gaussian"

    def __init__(
        self,
        covariance_type: str,
        min_std: float,
        fixed_std: bool,
        init_std: float,
    ):
        if covariance_type not in {"full", "diag", "spherical"}:
            raise ValueError("Unknown covariance_type {}".format(covariance_type))
        self.covariance_type = covariance_type
        self.min_std = float(min_std)
        self.reg_covar = self.min_std ** 2
        self.fixed_std = bool(fixed_std)
        self.init_std = float(init_std)
        self.mean_ = None
        self.covariance_ = None

    def fit(self, x: np.ndarray) -> "GaussianActionPrior":
        self.mean_ = np.mean(x, axis=0)
        if self.fixed_std:
            self.covariance_ = np.full(x.shape[1], max(self.init_std, self.min_std) ** 2)
            return self
        centered = x - self.mean_
        if self.covariance_type == "diag":
            cov = np.mean(centered * centered, axis=0)
            self.covariance_ = np.maximum(cov, self.reg_covar)
        elif self.covariance_type == "spherical":
            cov = float(np.mean(centered * centered))
            self.covariance_ = max(cov, self.reg_covar)
        else:
            cov = centered.T @ centered / x.shape[0]
            cov.flat[:: x.shape[1] + 1] += self.reg_covar
            self.covariance_ = cov
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        n, dim = x.shape
        log_2pi = dim * math.log(2.0 * math.pi)
        diff = x - self.mean_
        if self.covariance_type == "diag":
            cov = np.maximum(self.covariance_, self.reg_covar)
            maha = np.sum(diff * diff / cov[None, :], axis=1)
            log_det = np.sum(np.log(cov))
            return -0.5 * (log_2pi + log_det + maha)
        if self.covariance_type == "spherical":
            cov = max(float(self.covariance_), self.reg_covar)
            maha = np.sum(diff * diff, axis=1) / cov
            log_det = dim * math.log(cov)
            return -0.5 * (log_2pi + log_det + maha)
        chol = np.linalg.cholesky(self.covariance_)
        sol = np.linalg.solve(chol, diff.T).T
        maha = np.sum(sol * sol, axis=1)
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))
        return -0.5 * (log_2pi + log_det + maha)

    def score(self, x: np.ndarray) -> float:
        return float(np.mean(self.score_samples(x)))

    def num_parameters(self, dim: int) -> int:
        cov_params = {
            "full": dim * (dim + 1) // 2,
            "diag": dim,
            "spherical": 1,
        }[self.covariance_type]
        return dim + cov_params

    def information_criteria(self, x: np.ndarray) -> Tuple[float, float, float]:
        log_likelihood = float(np.sum(self.score_samples(x)))
        n, dim = x.shape
        n_params = self.num_parameters(dim)
        bic = -2.0 * log_likelihood + n_params * math.log(n)
        aic = -2.0 * log_likelihood + 2.0 * n_params
        return log_likelihood, bic, aic


def fit_best_of_inits(
    x: np.ndarray,
    n_components: int,
    covariance_type: str,
    min_std: float,
    args: argparse.Namespace,
) -> GaussianMixtureEM:
    best_model = None
    best_score = -np.inf
    for init_idx in range(args.n_init):
        model = GaussianMixtureEM(
            n_components=n_components,
            covariance_type=covariance_type,
            min_std=min_std,
            tol=args.tol,
            max_iter=args.max_iter,
            random_state=args.seed + 1009 * init_idx + 9176 * n_components,
            init_max_samples=args.init_max_samples,
            init_lloyd_iters=args.init_lloyd_iters,
        ).fit(x)
        if model.lower_bound_ > best_score:
            best_model = model
            best_score = model.lower_bound_
    return best_model


def fit_gaussian(
    x: np.ndarray,
    covariance_type: str,
    min_std: float,
    args: argparse.Namespace,
) -> GaussianActionPrior:
    return GaussianActionPrior(
        covariance_type=covariance_type,
        min_std=min_std,
        fixed_std=args.fixed_std,
        init_std=args.init_std,
    ).fit(x)


def candidate_key(row: Dict[str, object], criterion: str) -> float:
    if criterion == "bic":
        return float(row["bic"])
    if criterion == "aic":
        return float(row["aic"])
    if criterion == "val_nll":
        return float(row["val_nll"])
    raise ValueError("Unknown criterion {}".format(criterion))


def train_val_split(
    x: np.ndarray,
    val_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if val_fraction <= 0.0:
        return x, None
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val_fraction must be in [0, 1)")
    inds = np.arange(x.shape[0])
    rng.shuffle(inds)
    n_val = max(1, int(round(x.shape[0] * val_fraction)))
    assert n_val < x.shape[0], (n_val, x.shape[0])
    val_inds = inds[:n_val]
    train_inds = inds[n_val:]
    return x[np.sort(train_inds)], x[np.sort(val_inds)]


def split_actions_for_selection(
    dataset_paths: List[str],
    action_key: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, object]]:
    if args.filter_key is not None:
        actions, meta = load_actions(dataset_paths, action_key, args.filter_key)
        train_actions, val_actions = train_val_split(actions, args.val_fraction, rng)
        meta["split_mode"] = "filter_key"
        return train_actions, val_actions, meta

    if has_filter_key(dataset_paths, args.train_filter_key):
        train_actions, train_meta = load_actions(
            dataset_paths, action_key, args.train_filter_key
        )
        split_meta = {
            "split_mode": "hdf5_mask",
            "train": train_meta,
            "valid": None,
        }
        val_actions = None
        if args.valid_filter_key is not None and has_filter_key(
            dataset_paths, args.valid_filter_key
        ):
            val_actions, val_meta = load_actions(
                dataset_paths, action_key, args.valid_filter_key
            )
            split_meta["valid"] = val_meta
        return train_actions, val_actions, split_meta

    actions, meta = load_actions(dataset_paths, action_key, None)
    train_actions, val_actions = train_val_split(actions, args.val_fraction, rng)
    meta["split_mode"] = "random_action_split" if val_actions is not None else "all"
    return train_actions, val_actions, meta


def select_and_fit_dataset(
    name: str,
    dataset_paths: List[str],
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], object, object]:
    rng = np.random.default_rng(args.seed)
    train_actions, val_actions, dataset_meta = split_actions_for_selection(
        dataset_paths=dataset_paths,
        action_key=args.action_key,
        args=args,
        rng=rng,
    )
    if args.criterion == "val_nll" and val_actions is None:
        raise ValueError(
            "{} has no held-out validation actions. Provide --valid_filter_key, "
            "use --val_fraction with --filter_key, or choose --criterion bic/aic.".format(
                dataset_paths
            )
        )
    full_train_actions = train_actions
    train_actions = maybe_subsample(full_train_actions, args.max_selection_samples, rng)

    candidate_rows = []
    best_model = None
    best_row = None
    last_model = None
    last_row = None
    if args.distribution == "gmm":
        for covariance_type in args.covariance_types:
            for min_std in args.min_std:
                for n_components in args.num_modes:
                    model = fit_best_of_inits(
                        train_actions,
                        n_components=n_components,
                        covariance_type=covariance_type,
                        min_std=min_std,
                        args=args,
                    )
                    log_likelihood, bic, aic = model.information_criteria(train_actions)
                    row = {
                        "distribution": "gmm",
                        "n_components": int(n_components),
                        "num_modes": int(n_components),
                        "covariance_type": covariance_type,
                        "min_std": float(min_std),
                        "converged": bool(model.converged_),
                        "n_iter": int(model.n_iter_),
                        "train_log_likelihood": float(log_likelihood),
                        "train_nll": float(-model.score(train_actions)),
                        "bic": float(bic),
                        "aic": float(aic),
                    }
                    if val_actions is not None:
                        row["val_log_likelihood"] = float(np.sum(model.score_samples(val_actions)))
                        row["val_nll"] = float(-model.score(val_actions))
                    candidate_rows.append(row)
                    last_model = model
                    last_row = row
                    if best_row is None or candidate_key(row, args.criterion) < candidate_key(
                        best_row, args.criterion
                    ):
                        best_row = row
                        best_model = model
                    print(
                        "{name}: dist=gmm K={k:02d} cov={cov:<9s} min_std={min_std:.1e} "
                        "train_nll={nll:.4f} val_nll={val_nll} bic={bic:.1f}".format(
                            name=name,
                            k=n_components,
                            cov=covariance_type,
                            min_std=min_std,
                            nll=row["train_nll"],
                            val_nll=(
                                "{:.4f}".format(row["val_nll"])
                                if "val_nll" in row
                                else "n/a"
                            ),
                            bic=row["bic"],
                        ),
                        flush=True,
                    )
    else:
        for covariance_type in args.covariance_types:
            for min_std in args.min_std:
                model = fit_gaussian(
                    train_actions,
                    covariance_type=covariance_type,
                    min_std=min_std,
                    args=args,
                )
                log_likelihood, bic, aic = model.information_criteria(train_actions)
                row = {
                    "distribution": "gaussian",
                    "covariance_type": covariance_type,
                    "min_std": float(min_std),
                    "fixed_std": bool(args.fixed_std),
                    "init_std": float(args.init_std),
                    "train_log_likelihood": float(log_likelihood),
                    "train_nll": float(-model.score(train_actions)),
                    "bic": float(bic),
                    "aic": float(aic),
                }
                if val_actions is not None:
                    row["val_log_likelihood"] = float(np.sum(model.score_samples(val_actions)))
                    row["val_nll"] = float(-model.score(val_actions))
                candidate_rows.append(row)
                last_model = model
                last_row = row
                if best_row is None or candidate_key(row, args.criterion) < candidate_key(
                    best_row, args.criterion
                ):
                    best_row = row
                    best_model = model
                print(
                    "{name}: dist=gaussian cov={cov:<9s} min_std={min_std:.1e} "
                    "train_nll={nll:.4f} val_nll={val_nll} bic={bic:.1f}".format(
                        name=name,
                        cov=covariance_type,
                        min_std=min_std,
                        nll=row["train_nll"],
                        val_nll=(
                            "{:.4f}".format(row["val_nll"])
                            if "val_nll" in row
                            else "n/a"
                        ),
                        bic=row["bic"],
                    ),
                    flush=True,
                )

    if args.refit_full:
        refit_actions = full_train_actions
        if args.refit_include_valid and val_actions is not None:
            refit_actions = np.concatenate([refit_actions, val_actions], axis=0)
        if args.distribution == "gmm":
            best_model = fit_best_of_inits(
                refit_actions,
                n_components=int(best_row["n_components"]),
                covariance_type=str(best_row["covariance_type"]),
                min_std=float(best_row["min_std"]),
                args=args,
            )
        else:
            best_model = fit_gaussian(
                refit_actions,
                covariance_type=str(best_row["covariance_type"]),
                min_std=float(best_row["min_std"]),
                args=args,
            )
        log_likelihood, bic, aic = best_model.information_criteria(refit_actions)
        best_row = dict(best_row)
        best_row.update(
            {
                "refit_num_samples": int(refit_actions.shape[0]),
                "refit_includes_valid": bool(args.refit_include_valid),
                "refit_log_likelihood": float(log_likelihood),
                "refit_nll": float(-best_model.score(refit_actions)),
                "refit_bic": float(bic),
                "refit_aic": float(aic),
            }
        )
        if val_actions is not None:
            best_row["refit_valid_nll"] = float(-best_model.score(val_actions))

    summary = {
        "name": name,
        "distribution": args.distribution,
        "dataset": dataset_meta,
        "selection": {
            "criterion": args.criterion,
            "max_selection_samples": args.max_selection_samples,
            "val_fraction": args.val_fraction,
            "train_filter_key": args.train_filter_key,
            "valid_filter_key": args.valid_filter_key,
            "refit_full": bool(args.refit_full),
            "refit_include_valid": bool(args.refit_include_valid),
            "selected": best_row,
            "last": last_row,
            "candidates": candidate_rows,
        },
    }
    return summary, best_model, last_model


def save_model(
    output_dir: Path,
    name: str,
    model: object,
    summary: Dict[str, object],
    checkpoint_name: str,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    distribution = model.distribution
    model_path = output_dir / "{}_action_prior_{}_{}.npz".format(
        name, distribution, checkpoint_name
    )
    arrays = {
        "distribution": np.array(distribution),
        "covariance_type": np.array(model.covariance_type),
        "min_std": np.array(model.min_std),
        "reg_covar": np.array(model.reg_covar),
        "checkpoint_name": np.array(checkpoint_name),
        "metadata_json": np.array(json.dumps(summary, indent=2, sort_keys=True)),
    }
    if distribution == "gmm":
        arrays.update(
            weights=model.weights_,
            means=model.means_,
            covariances=model.covariances_,
        )
    else:
        arrays.update(
            mean=model.mean_,
            covariance=model.covariance_,
            fixed_std=np.array(model.fixed_std),
            init_std=np.array(model.init_std),
        )
    np.savez_compressed(model_path, **arrays)
    return str(model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS,
        description="Fit unconditional action prior distributions p(a)."
    )
    parser.add_argument("--config", help="Path to a flat JSON config for this script.")
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset spec name=path. May be repeated.",
    )
    parser.add_argument("--action_key")
    parser.add_argument(
        "--distribution",
        choices=["gmm", "gaussian"],
        help="Action prior distribution family.",
    )
    parser.add_argument(
        "--normalization",
        choices=["none"],
        help="Accepted for compatibility; actions are loaded as stored.",
    )
    parser.add_argument(
        "--filter_key",
        help=(
            "Optional single HDF5 mask to fit from. If set, train/valid mask "
            "arguments are ignored and --val_fraction can create a random split."
        ),
    )
    parser.add_argument("--train_filter_key")
    parser.add_argument("--valid_filter_key")
    parser.add_argument(
        "--output_dir",
        help="Directory for .npz model files and summary.json.",
    )
    parser.add_argument("--num_modes", nargs="+", type=int)
    parser.add_argument(
        "--covariance_types",
        nargs="+",
        choices=["full", "diag", "tied", "spherical"],
    )
    parser.add_argument(
        "--min_std",
        nargs="+",
        type=float,
        help="Minimum per-dimension standard deviations to evaluate.",
    )
    parser.add_argument(
        "--fixed_std",
        action="store_true",
        default=argparse.SUPPRESS,
        help="For Gaussian priors, use init_std as a fixed diagonal std.",
    )
    parser.add_argument("--init_std", type=float)
    parser.add_argument(
        "--criterion",
        choices=["bic", "aic", "val_nll"],
        help="Model-selection criterion. val_nll uses --valid_filter_key when present.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        help="Held-out action fraction; required for meaningful --criterion val_nll.",
    )
    parser.add_argument(
        "--max_selection_samples",
        type=int,
        help="Optional random action subsample for model selection.",
    )
    parser.add_argument(
        "--refit_full",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Refit the selected hyperparameters after selection.",
    )
    parser.add_argument(
        "--refit_include_valid",
        action="store_true",
        default=argparse.SUPPRESS,
        help="With --refit_full, include validation actions in the final fitted model.",
    )
    parser.add_argument("--n_init", type=int)
    parser.add_argument("--max_iter", type=int)
    parser.add_argument("--tol", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--init_max_samples", type=int)
    parser.add_argument("--init_lloyd_iters", type=int)
    cli_args = vars(parser.parse_args())
    config_path = cli_args.pop("config", None)

    config_args = {}
    if config_path is not None:
        config_args = json.loads(Path(config_path).read_text())
        config_args.pop("script", None)

    defaults = dict(
        action_key="actions",
        distribution="gmm",
        normalization="none",
        filter_key=None,
        train_filter_key="train",
        valid_filter_key="valid",
        num_modes=None,
        covariance_types=["diag"],
        min_std=None,
        fixed_std=False,
        init_std=0.1,
        criterion="val_nll",
        val_fraction=0.0,
        max_selection_samples=None,
        refit_full=False,
        refit_include_valid=False,
        n_init=3,
        max_iter=200,
        tol=1e-4,
        seed=1,
        init_max_samples=10000,
        init_lloyd_iters=10,
    )
    allowed_keys = set(defaults) | {"dataset", "output_dir"}
    unknown_keys = sorted(set(config_args) - allowed_keys)
    assert not unknown_keys, unknown_keys
    merged_args = defaults
    merged_args.update(config_args)
    merged_args.update(cli_args)
    return SimpleNamespace(**merged_args)


def validate_args(args: argparse.Namespace) -> None:
    assert hasattr(args, "dataset"), "At least one --dataset name=path argument or config dataset is required"
    assert hasattr(args, "output_dir"), "output_dir must be specified in config or with --output_dir"
    if isinstance(args.dataset, str):
        args.dataset = [args.dataset]
    if isinstance(args.num_modes, int):
        args.num_modes = [args.num_modes]
    if args.num_modes is None:
        args.num_modes = [5]
    if args.min_std is None:
        args.min_std = [1e-4] if args.distribution == "gmm" else [1e-2]
    if isinstance(args.min_std, float):
        args.min_std = [args.min_std]
    if isinstance(args.min_std, int):
        args.min_std = [float(args.min_std)]
    assert all(num_modes >= 1 for num_modes in args.num_modes), args.num_modes
    assert len(set(args.num_modes)) == len(args.num_modes), args.num_modes
    assert args.distribution in {"gmm", "gaussian"}, args.distribution
    if isinstance(args.covariance_types, str):
        args.covariance_types = [args.covariance_types]
    if args.distribution == "gaussian":
        assert "tied" not in args.covariance_types, args.covariance_types
        if args.fixed_std:
            assert args.covariance_types == ["diag"], args.covariance_types
    assert all(
        cov in {"full", "diag", "tied", "spherical"} for cov in args.covariance_types
    ), args.covariance_types
    assert len(set(args.covariance_types)) == len(args.covariance_types), args.covariance_types
    assert args.criterion in {"bic", "aic", "val_nll"}, args.criterion
    assert args.normalization == "none", args.normalization
    assert args.n_init >= 1, args.n_init
    assert args.max_iter >= 1, args.max_iter
    assert args.tol > 0.0, args.tol
    assert args.init_max_samples >= 1, args.init_max_samples
    assert args.init_lloyd_iters >= 0, args.init_lloyd_iters
    assert args.init_std > 0.0, args.init_std
    assert all(min_std > 0.0 for min_std in args.min_std), args.min_std
    assert 0.0 <= args.val_fraction < 1.0, args.val_fraction
    if args.max_selection_samples is not None:
        assert args.max_selection_samples >= 1, args.max_selection_samples
    if args.refit_include_valid:
        assert args.refit_full, "--refit_include_valid requires --refit_full"


def main() -> None:
    args = parse_args()
    validate_args(args)
    datasets = parse_dataset_specs(args.dataset)
    output_dir = Path(args.output_dir)
    all_summaries = []
    for name, dataset_paths in datasets.items():
        print("Fitting {} from {}".format(name, "+".join(dataset_paths)), flush=True)
        summary, model, last_model = select_and_fit_dataset(name, dataset_paths, args)
        selected_checkpoint_name = "best_{}".format(args.criterion)
        selected_model_path = save_model(
            output_dir, name, model, summary, selected_checkpoint_name
        )
        last_model_path = save_model(output_dir, name, last_model, summary, "last")
        summary["model_path"] = selected_model_path
        summary["selected_model_path"] = selected_model_path
        summary["last_model_path"] = last_model_path
        selected = summary["selection"]["selected"]
        if args.distribution == "gmm":
            print(
                "Selected {name}: dist=gmm K={k} cov={cov} min_std={min_std:.1e} criterion={criterion}".format(
                    name=name,
                    k=selected["n_components"],
                    cov=selected["covariance_type"],
                    min_std=selected["min_std"],
                    criterion=args.criterion,
                ),
                flush=True,
            )
        else:
            print(
                "Selected {name}: dist=gaussian cov={cov} min_std={min_std:.1e} criterion={criterion}".format(
                    name=name,
                    cov=selected["covariance_type"],
                    min_std=selected["min_std"],
                    criterion=args.criterion,
                ),
                flush=True,
            )
        all_summaries.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2, sort_keys=True))
    print("Wrote {}".format(summary_path), flush=True)


if __name__ == "__main__":
    main()
