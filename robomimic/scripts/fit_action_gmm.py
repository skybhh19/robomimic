"""
Fit unconditional Gaussian mixture models to robomimic action distributions.

This script estimates p(a) from HDF5 demonstrations, selecting the number of
mixture modes and covariance parameterization with held-out validation NLL by
default.

Example:
    python robomimic/scripts/fit_action_gmm.py \
        --output_dir robomimic/trained_models/square/action_gmm \
        --max_modes 20 \
        --covariance_types diag full
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
from scipy.special import logsumexp


DEFAULT_DATASETS = {
    "square_ph": "robomimic/datasets/square/ph/image.hdf5",
    "square_mh": "robomimic/datasets/square/mh/image.hdf5",
    "square_random_post": "robomimic/datasets/square/random_post/expert200/image.hdf5",
}


def parse_dataset_specs(specs: Optional[List[str]]) -> Dict[str, str]:
    if not specs:
        return DEFAULT_DATASETS.copy()
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
        datasets[name] = path
    return datasets


def sorted_demo_keys(keys: Iterable[str]) -> List[str]:
    def demo_index(key):
        if key.startswith("demo_"):
            return int(key[5:])
        return key

    return sorted(keys, key=demo_index)


def demos_from_filter(f: h5py.File, filter_key: Optional[str]) -> List[str]:
    if filter_key is None:
        return sorted_demo_keys(f["data"].keys())
    mask_key = "mask/{}".format(filter_key)
    if mask_key not in f:
        raise KeyError("Filter key {!r} not found in {}".format(filter_key, f.filename))
    return sorted_demo_keys(elem.decode("utf-8") for elem in np.array(f[mask_key]))


def has_filter_key(dataset_path: str, filter_key: Optional[str]) -> bool:
    if filter_key is None:
        return True
    with h5py.File(dataset_path, "r") as f:
        return "mask/{}".format(filter_key) in f


def load_actions(
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
    def __init__(
        self,
        n_components: int,
        covariance_type: str,
        reg_covar: float,
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
        self.reg_covar = float(reg_covar)
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


def fit_best_of_inits(
    x: np.ndarray,
    n_components: int,
    covariance_type: str,
    reg_covar: float,
    args: argparse.Namespace,
) -> GaussianMixtureEM:
    best_model = None
    best_score = -np.inf
    for init_idx in range(args.n_init):
        model = GaussianMixtureEM(
            n_components=n_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
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
    val_inds = inds[:n_val]
    train_inds = inds[n_val:]
    return x[np.sort(train_inds)], x[np.sort(val_inds)]


def split_actions_for_selection(
    dataset_path: str,
    action_key: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict[str, object]]:
    if args.filter_key is not None:
        actions, meta = load_actions(dataset_path, action_key, args.filter_key)
        train_actions, val_actions = train_val_split(actions, args.val_fraction, rng)
        meta["split_mode"] = "filter_key"
        return train_actions, val_actions, None, meta

    if has_filter_key(dataset_path, args.train_filter_key):
        train_actions, train_meta = load_actions(
            dataset_path, action_key, args.train_filter_key
        )
        split_meta = {
            "split_mode": "hdf5_mask",
            "train": train_meta,
            "valid": None,
            "test": None,
        }
        val_actions = None
        test_actions = None
        if args.valid_filter_key is not None and has_filter_key(
            dataset_path, args.valid_filter_key
        ):
            val_actions, val_meta = load_actions(
                dataset_path, action_key, args.valid_filter_key
            )
            split_meta["valid"] = val_meta
        if args.test_filter_key is not None and has_filter_key(
            dataset_path, args.test_filter_key
        ):
            test_actions, test_meta = load_actions(
                dataset_path, action_key, args.test_filter_key
            )
            split_meta["test"] = test_meta
        return train_actions, val_actions, test_actions, split_meta

    actions, meta = load_actions(dataset_path, action_key, None)
    train_actions, val_actions = train_val_split(actions, args.val_fraction, rng)
    meta["split_mode"] = "random_action_split" if val_actions is not None else "all"
    return train_actions, val_actions, None, meta


def select_and_fit_dataset(
    name: str,
    dataset_path: str,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], GaussianMixtureEM]:
    rng = np.random.default_rng(args.seed)
    train_actions, val_actions, test_actions, dataset_meta = split_actions_for_selection(
        dataset_path=dataset_path,
        action_key=args.action_key,
        args=args,
        rng=rng,
    )
    if args.criterion == "val_nll" and val_actions is None:
        raise ValueError(
            "{} has no held-out validation actions. Provide --valid_filter_key, "
            "use --val_fraction with --filter_key, or choose --criterion bic/aic.".format(
                dataset_path
            )
        )
    full_train_actions = train_actions
    train_actions = maybe_subsample(full_train_actions, args.max_selection_samples, rng)

    candidate_rows = []
    best_model = None
    best_row = None
    modes = list(range(args.min_modes, args.max_modes + 1))
    for covariance_type in args.covariance_types:
        for reg_covar in args.reg_covars:
            for n_components in modes:
                model = fit_best_of_inits(
                    train_actions,
                    n_components=n_components,
                    covariance_type=covariance_type,
                    reg_covar=reg_covar,
                    args=args,
                )
                log_likelihood, bic, aic = model.information_criteria(train_actions)
                row = {
                    "n_components": int(n_components),
                    "covariance_type": covariance_type,
                    "reg_covar": float(reg_covar),
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
                if test_actions is not None:
                    row["test_log_likelihood"] = float(np.sum(model.score_samples(test_actions)))
                    row["test_nll"] = float(-model.score(test_actions))
                candidate_rows.append(row)
                if best_row is None or candidate_key(row, args.criterion) < candidate_key(
                    best_row, args.criterion
                ):
                    best_row = row
                    best_model = model
                print(
                    "{name}: K={k:02d} cov={cov:<9s} reg={reg:.1e} "
                    "train_nll={nll:.4f} val_nll={val_nll} bic={bic:.1f}".format(
                        name=name,
                        k=n_components,
                        cov=covariance_type,
                        reg=reg_covar,
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
        best_model = fit_best_of_inits(
            refit_actions,
            n_components=int(best_row["n_components"]),
            covariance_type=str(best_row["covariance_type"]),
            reg_covar=float(best_row["reg_covar"]),
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
        if test_actions is not None:
            best_row["refit_test_nll"] = float(-best_model.score(test_actions))

    summary = {
        "name": name,
        "dataset": dataset_meta,
        "selection": {
            "criterion": args.criterion,
            "max_selection_samples": args.max_selection_samples,
            "val_fraction": args.val_fraction,
            "train_filter_key": args.train_filter_key,
            "valid_filter_key": args.valid_filter_key,
            "test_filter_key": args.test_filter_key,
            "refit_full": bool(args.refit_full),
            "refit_include_valid": bool(args.refit_include_valid),
            "selected": best_row,
            "candidates": candidate_rows,
        },
    }
    return summary, best_model


def save_model(
    output_dir: Path,
    name: str,
    model: GaussianMixtureEM,
    summary: Dict[str, object],
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "{}_action_gmm.npz".format(name)
    np.savez_compressed(
        model_path,
        weights=model.weights_,
        means=model.means_,
        covariances=model.covariances_,
        covariance_type=np.array(model.covariance_type),
        reg_covar=np.array(model.reg_covar),
        metadata_json=np.array(json.dumps(summary, indent=2, sort_keys=True)),
    )
    return str(model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit GMMs to square action distributions p(a)."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help=(
            "Dataset spec name=path. May be repeated. Defaults to square PH, MH, "
            "and random_post demo.hdf5 files."
        ),
    )
    parser.add_argument("--action_key", default="actions")
    parser.add_argument(
        "--filter_key",
        default=None,
        help=(
            "Optional single HDF5 mask to fit from. If set, train/valid mask "
            "arguments are ignored and --val_fraction can create a random split."
        ),
    )
    parser.add_argument("--train_filter_key", default="train")
    parser.add_argument("--valid_filter_key", default="valid")
    parser.add_argument("--test_filter_key", default="test")
    parser.add_argument(
        "--output_dir",
        default="robomimic/trained_models/square/action_gmm",
        help="Directory for .npz model files and summary.json.",
    )
    parser.add_argument("--min_modes", type=int, default=1)
    parser.add_argument("--max_modes", type=int, default=20)
    parser.add_argument(
        "--covariance_types",
        nargs="+",
        default=["diag", "full"],
        choices=["full", "diag", "tied", "spherical"],
    )
    parser.add_argument(
        "--reg_covars",
        nargs="+",
        type=float,
        default=[1e-6],
        help="Diagonal covariance regularizers to evaluate.",
    )
    parser.add_argument(
        "--criterion",
        default="val_nll",
        choices=["bic", "aic", "val_nll"],
        help="Model-selection criterion. val_nll uses --valid_filter_key when present.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.0,
        help="Held-out action fraction; required for meaningful --criterion val_nll.",
    )
    parser.add_argument(
        "--max_selection_samples",
        type=int,
        default=None,
        help="Optional random action subsample for model selection.",
    )
    parser.add_argument(
        "--refit_full",
        action="store_true",
        help="Refit the selected hyperparameters after selection.",
    )
    parser.add_argument(
        "--refit_include_valid",
        action="store_true",
        help="With --refit_full, include validation actions in the final fitted model.",
    )
    parser.add_argument("--n_init", type=int, default=3)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--init_max_samples", type=int, default=10000)
    parser.add_argument("--init_lloyd_iters", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_modes < 1 or args.max_modes < args.min_modes:
        raise ValueError("Require 1 <= --min_modes <= --max_modes")
    datasets = parse_dataset_specs(args.dataset)
    output_dir = Path(args.output_dir)
    all_summaries = []
    for name, dataset_path in datasets.items():
        print("Fitting {} from {}".format(name, dataset_path), flush=True)
        summary, model = select_and_fit_dataset(name, dataset_path, args)
        model_path = save_model(output_dir, name, model, summary)
        summary["model_path"] = model_path
        selected = summary["selection"]["selected"]
        print(
            "Selected {name}: K={k} cov={cov} reg={reg:.1e} criterion={criterion}".format(
                name=name,
                k=selected["n_components"],
                cov=selected["covariance_type"],
                reg=selected["reg_covar"],
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
