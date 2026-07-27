import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


SUMMARY_PATHS = [
    Path("robomimic/trained_models/square/density/mh/action_prior/instant/gaussian/std1em02_wd0/summary.json"),
    Path("robomimic/trained_models/square/density/mh/action_prior/instant/gmm/k5_std1em04_wd0/summary.json"),
    Path("robomimic/trained_models/square/density/ph/action_prior/instant/gaussian/std1em02_wd0/summary.json"),
    Path("robomimic/trained_models/square/density/ph/action_prior/instant/gmm/k5_std1em04_wd0/summary.json"),
    Path("robomimic/trained_models/square/density/rollout_1400_randomviews/action_prior/instant/gaussian/std1em02_wd0/summary.json"),
    Path("robomimic/trained_models/square/density/rollout_1400_randomviews/action_prior/instant/gmm/k5_std1em04_wd0/summary.json"),
    Path("robomimic/trained_models/transport/density/mh/action_prior/instant/gaussian/std1em02_wd0/summary.json"),
    Path("robomimic/trained_models/transport/density/mh/action_prior/instant/gmm/k10_std1em04_wd0/summary.json"),
]

OUTPUT = Path("robomimic/trained_models/action_prior_fit_visualization.png")
DIM_OUTPUT_DIR = Path("robomimic/trained_models/action_prior_dim_distributions")


def load_summary(path):
    data = json.loads(path.read_text())
    assert len(data) == 1, path
    return data[0]


def dataset_label(summary):
    path = Path(summary["dataset"]["valid"]["dataset_path"])
    task = path.parts[1]
    split = summary["name"]
    return "{}/{}".format(task, split)


def mask_demo_names(hdf5_file, filter_key):
    mask = hdf5_file["mask"][filter_key]
    return [name.decode("utf-8") for name in mask[:]]


def load_actions(dataset_meta, split="valid"):
    meta = dataset_meta[split]
    actions = []
    with h5py.File(meta["dataset_path"], "r") as f:
        for demo_key in mask_demo_names(f, meta["filter_key"]):
            actions.append(f["data"][demo_key][meta["action_key"]][:])
    return np.concatenate(actions, axis=0)


def pca_projector(actions):
    center = actions.mean(axis=0, keepdims=True)
    centered = actions - center
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    variance = singular_values**2 / max(actions.shape[0] - 1, 1)
    explained = variance[:2] / variance.sum()
    return center, components, explained


def project(actions, center, components):
    return (actions - center) @ components.T


def sample_gaussian(model, count, rng):
    mean = model["mean"]
    covariance = model["covariance"]
    std = np.sqrt(covariance)
    return rng.normal(mean, std, size=(count, mean.shape[0]))


def sample_gmm(model, count, rng):
    weights = model["weights"]
    means = model["means"]
    covariances = model["covariances"]
    modes = rng.choice(len(weights), size=count, p=weights / weights.sum())
    samples = np.empty((count, means.shape[1]), dtype=np.float64)
    for mode in range(len(weights)):
        inds = np.flatnonzero(modes == mode)
        if inds.size == 0:
            continue
        samples[inds] = rng.normal(means[mode], np.sqrt(covariances[mode]), size=(inds.size, means.shape[1]))
    return samples


def load_model(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files if key != "metadata_json"}


def subsample(points, count, rng):
    if points.shape[0] <= count:
        return points
    inds = rng.choice(points.shape[0], size=count, replace=False)
    return points[inds]


def configure_scatter_axis(ax, title, points, limits):
    ax.scatter(points[:, 0], points[:, 1], s=4, alpha=0.22, linewidths=0, color="#2f5f8f")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#bbbbbb")


def axis_limits(*point_sets):
    stacked = np.concatenate(point_sets, axis=0)
    x_lo, y_lo = np.percentile(stacked, 1, axis=0)
    x_hi, y_hi = np.percentile(stacked, 99, axis=0)
    x_pad = max((x_hi - x_lo) * 0.08, 1e-6)
    y_pad = max((y_hi - y_lo) * 0.08, 1e-6)
    return (x_lo - x_pad, x_hi + x_pad), (y_lo - y_pad, y_hi + y_pad)


def plot_nll(ax, gaussian_summary, gmm_summary):
    vals = [
        gaussian_summary["selection"]["selected"]["val_nll"],
        gmm_summary["selection"]["selected"]["val_nll"],
    ]
    train_vals = [
        gaussian_summary["selection"]["selected"]["train_nll"],
        gmm_summary["selection"]["selected"]["train_nll"],
    ]
    labels = ["Gaussian", "GMM"]
    colors = ["#8b8b8b", "#2f5f8f"]
    ax.bar(labels, vals, color=colors, width=0.62)
    ax.scatter(labels, train_vals, marker="_", s=180, color="#c0392b", linewidths=2, label="train")
    ax.set_title("NLL (lower is better)", fontsize=10)
    ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#bbbbbb")


def plot_dimension_distributions(label, actions, gaussian_samples, gmm_samples, gaussian_summary, gmm_summary):
    action_dim = actions.shape[1]
    ncols = 4
    nrows = int(np.ceil(action_dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    for dim in range(action_dim):
        ax = axes[dim]
        values = [actions[:, dim], gaussian_samples[:, dim], gmm_samples[:, dim]]
        lo, hi = np.percentile(np.concatenate(values), [0.5, 99.5])
        if lo == hi:
            lo -= 1.0
            hi += 1.0
        bins = np.linspace(lo, hi, 70)
        ax.hist(actions[:, dim], bins=bins, density=True, color="#222222", alpha=0.25, label="valid")
        ax.hist(gaussian_samples[:, dim], bins=bins, density=True, histtype="step", linewidth=1.4, color="#8b8b8b", label="Gaussian")
        ax.hist(gmm_samples[:, dim], bins=bins, density=True, histtype="step", linewidth=1.4, color="#2f5f8f", label="GMM")
        ax.set_title("action[{}]".format(dim), fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(axis="y", alpha=0.2)
        if dim == 0:
            ax.legend(fontsize=8, frameon=False)
    for ax in axes[action_dim:]:
        ax.axis("off")
    fig.suptitle(
        "{} action-dimension distributions | val NLL: Gaussian {:.2f}, GMM {:.2f}".format(
            label,
            gaussian_summary["selection"]["selected"]["val_nll"],
            gmm_summary["selection"]["selected"]["val_nll"],
        ),
        fontsize=13,
    )
    DIM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = DIM_OUTPUT_DIR / "{}.png".format(label.replace("/", "_"))
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main():
    rng = np.random.default_rng(1)
    summaries = {}
    for path in SUMMARY_PATHS:
        summary = load_summary(path)
        summaries.setdefault(dataset_label(summary), {})[summary["distribution"]] = summary

    rows = sorted(summaries)
    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.3 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = axes[None, :]

    for row, label in enumerate(rows):
        gaussian_summary = summaries[label]["gaussian"]
        gmm_summary = summaries[label]["gmm"]
        actions = load_actions(gaussian_summary["dataset"], split="valid")
        center, components, explained = pca_projector(actions)
        sample_count = min(6000, actions.shape[0])
        real_actions = subsample(actions, sample_count, rng)
        gaussian_samples = sample_gaussian(load_model(gaussian_summary["model_path"]), sample_count, rng)
        gmm_samples = sample_gmm(load_model(gmm_summary["model_path"]), sample_count, rng)
        real = project(real_actions, center, components)
        gaussian = project(gaussian_samples, center, components)
        gmm = project(gmm_samples, center, components)
        limits = axis_limits(real, gaussian, gmm)
        plot_dimension_distributions(label, real_actions, gaussian_samples, gmm_samples, gaussian_summary, gmm_summary)

        configure_scatter_axis(
            axes[row, 0],
            "{} valid actions\nn={} dim={} PCA {:.0f}%/{:.0f}%".format(
                label, actions.shape[0], actions.shape[1], 100 * explained[0], 100 * explained[1]
            ),
            real,
            limits,
        )
        configure_scatter_axis(
            axes[row, 1],
            "Gaussian samples\nval NLL {:.2f}".format(gaussian_summary["selection"]["selected"]["val_nll"]),
            gaussian,
            limits,
        )
        configure_scatter_axis(
            axes[row, 2],
            "GMM samples\nval NLL {:.2f}".format(gmm_summary["selection"]["selected"]["val_nll"]),
            gmm,
            limits,
        )
        plot_nll(axes[row, 3], gaussian_summary, gmm_summary)

    fig.suptitle("Action Prior Fit Diagnostics: Validation Actions vs Prior Samples", fontsize=14)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()
