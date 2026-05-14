"""
Generate a static HTML visualization for policy score-delta JSON files.

The input is produced by extract_policy_score_deltas.py and contains paired
visual-policy and robot-only NLL / entropy scores at trajectory and step level.
Both plain .json and gzip-compressed .json.gz files are supported.

Example:
    python robomimic/scripts/visualize_policy_score_deltas.py \
        --score-json vis/policy_score_deltas/square/example.json.gz \
        --output-dir vis/policy_score_delta_vis/example
"""

import argparse
import csv
import gzip
import html
import json
import os
import random
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import h5py
import numpy as np
from PIL import Image, ImageDraw


SCORE_FIELDS = OrderedDict(
    [
        ("nll_images_robot", {"label": "NLL images+robot", "short": "NLL img+robot", "color": "#177a78"}),
        ("entropy_images_robot", {"label": "Entropy images+robot", "short": "H img+robot", "color": "#7c5cbf"}),
        ("nll_robot", {"label": "NLL robot", "short": "NLL robot", "color": "#8c564b"}),
        ("entropy_robot", {"label": "Entropy robot", "short": "H robot", "color": "#2e7ab8"}),
        (
            "nll_images_robot_minus_robot",
            {"label": "NLL images+robot - robot", "short": "NLL delta", "color": "#b54708"},
        ),
        (
            "entropy_images_robot_minus_robot",
            {"label": "Entropy images+robot - robot", "short": "H delta", "color": "#2f7d32"},
        ),
        (
            "nll_nonconditional",
            {"label": "NLL non-conditional", "short": "NLL p(a)", "color": "#6f6a22"},
        ),
        (
            "entropy_nonconditional",
            {"label": "Entropy non-conditional", "short": "H p(a)", "color": "#6b7280"},
        ),
        (
            "nll_images_robot_minus_nonconditional",
            {"label": "NLL images+robot - non-conditional", "short": "NLL - p(a)", "color": "#c2410c"},
        ),
        (
            "entropy_images_robot_minus_nonconditional",
            {"label": "Entropy images+robot - non-conditional", "short": "H - p(a)", "color": "#15803d"},
        ),
        (
            "global_normalized_nll_images_robot",
            {"label": "Global norm NLL images+robot", "short": "norm NLL img+robot", "color": "#0096a6"},
        ),
        (
            "global_normalized_entropy_images_robot",
            {"label": "Global norm entropy images+robot", "short": "norm H img+robot", "color": "#9a6fd1"},
        ),
        (
            "global_normalized_nll_images_robot_minus_robot",
            {"label": "Global norm NLL images+robot - robot", "short": "norm NLL delta", "color": "#d06f1f"},
        ),
        (
            "global_normalized_entropy_images_robot_minus_robot",
            {"label": "Global norm entropy images+robot - robot", "short": "norm H delta", "color": "#4f9a48"},
        ),
    ]
)

PRIMARY_FIELDS = [
    "nll_images_robot",
    "entropy_images_robot",
    "nll_images_robot_minus_robot",
    "entropy_images_robot_minus_robot",
    "nll_images_robot_minus_nonconditional",
    "entropy_images_robot_minus_nonconditional",
]

DISTRIBUTION_FIELDS = PRIMARY_FIELDS + [
    "global_normalized_nll_images_robot",
    "global_normalized_entropy_images_robot",
    "global_normalized_nll_images_robot_minus_robot",
    "global_normalized_entropy_images_robot_minus_robot",
]

NLL_DISTRIBUTION_FIELDS = [key for key in DISTRIBUTION_FIELDS if "nll" in key]
ENTROPY_DISTRIBUTION_FIELDS = [key for key in DISTRIBUTION_FIELDS if "entropy" in key]

NORMALIZED_SCORE_SOURCES = OrderedDict(
    [
        ("global_normalized_nll_images_robot", "nll_images_robot"),
        ("global_normalized_entropy_images_robot", "entropy_images_robot"),
        ("global_normalized_nll_images_robot_minus_robot", "nll_images_robot_minus_robot"),
        ("global_normalized_entropy_images_robot_minus_robot", "entropy_images_robot_minus_robot"),
    ]
)

CSV_DATASET_ALIASES = {
    "ph": "square_ph",
    "mh": "square_mh",
    "random_post": "expert200",
}

MH_QUALITY_MASK_LABELS = OrderedDict(
    [
        ("worse", "1"),
        ("okay", "2"),
        ("better", "3"),
    ]
)


def natural_key(name):
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", str(name))]


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def artifact_stem(path):
    name = os.path.basename(str(path))
    for suffix in (".json.gz", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def camera_name_from_metadata(metadata):
    checkpoint = str(metadata.get("image_checkpoint") or metadata.get("image_dataset") or "")
    return "left_close_low" if "left_close_low" in checkpoint or "left_low_close" in checkpoint else "agentview"


def parse_run_stem(stem):
    pattern = re.compile(
        r"^policy_scores_(?P<dataset>.+)_(?P<policy>discrete_gaussian|discrete|gmm)_(?P<wd>wd[^_]+)_"
        r"v(?P<vkind>[^_]+)(?:_(?P<vepoch>e\d+))?_r(?P<rkind>[^_]+)(?:_(?P<repoch>e\d+))?_(?P<hash>[A-Za-z0-9]+)$"
    )
    match = pattern.match(stem)
    return match.groupdict() if match else {}


def readable_run_title(stem, metadata=None):
    info = parse_run_stem(stem)
    if not info:
        return stem
    camera = camera_name_from_metadata(metadata or {})
    pieces = [
        "{} dataset".format(info["dataset"]),
        "{} policy".format(info["policy"]),
        info["wd"],
        camera,
        "visual {}{}".format(info["vkind"], " " + info["vepoch"] if info.get("vepoch") else ""),
        "robot {}{}".format(info["rkind"], " " + info["repoch"] if info.get("repoch") else ""),
    ]
    return " · ".join(pieces)


def label_badge(label, kind):
    if label is None:
        return ""
    cls = "val-{}".format(safe_name(str(label)).lower())
    return '<span class="label-badge {} {}">{}</span>'.format(
        html.escape(kind),
        html.escape(cls),
        html.escape(str(label)),
    )


def iter_score_json_paths(directory):
    directory = Path(directory)
    by_stem = {}
    for path in directory.glob("*.json"):
        by_stem[artifact_stem(path)] = path
    for path in directory.glob("*.json.gz"):
        by_stem[artifact_stem(path)] = path
    return [by_stem[key] for key in sorted(by_stem, key=natural_key)]


def episode_index_from_key(episode_key):
    match = re.search(r"(\d+)$", str(episode_key))
    return int(match.group(1)) if match is not None else None


def dataset_group_from_path(path):
    parts = Path(str(path)).parts
    for idx, part in enumerate(parts[:-1]):
        if part == "square" and idx + 1 < len(parts) and parts[idx + 1] in CSV_DATASET_ALIASES:
            return parts[idx + 1]
    text = str(path)
    for dataset_group in CSV_DATASET_ALIASES:
        if dataset_group in text:
            return dataset_group
    return None


def resolve_path(path, base_dir):
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "r", encoding="utf-8")


def load_payload(score_json_path):
    with open_text(score_json_path) as f:
        return json.load(f)


def find_json_key(buffer, key):
    return buffer.find(json.dumps(key))


def json_object_end(buffer, start):
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(buffer)):
        char = buffer[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def load_json_field(score_json_path, key, opening_char):
    decoder = json.JSONDecoder()
    buffer = ""
    with open_text(score_json_path) as f:
        while True:
            chunk = f.read(1024 * 1024)
            if chunk == "":
                raise ValueError("Could not find {} in {}".format(key, score_json_path))
            buffer += chunk
            key_pos = find_json_key(buffer, key)
            if key_pos < 0:
                buffer = buffer[-128:]
                continue
            value_start = buffer.find(opening_char, key_pos)
            if value_start < 0:
                continue
            value_end = json_object_end(buffer, value_start)
            if value_end is None:
                continue
            return decoder.raw_decode(buffer[value_start:value_end])[0]


def load_metadata(score_json_path):
    return load_json_field(score_json_path, "metadata", "{")


def load_run_scores(score_json_path):
    return load_json_field(score_json_path, "run_scores", "{")


def load_trajectories(score_json_path):
    return load_json_field(score_json_path, "trajectories", "[")


def load_transitions(score_json_path):
    return load_json_field(score_json_path, "transitions", "[")


def stream_transitions(score_json_path):
    decoder = json.JSONDecoder()
    buffer = ""
    in_array = False
    with open_text(score_json_path) as f:
        while True:
            if not in_array:
                chunk = f.read(1024 * 1024)
                if chunk == "":
                    raise ValueError("Could not find transitions array in {}".format(score_json_path))
                buffer += chunk
                key_pos = find_json_key(buffer, "transitions")
                if key_pos < 0:
                    buffer = buffer[-128:]
                    continue
                array_start = buffer.find("[", key_pos)
                if array_start < 0:
                    continue
                buffer = buffer[array_start + 1 :]
                in_array = True

            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                return
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()

            try:
                transition, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = f.read(1024 * 1024)
                if chunk == "":
                    raise
                buffer += chunk
                continue

            yield transition
            buffer = buffer[end:]


def sample_transitions_streaming(score_json_path, num_episodes, steps_per_episode, seed):
    rng = random.Random(seed)
    grouped_samples = {}
    grouped_counts = defaultdict(int)

    for transition in stream_transitions(score_json_path):
        ep = transition["episode_key"]
        grouped_counts[ep] += 1
        samples = grouped_samples.setdefault(ep, [])
        if len(samples) < steps_per_episode:
            samples.append(transition)
            continue
        replace_index = rng.randrange(grouped_counts[ep])
        if replace_index < steps_per_episode:
            samples[replace_index] = transition

    episode_keys = sorted(grouped_samples.keys(), key=natural_key)
    if not episode_keys:
        raise ValueError("No transitions available to sample.")
    sampled_eps = rng.sample(episode_keys, k=min(num_episodes, len(episode_keys)))

    samples = []
    for ep in sampled_eps:
        sampled = grouped_samples[ep]
        sampled.sort(key=lambda item: item["step"])
        samples.extend(sampled)
    return samples


def sample_transitions(transitions, num_episodes, steps_per_episode, seed):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for transition in transitions:
        grouped[transition["episode_key"]].append(transition)
    episode_keys = sorted(grouped.keys(), key=natural_key)
    if not episode_keys:
        raise ValueError("No transitions available to sample.")
    sampled_eps = rng.sample(episode_keys, k=min(num_episodes, len(episode_keys)))
    samples = []
    for ep in sampled_eps:
        ep_transitions = grouped[ep]
        sampled = rng.sample(ep_transitions, k=min(steps_per_episode, len(ep_transitions)))
        sampled.sort(key=lambda item: item["step"])
        samples.extend(sampled)
    return samples


def collect_episode_step_series(score_json_path, episode_keys):
    wanted = set(episode_keys)
    grouped = defaultdict(list)
    for transition in stream_transitions(score_json_path):
        ep = transition["episode_key"]
        if ep in wanted:
            grouped[ep].append(transition)
    for ep in grouped:
        grouped[ep].sort(key=lambda item: item["step"])
    return grouped


def collect_episode_step_series_from_transitions(transitions, episode_keys):
    wanted = set(episode_keys)
    grouped = defaultdict(list)
    for transition in transitions:
        ep = transition["episode_key"]
        if ep in wanted:
            grouped[ep].append(transition)
    for ep in grouped:
        grouped[ep].sort(key=lambda item: item["step"])
    return grouped


def default_observability_csv(repo_root):
    path = Path(repo_root) / "robomimic" / "datasets" / "square" / "observability_annotations.csv"
    return str(path) if path.exists() else None


def load_observability_labels(csv_path, dataset_group):
    if csv_path is None or dataset_group is None:
        return {}
    dataset_id = CSV_DATASET_ALIASES.get(dataset_group, dataset_group)
    labels = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dataset") != dataset_id:
                continue
            labels[int(row["ep_idx"])] = row["label"].strip()
    return labels


def decode_hdf5_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_mh_quality_labels(dataset_path):
    labels = {}
    with h5py.File(dataset_path, "r") as f:
        if "mask" not in f:
            return labels
        for mask_key, label in MH_QUALITY_MASK_LABELS.items():
            if "mask/{}".format(mask_key) not in f:
                continue
            for episode_key in f["mask/{}".format(mask_key)][:]:
                labels[decode_hdf5_string(episode_key)] = label
    return labels


def annotate_labels(items, observability_labels, mh_quality_labels):
    annotated = []
    for item in items:
        item = dict(item)
        episode_key = item.get("episode_key")
        episode_index = episode_index_from_key(episode_key)
        if observability_labels:
            item["observability_label"] = observability_labels.get(episode_index, "unlabeled")
        if mh_quality_labels:
            item["mh_quality_label"] = mh_quality_labels.get(str(episode_key), "unlabeled")
        annotated.append(item)
    return annotated


def default_obs_keys_for_visualization(dataset_path, metadata):
    context = " ".join(
        str(value)
        for value in [
            dataset_path,
            metadata.get("image_dataset") if metadata else None,
            metadata.get("image_checkpoint") if metadata else None,
        ]
        if value
    ).lower()
    third_person_key = (
        "left_close_low_image"
        if "left_close_low" in context or "left_low_close" in context
        else "agentview_image"
    )
    return (third_person_key, "robot0_eye_in_hand_image")


def third_person_camera_from_context(dataset_path, metadata):
    context = " ".join(
        str(value)
        for value in [
            dataset_path,
            metadata.get("image_dataset") if metadata else None,
            metadata.get("image_checkpoint") if metadata else None,
        ]
        if value
    ).lower()
    return "left_close_low" if "left_close_low" in context or "left_low_close" in context else "agent"


def camera_label(camera_key):
    return {
        "agent": "agent",
        "left_close_low": "left close low",
        "wrist": "wrist",
        "wrist_agent": "wrist + agent",
        "wrist_left_close_low": "wrist + left close low",
    }.get(camera_key, camera_key)


def video_path_for_episode(repo_root, dataset_group, episode, camera_key):
    if dataset_group is None or episode is None:
        return None
    dataset_id = "square_{}".format(dataset_group)
    videos_dir = Path(repo_root) / "vis" / "dataset_videos" / "videos" / "square" / dataset_group
    for split in ("train", "val"):
        path = videos_dir / "{}_{}_ep{:04d}_{}.mp4".format(dataset_id, split, int(episode), camera_key)
        if path.exists():
            return path
    return None


def video_urls_for_episode(repo_root, output_dir, dataset_group, episode, third_person_camera):
    combined_camera = "wrist_left_close_low" if third_person_camera == "left_close_low" else "wrist_agent"
    urls = {}
    for key in (combined_camera, "wrist", third_person_camera, "agent", "left_close_low"):
        path = video_path_for_episode(repo_root, dataset_group, episode, key)
        if path is not None:
            urls[key] = os.path.relpath(path, output_dir)
    return {
        "combined": urls.get(combined_camera),
        "combined_key": combined_camera,
        "wrist": urls.get("wrist"),
        "third_person": urls.get(third_person_camera) or urls.get("agent") or urls.get("left_close_low"),
        "third_person_key": third_person_camera if urls.get(third_person_camera) else "agent" if urls.get("agent") else "left_close_low",
    }


def get_frame(hdf5_file, episode_key, step, obs_keys):
    frames = []
    for key in obs_keys:
        dataset_path = "data/{}/obs/{}".format(episode_key, key)
        if dataset_path not in hdf5_file:
            continue
        frame = hdf5_file[dataset_path][step]
        if frame.ndim == 3 and frame.shape[-1] in (1, 3):
            frame = frame.astype(np.uint8)
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            frames.append((key, Image.fromarray(frame)))
    if not frames:
        raise ValueError("No RGB observation image found for {} step {}".format(episode_key, step))
    return frames


def compose_frames(frames):
    label_h = 18
    gap = 4
    width = sum(img.width for _, img in frames) + gap * (len(frames) - 1)
    height = max(img.height for _, img in frames) + label_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for key, img in frames:
        canvas.paste(img.convert("RGB"), (x, label_h))
        draw.text((x + 2, 2), key, fill=(20, 20, 20))
        x += img.width + gap
    return canvas


def save_transition_image(hdf5_file, image_cache, assets_dir, episode_key, step, obs_keys):
    cache_key = (episode_key, int(step))
    if cache_key in image_cache:
        return image_cache[cache_key]
    image = compose_frames(get_frame(hdf5_file, episode_key, int(step), obs_keys))
    filename = "{}_step_{:04d}.png".format(safe_name(episode_key), int(step))
    path = assets_dir / filename
    image.save(path)
    rel_path = "assets/{}".format(filename)
    image_cache[cache_key] = rel_path
    return rel_path


def format_value(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "{:.6g}".format(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(format_value(v) for v in value)
    return str(value)


def metric_boxes(item, fields=PRIMARY_FIELDS):
    boxes = []
    for key in fields:
        spec = SCORE_FIELDS[key]
        value = item.get(key)
        tone = "pos" if isinstance(value, (int, float)) and value > 0 else "neg" if isinstance(value, (int, float)) and value < 0 else "neutral"
        boxes.append(
            '<div class="metric {}"><span>{}</span><strong>{}</strong></div>'.format(
                tone,
                html.escape(spec["short"]),
                html.escape(format_value(value)),
            )
        )
    return "\n".join(boxes)


def metadata_table(metadata):
    rows = []
    for key, value in metadata.items():
        if isinstance(value, dict):
            value = json.dumps(value, sort_keys=True)
        rows.append(
            "<tr><th>{}</th><td>{}</td></tr>".format(
                html.escape(str(key)),
                html.escape(format_value(value)),
            )
        )
    return "<table>{}</table>".format("\n".join(rows))


def score_card_html(label, scores):
    return """
    <article class="summary-card">
      <h3>{label}</h3>
      <div class="metric-grid">{metrics}</div>
    </article>
    """.format(
        label=html.escape(label),
        metrics=metric_boxes(scores, fields=list(SCORE_FIELDS.keys())),
    )


def mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return None
    return sum(values) / len(values)


def percentile_normalize_values(values, lower_percentile=1.0, upper_percentile=99.0):
    values = np.asarray(values, dtype=np.float64)
    lower = float(np.percentile(values, lower_percentile))
    upper = float(np.percentile(values, upper_percentile))
    denom = upper - lower
    if denom <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - lower) / denom, 0.0, 1.0)


def ensure_normalized_scores(run_scores, trajectories, transitions):
    trajectory_by_key = {traj.get("episode_key"): traj for traj in trajectories}
    for normalized_key, raw_key in NORMALIZED_SCORE_SOURCES.items():
        if all(isinstance(traj.get(normalized_key), (int, float)) for traj in trajectories):
            continue
        raw_values = [
            transition.get(raw_key)
            for transition in transitions
            if isinstance(transition.get(raw_key), (int, float))
        ]
        if not raw_values:
            continue
        normalized_values = percentile_normalize_values(raw_values)
        episode_values = defaultdict(list)
        value_index = 0
        for transition in transitions:
            if not isinstance(transition.get(raw_key), (int, float)):
                continue
            normalized_value = float(normalized_values[value_index])
            value_index += 1
            transition.setdefault(normalized_key, normalized_value)
            episode_values[transition.get("episode_key")].append(normalized_value)
        for episode_key, values in episode_values.items():
            trajectory = trajectory_by_key.get(episode_key)
            if trajectory is not None:
                trajectory.setdefault(normalized_key, mean(values))
        if run_scores is not None:
            run_scores.setdefault(normalized_key, mean(normalized_values.tolist()))


def smooth_count_curve(values, x_min, x_max, points=120):
    if not values:
        return []
    if len(values) == 1 or x_min == x_max:
        center = values[0]
        width = max(abs(center) * 0.05, 0.5)
        x_min = center - width
        x_max = center + width
    value_range = x_max - x_min
    mean_value = mean(values) or 0.0
    variance = mean([(value - mean_value) ** 2 for value in values]) or 0.0
    std = variance ** 0.5
    bandwidth = max(1.06 * std * (len(values) ** -0.2), value_range / 32.0, 1e-6)
    bin_width = value_range / 24.0
    norm = 1.0 / (bandwidth * (2.0 * np.pi) ** 0.5)
    curve = []
    for i in range(points):
        x = x_min + (i / max(points - 1, 1)) * value_range
        density = sum(norm * np.exp(-0.5 * ((x - value) / bandwidth) ** 2) for value in values)
        curve.append((x, density * bin_width))
    return curve


def distribution_chart_svg(groups, labels, score_key, colors):
    grouped_values = []
    all_values = []
    for label in labels:
        values = [
            float(traj[score_key])
            for traj in groups[label]
            if isinstance(traj.get(score_key), (int, float))
        ]
        grouped_values.append((label, values))
        all_values.extend(values)
    if not all_values:
        return ""

    raw_x_min, raw_x_max = min(all_values), max(all_values)
    if raw_x_min == raw_x_max:
        pad = max(abs(raw_x_min) * 0.05, 0.5)
    else:
        pad = 0.05 * (raw_x_max - raw_x_min)
    x_min, x_max = raw_x_min - pad, raw_x_max + pad

    curves = [
        (label, smooth_count_curve(values, x_min, x_max), values)
        for label, values in grouped_values
        if values
    ]
    y_max = max([point[1] for _, curve, _ in curves for point in curve] + [1.0])

    width, height = 520, 280
    pl, pr, pt, pb = 58, 16, 28, 56
    iw, ih = width - pl - pr, height - pt - pb

    def to_x(x):
        return pl + ((x - x_min) / (x_max - x_min or 1.0)) * iw

    def to_y(y):
        return pt + ih - (y / y_max) * ih

    paths = []
    legend = []
    for idx, (label, curve, values) in enumerate(curves):
        color = colors[idx % len(colors)]
        points = " ".join("{:.1f},{:.1f}".format(to_x(x), to_y(y)) for x, y in curve)
        paths.append(
            '<polyline points="{}" fill="none" stroke="{}" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round"/>'.format(
                points,
                color,
            )
        )
        legend.append(
            '<span><i style="background:{}"></i>{} (n={})</span>'.format(
                color,
                html.escape(label),
                len(values),
            )
        )

    grid = "\n".join(
        '<line x1="{pl}" y1="{y:.1f}" x2="{x2}" y2="{y:.1f}" stroke="#dde3eb" stroke-dasharray="4 4"/>'.format(
            pl=pl,
            x2=pl + iw,
            y=pt + frac * ih,
        )
        for frac in (0.25, 0.5, 0.75)
    )
    return """
    <article class="dist-card">
      <h3>{score_label}</h3>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{score_label} trajectory distribution">
        <rect x="{pl}" y="{pt}" width="{iw}" height="{ih}" fill="#fbfcfe" rx="4"/>
        {grid}
        <line x1="{pl}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#17202a" stroke-width="1.2"/>
        <line x1="{pl}" y1="{pt}" x2="{pl}" y2="{bottom}" stroke="#17202a" stroke-width="1.2"/>
        <text x="{pl}" y="{height_minus_18}" text-anchor="middle">trajectory score</text>
        <text x="14" y="{mid_y}" text-anchor="middle" transform="rotate(-90 14 {mid_y})">episodes</text>
        <text x="{label_x}" y="{top_label_y}" text-anchor="end">{y_max:.3g}</text>
        <text x="{label_x}" y="{bottom_label_y}" text-anchor="end">0</text>
        <text x="{pl}" y="{x_label_y}" text-anchor="middle">{x_min:.3g}</text>
        <text x="{right}" y="{x_label_y}" text-anchor="middle">{x_max:.3g}</text>
        {paths}
      </svg>
      <div class="chart-legend">{legend}</div>
    </article>
    """.format(
        score_label=html.escape(SCORE_FIELDS[score_key]["label"]),
        width=width,
        height=height,
        pl=pl,
        pt=pt,
        iw=iw,
        ih=ih,
        bottom=pt + ih,
        right=pl + iw,
        height_minus_18=height - 18,
        mid_y=pt + ih / 2,
        label_x=pl - 7,
        top_label_y=pt + 5,
        bottom_label_y=pt + ih + 4,
        x_label_y=pt + ih + 18,
        y_max=y_max,
        x_min=x_min,
        x_max=x_max,
        grid=grid,
        paths="\n".join(paths),
        legend="".join(legend),
    )


def label_quality_value(label, label_key):
    if label is None or label == "unlabeled":
        return None
    if label_key == "observability_label":
        return {"partial": 1.0, "full": 2.0}.get(str(label))
    if label_key == "mh_quality_label":
        try:
            value = float(label)
        except (TypeError, ValueError):
            return None
        return value if value in (1.0, 2.0, 3.0) else None
    return None


def average_quality_after_filter_curve(trajectories, label_key, score_key):
    rows = []
    for traj in trajectories:
        quality = label_quality_value(traj.get(label_key), label_key)
        score = traj.get(score_key)
        if quality is None or not isinstance(score, (int, float)):
            continue
        rows.append((float(score), float(quality), str(traj.get("episode_key", ""))))
    if not rows:
        return []
    rows.sort(key=lambda item: (-item[0], natural_key(item[2])))
    total = sum(item[1] for item in rows)
    curve = []
    for filtered_count in range(len(rows)):
        remaining = len(rows) - filtered_count
        curve.append((filtered_count, total / remaining))
        total -= rows[filtered_count][1]
    return curve


def oracle_quality_after_filter_curve(trajectories, label_key):
    rows = []
    for traj in trajectories:
        quality = label_quality_value(traj.get(label_key), label_key)
        if quality is None:
            continue
        rows.append((float(quality), str(traj.get("episode_key", ""))))
    if not rows:
        return []
    rows.sort(key=lambda item: (item[0], natural_key(item[1])))
    total = sum(item[0] for item in rows)
    curve = []
    for filtered_count in range(len(rows)):
        remaining = len(rows) - filtered_count
        curve.append((filtered_count, total / remaining))
        total -= rows[filtered_count][0]
    return curve


def filter_quality_chart_svg(trajectories, label_key, score_key):
    score_curve = average_quality_after_filter_curve(trajectories, label_key, score_key)
    oracle_curve = oracle_quality_after_filter_curve(trajectories, label_key)
    if not score_curve or not oracle_curve:
        return ""

    y_values = [y for _, y in score_curve] + [y for _, y in oracle_curve]
    y_min, y_max = min(y_values), max(y_values)
    if y_min == y_max:
        y_min -= 0.25
        y_max += 0.25
    else:
        pad = 0.06 * (y_max - y_min)
        y_min -= pad
        y_max += pad
    if label_key == "observability_label":
        y_min = min(y_min, 1.0)
        y_max = max(y_max, 2.0)
        quality_label = "avg quality (partial=1, full=2)"
    else:
        y_min = min(y_min, 1.0)
        y_max = max(y_max, 3.0)
        quality_label = "avg quality label"

    x_max = max(score_curve[-1][0], oracle_curve[-1][0], 1)
    width, height = 520, 280
    pl, pr, pt, pb = 58, 16, 28, 56
    iw, ih = width - pl - pr, height - pt - pb

    def to_x(x):
        return pl + (x / x_max) * iw

    def to_y(y):
        return pt + ih - ((y - y_min) / (y_max - y_min or 1.0)) * ih

    def points(curve):
        return " ".join("{:.1f},{:.1f}".format(to_x(x), to_y(y)) for x, y in curve)

    grid = "\n".join(
        '<line x1="{pl}" y1="{y:.1f}" x2="{x2}" y2="{y:.1f}" stroke="#dde3eb" stroke-dasharray="4 4"/>'.format(
            pl=pl,
            x2=pl + iw,
            y=pt + frac * ih,
        )
        for frac in (0.25, 0.5, 0.75)
    )
    score_color = SCORE_FIELDS[score_key]["color"]
    return """
    <article class="dist-card">
      <h3>{score_label}</h3>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{score_label} quality after filtering">
        <rect x="{pl}" y="{pt}" width="{iw}" height="{ih}" fill="#fbfcfe" rx="4"/>
        {grid}
        <line x1="{pl}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#17202a" stroke-width="1.2"/>
        <line x1="{pl}" y1="{pt}" x2="{pl}" y2="{bottom}" stroke="#17202a" stroke-width="1.2"/>
        <text x="{pl}" y="{height_minus_18}" text-anchor="middle">episodes filtered</text>
        <text x="14" y="{mid_y}" text-anchor="middle" transform="rotate(-90 14 {mid_y})">{quality_label}</text>
        <text x="{label_x}" y="{top_label_y}" text-anchor="end">{y_max:.3g}</text>
        <text x="{label_x}" y="{bottom_label_y}" text-anchor="end">{y_min:.3g}</text>
        <text x="{pl}" y="{x_label_y}" text-anchor="middle">0</text>
        <text x="{right}" y="{x_label_y}" text-anchor="middle">{x_max}</text>
        <polyline points="{score_points}" fill="none" stroke="{score_color}" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round"/>
        <polyline points="{oracle_points}" fill="none" stroke="#221e1b" stroke-width="2.6" stroke-dasharray="6 5" stroke-linejoin="round" stroke-linecap="round"/>
      </svg>
      <div class="chart-legend">
        <span><i style="background:{score_color}"></i>filter highest {score_short} first</span>
        <span><i style="background:#221e1b"></i>oracle: lowest quality labels first</span>
      </div>
    </article>
    """.format(
        score_label=html.escape(SCORE_FIELDS[score_key]["label"]),
        score_short=html.escape(SCORE_FIELDS[score_key]["short"]),
        quality_label=html.escape(quality_label),
        width=width,
        height=height,
        pl=pl,
        pt=pt,
        iw=iw,
        ih=ih,
        bottom=pt + ih,
        right=pl + iw,
        height_minus_18=height - 18,
        mid_y=pt + ih / 2,
        label_x=pl - 7,
        top_label_y=pt + 5,
        bottom_label_y=pt + ih + 4,
        x_label_y=pt + ih + 18,
        y_max=y_max,
        y_min=y_min,
        x_max=int(x_max),
        grid=grid,
        score_points=points(score_curve),
        oracle_points=points(oracle_curve),
        score_color=score_color,
    )


def combined_filter_quality_chart_svg(trajectories, label_key, score_keys, title):
    curves = []
    for score_key in score_keys:
        score_curve = average_quality_after_filter_curve(trajectories, label_key, score_key)
        if score_curve:
            curves.append((score_key, score_curve))
    oracle_curve = oracle_quality_after_filter_curve(trajectories, label_key)
    if not curves or not oracle_curve:
        return ""

    y_values = [y for _, curve in curves for _, y in curve] + [y for _, y in oracle_curve]
    y_min, y_max = min(y_values), max(y_values)
    if y_min == y_max:
        y_min -= 0.25
        y_max += 0.25
    else:
        pad = 0.06 * (y_max - y_min)
        y_min -= pad
        y_max += pad
    if label_key == "observability_label":
        y_min = min(y_min, 1.0)
        y_max = max(y_max, 2.0)
        quality_label = "avg quality (partial=1, full=2)"
    else:
        y_min = min(y_min, 1.0)
        y_max = max(y_max, 3.0)
        quality_label = "avg quality label"

    x_max = max([curve[-1][0] for _, curve in curves] + [oracle_curve[-1][0], 1])
    width, height = 920, 360
    pl, pr, pt, pb = 58, 18, 26, 56
    iw, ih = width - pl - pr, height - pt - pb

    def to_x(x):
        return pl + (x / x_max) * iw

    def to_y(y):
        return pt + ih - ((y - y_min) / (y_max - y_min or 1.0)) * ih

    def points(curve):
        return " ".join("{:.1f},{:.1f}".format(to_x(x), to_y(y)) for x, y in curve)

    grid = "\n".join(
        '<line x1="{pl}" y1="{y:.1f}" x2="{x2}" y2="{y:.1f}" stroke="#dde3eb" stroke-dasharray="4 4"/>'.format(
            pl=pl,
            x2=pl + iw,
            y=pt + frac * ih,
        )
        for frac in (0.25, 0.5, 0.75)
    )
    paths = []
    legend = []
    for score_key, curve in curves:
        spec = SCORE_FIELDS[score_key]
        color = spec["color"]
        paths.append(
            '<polyline points="{}" fill="none" stroke="{}" stroke-width="2.2" stroke-opacity="0.92" stroke-linejoin="round" stroke-linecap="round"/>'.format(
                points(curve),
                color,
            )
        )
        legend.append(
            '<span><i style="background:{}"></i>{}</span>'.format(
                color,
                html.escape(spec["short"]),
            )
        )
    legend.append('<span><i style="background:#221e1b"></i>oracle</span>')

    return """
    <article class="dist-card combined">
      <h3>{title}</h3>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{title} quality after filtering">
        <rect x="{pl}" y="{pt}" width="{iw}" height="{ih}" fill="#fbfcfe" rx="4"/>
        {grid}
        <line x1="{pl}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#17202a" stroke-width="1.2"/>
        <line x1="{pl}" y1="{pt}" x2="{pl}" y2="{bottom}" stroke="#17202a" stroke-width="1.2"/>
        <text x="{pl}" y="{height_minus_18}" text-anchor="middle">episodes filtered</text>
        <text x="14" y="{mid_y}" text-anchor="middle" transform="rotate(-90 14 {mid_y})">{quality_label}</text>
        <text x="{label_x}" y="{top_label_y}" text-anchor="end">{y_max:.3g}</text>
        <text x="{label_x}" y="{bottom_label_y}" text-anchor="end">{y_min:.3g}</text>
        <text x="{pl}" y="{x_label_y}" text-anchor="middle">0</text>
        <text x="{right}" y="{x_label_y}" text-anchor="middle">{x_max}</text>
        {paths}
        <polyline points="{oracle_points}" fill="none" stroke="#221e1b" stroke-width="2.6" stroke-dasharray="6 5" stroke-linejoin="round" stroke-linecap="round"/>
      </svg>
      <div class="chart-legend compact">{legend}</div>
    </article>
    """.format(
        title=html.escape(title),
        quality_label=html.escape(quality_label),
        width=width,
        height=height,
        pl=pl,
        pt=pt,
        iw=iw,
        ih=ih,
        bottom=pt + ih,
        right=pl + iw,
        height_minus_18=height - 18,
        mid_y=pt + ih / 2,
        label_x=pl - 7,
        top_label_y=pt + 5,
        bottom_label_y=pt + ih + 4,
        x_label_y=pt + ih + 18,
        y_max=y_max,
        y_min=y_min,
        x_max=int(x_max),
        grid=grid,
        paths="\n".join(paths),
        oracle_points=points(oracle_curve),
        legend="\n".join(legend),
    )


def grouped_score_panel(trajectories, label_key, title, label_order=None):
    groups = defaultdict(list)
    for traj in trajectories:
        label = traj.get(label_key)
        if label is not None and label != "unlabeled":
            groups[str(label)].append(traj)
    if not groups:
        return ""

    labels = list(label_order or [])
    labels.extend(sorted(label for label in groups if label not in labels))
    labels = [label for label in labels if label in groups]
    colors = ["#177a78", "#c35a2d", "#7c5cbf", "#2e7ab8", "#7b6f23", "#a6425f"]
    charts = [
        distribution_chart_svg(groups, labels, score_key, colors)
        for score_key in DISTRIBUTION_FIELDS
    ]
    charts = [chart for chart in charts if chart]
    if not charts:
        return ""
    group_counts = ", ".join(
        "{} n={}".format(label, len(groups[label]))
        for label in labels
    )
    return """
    <section class="panel">
      <div class="panel-title">
        <h2>{title}</h2>
        <span>trajectory score distributions; {group_counts}</span>
      </div>
      <div class="label-strip">{label_badges}</div>
      <div class="dist-grid">{charts}</div>
    </section>
    """.format(
        title=html.escape(title),
        group_counts=html.escape(group_counts),
        label_badges="".join(label_badge("{} · {} episodes".format(label, len(groups[label])), "group") for label in labels),
        charts="\n".join(charts),
    )


def filter_quality_panel(trajectories, label_key, title):
    labeled_count = sum(1 for traj in trajectories if label_quality_value(traj.get(label_key), label_key) is not None)
    if labeled_count == 0:
        return ""
    combined_charts = [
        combined_filter_quality_chart_svg(trajectories, label_key, NLL_DISTRIBUTION_FIELDS, "NLL score filters"),
        combined_filter_quality_chart_svg(trajectories, label_key, ENTROPY_DISTRIBUTION_FIELDS, "Entropy score filters"),
    ]
    individual_charts = [
        filter_quality_chart_svg(trajectories, label_key, score_key)
        for score_key in DISTRIBUTION_FIELDS
    ]
    charts = [chart for chart in combined_charts + individual_charts if chart]
    if not charts:
        return ""
    return """
    <section class="panel">
      <div class="panel-title">
        <h2>{title}</h2>
        <span>remove high-score episodes first; oracle removes low-quality labels first · n={count}</span>
      </div>
      <div class="dist-grid">{charts}</div>
    </section>
    """.format(
        title=html.escape(title),
        count=labeled_count,
        charts="\n".join(charts),
    )


def trajectory_table(trajectories, sort_key, limit):
    sorted_traj = sorted(
        trajectories,
        key=lambda item: item.get(sort_key, -float("inf")) if isinstance(item.get(sort_key), (int, float)) else -float("inf"),
        reverse=True,
    )
    label_headers = []
    if any("observability_label" in traj for traj in trajectories):
        label_headers.append("observability_label")
    if any("mh_quality_label" in traj for traj in trajectories):
        label_headers.append("mh_quality_label")
    headers = ["episode_key"] + label_headers + ["num_steps"] + list(SCORE_FIELDS.keys())
    head = "".join("<th>{}</th>".format(html.escape(SCORE_FIELDS.get(h, {}).get("short", h))) for h in headers)
    score_options = "\n".join(
        '<option value="{key}"{selected}>{label}</option>'.format(
            key=html.escape(key),
            selected=" selected" if key == sort_key else "",
            label=html.escape(spec["label"]),
        )
        for key, spec in SCORE_FIELDS.items()
    )
    body_rows = []
    for traj in sorted_traj:
        cells = []
        for key in headers:
            value = traj.get(key)
            if key == "observability_label":
                cell = label_badge(value, "observability")
            elif key == "mh_quality_label":
                cell = label_badge(value, "quality")
            else:
                cell = html.escape(format_value(value))
            cells.append("<td>{}</td>".format(cell))
        score_attrs = " ".join(
            'data-{}="{}"'.format(key.replace("_", "-"), html.escape(format_value(traj.get(key))))
            for key in SCORE_FIELDS
        )
        body_rows.append("<tr {}>{}</tr>".format(score_attrs, "".join(cells)))
    return """
    <section class="panel">
      <div class="panel-title">
        <h2>Trajectory Scores</h2>
        <span>{count} episodes</span>
      </div>
      <div class="table-controls">
        <label>
          <span>Sort score</span>
          <select id="trajectory-sort-score">{score_options}</select>
        </label>
        <label>
          <span>Direction</span>
          <select id="trajectory-sort-direction">
            <option value="desc" selected>high to low</option>
            <option value="asc">low to high</option>
          </select>
        </label>
      </div>
      <div class="table-scroll">
        <table class="score-table"><thead><tr>{head}</tr></thead><tbody id="trajectory-table-body">{body}</tbody></table>
      </div>
    </section>
    """.format(
        count=len(sorted_traj),
        score_options=score_options,
        head=head,
        body="\n".join(body_rows),
    )


def score_data_attrs(item):
    return " ".join(
        'data-{}="{}"'.format(key.replace("_", "-"), html.escape(format_value(item.get(key))))
        for key in SCORE_FIELDS
    )


def step_chart_svg(series, selected_step):
    valid = [(key, values) for key, values in series.items() if values]
    if not valid:
        return ""
    width, height = 780, 230
    pl, pr, pt, pb = 54, 20, 18, 42
    iw, ih = width - pl - pr, height - pt - pb
    all_values = [value for _, values in valid for value in values if isinstance(value, (int, float))]
    if not all_values:
        return ""
    y_min, y_max = min(all_values), max(all_values)
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    x_len = max(len(values) for _, values in valid)

    def to_x(i):
        return pl + (i / max(x_len - 1, 1)) * iw

    def to_y(v):
        return pt + ih - ((v - y_min) / (y_max - y_min)) * ih

    selected_step = max(0, min(int(selected_step), x_len - 1))
    play_x = to_x(selected_step)
    zero_y = to_y(0)
    lines = []
    legend = []
    for idx, (key, values) in enumerate(valid):
        color = SCORE_FIELDS[key]["color"]
        pts = " ".join("{:.1f},{:.1f}".format(to_x(i), to_y(v)) for i, v in enumerate(values))
        lines.append(
            '<polyline points="{}" fill="none" stroke="{}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>'.format(
                pts,
                color,
            )
        )
        selected_value = values[selected_step] if selected_step < len(values) else None
        legend.append(
            '<span><i style="background:{}"></i>{}: <b>{}</b></span>'.format(
                color,
                html.escape(SCORE_FIELDS[key]["short"]),
                html.escape(format_value(selected_value)),
            )
        )

    grid = "\n".join(
        '<line x1="{pl}" y1="{y:.1f}" x2="{x2}" y2="{y:.1f}" stroke="#dde3eb" stroke-dasharray="4 4"/>'.format(
            pl=pl,
            x2=pl + iw,
            y=pt + frac * ih,
        )
        for frac in (0.25, 0.5, 0.75)
    )
    return """
    <div class="chart-wrap">
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="step score curves" data-step-count="{x_len}" data-plot-left="{pl}" data-plot-width="{iw}">
        <rect x="{pl}" y="{pt}" width="{iw}" height="{ih}" fill="#fbfcfe" rx="4"/>
        {grid}
        <line x1="{pl}" y1="{zero_y:.1f}" x2="{x2}" y2="{zero_y:.1f}" stroke="#6b7785" stroke-width="1.2"/>
        <line x1="{pl}" y1="{pt}" x2="{pl}" y2="{bottom}" stroke="#17202a" stroke-width="1.2"/>
        <text x="{label_x}" y="{top_label_y}" text-anchor="end">{y_max:.3g}</text>
        <text x="{label_x}" y="{bottom_label_y}" text-anchor="end">{y_min:.3g}</text>
        {lines}
        <line class="step-playhead" x1="{play_x:.1f}" y1="{pt}" x2="{play_x:.1f}" y2="{bottom}" stroke="#17202a" stroke-width="2.0" stroke-dasharray="3 3"/>
      </svg>
      <div class="chart-legend">{legend}</div>
    </div>
    """.format(
        width=width,
        height=height,
        x_len=x_len,
        pl=pl,
        pt=pt,
        iw=iw,
        ih=ih,
        x2=pl + iw,
        bottom=pt + ih,
        zero_y=zero_y,
        label_x=pl - 7,
        top_label_y=pt + 5,
        bottom_label_y=pt + ih + 4,
        y_max=y_max,
        y_min=y_min,
        play_x=play_x,
        grid=grid,
        lines="\n".join(lines),
        legend="".join(legend),
    )


def video_pane_html(src, camera_key, missing_label):
    if src:
        return """
        <div class="video-pane">
          <div class="camera-badge">{label} view</div>
          <video src="{src}" controls preload="metadata" playsinline muted></video>
        </div>
        """.format(
            label=html.escape(camera_label(camera_key)),
            src=html.escape(src),
        )
    return '<div class="video-pane missing">{}</div>'.format(html.escape(missing_label))


def trajectory_card_html(traj, episode_series, video_urls):
    series = {
        key: [transition.get(key) for transition in episode_series if isinstance(transition.get(key), (int, float))]
        for key in PRIMARY_FIELDS
    }
    label_bits = []
    if traj.get("observability_label"):
        label_bits.append(label_badge(traj.get("observability_label"), "observability"))
    if traj.get("mh_quality_label"):
        label_bits.append(label_badge(traj.get("mh_quality_label"), "quality"))
    label_html = '<div class="trajectory-labels">{}</div>'.format("".join(label_bits)) if label_bits else ""

    if video_urls.get("combined"):
        media_html = """
        <div class="demo-media combined">{combined}</div>
        """.format(
            combined=video_pane_html(video_urls["combined"], video_urls["combined_key"], "Combined video unavailable"),
        )
    else:
        media_html = """
        <div class="demo-media two-up">
          {wrist}
          {third_person}
        </div>
        """.format(
            wrist=video_pane_html(video_urls.get("wrist"), "wrist", "Wrist video unavailable"),
            third_person=video_pane_html(
                video_urls.get("third_person"),
                video_urls.get("third_person_key", "agent"),
                "Third-person video unavailable",
            ),
        )

    return """
    <article class="trajectory-card" {score_attrs}>
      {media_html}
      <div class="transition-body">
        <div class="transition-head">
          <h3>{episode_key}</h3>
          <span>{num_steps} steps</span>
        </div>
        {label_html}
        <div class="metric-grid">{metrics}</div>
        {chart}
      </div>
    </article>
    """.format(
        score_attrs=score_data_attrs(traj),
        media_html=media_html,
        episode_key=html.escape(str(traj.get("episode_key", "episode"))),
        num_steps=html.escape(format_value(traj.get("num_steps", len(episode_series)))),
        label_html=label_html,
        metrics=metric_boxes(traj, fields=list(SCORE_FIELDS.keys())),
        chart=step_chart_svg(series, 0),
    )


def transition_card_html(item, image_path, episode_series, video_urls):
    series = {
        key: [transition.get(key) for transition in episode_series if isinstance(transition.get(key), (int, float))]
        for key in PRIMARY_FIELDS
    }
    labels = []
    if item.get("observability_label"):
        labels.append("observability: {}".format(item["observability_label"]))
    if item.get("mh_quality_label"):
        labels.append("MH label: {}".format(item["mh_quality_label"]))
    label_text = " · ".join(labels)
    label_html = '<span class="tag-line">{}</span>'.format(html.escape(label_text)) if label_text else ""
    if video_urls.get("combined"):
        media_html = """
        <div class="demo-media combined">{combined}</div>
        """.format(
            combined=video_pane_html(video_urls["combined"], video_urls["combined_key"], "Combined video unavailable"),
        )
    elif video_urls.get("wrist") or video_urls.get("third_person"):
        media_html = """
        <div class="demo-media two-up">
          {wrist}
          {third_person}
        </div>
        """.format(
            wrist=video_pane_html(video_urls.get("wrist"), "wrist", "Wrist video unavailable"),
            third_person=video_pane_html(
                video_urls.get("third_person"),
                video_urls.get("third_person_key", "agent"),
                "Third-person video unavailable",
            ),
        )
    else:
        media_html = """
        <div class="transition-media">
          <img src="{image_path}" alt="{alt}">
        </div>
        """.format(
            image_path=html.escape(image_path),
            alt=html.escape("{} step {}".format(item["episode_key"], item["step"])),
        )
    return """
    <article class="transition-card">
      {media_html}
      <div class="transition-body">
        <div class="transition-head">
          <h3>{episode_key} step {step}</h3>
          <span>{num_steps} steps in episode</span>
        </div>
        {label_html}
        <div class="metric-grid">{metrics}</div>
        {chart}
      </div>
    </article>
    """.format(
        media_html=media_html,
        episode_key=html.escape(str(item["episode_key"])),
        step=html.escape(str(item["step"])),
        num_steps=len(episode_series),
        label_html=label_html,
        metrics=metric_boxes(item),
        chart=step_chart_svg(series, item["step"]),
    )


def build_html(
    score_json_path,
    dataset_path,
    metadata,
    run_scores,
    trajectories,
    trajectory_cards,
    sort_key,
    summary_only=False,
):
    title = readable_run_title(artifact_stem(score_json_path), metadata)
    trajectory_card_section = ""
    if not summary_only:
        trajectory_card_section = """
    <section class="panel">
      <div class="panel-title">
        <h2>Trajectory Videos And Step Scores</h2>
        <span>all trajectories, ordered by the score selector above</span>
      </div>
      <div class="sample-grid" data-sortable-score-cards>{trajectory_cards}</div>
    </section>
        """.format(
            trajectory_cards="\n".join(trajectory_cards),
        )
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #efe6d4;
      --panel: #fffaf1;
      --panel-2: #f8f2e8;
      --ink: #221e1b;
      --muted: #70665c;
      --line: #d4c4ab;
      --line-strong: #bea888;
      --pos: #177a78;
      --neg: #c35a2d;
      --neutral: #4d5b6a;
      --purple: #7c5cbf;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #f3ecde 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    header {{
      padding: 24px 28px 12px;
      background: rgba(255,250,241,0.92);
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 8px; font-size: 25px; letter-spacing: 0; }}
    h2, h3 {{ letter-spacing: 0; }}
    .subhead {{ margin: 0; color: var(--muted); font-size: 14px; overflow-wrap: anywhere; }}
    main {{ padding: 18px 24px 36px; display: grid; gap: 18px; }}
    .summary-grid, .sample-grid {{ display: grid; gap: 14px; }}
    .summary-grid {{ grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    .summary-card, .panel, .transition-card, .trajectory-card, details {{
      background: rgba(255,250,241,0.94);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .summary-card {{ padding: 16px; }}
    .summary-card h3 {{ margin: 0 0 12px; font-size: 16px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; }}
    .metric {{
      border: 1px solid var(--line);
      background: var(--panel-2);
      border-radius: 8px;
      padding: 9px 10px;
      display: grid;
      gap: 4px;
    }}
    .metric span {{ color: var(--muted); font-size: 12px; }}
    .metric strong {{ font-size: 17px; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }}
    .metric.pos strong {{ color: var(--pos); }}
    .metric.neg strong {{ color: var(--neg); }}
    .metric.neutral strong {{ color: var(--neutral); }}
    details {{ padding: 14px 16px; }}
    summary {{ cursor: pointer; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ padding: 7px 8px; border-bottom: 1px solid #eadfce; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 700; }}
    td {{ font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }}
    .panel-title {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }}
    .panel-title h2 {{ margin: 0; font-size: 18px; }}
    .panel-title span {{ color: var(--muted); font-size: 13px; }}
    .table-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
    }}
    .table-controls label {{ display: grid; gap: 6px; min-width: 220px; }}
    .table-controls span {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }}
    .table-controls select {{
      border: 1px solid var(--line-strong);
      border-radius: 999px;
      background: var(--panel);
      padding: 10px 13px;
      color: var(--ink);
      font: inherit;
    }}
    .table-scroll {{ overflow-x: auto; }}
    .score-table th, .score-table td {{ white-space: nowrap; }}
    .sample-grid {{ grid-template-columns: 1fr; }}
    .transition-card, .trajectory-card {{ display: grid; grid-template-columns: minmax(360px, 0.85fr) 1fr; }}
    .transition-media {{ border-right: 1px solid var(--line); background: #eef2f6; }}
    .demo-media {{
      display: grid;
      gap: 1px;
      background: var(--line);
      border-right: 1px solid var(--line);
      align-content: start;
    }}
    .demo-media.two-up {{ grid-template-columns: 1fr 1fr; }}
    .demo-media.combined {{ grid-template-columns: 1fr; }}
    .video-pane {{ position: relative; min-height: 180px; background: #eef2f6; display: grid; place-items: center; }}
    .video-pane video {{ width: 100%; height: auto; display: block; }}
    .video-pane.missing {{ color: var(--muted); font-size: 13px; padding: 16px; }}
    .camera-badge {{
      position: absolute;
      top: 8px;
      left: 8px;
      z-index: 1;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid var(--line);
      padding: 5px 8px;
      font-size: 11px;
      color: var(--muted);
    }}
    img {{ display: block; width: 100%; height: auto; }}
    .transition-body {{ padding: 14px; display: grid; gap: 12px; }}
    .transition-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }}
    .transition-head h3 {{ margin: 0; font-size: 17px; }}
    .transition-head span {{ color: var(--muted); font-size: 13px; }}
    .tag-line {{ color: var(--muted); font-size: 13px; display: flex; flex-wrap: wrap; gap: 6px; }}
    .trajectory-labels {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .label-badge {{
      display: inline-flex;
      align-items: center;
      width: max-content;
      border-radius: 999px;
      border: 1px solid var(--line-strong);
      background: #fff7ea;
      color: var(--ink);
      padding: 5px 9px;
      font-size: 12px;
      font-weight: 700;
    }}
    .label-badge.val-full, .label-badge.observability.val-full {{ background: #d9ebe1; color: #145b58; }}
    .label-badge.val-partial, .label-badge.observability.val-partial {{ background: #f1dfd0; color: #8d3f1f; }}
    .label-badge.val-1, .label-badge.quality.val-1 {{ background: #f1dfd0; color: #8d3f1f; }}
    .label-badge.val-2, .label-badge.quality.val-2 {{ background: #fff0bd; color: #725a00; }}
    .label-badge.val-3, .label-badge.quality.val-3 {{ background: #d9ebe1; color: #145b58; }}
    .label-strip {{ display: flex; flex-wrap: wrap; gap: 8px; padding: 12px 16px 0; }}
    .chart-wrap {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px; overflow-x: auto; background: var(--panel-2); }}
    .dist-grid {{
      padding: 14px;
      display: grid;
      grid-template-columns: repeat(2, minmax(320px, 1fr));
      gap: 14px;
    }}
    .dist-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--panel-2);
      overflow-x: auto;
    }}
    .dist-card.combined {{ grid-column: 1 / -1; }}
    .dist-card h3 {{ margin: 0 0 8px; font-size: 14px; }}
    svg {{ display: block; min-width: 620px; width: 100%; height: auto; }}
    .dist-card.combined svg {{ min-width: 900px; }}
    svg text {{ font-size: 11px; fill: var(--muted); }}
    .chart-legend {{ display: flex; flex-wrap: wrap; gap: 10px 16px; margin-top: 8px; font-size: 12px; color: var(--muted); }}
    .chart-legend.compact {{ gap: 8px 12px; }}
    .chart-legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .chart-legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; }}
    .chart-legend b {{ color: var(--ink); font-variant-numeric: tabular-nums; }}
    @media (max-width: 820px) {{
      main {{ padding: 12px; }}
      .dist-grid {{ grid-template-columns: 1fr; padding: 10px; }}
      .transition-card {{ grid-template-columns: 1fr; }}
      .transition-media, .demo-media {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .demo-media.two-up {{ grid-template-columns: 1fr; }}
    }}
  </style>
  <script>
    function numericAttr(row, key) {{
      const raw = row.getAttribute(`data-${{key.replaceAll("_", "-")}}`);
      const value = Number(raw);
      return Number.isFinite(value) ? value : null;
    }}
    function sortTrajectoryTable() {{
      const score = document.getElementById("trajectory-sort-score");
      const direction = document.getElementById("trajectory-sort-direction");
      const body = document.getElementById("trajectory-table-body");
      if (!score || !direction || !body) return;
      const rows = Array.from(body.querySelectorAll("tr"));
      const desc = direction.value === "desc";
      rows.sort((a, b) => {{
        const av = numericAttr(a, score.value);
        const bv = numericAttr(b, score.value);
        if (av === null && bv === null) return 0;
        if (av === null) return 1;
        if (bv === null) return -1;
        return desc ? bv - av : av - bv;
      }});
      rows.forEach((row) => body.appendChild(row));
      document.querySelectorAll("[data-sortable-score-cards]").forEach((container) => {{
        const cards = Array.from(container.querySelectorAll(".trajectory-card"));
        cards.sort((a, b) => {{
          const av = numericAttr(a, score.value);
          const bv = numericAttr(b, score.value);
          if (av === null && bv === null) return 0;
          if (av === null) return 1;
          if (bv === null) return -1;
          return desc ? bv - av : av - bv;
        }});
        cards.forEach((card) => container.appendChild(card));
      }});
    }}
    function syncStepPlayheads(card, sourceVideo) {{
      if (!card || !sourceVideo) return;
      const duration = Number.isFinite(sourceVideo.duration) && sourceVideo.duration > 0 ? sourceVideo.duration : null;
      const progress = duration ? Math.max(0, Math.min(1, sourceVideo.currentTime / duration)) : 0;
      card.querySelectorAll("svg[data-step-count]").forEach((svg) => {{
        const stepCount = Math.max(1, Number(svg.dataset.stepCount) || 1);
        const left = Number(svg.dataset.plotLeft) || 0;
        const width = Number(svg.dataset.plotWidth) || 1;
        const stepIndex = Math.round(progress * Math.max(stepCount - 1, 0));
        const x = left + (stepIndex / Math.max(stepCount - 1, 1)) * width;
        svg.querySelectorAll(".step-playhead").forEach((line) => {{
          line.setAttribute("x1", x.toFixed(1));
          line.setAttribute("x2", x.toFixed(1));
        }});
      }});
    }}
    function setupTrajectoryVideoSync() {{
      document.querySelectorAll(".trajectory-card").forEach((card) => {{
        const videos = Array.from(card.querySelectorAll("video"));
        if (!videos.length) return;
        let syncing = false;
        const syncPeers = (source, action) => {{
          if (syncing) return;
          syncing = true;
          videos.forEach((video) => {{
            if (video === source) return;
            if (Number.isFinite(source.currentTime) && Math.abs((video.currentTime || 0) - source.currentTime) > 0.08) {{
              video.currentTime = source.currentTime;
            }}
            if (action === "play" && video.paused) {{
              video.play().catch(() => {{}});
            }} else if (action === "pause" && !video.paused) {{
              video.pause();
            }}
          }});
          syncing = false;
        }};
        videos.forEach((video) => {{
          video.addEventListener("play", () => {{
            syncStepPlayheads(card, video);
            syncPeers(video, "play");
          }});
          video.addEventListener("pause", () => syncPeers(video, "pause"));
          video.addEventListener("seeked", () => {{
            syncStepPlayheads(card, video);
            syncPeers(video, "seek");
          }});
          video.addEventListener("timeupdate", () => {{
            syncStepPlayheads(card, video);
            syncPeers(video, "time");
          }});
          video.addEventListener("loadedmetadata", () => syncStepPlayheads(card, video));
        }});
      }});
    }}
    window.addEventListener("DOMContentLoaded", () => {{
      document.getElementById("trajectory-sort-score")?.addEventListener("change", sortTrajectoryTable);
      document.getElementById("trajectory-sort-direction")?.addEventListener("change", sortTrajectoryTable);
      setupTrajectoryVideoSync();
    }});
  </script>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p class="subhead">Dataset: {dataset_path} · trajectories: {num_trajectories}</p>
  </header>
  <main>
    <section class="summary-grid">
      {run_card}
    </section>
    <details>
      <summary>Metadata</summary>
      {metadata_table}
    </details>
    {observability_panel}
    {observability_filter_panel}
    {mh_quality_panel}
    {mh_quality_filter_panel}
    {trajectory_table}
    {trajectory_card_section}
  </main>
</body>
</html>
""".format(
        title=html.escape(title),
        dataset_path=html.escape(dataset_path),
        num_trajectories=len(trajectories),
        run_card=score_card_html("Run Mean Scores", run_scores),
        metadata_table=metadata_table(metadata),
        observability_panel=grouped_score_panel(
            trajectories,
            label_key="observability_label",
            title="Full / Partial Observability Scores",
            label_order=["full", "partial"],
        ),
        observability_filter_panel=filter_quality_panel(
            trajectories,
            label_key="observability_label",
            title="Full / Partial Quality After Filtering",
        ),
        mh_quality_panel=grouped_score_panel(
            trajectories,
            label_key="mh_quality_label",
            title="MH Label 1 / 2 / 3 Scores",
            label_order=["1", "2", "3"],
        ),
        mh_quality_filter_panel=filter_quality_panel(
            trajectories,
            label_key="mh_quality_label",
            title="MH Label Quality After Filtering",
        ),
        trajectory_table=trajectory_table(trajectories, sort_key=sort_key, limit=50),
        trajectory_card_section=trajectory_card_section,
    )


def build_index_html(entries, source_dir):
    cards = []
    for entry in entries:
        run_scores = entry["run_scores"]
        metadata = entry["metadata"]
        info = parse_run_stem(entry["title"])
        title = readable_run_title(entry["title"], metadata)
        pieces = [
            "{} dataset".format(info.get("dataset")) if info.get("dataset") else None,
            "{} policy".format(info.get("policy") or metadata.get("policy_type")),
            info.get("wd"),
            camera_name_from_metadata(metadata),
            "visual {}{}".format(info.get("vkind"), " " + info.get("vepoch") if info.get("vepoch") else "") if info.get("vkind") else None,
            "robot {}{}".format(info.get("rkind"), " " + info.get("repoch") if info.get("repoch") else "") if info.get("rkind") else None,
            "steps={}".format(run_scores.get("num_steps", "n/a")),
        ]
        chips = "".join(
            '<span class="run-chip">{}</span>'.format(html.escape(str(piece)))
            for piece in pieces
            if piece
        )
        cards.append(
            """
            <article class="run-card">
              <a href="{href}">{title}</a>
              <div class="run-meta">{chips}</div>
              <div class="score-row">
                <span>NLL img+robot <b>{nll_img}</b></span>
                <span>H img+robot <b>{h_img}</b></span>
                <span>NLL delta <b>{nll_delta}</b></span>
                <span>H delta <b>{h_delta}</b></span>
              </div>
            </article>
            """.format(
                href=html.escape(entry["href"]),
                title=html.escape(title),
                chips=chips,
                nll_img=html.escape(format_value(run_scores.get("nll_images_robot"))),
                h_img=html.escape(format_value(run_scores.get("entropy_images_robot"))),
                nll_delta=html.escape(format_value(run_scores.get("nll_images_robot_minus_robot"))),
                h_delta=html.escape(format_value(run_scores.get("entropy_images_robot_minus_robot"))),
            )
        )

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Policy Score Delta Runs</title>
  <style>
    :root {{
      --bg: #efe6d4;
      --panel: #fffaf1;
      --panel-2: #f8f2e8;
      --ink: #221e1b;
      --muted: #70665c;
      --line: #d4c4ab;
      --line-strong: #bea888;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #f3ecde 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    header {{ padding: 26px 30px 14px; background: rgba(255,250,241,0.92); border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0 0 8px; font-size: 25px; letter-spacing: 0; }}
    .subhead {{ margin: 0; color: var(--muted); overflow-wrap: anywhere; }}
    main {{ padding: 22px 24px 42px; display: grid; gap: 12px; }}
    .run-card {{
      background: rgba(255,250,241,0.94);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      display: grid;
      gap: 8px;
    }}
    .run-card a {{ color: var(--ink); font-weight: 750; text-decoration: none; overflow-wrap: anywhere; }}
    .run-card a:hover {{ text-decoration: underline; }}
    .run-meta {{ display: flex; flex-wrap: wrap; gap: 8px; color: var(--muted); font-size: 13px; }}
    .run-chip {{ border: 1px solid var(--line-strong); border-radius: 999px; background: var(--panel-2); padding: 5px 9px; }}
    .score-row {{ display: flex; flex-wrap: wrap; gap: 8px; font-size: 13px; }}
    .score-row span {{ border: 1px solid var(--line); background: var(--panel-2); border-radius: 8px; padding: 7px 9px; }}
    .score-row b {{ font-variant-numeric: tabular-nums; }}
  </style>
</head>
<body>
  <header>
    <h1>Policy Score Delta Runs</h1>
    <p class="subhead">Source: {source_dir} · rendered runs: {count}</p>
  </header>
  <main>
    {cards}
  </main>
</body>
</html>
""".format(
        source_dir=html.escape(str(source_dir)),
        count=len(entries),
        cards="\n".join(cards) if cards else "<p>No score JSON files found.</p>",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a static visualization page for policy score-delta JSON.")
    parser.add_argument("--score-json", default=None, help="Path to score-delta JSON file, plain .json or .json.gz.")
    parser.add_argument(
        "--score-json-dir",
        default=None,
        help="Directory of score-delta JSON files. Renders one page per file plus an index page.",
    )
    parser.add_argument("--dataset", default=None, help="Optional image HDF5 dataset path. Defaults to metadata image_dataset.")
    parser.add_argument("--output-dir", default=None, help="Directory for HTML and image assets.")
    parser.add_argument(
        "--observability-csv",
        default=None,
        help=(
            "CSV with dataset,ep_idx,label columns. Defaults to "
            "robomimic/datasets/square/observability_annotations.csv."
        ),
    )
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes to sample.")
    parser.add_argument("--steps-per-episode", type=int, default=5, help="Number of transitions to sample per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument(
        "--trajectory-sort-score",
        choices=list(SCORE_FIELDS.keys()),
        default="nll_images_robot_minus_robot",
        help="Score used to sort the trajectory table.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Render only run scores, metadata, label panels, filter curves, and trajectory table; skip videos and step charts.",
    )
    parser.add_argument(
        "--obs-keys",
        nargs="+",
        default=None,
        help="RGB observation keys to compose into sampled step images.",
    )
    return parser.parse_args()


def render_score_json(args, repo_root, score_json_path, output_dir):
    payload = load_payload(score_json_path)
    metadata = payload["metadata"]
    run_scores = payload["run_scores"]
    trajectories = payload["trajectories"]
    transitions = payload["transitions"]
    ensure_normalized_scores(run_scores, trajectories, transitions)

    dataset_path = args.dataset or metadata.get("image_dataset")
    if dataset_path is None:
        raise ValueError("Dataset path must be passed via --dataset or present as metadata.image_dataset.")
    dataset_path = resolve_path(dataset_path, repo_root)
    obs_keys = args.obs_keys or default_obs_keys_for_visualization(dataset_path, metadata)
    dataset_group = dataset_group_from_path(dataset_path) or dataset_group_from_path(metadata.get("image_dataset", ""))
    third_person_camera = third_person_camera_from_context(dataset_path, metadata)
    observability_csv = (
        resolve_path(args.observability_csv, repo_root)
        if args.observability_csv is not None
        else default_observability_csv(repo_root)
    )
    observability_labels = load_observability_labels(observability_csv, dataset_group)
    mh_quality_labels = load_mh_quality_labels(dataset_path) if dataset_group == "mh" else {}
    metadata = dict(metadata)
    if observability_csv is not None:
        metadata["observability_csv"] = observability_csv
        metadata["observability_dataset_id"] = CSV_DATASET_ALIASES.get(dataset_group, dataset_group)
        metadata["num_observability_labels"] = len(observability_labels)
    if mh_quality_labels:
        metadata["mh_quality_label_source"] = "{}/mask/{}".format(dataset_path, ",".join(MH_QUALITY_MASK_LABELS.keys()))
        metadata["num_mh_quality_labels"] = len(mh_quality_labels)
    trajectories = annotate_labels(trajectories, observability_labels, mh_quality_labels)

    output_dir = Path(resolve_path(output_dir, repo_root))
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    sorted_trajectories = sorted(
        trajectories,
        key=lambda item: item.get(args.trajectory_sort_score, -float("inf"))
        if isinstance(item.get(args.trajectory_sort_score), (int, float))
        else -float("inf"),
        reverse=True,
    )
    cards = []
    if not args.summary_only:
        episode_series = collect_episode_step_series_from_transitions(
            transitions,
            [item["episode_key"] for item in sorted_trajectories],
        )

        for traj in sorted_trajectories:
            episode = traj.get("episode")
            if episode is None:
                episode = episode_index_from_key(traj.get("episode_key"))
            video_urls = video_urls_for_episode(
                repo_root=repo_root,
                output_dir=output_dir,
                dataset_group=dataset_group,
                episode=episode,
                third_person_camera=third_person_camera,
            )
            cards.append(trajectory_card_html(traj, episode_series[traj["episode_key"]], video_urls))

    page = build_html(
        score_json_path=score_json_path,
        dataset_path=dataset_path,
        metadata=metadata,
        run_scores=run_scores,
        trajectories=trajectories,
        trajectory_cards=cards,
        sort_key=args.trajectory_sort_score,
        summary_only=args.summary_only,
    )
    output_path = output_dir / "index.html"
    output_path.write_text(page, encoding="utf-8")
    print("Wrote {}".format(output_path))
    return {
        "title": artifact_stem(score_json_path),
        "href": output_path.name if output_path.parent == output_dir.parent else str(output_path),
        "output_path": output_path,
        "metadata": metadata,
        "run_scores": run_scores,
    }


def main():
    args = parse_args()
    if args.num_episodes <= 0 or args.steps_per_episode <= 0:
        raise ValueError("--num-episodes and --steps-per-episode must be positive.")
    if (args.score_json is None) == (args.score_json_dir is None):
        raise ValueError("Pass exactly one of --score-json or --score-json-dir.")

    repo_root = os.getcwd()
    if args.score_json is not None:
        score_json_path = resolve_path(args.score_json, repo_root)
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = os.path.join("policy_score_delta_visualizations", artifact_stem(score_json_path))
        render_score_json(args, repo_root, score_json_path, output_dir)
        return

    score_json_dir = Path(resolve_path(args.score_json_dir, repo_root))
    output_root = args.output_dir
    if output_root is None:
        output_root = os.path.join("policy_score_delta_visualizations", safe_name(score_json_dir.name))
    output_root = Path(resolve_path(output_root, repo_root))
    output_root.mkdir(parents=True, exist_ok=True)

    entries = []
    for score_json_path in iter_score_json_paths(score_json_dir):
        run_dir = output_root / artifact_stem(score_json_path)
        entry = render_score_json(args, repo_root, str(score_json_path), str(run_dir))
        entry["href"] = "{}/index.html".format(run_dir.name)
        entries.append(entry)

    index_path = output_root / "index.html"
    index_path.write_text(build_index_html(entries, score_json_dir), encoding="utf-8")
    print("Wrote {}".format(index_path))


if __name__ == "__main__":
    main()
