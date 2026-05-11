"""
Generate a static HTML visualization for a robomimic transition KNN JSON file.
Both plain .json and gzip-compressed .json.gz KNN files are supported.

The page samples a small set of query transitions, extracts their observation
images from the dataset HDF5, and lays each query plus its KNNs in one row.

Example:
    python robomimic/scripts/visualize_knn_json.py \
        --knn-json square_ph_discrete_knn.json.gz \
        --output-dir vis/knn_vis/square_ph_discrete
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


def natural_key(name):
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", name)]


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def knn_artifact_stem(path):
    name = os.path.basename(str(path))
    for suffix in (".json.gz", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def resolve_path(path, base_dir):
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r")


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


def load_knn_metadata(knn_json_path):
    decoder = json.JSONDecoder()
    buffer = ""
    with open_text(knn_json_path) as f:
        while True:
            chunk = f.read(1024 * 1024)
            if chunk == "":
                raise ValueError("Could not find metadata in {}".format(knn_json_path))
            buffer += chunk
            key_pos = find_json_key(buffer, "metadata")
            if key_pos < 0:
                continue
            object_start = buffer.find("{", key_pos)
            if object_start < 0:
                continue
            object_end = json_object_end(buffer, object_start)
            if object_end is None:
                continue
            return decoder.raw_decode(buffer[object_start:object_end])[0]


def stream_knn_transitions(knn_json_path):
    decoder = json.JSONDecoder()
    buffer = ""
    in_array = False
    with open_text(knn_json_path) as f:
        while True:
            if not in_array:
                chunk = f.read(1024 * 1024)
                if chunk == "":
                    raise ValueError("Could not find transitions array in {}".format(knn_json_path))
                buffer += chunk
                key_pos = find_json_key(buffer, "transitions")
                if key_pos < 0:
                    buffer = buffer[-64:]
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


def sample_queries_streaming(knn_json_path, labels, label, num_episodes, steps_per_episode, seed):
    rng = random.Random(seed)
    grouped_samples = {}
    grouped_counts = defaultdict(int)

    for transition in stream_knn_transitions(knn_json_path):
        if labels is not None and label is not None:
            episode_index = episode_index_from_key(transition["episode_key"])
            if labels.get(episode_index) != label:
                continue

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
    if len(episode_keys) == 0:
        raise ValueError("No transitions available to sample after filtering.")
    sampled_eps = rng.sample(episode_keys, k=min(num_episodes, len(episode_keys)))

    queries = []
    for ep in sampled_eps:
        sampled = grouped_samples[ep]
        sampled.sort(key=lambda item: item["step"])
        queries.extend(sampled)
    return queries, sum(grouped_counts.values())


def group_transitions_by_episode(transitions):
    grouped = defaultdict(list)
    for transition in transitions:
        grouped[transition["episode_key"]].append(transition)
    for ep in grouped:
        grouped[ep].sort(key=lambda item: item["step"])
    return grouped


def episode_index_from_key(episode_key):
    match = re.search(r"(\d+)$", str(episode_key))
    return int(match.group(1)) if match is not None else None


def load_observability_labels(csv_path):
    labels = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "ep_idx" not in row or "label" not in row:
                raise ValueError("Expected observability CSV columns: ep_idx,label")
            labels[int(row["ep_idx"])] = row["label"].strip()
    return labels


def default_observability_csv_for_dataset(dataset_path):
    candidate = os.path.join(os.path.dirname(dataset_path), "observability_annotations.csv")
    if os.path.exists(candidate):
        return candidate

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_parts = Path(dataset_path).parts
    if "square" not in dataset_parts or "ph" not in dataset_parts:
        return None

    fallback = os.path.join(
        repo_root,
        "vis",
        "dataset_videos",
        "videos",
        "square",
        "ph",
        "observability_annotations.csv",
    )
    return fallback if os.path.exists(fallback) else None


def annotate_transition_label(transition, labels):
    if labels is None:
        return transition
    episode_index = episode_index_from_key(transition["episode_key"])
    label = labels.get(episode_index, "unlabeled")
    transition = dict(transition)
    transition["observability_label"] = label
    neighbors = []
    for neighbor in transition["neighbors"]:
        neighbor_index = episode_index_from_key(neighbor["episode_key"])
        neighbor = dict(neighbor)
        neighbor["observability_label"] = labels.get(neighbor_index, "unlabeled")
        neighbors.append(neighbor)
    transition["neighbors"] = neighbors
    return transition


def filter_transitions_by_label(transitions, labels, label):
    if labels is None or label is None:
        return transitions
    filtered = []
    for transition in transitions:
        episode_index = episode_index_from_key(transition["episode_key"])
        if labels.get(episode_index) == label:
            filtered.append(transition)
    return filtered


def sample_queries(transitions, num_episodes, steps_per_episode, seed):
    rng = random.Random(seed)
    grouped = group_transitions_by_episode(transitions)
    episode_keys = sorted(grouped.keys(), key=natural_key)
    if len(episode_keys) == 0:
        raise ValueError("No transitions available to sample after filtering.")
    sampled_eps = rng.sample(episode_keys, k=min(num_episodes, len(episode_keys)))

    queries = []
    for ep in sampled_eps:
        ep_transitions = grouped[ep]
        sampled = rng.sample(ep_transitions, k=min(steps_per_episode, len(ep_transitions)))
        sampled.sort(key=lambda item: item["step"])
        queries.extend(sampled)
    return queries


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


def default_obs_keys_for_visualization(dataset_path, metadata):
    context = " ".join(
        str(value)
        for value in [
            dataset_path,
            metadata.get("dataset") if metadata else None,
            metadata.get("checkpoint") if metadata else None,
        ]
        if value
    ).lower()
    third_person_key = (
        "left_close_low_image"
        if "left_close_low" in context or "left_low_close" in context
        else "agentview_image"
    )
    return (third_person_key, "robot0_eye_in_hand_image")


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

    frames = get_frame(
        hdf5_file=hdf5_file,
        episode_key=episode_key,
        step=int(step),
        obs_keys=obs_keys,
    )
    image = compose_frames(frames)
    filename = "{}_step_{:04d}.png".format(safe_name(episode_key), int(step))
    path = assets_dir / filename
    image.save(path)
    rel_path = "assets/{}".format(filename)
    image_cache[cache_key] = rel_path
    return rel_path


def format_value(value):
    if isinstance(value, float):
        return "{:.6g}".format(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(format_value(v) for v in value)
    return str(value)


SCORE_FIELDS = OrderedDict(
    [
        (
            "knn_score",
            {
                "label": "KNN score",
                "default_order": "desc",
            },
        ),
        (
            "negative_log_likelihood",
            {
                "label": "raw NLL",
                "default_order": "asc",
            },
        ),
        (
            "policy_sample_entropy",
            {
                "label": "raw entropy",
                "default_order": "asc",
            },
        ),
        (
            "global_normalized_negative_log_likelihood",
            {
                "label": "global norm NLL",
                "default_order": "asc",
            },
        ),
        (
            "knn_batch_normalized_negative_log_likelihood",
            {
                "label": "KNN norm NLL",
                "default_order": "asc",
            },
        ),
        (
            "global_normalized_policy_sample_entropy",
            {
                "label": "global norm entropy",
                "default_order": "asc",
            },
        ),
        (
            "knn_batch_normalized_policy_sample_entropy",
            {
                "label": "KNN norm entropy",
                "default_order": "asc",
            },
        ),
    ]
)


def get_score(item, score_key):
    value = item.get(score_key, None)
    return value if isinstance(value, (int, float)) else None


def info_table(item, role):
    rows = [("role", role)]
    for key, value in item.items():
        if key == "neighbors":
            rows.append(("num_neighbors", len(value)))
        else:
            rows.append((key, value))

    body = []
    for key, value in rows:
        body.append(
            "<tr><th>{}</th><td>{}</td></tr>".format(
                html.escape(str(key)),
                html.escape(format_value(value)),
            )
        )
    return "<table>{}</table>".format("\n".join(body))


def score_metrics_html(item):
    metric_keys = [
        ("raw NLL", "negative_log_likelihood", "metric-raw"),
        ("global norm NLL", "global_normalized_negative_log_likelihood", "metric-nll"),
        ("KNN norm NLL", "knn_batch_normalized_negative_log_likelihood", "metric-nll"),
        ("raw entropy", "policy_sample_entropy", "metric-raw"),
        ("global norm entropy", "global_normalized_policy_sample_entropy", "metric-entropy"),
        ("KNN norm entropy", "knn_batch_normalized_policy_sample_entropy", "metric-entropy"),
    ]
    rows = []
    for label, key, class_name in metric_keys:
        if key not in item:
            continue
        rows.append(
            '<div class="metric {}"><span>{}</span><strong>{}</strong></div>'.format(
                class_name,
                html.escape(label),
                html.escape(format_value(item[key])),
            )
        )
    if not rows:
        return ""
    return '<div class="image-metrics">{}</div>'.format("\n".join(rows))


def card_html(item, role, image_path):
    return """
    <article class="card {role_class}">
      <img src="{image_path}" alt="{alt}">
      {metrics}
      {table}
    </article>
    """.format(
        role_class="query" if role == "query" else "neighbor",
        image_path=html.escape(image_path),
        alt=html.escape("{} {} step {}".format(role, item["episode_key"], item["step"])),
        metrics=score_metrics_html(item),
        table=info_table(item, role),
    )


def sort_neighbors(neighbors, score_key, sort_order):
    descending = sort_order == "desc"
    missing = -float("inf") if descending else float("inf")

    def sort_key(item):
        score = get_score(item, score_key)
        if score is None:
            score = missing
        if descending:
            score = -score
        return score, item.get("episode_key", ""), item.get("step", 0)

    return sorted(
        neighbors,
        key=sort_key,
    )


def build_html(knn_json_path, dataset_path, metadata, rows_html, num_queries, neighbor_sort_score, neighbor_sort_order):
    title = "KNN Visualization: {}".format(os.path.basename(knn_json_path))
    metadata_rows = "\n".join(
        "<tr><th>{}</th><td>{}</td></tr>".format(
            html.escape(str(k)),
            html.escape(format_value(v)),
        )
        for k, v in metadata.items()
    )
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #657282;
      --line: #d9dee6;
      --query: #1f7a5a;
      --neighbor: #315f9f;
      --nll: #7956a5;
      --entropy: #b65f2a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      padding: 24px 28px 10px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .subhead {{
      margin: 0 0 16px;
      color: var(--muted);
      font-size: 14px;
    }}
    main {{ padding: 18px 24px 36px; }}
    details {{
      margin: 0 0 18px;
      padding: 14px 16px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    summary {{ cursor: pointer; font-weight: 650; }}
    .sample-row {{
      margin: 0 0 18px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .row-title {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      font-weight: 700;
      font-size: 14px;
    }}
    .row-title-secondary {{
      padding-top: 8px;
      padding-bottom: 8px;
      color: var(--muted);
      font-weight: 550;
      font-size: 12px;
      background: #fbfcfe;
    }}
    .strip {{
      display: flex;
      gap: 12px;
      overflow-x: auto;
      padding: 14px;
      align-items: flex-start;
    }}
    .card {{
      flex: 0 0 238px;
      border: 1px solid var(--line);
      border-top: 4px solid var(--neighbor);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }}
    .card.query {{ border-top-color: var(--query); }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      image-rendering: auto;
      border-bottom: 1px solid var(--line);
    }}
    .image-metrics {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 4px;
      padding: 8px 9px;
      border-bottom: 1px solid var(--line);
      background: #f9fafc;
      font-size: 12px;
    }}
    .metric {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: baseline;
    }}
    .metric span {{
      color: var(--muted);
      overflow-wrap: anywhere;
    }}
    .metric strong {{
      flex: 0 0 auto;
      font-variant-numeric: tabular-nums;
      font-weight: 700;
    }}
    .metric-nll strong {{ color: var(--nll); }}
    .metric-entropy strong {{ color: var(--entropy); }}
    .metric-raw span, .metric-raw strong {{ color: #4d5b6a; }}
    .row-sort {{
      color: var(--muted);
      font-weight: 550;
      margin-left: auto;
      font-size: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      padding: 5px 7px;
      border-bottom: 1px solid #edf0f4;
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    th {{
      width: 45%;
      color: var(--muted);
      font-weight: 650;
    }}
    td {{ font-variant-numeric: tabular-nums; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p class="subhead">Dataset: {dataset_path} · sampled query transitions: {num_queries} · neighbors sorted by {neighbor_sort_score} ({neighbor_sort_order})</p>
  </header>
  <main>
    <details>
      <summary>Metadata</summary>
      <table>{metadata_rows}</table>
    </details>
    {rows_html}
  </main>
</body>
</html>
""".format(
        title=html.escape(title),
        dataset_path=html.escape(dataset_path),
        num_queries=num_queries,
        neighbor_sort_score=html.escape(neighbor_sort_score),
        neighbor_sort_order=html.escape(neighbor_sort_order),
        metadata_rows=metadata_rows,
        rows_html="\n".join(rows_html),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a static sampled KNN visualization page.")
    parser.add_argument("--knn-json", required=True, help="Path to KNN JSON file, plain .json or compressed .json.gz.")
    parser.add_argument("--dataset", default=None, help="Optional HDF5 dataset path. Defaults to JSON metadata dataset.")
    parser.add_argument("--output-dir", default=None, help="Directory for HTML and image assets.")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes to sample.")
    parser.add_argument("--steps-per-episode", type=int, default=5, help="Number of query steps to sample per episode.")
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=100,
        help="Maximum neighbors to visualize per query after sorting by --neighbor-sort-score.",
    )
    parser.add_argument(
        "--neighbor-sort-score",
        choices=list(SCORE_FIELDS.keys()),
        default="knn_score",
        help=(
            "Score used to order displayed neighbors. The JSON stores neighbors in KNN order; "
            "the default keeps nearest neighbors first."
        ),
    )
    parser.add_argument(
        "--neighbor-sort-order",
        choices=["auto", "asc", "desc"],
        default="auto",
        help="Neighbor sort direction. 'auto' uses descending for KNN similarity and ascending for uncertainty scores.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument(
        "--observability-csv",
        default=None,
        help="Optional trajectory label CSV. Defaults to observability_annotations.csv beside the dataset if present.",
    )
    parser.add_argument(
        "--sample-label",
        default="partial",
        help="Only sample query trajectories with this label. Use 'all' to disable label filtering.",
    )
    parser.add_argument(
        "--obs-keys",
        nargs="+",
        default=None,
        help=(
            "RGB observation keys to compose into each image. Defaults to the policy's "
            "third-person camera when it can be inferred, plus robot0_eye_in_hand_image."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_neighbors <= 0:
        raise ValueError("--max-neighbors must be positive.")
    neighbor_sort_order = args.neighbor_sort_order
    if neighbor_sort_order == "auto":
        neighbor_sort_order = SCORE_FIELDS[args.neighbor_sort_score]["default_order"]

    repo_root = os.getcwd()
    knn_json_path = resolve_path(args.knn_json, repo_root)

    metadata = load_knn_metadata(knn_json_path)
    dataset_path = args.dataset or metadata.get("dataset")
    if dataset_path is None:
        raise ValueError("Dataset path must be passed via --dataset or present in JSON metadata.")
    dataset_path = resolve_path(dataset_path, repo_root)
    obs_keys = args.obs_keys or default_obs_keys_for_visualization(dataset_path, metadata)

    observability_csv = args.observability_csv
    if observability_csv is None:
        observability_csv = default_observability_csv_for_dataset(dataset_path)
    else:
        observability_csv = resolve_path(observability_csv, repo_root)

    labels = load_observability_labels(observability_csv) if observability_csv is not None else None
    sample_label = None if args.sample_label.lower() == "all" else args.sample_label
    output_dir = args.output_dir
    if output_dir is None:
        stem = knn_artifact_stem(knn_json_path)
        output_dir = os.path.join("knn_visualizations", stem)
    output_dir = Path(resolve_path(output_dir, repo_root))
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    queries, num_sample_candidate_transitions = sample_queries_streaming(
        knn_json_path=knn_json_path,
        labels=labels,
        label=sample_label,
        num_episodes=args.num_episodes,
        steps_per_episode=args.steps_per_episode,
        seed=args.seed,
    )
    queries = [annotate_transition_label(q, labels) for q in queries]
    if labels is not None:
        metadata = dict(metadata)
        metadata["observability_csv"] = observability_csv
        metadata["sample_label"] = sample_label if sample_label is not None else "all"
        metadata["num_sample_candidate_transitions"] = num_sample_candidate_transitions

    rows_html = []
    image_cache = {}
    with h5py.File(dataset_path, "r") as hdf5_file:
        for query_idx, query in enumerate(queries):
            sorted_neighbors = sort_neighbors(
                query["neighbors"],
                score_key=args.neighbor_sort_score,
                sort_order=neighbor_sort_order,
            )[: args.max_neighbors]
            query_card = dict(query)
            query_card["neighbors"] = sorted_neighbors
            query_image = save_transition_image(
                hdf5_file=hdf5_file,
                image_cache=image_cache,
                assets_dir=assets_dir,
                episode_key=query["episode_key"],
                step=query["step"],
                obs_keys=obs_keys,
            )
            cards = [card_html(query_card, "query", query_image)]
            for neighbor_rank, neighbor in enumerate(sorted_neighbors, start=1):
                neighbor_image = save_transition_image(
                    hdf5_file=hdf5_file,
                    image_cache=image_cache,
                    assets_dir=assets_dir,
                    episode_key=neighbor["episode_key"],
                    step=neighbor["step"],
                    obs_keys=obs_keys,
                )
                cards.append(card_html(neighbor, "neighbor {}".format(neighbor_rank), neighbor_image))

            rows_html.append(
                """
                <section class="sample-row">
                  <div class="row-title">Query {query_idx}: {episode_key} step {step}</div>
                  <div class="row-title row-title-secondary">
                    <span>{sort_label}: {sort_order}</span>
                  </div>
                  <div class="strip">{cards}</div>
                </section>
                """.format(
                    query_idx=query_idx + 1,
                    episode_key=html.escape(str(query["episode_key"])),
                    step=html.escape(str(query["step"])),
                    sort_label=html.escape(SCORE_FIELDS[args.neighbor_sort_score]["label"]),
                    sort_order=html.escape(neighbor_sort_order),
                    cards="\n".join(cards),
                )
            )

    page = build_html(
        knn_json_path=knn_json_path,
        dataset_path=dataset_path,
        metadata=metadata,
        rows_html=rows_html,
        num_queries=len(queries),
        neighbor_sort_score=SCORE_FIELDS[args.neighbor_sort_score]["label"],
        neighbor_sort_order=neighbor_sort_order,
    )
    output_path = output_dir / "index.html"
    output_path.write_text(page)
    print("Wrote {}".format(output_path))


if __name__ == "__main__":
    main()
