# Square Policy Score Deltas

This workflow compares visual policies against matching robot-only policies on
the same Square episodes. It saves raw per-step and per-trajectory policy scores
without using the KNN local baseline from `extract_policy_latent_knn.py`.

## Files

- `extract_policy_score_deltas.py`
  - Scores one paired visual-policy / robot-only-policy checkpoint pair.
  - Writes a `.json` or `.json.gz` file with transition-level, trajectory-level,
    and run-level scores.
- `generate_square_policy_score_delta_commands.py`
  - Discovers Square sweep checkpoints and writes scoring commands.
  - Matches visual and robot-only runs by dataset group, policy type, and weight
    decay.
- `square_policy_score_delta_commands.txt`
  - Generated command manifest.
- `visualize_policy_score_deltas.py`
  - Builds a static HTML review page from one score JSON, including trajectory
    summaries, sampled step images, and label-group score curves.

## Score Definitions

For each aligned transition, the scorer reports:

- `nll_images_robot`: `-log p(a | s)` from the visual policy that uses images and
  robot state.
- `entropy_images_robot`: `H(pi(. | s))` from the visual policy.
- `nll_robot`: `-log p(a | s)` from the robot-only policy.
- `entropy_robot`: `H(pi(. | s))` from the robot-only policy.
- `nll_images_robot_minus_robot`: `nll_images_robot - nll_robot`.
- `entropy_images_robot_minus_robot`: `entropy_images_robot - entropy_robot`.

Entropy is computed by policy type:

- Discrete policy: exact summed categorical entropy over action dimensions.
- Discrete Gaussian target policy: exact summed categorical entropy over action
  dimensions. The Gaussian target affects the dataset-action NLL, not policy
  entropy.
- Gaussian policy: exact `torch.distributions` entropy.
- GMM policy: Monte Carlo estimate `E[-log pi(a | s)]` using policy samples,
  controlled by `--entropy-num-samples`.

## Generate Commands

Run from the repository root:

```bash
python robomimic/scripts/generate_square_policy_score_delta_commands.py
```

By default this writes:

```text
robomimic/scripts/square_policy_score_delta_commands.txt
```

The default checkpoint modes are:

```text
best-validation,last,early-after-best-validation
```

For `last`, the generator normally uses `last.pth`. If the run is missing its
final epoch checkpoint, for example `model_epoch_600*.pth` for visual policies or
`model_epoch_2000*.pth` for robot-only policies, `last` falls back to the highest
available `models/model_epoch_*.pth` checkpoint so incomplete runs can still be
scored.

The default pairing mode is `same-label` to only
emit best-vs-best, last-vs-last, and early-vs-early. Use 'cross-product', so each valid visual checkpoint mode
is paired with each valid robot-only checkpoint mode. Use `cross-product` :

```bash
python robomimic/scripts/generate_square_policy_score_delta_commands.py \
    --checkpoint-pairing cross-product
```

Use `--verify-only` to check pairing without rewriting the manifest:

```bash
python robomimic/scripts/generate_square_policy_score_delta_commands.py --verify-only
```

## Run Scores

Use the `robomimic_py310` environment; the default Python on this machine may not
have all robomimic import dependencies.

```bash
conda activate robomimic_py310
bash robomimic/scripts/square_policy_score_delta_commands.txt
```

For a small CPU smoke test:

```bash
python robomimic/scripts/extract_policy_score_deltas.py \
    --image-dataset robomimic/datasets/square/ph/image.hdf5 \
    --robot-dataset robomimic/datasets/square/ph/low_dim_v15.hdf5 \
    --image-checkpoint robomimic/trained_models/square/sweep/ph/gmm/wd0/bc_gmm_square_ph_image_wd0/20260504165026/last.pth \
    --robot-checkpoint robomimic/trained_models/square/sweep_robot_only/ph/bc_gmm/wd0/bc_gmm_square_ph_robot_wd0/20260511040057/last.pth \
    --output /tmp/policy_score_delta_smoke.json.gz \
    --entropy-num-samples 2 \
    --batch-size 64 \
    --num-demos 1 \
    --device cpu
```

## Output

Generated outputs go under:

```text
vis/policy_score_deltas/square/
```

Each output JSON contains:

- `metadata`: input datasets, checkpoints, policy type, entropy estimator, and
  scoring definitions.
- `run_scores`: mean scores over all included transitions.
- `trajectories`: mean scores per episode.
- `transitions`: step-level scores for every aligned transition.

The scorer checks that visual and robot-only transitions are aligned by episode
key and step before writing output.

## Visualization Pipeline

The visualization pipeline starts from the score JSON files written by
`extract_policy_score_deltas.py` and produces a static HTML page with sampled
transition images, trajectory tables, and label-group score curves.

1. Generate or refresh the score extraction commands:

```bash
python robomimic/scripts/generate_square_policy_score_delta_commands.py
```

2. Run the generated scoring commands:

```bash
conda activate robomimic_py310
bash robomimic/scripts/square_policy_score_delta_commands.txt
```

To run the same command manifest through Slurm, use `run_slurm.py` with
`--commands_file`. In this mode each non-comment line in the manifest is emitted
directly into the Slurm script instead of being wrapped as `train.py --config`.

```bash
python robomimic/scripts/run_slurm.py \
    --commands_file robomimic/scripts/square_policy_score_delta_commands.txt \
    --scripts-per-job 2 \
    --job-name square-score-delta \
    --time 24:00:00 \
    --mem 64G \
    --gpus 1
```

Use `--dry_run` first to inspect the generated Slurm scripts without submitting:

```bash
python robomimic/scripts/run_slurm.py \
    --commands_file robomimic/scripts/square_policy_score_delta_commands.txt \
    --scripts-per-job 2 \
    --job-name square-score-delta \
    --dry_run
```

This writes score JSON files under:

```text
vis/policy_score_deltas/square/
```

3. Render one score JSON into an HTML review page:

```bash
python robomimic/scripts/visualize_policy_score_deltas.py \
    --score-json vis/policy_score_deltas/square/<score-file>.json.gz \
    --output-dir vis/policy_score_delta_vis/<run-name>
```

The rendered page is:

```text
vis/policy_score_delta_vis/<run-name>/index.html
```

Open that file in a browser to review the result. The visualizer also writes
sampled frame images to:

```text
vis/policy_score_delta_vis/<run-name>/assets/
```

The page includes:

- run-level mean scores for all six saved score fields.
- a trajectory table sorted by `nll_images_robot_minus_robot` by default.
- sampled step cards with composed camera frames and per-step scores.
- per-episode step curves for the four requested primary scores:
  `nll_images_robot`, `entropy_images_robot`,
  `nll_images_robot_minus_robot`, and
  `entropy_images_robot_minus_robot`.
- label-group trajectory curves for full / partial observability.
- MH-only label-group trajectory curves for labels `1`, `2`, and `3`.

The visualizer reads full / partial observability labels from:

```text
robomimic/datasets/square/observability_annotations.csv
```

The CSV dataset ids are mapped as:

- PH: `square_ph`
- MH: `square_mh`
- random_post: `expert200`

For MH runs, it also reads the HDF5 masks `worse`, `okay`, and `better` from the
image dataset and plots the extra label curves as `1`, `2`, and `3`,
respectively.

Useful visualization options:

```bash
python robomimic/scripts/visualize_policy_score_deltas.py \
    --score-json vis/policy_score_deltas/square/<score-file>.json.gz \
    --output-dir vis/policy_score_delta_vis/<run-name> \
    --num-episodes 8 \
    --steps-per-episode 6 \
    --trajectory-sort-score entropy_images_robot_minus_robot
```

Use `--observability-csv` to override the default label CSV, and use
`--obs-keys` to override which RGB observation keys are composed into sampled
step images.

## Notes

- The generator skips runs that do not have a matching robot-only config or a
  requested checkpoint mode. Skips are written as comments at the end of the
  generated command file.
- `run_slurm.py` supports two launch modes: `--config_paths_file` for training
  config manifests and `--commands_file` for full shell command manifests such
  as `square_policy_score_delta_commands.txt`.
