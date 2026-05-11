# Square BC Sweep

Generate Square visual BC sweep configs:

```bash
python robomimic/scripts/generate_square_bc_sweep_configs.py
```

By default, this sweeps six datasets: `ph`, `mh`, `ph_left_low_close`,
`random_post`, `random_post_left_low_close`, and `mh_left_low_close`. Each
generated config sets the RGB observation keys and rollout camera names to match
that dataset. This writes configs under
`robomimic/exps/square/sweep/` and saves the generated config paths to:

```text
robomimic/exps/square/sweep/config_paths.txt
```

Generate a smaller or custom sweep:

```bash
python robomimic/scripts/generate_square_bc_sweep_configs.py \
  --datasets ph mh \
  --policies gmm discrete \
  --weight_decays 0 1e-4 \
  --config_paths_file robomimic/exps/square/sweep/config_paths.txt
```

Generate configs that include the `left_close_low_image` observation:

```bash
python robomimic/scripts/generate_square_bc_sweep_configs.py \
  --use_left_close_low_obs
```

Inspect Slurm scripts without submitting jobs:

```bash
python robomimic/scripts/run_slurm.py \
  --config_paths_file robomimic/exps/square/sweep/config_paths.txt \
  --scripts-per-job 1 \
  --dry_run
```

Submit the generated configs to Slurm:

```bash
python robomimic/scripts/run_slurm.py \
  --config_paths_file robomimic/exps/square/sweep/config_paths.txt \
  --gpus 1 \
  --cpus 20 \
  --mem 64G \
  --scripts-per-job 1
```

Overwrite existing experiment directories instead of prompting:

```bash
python robomimic/scripts/run_slurm.py \
  --config_paths_file robomimic/exps/square/sweep/config_paths.txt \
  --overwrite \
  --gpus 1 \
  --cpus 20 \
  --mem 64G \
  --scripts-per-job 1
```

`--overwrite` removes the existing output directory for each matching
`config.train.output_dir/config.experiment.name` before starting a new run. Use
`--resume` instead when you want to continue from the latest checkpoint; `--resume`
and `--overwrite` cannot be used together.

If the cluster environment needs setup before training, pass a shell setup script:

```bash
python robomimic/scripts/run_slurm.py \
  --config_paths_file robomimic/exps/square/sweep/config_paths.txt \
  --env_setup_script /path/to/setup_env.sh \
  --gpus 1 \
  --cpus 20 \
  --mem 64G
```
