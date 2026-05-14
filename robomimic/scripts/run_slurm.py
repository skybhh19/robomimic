"""
Launch robomimic training jobs on Slurm from a generated config path manifest.
It can also launch arbitrary command manifests with --commands_file.

Example:
    python robomimic/scripts/generate_square_bc_sweep_configs.py

    python robomimic/scripts/run_slurm.py \
        --config_paths_file robomimic/exps/square/sweep/config_paths.txt \
        --scripts-per-job 2 \
        --dry_run
"""

import argparse
import copy
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import List, TextIO


SLURM_ARGS = {
    "partition": {"type": str, "default": "sc-loprio"},
    "constraint": {"type": str, "required": False, "default": "[ampere|hopper|ada]"},
    # "partition": {"type": str, "default": "iris-hi"},
    "time": {"type": str, "default": "72:00:00"},
    "nodes": {"type": int, "default": 1},
    "ntasks-per-node": {"type": int, "default": 1},
    "cpus": {"type": int, "default": 12},
    "gpus": {"type": str, "required": False, "default": "1"},
    "mem": {"type": str, "default": "64G"},
    "output": {"type": str, "default": "slurm_logs"},
    "error": {"type": str, "default": "slurm_logs"},
    "job-name": {"type": str, "default": "square-bc"},
    "exclude": {
        "type": str,
        "required": False,
        "default": (
            "iris1,iris2,iris3,iris4,iris5,iris6,iris8,"
            "iris-hgx-1,iris-hgx-2,iris-hp-z8,"
            "iliad1,iliad2,iliad3,iliad4,iliad5,iliad6,iliad-hgx-1"
        ),
    },
    "nodelist": {"type": str, "required": False, "default": None},
    "account": {"type": str, "required": False, "default": "iliad"},
    "mail-user": {"type": str, "required": False, "default": "tiangao@stanford.edu"},
    "mail-type": {"type": str, "required": False, "default": "END,FAIL,REQUEUE,TIME_LIMIT_80"},
}

SLURM_NAME_OVERRIDES = {"gpus": "gres", "cpus": "cpus-per-task"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_paths_file",
        default="robomimic/exps/square/sweep/config_paths.txt",
        help="Newline-delimited file containing config paths to launch.",
    )
    parser.add_argument(
        "--commands_file",
        default=None,
        help=(
            "Optional newline-delimited file containing full shell commands to launch. "
            "When set, lines are emitted directly instead of wrapped as train.py --config commands."
        ),
    )
    parser.add_argument(
        "--entry_point",
        default="robomimic/scripts/train.py",
        help="Training entry point to call for each config.",
    )
    parser.add_argument(
        "--python",
        default="python",
        help="Python executable used inside each Slurm job.",
    )
    parser.add_argument(
        "--env_setup_script",
        default=None,
        help="Optional shell script to source before launching training.",
    )
    parser.add_argument(
        "--remainder",
        default="new",
        choices=["split", "new"],
        help="How to place jobs that do not divide evenly by --scripts-per-job.",
    )
    parser.add_argument(
        "--scripts-per-job",
        type=int,
        default=1,
        help="Number of training scripts to run per Slurm job.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=2,
        help="Stagger submitted batch scripts by this many seconds times their index.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pass --resume through to robomimic/scripts/train.py.",
    )
    parser.add_argument(
        "--overwrite",
        "--overwirte",
        dest="overwrite",
        action="store_true",
        help="Pass --overwrite through to robomimic/scripts/train.py.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Pass --debug through to robomimic/scripts/train.py.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Write Slurm scripts and print their paths without calling sbatch.",
    )
    for key, value in SLURM_ARGS.items():
        parser.add_argument("--" + key, **value)
    return parser.parse_args()


def read_config_paths(config_paths_file: str) -> List[str]:
    path = Path(config_paths_file)
    if not path.exists():
        raise FileNotFoundError("Config paths file does not exist: {}".format(path))
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def read_command_lines(commands_file: str) -> List[str]:
    path = Path(commands_file)
    if not path.exists():
        raise FileNotFoundError("Commands file does not exist: {}".format(path))
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def make_batch_sizes(num_scripts: int, scripts_per_job: int, remainder: str) -> List[int]:
    if scripts_per_job <= 0:
        raise ValueError("--scripts-per-job must be positive")
    if num_scripts == 0:
        return []

    num_full_jobs = num_scripts // scripts_per_job
    remainder_scripts = num_scripts - num_full_jobs * scripts_per_job

    if remainder == "new":
        sizes = [scripts_per_job for _ in range(num_full_jobs)]
        if remainder_scripts:
            sizes.append(remainder_scripts)
        return sizes

    if num_full_jobs == 0:
        return [remainder_scripts]
    sizes = [scripts_per_job for _ in range(num_full_jobs)]
    for index in range(remainder_scripts):
        sizes[index] += 1
    return sizes


def sanitize_job_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
    return name.strip("-")


def sweep_name_from_config_path(config_path: str) -> str:
    path = Path(config_path)
    parts = path.with_suffix("").parts
    if len(parts) >= 3:
        return sanitize_job_name("_".join(parts[-3:]))
    return sanitize_job_name(path.stem)


def command_name(command: str) -> str:
    parts = shlex.split(command)
    if "--output" in parts:
        output = Path(parts[parts.index("--output") + 1])
        return sanitize_job_name(output.stem)
    if len(parts) >= 2:
        return sanitize_job_name(Path(parts[1]).stem)
    return sanitize_job_name(command[:48])


def job_name_for_batch(base_job_name: str, config_paths: List[str], batch_index: int) -> str:
    sweep_names = [sweep_name_from_config_path(path) for path in config_paths]
    if len(sweep_names) == 1:
        suffix = sweep_names[0]
    else:
        suffix = "{}-to-{}_batch{}".format(sweep_names[0], sweep_names[-1], batch_index)
    return sanitize_job_name("{}_{}".format(base_job_name, suffix))


def job_name_for_command_batch(base_job_name: str, commands: List[str], batch_index: int) -> str:
    names = [command_name(command) for command in commands]
    if len(names) == 1:
        suffix = names[0]
    else:
        suffix = "{}-to-{}_batch{}".format(names[0], names[-1], batch_index)
    return sanitize_job_name("{}_{}".format(base_job_name, suffix))[:180]


def write_slurm_header(f: TextIO, args: argparse.Namespace) -> None:
    args = copy.deepcopy(args)
    for key in SLURM_ARGS:
        assert key.replace("-", "_") in args, "Key {} not found.".format(key)

    if "%" not in args.output:
        os.makedirs(args.output, exist_ok=True)
        args.output = os.path.join(args.output, args.job_name + "_%A.out")
    if "%" not in args.error:
        os.makedirs(args.error, exist_ok=True)
        args.error = os.path.join(args.error, args.job_name + "_%A.err")
    args.gpus = "gpu:" + str(args.gpus) if args.gpus is not None else args.gpus

    f.write("#!/bin/bash\n\n")
    for arg_name in SLURM_ARGS:
        arg_value = vars(args)[arg_name.replace("-", "_")]
        if arg_value is not None:
            slurm_name = SLURM_NAME_OVERRIDES.get(arg_name, arg_name)
            f.write("#SBATCH --{}={}\n".format(slurm_name, arg_value))

    f.write("\n")
    f.write('echo "SLURM_JOBID = "$SLURM_JOBID\n')
    f.write('echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST\n')
    f.write('echo "SLURM_NNODES = "$SLURM_NNODES\n')
    f.write('echo "SLURMTMPDIR = "$SLURMTMPDIR\n')
    f.write('echo "working directory = "$SLURM_SUBMIT_DIR\n\n')
    if args.env_setup_script is not None:
        f.write(". {}\n\n".format(shlex.quote(args.env_setup_script)))


def train_command(config_path: str, args: argparse.Namespace) -> str:
    command = [
        args.python,
        args.entry_point,
        "--config",
        config_path,
    ]
    if args.resume:
        command.append("--resume")
    if args.overwrite:
        command.append("--overwrite")
    if args.debug:
        command.append("--debug")
    return " ".join(shlex.quote(part) for part in command)


def main() -> None:
    args = parse_args()
    if args.commands_file is not None and args.config_paths_file != "robomimic/exps/square/sweep/config_paths.txt":
        raise ValueError("Use either --commands_file or --config_paths_file, not both")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    use_commands_file = args.commands_file is not None
    scripts = read_command_lines(args.commands_file) if use_commands_file else read_config_paths(args.config_paths_file)
    batch_sizes = make_batch_sizes(
        num_scripts=len(scripts),
        scripts_per_job=args.scripts_per_job,
        remainder=args.remainder,
    )
    if not batch_sizes:
        source = args.commands_file if use_commands_file else args.config_paths_file
        print("No launch lines found in {}".format(source))
        return

    script_index = 0
    procs = []
    for batch_index, num_scripts in enumerate(batch_sizes):
        current_scripts = scripts[script_index : script_index + num_scripts]
        script_index += num_scripts
        slurm_args = copy.deepcopy(args)
        slurm_args.job_name = (
            job_name_for_command_batch(args.job_name, current_scripts, batch_index)
            if use_commands_file
            else job_name_for_batch(args.job_name, current_scripts, batch_index)
        )

        _, slurm_file = tempfile.mkstemp(text=True, prefix="job_", suffix=".sh")
        with open(slurm_file, "w+") as f:
            write_slurm_header(f, slurm_args)
            f.write("sleep {}\n".format(args.sleep_seconds * batch_index))
            for script in current_scripts:
                command = script if use_commands_file else train_command(script, args)
                if len(current_scripts) != 1:
                    command += " &"
                f.write(command + "\n")
            if len(current_scripts) != 1:
                f.write("wait\n")

        print("Prepared Slurm script: {} ({})".format(slurm_file, slurm_args.job_name))
        if args.dry_run:
            continue
        procs.append(subprocess.Popen(["sbatch", slurm_file]))

    exit_codes = [proc.wait() for proc in procs]
    if exit_codes and any(code != 0 for code in exit_codes):
        raise SystemExit("At least one sbatch call failed: {}".format(exit_codes))


if __name__ == "__main__":
    main()
