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
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, TextIO


DEFAULT_MAX_JOBS_FILE = Path(__file__).with_name("slurm_max_jobs.txt")
ACTIVE_SLURM_STATES = ["PENDING", "RUNNING", "CONFIGURING", "COMPLETING"]
SUBMITTER_JOB_PREFIX = "submitter-"

SLURM_ARGS = {
    "partition": {"type": str, "default": "sc-loprio"},
    "constraint": {"type": str, "required": False, "default": "[ampere|hopper|ada]"},
    # "partition": {"type": str, "default": "iliad"},
    "time": {"type": str, "default": "48:00:00"},
    "nodes": {"type": int, "default": 1},
    "ntasks-per-node": {"type": int, "default": 1},
    "cpus": {"type": int, "default": 16},
    "gpus": {"type": str, "required": False, "default": "1"},
    "mem": {"type": str, "default": "64G"},
    "output": {"type": str, "default": "slurm_logs"},
    "error": {"type": str, "default": "slurm_logs"},
    "job-name": {"type": str, "default": "square-bc"},
    "exclude": {
        "type": str,
        "required": False,
        "default": (
            "iris1,iris2,iris3,iris5,"
            "iris-hgx-1,iris-hgx-2,iris-hp-z8,"
            "iliad1,iliad2,iliad3,iliad4,iliad5,iliad-hgx-1"
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
        default=sys.executable,
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
        "--max-jobs-file",
        default=str(DEFAULT_MAX_JOBS_FILE),
        help="Text file containing the maximum number of active Slurm jobs to allow.",
    )
    parser.add_argument(
        "--max-jobs-poll-seconds",
        type=int,
        default=60,
        help="Seconds to wait between Slurm job count checks when the limit is reached.",
    )
    parser.add_argument(
        "--slurm-query-timeout-seconds",
        type=int,
        default=30,
        help="Seconds to wait for each Slurm job-count query.",
    )
    parser.add_argument(
        "--slurm-query-retries",
        type=int,
        default=3,
        help="Number of Slurm job-count query attempts before waiting for the next poll.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        "--assign-gpus",
        action="store_true",
        help="Assign one CUDA_VISIBLE_DEVICES entry to each script in a multi-script job.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Write Slurm scripts and print their paths without calling sbatch.",
    )
    parser.add_argument(
        "--submitter-as-slurm",
        action="store_true",
        help="Submit this launcher itself as a CPU-only Slurm controller job and exit.",
    )
    parser.add_argument(
        "--submitter-partitions",
        default="iliad",
        help="Comma-delimited non-preemptible controller-job partitions to try in order.",
    )
    parser.add_argument(
        "--submitter-time",
        default="48:00:00",
        help="Wall time for the Slurm controller job.",
    )
    parser.add_argument(
        "--submitter-cpus",
        type=int,
        default=1,
        help="CPU count for the Slurm controller job.",
    )
    parser.add_argument(
        "--submitter-mem",
        default="4G",
        help="Memory for the Slurm controller job.",
    )
    parser.add_argument(
        "--submitter-output",
        default="slurm_logs",
        help="Output directory or pattern for the Slurm controller job.",
    )
    parser.add_argument(
        "--submitter-account",
        default=None,
        help="Optional account override for the Slurm controller job.",
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


def read_max_slurm_jobs(max_jobs_file: str):
    path = Path(max_jobs_file)
    assert path.exists(), "Slurm max jobs file does not exist: {}".format(path)
    max_jobs = json.loads(path.read_text())
    assert isinstance(max_jobs, int), "Slurm max jobs file must contain an integer: {}".format(path)
    assert max_jobs > 0, "Slurm max jobs must be positive: {}".format(max_jobs)
    return max_jobs


def active_slurm_job_count(timeout_seconds):
    output = subprocess.check_output(
        [
            "squeue",
            "--noheader",
            "--me",
            "--states={}".format(",".join(ACTIVE_SLURM_STATES)),
            "--format=%i|%j",
        ],
        text=True,
        timeout=timeout_seconds,
    )
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        _, name = line.split("|", 1)
        if not name.startswith(SUBMITTER_JOB_PREFIX):
            rows.append(line)
    return len(rows)


def wait_for_slurm_slot(max_jobs, poll_seconds, query_timeout_seconds, query_retries):
    assert poll_seconds > 0, "--max-jobs-poll-seconds must be positive"
    assert query_timeout_seconds > 0, "--slurm-query-timeout-seconds must be positive"
    assert query_retries > 0, "--slurm-query-retries must be positive"
    current_jobs = None
    while current_jobs is None or current_jobs >= max_jobs:
        for attempt in range(query_retries):
            try:
                current_jobs = active_slurm_job_count(query_timeout_seconds)
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(
                    "Slurm job-count query failed ({}/{}): {}; retrying.".format(
                        attempt + 1,
                        query_retries,
                        e,
                    ),
                    flush=True,
                )
                time.sleep(min(10, poll_seconds))
        else:
            print(
                "Slurm job-count query failed after {} attempts; waiting {} seconds before trying again.".format(
                    query_retries,
                    poll_seconds,
                ),
                flush=True,
            )
            time.sleep(poll_seconds)
            continue
        if current_jobs < max_jobs:
            break
        print(
            "Found {} active Slurm jobs; waiting for fewer than {} before submitting.".format(
                current_jobs,
                max_jobs,
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


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
    if len(parts) >= 5:
        return sanitize_job_name("_".join(parts[-5:]))
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
    return sanitize_job_name("{}_{}".format(base_job_name, suffix))[:180]


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
        if arg_value is not None and arg_value != "":
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


def submitter_account(partition: str, args: argparse.Namespace) -> str:
    if args.submitter_account is not None:
        return args.submitter_account
    if partition.startswith("iris"):
        return "iris"
    return "iliad"


def without_submitter_flag(argv: List[str]) -> List[str]:
    filtered = []
    for arg in argv:
        if arg == "--submitter-as-slurm":
            continue
        filtered.append(arg)
    return filtered


def write_submitter_script(path: str, partition: str, args: argparse.Namespace) -> None:
    output = args.submitter_output
    if "%" not in output:
        os.makedirs(output, exist_ok=True)
        output = os.path.join(output, SUBMITTER_JOB_PREFIX + sanitize_job_name(args.job_name) + "_%A.out")
    command = [sys.executable, str(Path(__file__).resolve())] + without_submitter_flag(sys.argv[1:])
    with open(path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("#SBATCH --partition={}\n".format(partition))
        f.write("#SBATCH --account={}\n".format(submitter_account(partition, args)))
        f.write("#SBATCH --time={}\n".format(args.submitter_time))
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --ntasks-per-node=1\n")
        f.write("#SBATCH --cpus-per-task={}\n".format(args.submitter_cpus))
        f.write("#SBATCH --mem={}\n".format(args.submitter_mem))
        f.write("#SBATCH --output={}\n".format(output))
        f.write("#SBATCH --error={}\n".format(output))
        f.write("#SBATCH --job-name={}\n".format(SUBMITTER_JOB_PREFIX + sanitize_job_name(args.job_name)))
        f.write("#SBATCH --mail-user={}\n".format(args.mail_user))
        f.write("#SBATCH --mail-type={}\n\n".format(args.mail_type))
        f.write('echo "SLURM_JOBID = "$SLURM_JOBID\n')
        f.write('echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST\n')
        f.write('echo "working directory = "$SLURM_SUBMIT_DIR\n\n')
        f.write("cd {}\n".format(shlex.quote(os.getcwd())))
        f.write(" ".join(shlex.quote(part) for part in command) + "\n")


def submit_self_as_slurm(args: argparse.Namespace) -> None:
    partitions = [item.strip() for item in args.submitter_partitions.split(",") if item.strip()]
    assert partitions, args.submitter_partitions
    for partition in partitions:
        _, slurm_file = tempfile.mkstemp(text=True, prefix="submitter_", suffix=".sh")
        write_submitter_script(slurm_file, partition, args)
        print("Prepared submitter Slurm script: {} ({})".format(slurm_file, partition), flush=True)
        if args.dry_run:
            continue
        try:
            output = subprocess.check_output(["sbatch", slurm_file], text=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print("Submitter sbatch failed on {}: {}".format(partition, e.output.strip()), flush=True)
            continue
        print(output.strip(), flush=True)
        return
    if not args.dry_run:
        raise SystemExit("Failed to submit controller job on any partition: {}".format(partitions))


def train_command(config_path: str, args: argparse.Namespace) -> str:
    command = [
        args.python,
        args.entry_point,
        "--config",
        config_path,
    ]
    if args.resume and Path(args.entry_point).name == "train.py":
        command.append("--resume")
    if args.overwrite:
        command.append("--overwrite")
    if args.debug:
        command.append("--debug")
    return " ".join(shlex.quote(part) for part in command)


def command_with_default_resume(command: str, args: argparse.Namespace) -> str:
    if not args.resume:
        return command
    parts = shlex.split(command)
    if not any(Path(part).name == "train.py" for part in parts):
        return command
    if "--resume" in parts or "--overwrite" in parts:
        return command
    return command + " --resume"


def assign_gpu(command: str, index: int) -> str:
    visible = '"${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"'
    gpu = '$(echo {} | cut -d, -f{})'.format(visible, index + 1)
    return "CUDA_VISIBLE_DEVICES={} {}".format(gpu, command)


def main() -> None:
    args = parse_args()
    if args.submitter_as_slurm:
        submit_self_as_slurm(args)
        return
    if args.commands_file is not None and args.config_paths_file != "robomimic/exps/square/sweep/config_paths.txt":
        raise ValueError("Use either --commands_file or --config_paths_file, not both")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    use_commands_file = args.commands_file is not None
    scripts = read_command_lines(args.commands_file) if use_commands_file else read_config_paths(args.config_paths_file)
    max_jobs = None if args.dry_run else read_max_slurm_jobs(args.max_jobs_file)
    batch_sizes = make_batch_sizes(
        num_scripts=len(scripts),
        scripts_per_job=args.scripts_per_job,
        remainder=args.remainder,
    )
    if not batch_sizes:
        source = args.commands_file if use_commands_file else args.config_paths_file
        print("No launch lines found in {}".format(source))
        return
    if args.assign_gpus:
        assert args.gpus.isdigit(), "--assign-gpus expects numeric --gpus"
        assert max(batch_sizes) <= int(args.gpus), (max(batch_sizes), args.gpus)

    script_index = 0
    exit_codes = []
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
            for local_index, script in enumerate(current_scripts):
                command = command_with_default_resume(script, args) if use_commands_file else train_command(script, args)
                if args.assign_gpus:
                    command = assign_gpu(command, local_index)
                if len(current_scripts) != 1:
                    command += " &"
                f.write(command + "\n")
            if len(current_scripts) != 1:
                f.write("wait\n")

        print("Prepared Slurm script: {} ({})".format(slurm_file, slurm_args.job_name))
        if args.dry_run:
            continue
        wait_for_slurm_slot(
            max_jobs,
            args.max_jobs_poll_seconds,
            args.slurm_query_timeout_seconds,
            args.slurm_query_retries,
        )
        exit_codes.append(subprocess.call(["sbatch", slurm_file]))
        time.sleep(1)

    if exit_codes and any(code != 0 for code in exit_codes):
        raise SystemExit("At least one sbatch call failed: {}".format(exit_codes))


if __name__ == "__main__":
    main()
