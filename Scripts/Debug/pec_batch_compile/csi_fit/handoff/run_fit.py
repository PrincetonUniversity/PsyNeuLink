#!/usr/bin/env python3
"""Research handoff runner. Use run.sh to set scratch/cache paths first."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys


CONDITIONS = {"NoInstruction", "RealRare", "RealFrequent"}


def inspect_data(path, subject):
    """Preserve CSV order and translate the actual ID to the legacy GPU index."""
    required = {"subject_nr", "sequence", "T1", "T2", "S1", "S2", "S3", "S4",
                "correct_response", "decision", "response_time", "likelihood_include_mask"}
    subjects, selected, mask_values = [], [], set()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for row in reader:
            value = float(row["subject_nr"])
            if not math.isfinite(value) or not value.is_integer():
                raise ValueError("subject_nr must contain finite integer IDs.")
            actual = int(value)
            if actual not in subjects:
                subjects.append(actual)
            mask_values.add(row["likelihood_include_mask"].lower())
            if actual == subject and row["sequence"] in CONDITIONS:
                selected.append(row)
    if not (mask_values <= {"0", "1"} or mask_values <= {"true", "false"}):
        raise ValueError("Use a consistent 0/1 or True/False likelihood_include_mask column; no blanks.")
    if not selected:
        raise ValueError(f"No retained rows for subject {subject}. Available IDs: {subjects}")
    for row in selected:
        for name in ("T1", "T2", "S1", "S2", "S3", "S4", "correct_response", "decision", "response_time"):
            if not math.isfinite(float(row[name])):
                raise ValueError(f"Nonfinite {name} in subject {subject}.")
        if float(row["decision"]) not in (0, 1) or float(row["correct_response"]) not in (-1, 1):
            raise ValueError("decision must be 0/1; correct_response must be -1/+1.")
        if float(row["response_time"]) <= 0:
            raise ValueError("All retained RTs, including masked rows, must be positive seconds.")
    included = [row for row in selected if row["likelihood_include_mask"].lower() in ("1", "true")]
    if {row["sequence"] for row in included} != CONDITIONS:
        raise ValueError("This 13-parameter workflow requires included rows in all three CSI conditions.")
    return {"subject_nr": subject, "gpu_subject_index": subjects.index(subject) + 1,
            "retained_rows": len(selected), "included_rows": len(included)}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("cpu", "gpu"))
    parser.add_argument("--subject", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "1")),
                        help="Actual subject_nr, on BOTH backends; default Slurm array ID or 1.")
    parser.add_argument("--data", type=Path, default=Path(os.environ["CSI_DATA_FILE"]))
    parser.add_argument("--output", type=Path, help="New run directory; must not already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and print commands; no fit or writes.")
    parser.add_argument("--smoke", action="store_true", help="Tiny optimizer/simulation budget; not a scientific fit.")
    parser.add_argument("--seed", type=int, default=1, help="Optimizer seed (both backends).")
    parser.add_argument("--iterations", type=int, help="CPU iterations per start; GPU candidate-evaluation budget.")
    bounds = parser.add_argument_group("Search bounds (both backends)")
    bounds.add_argument("--gain-upper-bound", type=float, default=120.0)
    bounds.add_argument("--threshold-upper-bound", type=float, default=0.30)
    bounds.add_argument("--non-decision-time-upper-bound", type=float, default=0.50)
    cpu = parser.add_argument_group("CPU direct solver")
    cpu.add_argument("--starts", type=int, default=4)
    cpu.add_argument("--random-start-candidates", type=int, default=32)
    cpu.add_argument("--initial-parameters", type=Path, action="append", default=[])
    cpu.add_argument("--ddm-time-step", type=float, default=0.001)
    cpu.add_argument("--ddm-spatial-points", type=int, default=65)
    cpu.add_argument("--lca-max-step", type=float, default=0.01)
    gpu = parser.add_argument_group("GPU generated batched likelihood")
    gpu.add_argument("--time-step", type=float, default=0.001)
    gpu.add_argument("--horizon", type=float, default=12.0, help="Maximum DDM simulation time in seconds.")
    gpu.add_argument("--strict-truncation", action=argparse.BooleanOptionalAction, default=False,
                     help="Require every trajectory to finish; default uses checked histogram-window stopping.")
    gpu.add_argument("--estimates", type=int, default=100000)
    gpu.add_argument("--batch-size", type=int, default=11)
    gpu.add_argument("--buffer-mib", type=int, default=1024)
    gpu.add_argument("--bins", type=int, default=100)
    gpu.add_argument("--smoothing-sigma", type=float, default=0.5)
    gpu.add_argument("--pseudocount", type=float, default=0.1)
    gpu.add_argument("--simulation-seed", type=int, default=1)
    gpu.add_argument("--predictive-simulations", type=int, default=0)
    gpu.add_argument("--rescore", type=Path, help="GPU fit.csv to rescore instead of fitting; reuse its estimator settings.")
    gpu.add_argument("--rescore-seeds", nargs="+", type=int, default=[101, 102, 103])
    args = parser.parse_args()
    if args.rescore and args.backend != "gpu":
        parser.error("--rescore is for GPU CSVs. CPU fits are automatically rescored.")
    for name in ("starts", "estimates", "batch_size", "buffer_mib", "bins", "random_start_candidates"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.iterations is not None and args.iterations < 1:
        parser.error("--iterations must be positive")
    return args


def make_commands(args, repo, data, output, selection):
    fit_dir = repo / "Scripts/Debug/pec_batch_compile/csi_fit"
    if args.backend == "cpu":
        driver = [sys.executable, "-u", str(fit_dir / "csi_direct_likelihood.py")]
        shared = ["--data", str(data), "--subject", str(args.subject), "--device", "cpu",
                  "--ddm-time-step", str(args.ddm_time_step), "--ddm-spatial-points", str(args.ddm_spatial_points),
                  "--lca-max-step", str(args.lca_max_step), "--lca-integration-method", "rk4",
                  "--ddm-bucket-size", "256", "--native-lca-scan", "--native-ddm-forward"]
        command = driver + ["fit"] + shared + [
            "--optimizer", "lbfgsb", "--gradient-method", "autograd", "--seed", str(args.seed),
            "--starts", str(1 if args.smoke else args.starts),
            "--max-iterations", str(args.iterations or (1 if args.smoke else 200)),
            "--random-start-candidates", str(1 if args.smoke else args.random_start_candidates),
            "--gain-upper-bound", str(args.gain_upper_bound),
            "--threshold-upper-bound", str(args.threshold_upper_bound),
            "--non-decision-time-upper-bound", str(args.non_decision_time_upper_bound),
            "--output", str(output / "fit.partial.json")]
        for path in args.initial_parameters:
            command += ["--initial-parameters", str(path.expanduser().resolve(strict=True))]
        if args.smoke:
            command += ["--polish-restarts", "0", "--no-coordinate-polish"]
        score = driver + ["score"] + shared + ["--parameters", str(output / "fit.partial.json"),
                                                "--output", str(output / "fresh-score.json")]
        return [command, score]
    command = [sys.executable, "-u", str(fit_dir / "data fitting/expectation_fit_study3.2_real_sequences_single_csi_leak12.py"),
               "--backend", "triton", "--subject-id", str(selection["gpu_subject_index"]), "--data-file", str(data),
               "--condition-observed-history", "--deterministic-observed-history", "--history-implementation", "generated",
               "--strict-truncation" if args.strict_truncation else "--no-strict-truncation",
               "--maximum-simulation-time", str(args.horizon),
               "--gain-upper-bound", str(args.gain_upper_bound),
               "--threshold-upper-bound", str(args.threshold_upper_bound),
               "--non-decision-time-upper-bound", str(args.non_decision_time_upper_bound),
               "--model-time-step", str(args.time_step), "--num-estimates", str(64 if args.smoke else args.estimates),
               "--max-iterations", str(args.iterations or (22 if args.smoke else 5000)),
               "--parameter-batch-size", str(args.batch_size), "--likelihood-buffer-mib", str(args.buffer_mib),
               "--bins", str(args.bins), "--smoothing-sigma", str(args.smoothing_sigma), "--pseudocount", str(args.pseudocount),
               "--optimizer-seed", str(args.seed), "--simulation-seed", str(args.simulation_seed),
               "--cpu-count", os.environ["OMP_NUM_THREADS"], "--triton-block-size", "32", "--triton-num-warps", "1",
               "--output-dir", str(output), "--fit-output", str(output / "fit.csv"), "--run-label", output.name]
    if args.predictive_simulations and not args.smoke and not args.rescore:
        command += ["--posterior-predictive-simulations", str(args.predictive_simulations)]
    else:
        command += ["--skip-posterior-predictive"]
    if args.rescore:
        command += ["--rescore-parameter-file", str(args.rescore.expanduser().resolve(strict=True)),
                    "--rescore-simulation-seeds", *map(str, args.rescore_seeds)]
    return [command]


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main():
    args = arguments()
    repo = Path(os.environ["CSI_REPO_ROOT"]).resolve(strict=True)
    data = args.data.expanduser().resolve(strict=True)
    selection = inspect_data(data, args.subject)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    label = f"{args.backend}-{'smoke' if args.smoke else 'run'}-{os.environ.get('SLURM_JOB_ID', 'local')}-{stamp}"
    output = (args.output or Path(os.environ["CSI_RESULTS_ROOT"]) / label / f"subject-{args.subject}").expanduser().resolve()
    if output.is_relative_to(Path.home().resolve()):
        raise ValueError("Output must be outside the home directory.")
    if output.exists():
        raise ValueError(f"Refusing to reuse an output directory: {output}")
    commands = make_commands(args, repo, data, output, selection)
    print(json.dumps(selection, indent=2), flush=True)
    print(f"Output: {output}", flush=True)
    for command in commands:
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    # Do this before importing Torch/Triton; their startup may create caches.
    for name in ("TMPDIR", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "MPLCONFIGDIR"):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    import torch
    if args.backend == "gpu":
        import triton  # noqa: F401
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use an allocated GPU node and a CUDA-enabled Torch environment.")
    elif not shutil.which("ninja") or not shutil.which(os.environ.get("CXX", "c++")):
        raise RuntimeError("The native direct solver requires Ninja and a C++ compiler with OpenMP on PATH.")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "started_utc": stamp, "backend": args.backend, "smoke": args.smoke,
                "selection": selection, "data": str(data), "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "commands": commands, "git_commit": git(repo, "rev-parse", "HEAD"),
                "git_status": git(repo, "status", "--short"), "host": socket.gethostname(),
                "python": sys.version, "torch_cuda": torch.version.cuda,
                "torch_num_threads": torch.get_num_threads(),
                "gpu": torch.cuda.get_device_name(0) if args.backend == "gpu" else None,
                "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()},
                "environment": {k: v for k, v in os.environ.items() if k.startswith(("CSI_", "SLURM_")) or k in
                                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CXX", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR")}}
    (output / "source.diff").write_text(git(repo, "diff", "HEAD"))
    manifest_path = output / "run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for index, command in enumerate(commands, 1):
            print(f"Running step {index}/{len(commands)}", flush=True)
            subprocess.run(command, cwd=output, check=True)
        if args.backend == "cpu":
            fit = json.loads((output / "fit.partial.json").read_text())
            score = json.loads((output / "fresh-score.json").read_text())
            if not all(math.isfinite(float(item["log_likelihood"])) for item in (fit, score)):
                raise RuntimeError("Nonfinite direct likelihood.")
            if abs(fit["log_likelihood"] - score["log_likelihood"]) > 1e-8:
                raise RuntimeError("Fresh direct score does not reproduce the fit.")
            if any(score["diagnostics"][key] for key in ("invalid_included_rows", "zero_probability_included_rows")):
                raise RuntimeError("Fresh direct score has invalid or zero-probability included rows.")
            (output / "fit.partial.json").rename(output / "fit.json")
        else:
            result = next(output.glob("*_rescore.csv")) if args.rescore else output / "fit.csv"
            with result.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            key = "validation_log_likelihood" if args.rescore else "log_likelihood"
            if not rows or not all(math.isfinite(float(row[key])) and int(float(row["subject_nr"])) == args.subject for row in rows):
                raise RuntimeError("GPU output has a nonfinite likelihood or incorrect subject.")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed: {output}. Inspect optimizer diagnostics before accepting the fit.", flush=True)


if __name__ == "__main__":
    main()
