#!/usr/bin/env python3
"""
Parse condor .err log files for GenSBI training runs.
Extract training time and time per step for runs that completed (100%).
Output a CSV table grouped by task, model (architecture), and methodology.

Usage:
    python parse_training_times.py [LOG_DIR] > training_times.csv
"""

import os
import re
import csv
import sys
import pandas as pd

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "sub", "sbibm", "condor_logs")

GPU_MAPPING = {
    "mlwn01.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn02.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn03.ific.uv.es": "Tesla V100-SXM2-32GB",
    "mlwn04.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn05.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn06.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn07.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn08.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn09.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn10.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn11.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn12.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn13.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn14.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn15.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn16.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn17.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn18.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn19.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn20.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn21.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn22.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn23.ific.uv.es": "Tesla V100-PCIE-32GB",
    "mlwn24.ific.uv.es": "NVIDIA A100-SXM4-40GB",
    "mlwn25.ific.uv.es": "NVIDIA A100-PCIE-40GB",
    "mlwn26.ific.uv.es": "NVIDIA A100-PCIE-40GB",
    "mlwn27.ific.uv.es": "NVIDIA A100-PCIE-40GB",
    "mlwn28.ific.uv.es": "NVIDIA A100-PCIE-40GB",
    "mlwn29.ific.uv.es": "NVIDIA A100-PCIE-40GB",
    "mlwn30.ific.uv.es": "NVIDIA A100-PCIE-40GB",
    "mlwn31.ific.uv.es": "NVIDIA A100-PCIE-40GB",
    "mlwn32.ific.uv.es": "NVIDIA A100-PCIE-40GB",
    "mlwn33.ific.uv.es": "NVIDIA A100-PCIE-40GB",
    "mlwn34.ific.uv.es": "NVIDIA A100-PCIE-40GB",
    "mlwn36.ific.uv.es": "NVIDIA H100 NVL",
    "mlwn37.ific.uv.es": "NVIDIA H100 NVL",
}

def get_gpu_from_log(log_path):
    if not os.path.exists(log_path):
        return "Unknown Host", "Unknown GPU"
        
    try:
        with open(log_path, "r") as f:
            content = f.read()
            m = re.search(r'(mlwn\d\d\.ific\.uv\.es)', content)
            if m:
                host = m.group(1)
                return host, GPU_MAPPING.get(host, "Unknown GPU")
    except Exception as e:
        print(f"Error reading {log_path}: {e}", file=sys.stderr)
        
    return "Unknown Host", "Unknown GPU"


# Regex to parse the LAST meaningful tqdm line
# Format: XX%|... | steps/total [elapsed<remaining, ...]
TQDM_PATTERN = re.compile(
    r'\s*(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:]+)'
)


def parse_duration(s):
    """Convert HH:MM:SS or MM:SS to seconds."""
    parts = s.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return float(parts[0])


def format_duration(secs):
    """Format seconds as HH:MM:SS."""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_filename(fname):
    """
    Parse filename like: errors_{task}_{method}_{model}_{budget}.err
    Methods: diffusion (EDM), flow (FM), score_matching (SM)
    Models: flux (Flux1), flux1joint (Flux1Joint)
    Tasks: bernoulli_glm, gaussian_linear, gaussian_mixture, slcp, two_moons, ...
    """
    name = fname.replace("errors_", "").replace(".err", "")
    # Budget is the last part (a number)
    parts = name.rsplit("_", 1)
    if len(parts) != 2:
        return None
    rest, budget_str = parts
    if not budget_str.isdigit():
        return None
    budget = int(budget_str)

    # Model is one of: flux1joint, flux — check flux1joint first (longer match)
    if rest.endswith("_flux1joint"):
        model = "Flux1Joint"
        rest = rest[: -len("_flux1joint")]
    elif rest.endswith("_flux"):
        model = "Flux1"
        rest = rest[: -len("_flux")]
    else:
        return None

    # Method is one of: diffusion, flow, score_matching
    methods = ["score_matching", "diffusion", "flow"]
    method = None
    for m in methods:
        if rest.endswith(f"_{m}"):
            method = m
            rest = rest[: -len(f"_{m}")]
            break
    if method is None:
        return None

    task = rest  # remaining is the task name

    method_map = {"diffusion": "EDM", "flow": "FM", "score_matching": "SM"}

    return {
        "task": task,
        "model": model,
        "method": method_map.get(method, method),
        "budget": budget,
        "filename": fname,
    }


def get_last_tqdm_line(filepath):
    """Read the last portion of a file and return the last non-empty tqdm line."""
    chunk_size = 64 * 1024  # 64 KB from end
    try:
        fsize = os.path.getsize(filepath)
        with open(filepath, "rb") as f:
            seek_pos = max(0, fsize - chunk_size)
            f.seek(seek_pos)
            data = f.read()

        text = data.decode("utf-8", errors="replace")
        lines = re.split(r"[\r\n]+", text)
        lines = [l for l in lines if l.strip()]
        return lines[-1] if lines else None
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return None


def parse_tqdm_line(line):
    """Parse a tqdm line and extract percentage, elapsed time, and step count."""
    # Remove ANSI colour codes
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    line = ansi_escape.sub("", line)

    m = TQDM_PATTERN.search(line)
    if not m:
        return None

    return {
        "percent": int(m.group(1)),
        "steps_done": int(m.group(2)),
        "steps_total": int(m.group(3)),
        "elapsed_secs": parse_duration(m.group(4)),
        "remaining_secs": parse_duration(m.group(5)),
    }


def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else LOG_DIR
    log_dir = os.path.abspath(log_dir)

    results = []
    skipped = []
    errors = []

    err_files = sorted(f for f in os.listdir(log_dir) if f.endswith(".err"))
    print(f"Found {len(err_files)} .err files in {log_dir}", file=sys.stderr)

    for fname in err_files:
        parsed = parse_filename(fname)
        if parsed is None:
            errors.append(f"Could not parse filename: {fname}")
            continue

        log_fname = fname.replace("errors_", "logs_").replace(".err", ".log")
        host, gpu = get_gpu_from_log(os.path.join(log_dir, log_fname))

        last_line = get_last_tqdm_line(os.path.join(log_dir, fname))
        if last_line is None:
            skipped.append(f"No content: {fname}")
            continue

        tqdm_info = parse_tqdm_line(last_line)
        if tqdm_info is None:
            skipped.append(f"Could not parse tqdm line in: {fname}")
            print(f"  Last line sample: {last_line[:200]}", file=sys.stderr)
            continue

        # Compute derived timing metrics (for all runs, including early stopped)
        elapsed = tqdm_info["elapsed_secs"]
        steps = tqdm_info["steps_done"]
        # Derive it/s from total_time / total_steps (more reliable than instantaneous tqdm value)
        its = steps / elapsed if elapsed > 0 else 0.0
        time_per_step_ms = (elapsed / steps) * 1000 if steps > 0 else 0.0

        results.append(
            {
                "task": parsed["task"],
                "model": parsed["model"],
                "method": parsed["method"],
                "budget": parsed["budget"],
                "host": host,
                "gpu": gpu,
                "percent": tqdm_info["percent"],
                "completed": tqdm_info["percent"] == 100,
                "training_time_hms": format_duration(elapsed),
                "training_time_s": round(elapsed, 1),
                "time_per_step_ms": round(time_per_step_ms, 3),
                "its": round(its, 3),
                "steps": steps,
            }
        )

    print(f"\n=== Skipped ({len(skipped)}) ===", file=sys.stderr)
    for s in skipped:
        print(f"  {s}", file=sys.stderr)

    print(f"\n=== Parse Errors ({len(errors)}) ===", file=sys.stderr)
    for e in errors:
        print(f"  {e}", file=sys.stderr)

    print(f"\n=== Parsed runs: {len(results)} ===", file=sys.stderr)

    if not results:
        print("No runs found!", file=sys.stderr)
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by=["task", "model", "method", "budget"])
    
    # Save to standard CSV file location
    out_file = "training_times.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved {len(results)} rows to {os.path.abspath(out_file)}", file=sys.stderr)


if __name__ == "__main__":
    main()
