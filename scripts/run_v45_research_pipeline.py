"""Run the isolated v45 research chain with visible logs and verified resume."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "output/reports"
MANIFEST = REPORTS / "v45_pipeline_manifest.json"


def stamp():
    return datetime.now().isoformat(timespec="seconds")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(value):
    REPORTS.mkdir(parents=True, exist_ok=True)
    temp = MANIFEST.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(MANIFEST)


def process_alive(pid):
    if not pid:
        return False
    if sys.platform == "win32":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong)
        kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def stage_spec(config, anchor_policy):
    common = ["src/v45_solver.py", "src/portfolio_accounting.py", str(config)]
    return {
        "solver": dict(
            commands=[["scripts/test_v45_solver_contract.py"],
                      ["scripts/audit_v45_solver_path.py", "--anchor-policy", anchor_policy]],
            inputs=common + ["scripts/audit_v45_solver_path.py", "scripts/test_v45_solver_contract.py"],
            outputs=[f"output/reports/{'v45_anchor_smoothing' if anchor_policy == 'causal_projection' else 'v45_solver'}_stage_gate.json"]),
        "data": dict(commands=[["scripts/prepare_v45_data.py"], ["scripts/test_v45_data.py"]],
            inputs=common + ["src/v45_data.py", "scripts/prepare_v45_data.py", "scripts/test_v45_data.py",
                "data_processed/hierarchical_asset_itransformer_v43_clean_calendar_dataset.npz",
                "output/reports/hierarchical_asset_itransformer_v43_clean_calendar_metadata.json",
                "data_processed/v44_frozen_returns_daily.parquet", "data_processed/v44_state_cards_daily.parquet"],
            outputs=["data_processed/v45_dataset.npz", "output/reports/v45_metadata.json", "output/reports/v45_split_dates.csv"]),
        "forecast": dict(commands=[["scripts/test_v45_forecast.py"],
                ["scripts/train_v45_forecasts.py", "--config", str(config), "--preflight-only"],
                ["scripts/train_v45_forecasts.py", "--config", str(config), "--benchmark-only"],
                ["scripts/train_v45_forecasts.py", "--config", str(config)]],
            inputs=common + ["src/v45_forecast.py", "scripts/train_v45_forecasts.py", "scripts/test_v45_forecast.py",
                "src/v45_data.py", "src/hierarchical_asset_itransformer.py", "data_processed/v45_dataset.npz", "output/reports/v45_metadata.json"],
            outputs=["data_processed/v45_predictions.parquet", "output/reports/v45_forecast_report.json"]),
        "portfolio": dict(commands=[["scripts/test_v45_portfolio.py"],
                ["scripts/run_v45_portfolios.py", "--config", str(config)]],
            inputs=common + ["src/v45_portfolio.py", "scripts/run_v45_portfolios.py", "scripts/test_v45_portfolio.py",
                "data_processed/v45_predictions.parquet", "output/reports/v45_forecast_report.json",
                "data_processed/v45_anchor_smoothing_validated_targets.parquet"],
            outputs=[f"output/reports/v45_{p}_A3_daily.csv" for p in ("low_vol", "balanced", "high_vol")]),
        "report": dict(commands=[["scripts/test_v45_report.py"], ["scripts/report_v45_research.py"]],
            inputs=common + ["scripts/report_v45_research.py", "scripts/report_v45_workbook.mjs", "scripts/test_v45_report.py"] +
                [f"output/reports/v45_{p}_{a}_daily.csv" for p in ("low_vol", "balanced", "high_vol") for a in ("A0", "A1", "A2", "A3")],
            outputs=["output/reports/v45_research_report.md"]),
        "audit": dict(commands=[["scripts/validate_v45_pipeline.py"]],
            inputs=common + ["scripts/validate_v45_pipeline.py", "data_processed/v45_predictions.parquet"] +
                [f"output/reports/v45_{p}_A3_daily.csv" for p in ("low_vol", "balanced", "high_vol")],
            outputs=["output/reports/v45_final_integrity_audit.json"]),
    }


def fingerprint(spec):
    paths = [ROOT / p for p in spec["inputs"]]
    metadata = REPORTS / "v45_metadata.json"
    if metadata.exists() and "src/v45_data.py" in spec["inputs"]:
        raw = json.loads(metadata.read_text(encoding="utf-8"))
        paths += [ROOT / item["path"] for item in raw.get("source_manifest", {}).values()]
    signatures = {str(p): digest(p) for p in paths if p.exists()}
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Stage prerequisites missing: " + ", ".join(missing))
    return hashlib.sha256(json.dumps(dict(inputs=signatures, commands=spec["commands"]), sort_keys=True).encode()).hexdigest()


def run_command(arguments, log, state):
    command = [sys.executable, "-u", "-X", "utf8", *arguments]
    print(f"[{stamp()}] {' '.join(arguments)}", flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{stamp()}] {command!r}\n")
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        state["child_pid"] = process.pid
        save(state)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
                else:
                    process.terminate()
                process.wait()
            raise
        finally:
            state.pop("child_pid", None)
    if code:
        raise subprocess.CalledProcessError(code, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "solver", "data", "forecast", "portfolio", "report", "audit"], default="all")
    parser.add_argument("--config", type=Path, default=ROOT / "config/v45_research.yaml")
    parser.add_argument("--anchor-policy", choices=["strict_raw", "causal_projection"], default="causal_projection")
    parser.add_argument("--resume", action="store_true", help="Verify source/output hashes before skipping completed stages")
    args = parser.parse_args()
    config = args.config.resolve()
    existing = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    if existing.get("status") == "running" and process_alive(existing.get("parent_pid", existing.get("child_pid"))):
        raise SystemExit("Another v45 pipeline is active; use scripts/show_v45_progress.py instead of launching a duplicate.")
    previous = existing if args.resume else {}
    old = {row["stage"]: row for row in previous.get("stages", [])}
    versions = {}
    for package in ("numpy", "pandas", "torch", "skfolio", "scipy", "cvxpy-base", "openpyxl", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_found"
    state = dict(started=stamp(), requested_stage=args.stage, config=str(config),
        anchor_policy=args.anchor_policy, python=sys.executable, versions=versions,
        status="running", stages=[], research_only=True, training_started=False, parent_pid=os.getpid())
    specifications = stage_spec(config, args.anchor_policy)
    stages = list(specifications) if args.stage == "all" else [args.stage]
    (ROOT / "output/logs").mkdir(parents=True, exist_ok=True)
    try:
        for name in stages:
            spec = specifications[name]
            key = fingerprint(spec)
            cached = old.get(name, {})
            outputs = {p: digest(ROOT / p) for p in spec["outputs"] if (ROOT / p).exists()}
            if (args.resume and cached.get("status") == "completed" and cached.get("input_signature") == key
                    and len(outputs) == len(spec["outputs"]) and cached.get("output_hashes") == outputs):
                state["stages"].append(dict(cached, resumed_at=stamp()))
                print(f"[{stamp()}] {name}: verified completed stage reused", flush=True)
                save(state)
                continue
            row = dict(stage=name, started=stamp(), status="running", input_signature=key,
                log=f"output/logs/v45_pipeline_{name}.log")
            state["stages"].append(row)
            state["current_stage"] = name
            state["training_started"] |= name in ("forecast", "portfolio")
            save(state)
            start = time.monotonic()
            for command in spec["commands"]:
                command = list(command)
                if args.resume and command[0] in ("scripts/train_v45_forecasts.py", "scripts/run_v45_portfolios.py") and "--benchmark-only" not in command:
                    command.append("--resume")
                run_command(command, ROOT / row["log"], state)
            row["output_hashes"] = {p: digest(ROOT / p) for p in spec["outputs"]}
            row.update(status="completed", finished=stamp(), seconds=time.monotonic()-start)
            save(state)
        state.update(status="completed", finished=stamp(),
            note="Pipeline completion is not a formal promotion or source-PIT certification.")
        save(state)
        print(f"[{stamp()}] requested stages completed; manifest={MANIFEST}", flush=True)
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            finished=stamp(), error=f"{type(exc).__name__}: {exc}")
        if state["stages"] and state["stages"][-1]["status"] == "running":
            state["stages"][-1].update(status=state["status"], error=state["error"])
        save(state)
        raise


if __name__ == "__main__":
    main()
