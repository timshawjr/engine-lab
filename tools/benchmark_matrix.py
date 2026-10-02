#!/usr/bin/env python3
"""Run the Phase 3 scenario × density acceptance matrix and update bench_report.md."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
REPORT = ROOT / "bench_report.md"
SCENARIOS = (
    "retail",
    "metro",
    "manufacturing",
    "education",
    "health",
    "federal",
)
START_MARKER = "<!-- PHASE3_BENCHMARK_START -->"
END_MARKER = "<!-- PHASE3_BENCHMARK_END -->"


@dataclass(frozen=True)
class RunSpec:
    name: str
    scenario: str
    density: int
    seconds: float
    attract: bool = False


@dataclass(frozen=True)
class RunResult:
    spec: RunSpec
    passed: bool
    detail: str
    payload: dict[str, Any]


def _run(
    spec: RunSpec,
    timeout_padding: float,
    *,
    require_rss_checkpoint: bool,
) -> RunResult:
    output = LOGS / f"phase3-benchmark-{spec.name}.json"
    command = [
        sys.executable,
        "-m",
        "app.main",
        "--scenario",
        spec.scenario,
        "--mode",
        "spread",
        "--density",
        str(spec.density),
        "--exit-after",
        f"{spec.seconds:g}",
        "--diagnostic-output",
        str(output),
    ]
    if spec.attract:
        command.append("--attract")
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["OV_TELEMETRY_OPT_IN"] = "0"
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=spec.seconds + timeout_padding,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return RunResult(spec, False, f"timeout after {spec.seconds + timeout_padding:g}s", {})
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout)[-1000:]
        return RunResult(spec, False, f"exit {completed.returncode}: {tail}", {})
    runtime_log = (completed.stderr or "") + (completed.stdout or "")
    error_markers = (
        "Traceback (most recent call last)",
        "Error calling Python override",
        "QThread: Destroyed while thread is still running",
    )
    if any(marker in runtime_log for marker in error_markers):
        return RunResult(
            spec,
            False,
            f"runtime exception marker: {runtime_log[-1000:]}",
            {},
        )
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return RunResult(spec, False, f"invalid diagnostic JSON: {exc}", {})

    duration = float(payload["captured_at"]) - float(payload["started_at"])
    if payload.get("last_error"):
        return RunResult(spec, False, f"application error: {payload['last_error']}", payload)
    if duration < spec.seconds * 0.90:
        return RunResult(spec, False, f"ran only {duration:.1f}s", payload)
    if int(payload.get("density", -1)) != spec.density:
        return RunResult(spec, False, "density changed during run", payload)
    processed = int(
        payload.get("measurement_summary", {})
        .get("streams", {})
        .get("processed", 0)
    )
    if processed < spec.density:
        return RunResult(
            spec,
            False,
            f"only {processed}/{spec.density} streams processed",
            payload,
        )
    if not spec.attract and payload.get("scenario") != spec.scenario:
        return RunResult(spec, False, "final scenario mismatch", payload)
    if spec.attract:
        visited = {item.get("scenario") for item in payload.get("scenario_history", [])}
        missing = sorted(set(SCENARIOS) - visited)
        if missing:
            return RunResult(spec, False, f"attract missed scenarios: {missing}", payload)
        timeline = payload.get("rss_timeline", [])
        checkpoint = min(
            (
                item
                for item in timeline
                if float(item.get("elapsed_s", 0.0)) >= 120.0
            ),
            key=lambda item: abs(float(item["elapsed_s"]) - 120.0),
            default=None,
        )
        if not timeline:
            return RunResult(spec, False, "missing RSS timeline", payload)
        if checkpoint is None:
            if require_rss_checkpoint:
                return RunResult(spec, False, "missing two-minute RSS checkpoint", payload)
            checkpoint = timeline[0]
        rss_2m = float(checkpoint["rss_bytes"])
        rss_final = float(timeline[-1]["rss_bytes"])
        growth = (rss_final - rss_2m) / rss_2m
        if require_rss_checkpoint and growth >= 0.05:
            return RunResult(
                spec,
                False,
                f"RSS growth {growth:.2%} from 2m to final",
                payload,
            )
        detail = (
            f"{duration:.1f}s, scenarios={len(visited)}, "
            f"RSS growth={growth:.2%}"
        )
        return RunResult(spec, True, detail, payload)
    frames = len(payload.get("frame_samples", []))
    return RunResult(
        spec,
        True,
        f"{duration:.1f}s, {frames} measured frames, no application error",
        payload,
    )


def _load_existing(spec: RunSpec) -> RunResult:
    path = LOGS / f"phase3-benchmark-{spec.name}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    passed = not payload.get("last_error")
    return RunResult(
        spec,
        passed,
        "existing diagnostic has no application error" if passed else "existing diagnostic has an error",
        payload,
    )


def _markdown(results: list[RunResult]) -> str:
    lines = [
        START_MARKER,
        "## Phase 3 automated scenario matrix",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Run | Duration target | Result | Evidence |",
        "|---|---:|---|---|",
    ]
    for result in results:
        evidence = f"`logs/phase3-benchmark-{result.spec.name}.json`"
        lines.append(
            f"| {result.spec.name} | {result.spec.seconds:g}s | "
            f"{'PASS' if result.passed else 'FAIL'} — {result.detail} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "The RSS acceptance comparison uses the first measured sample at or after two minutes "
            "and the final sample from the 10-minute attract run.",
            END_MARKER,
        ]
    )
    return "\n".join(lines)


def _update_report(markdown: str) -> None:
    text = REPORT.read_text(encoding="utf-8")
    if START_MARKER in text and END_MARKER in text:
        prefix, remainder = text.split(START_MARKER, 1)
        _, suffix = remainder.split(END_MARKER, 1)
        text = f"{prefix}{markdown}{suffix}"
    else:
        text = text.rstrip() + "\n\n" + markdown + "\n"
    temporary = REPORT.with_suffix(".md.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, REPORT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="development smoke durations instead of the 5m/1m/10m acceptance durations",
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="analyze existing benchmark JSON files without launching the app",
    )
    parser.add_argument(
        "--timeout-padding",
        type=float,
        default=120.0,
        help="extra subprocess seconds allowed for startup and shutdown",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    d1_seconds = 5.0 if args.quick else 300.0
    d4_seconds = 5.0 if args.quick else 60.0
    attract_seconds = 55.0 if args.quick else 600.0
    specs: list[RunSpec] = []
    for scenario in SCENARIOS:
        specs.append(
            RunSpec(
                name=f"{scenario}-d1",
                scenario=scenario,
                density=1,
                seconds=d1_seconds,
            )
        )
    for scenario in SCENARIOS:
        specs.append(
            RunSpec(
                name=f"{scenario}-d4",
                scenario=scenario,
                density=4,
                seconds=d4_seconds,
            )
        )
    specs.append(
        RunSpec(
            name="attract-d4",
            scenario="retail",
            density=4,
            seconds=attract_seconds,
            attract=True,
        )
    )
    LOGS.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []
    for spec in specs:
        print(f"{'Analyzing' if args.skip_runs else 'Running'} {spec.name} ({spec.seconds:g}s)", flush=True)
        result = (
            _load_existing(spec)
            if args.skip_runs
            else _run(
                spec,
                args.timeout_padding,
                require_rss_checkpoint=not args.quick,
            )
        )
        results.append(result)
        print(f"  {'PASS' if result.passed else 'FAIL'}: {result.detail}", flush=True)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "quick": args.quick,
        "passed": sum(result.passed for result in results),
        "failed": sum(not result.passed for result in results),
        "runs": [
            {
                "name": result.spec.name,
                "passed": result.passed,
                "detail": result.detail,
            }
            for result in results
        ],
    }
    output = LOGS / "phase3-benchmark-summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _update_report(_markdown(results))
    print(f"Wrote {output} and updated {REPORT.name}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
