"""Per-scenario detection-quality report across all six verticals.

Run the app once per scenario, then summarise the frame samples. The figures that
matter for the reported regression:

  det/frame      mean detections per frame. A drop means detection got worse.
  jumpiness      mean |change in detection count| between consecutive frames.
                 Phantom objects show up here as large frame-to-frame swings.
  index gaps     how often the worker's frame_index skips. Non-zero means the
                 capture is being seeked, which discards frames the tracker
                 depends on. Measured 0 on a healthy run.
  DET/s          detections per second over the window.

Usage: qa_detection.py <seconds> <tag> [scenario ...]
"""
from __future__ import annotations

import json
import subprocess
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("retail", "metro", "manufacturing", "education", "health", "federal")


def run(scenario: str, seconds: int, tag: str) -> Path:
    out = ROOT / "logs" / f"_qa_{tag}_{scenario}.json"
    if out.exists():
        out.unlink()
    cmd = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "app.main",
        "--scenario", scenario, "--density", "1",
        "--exit-after", str(seconds), "--diagnostic-output", str(out),
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 300, cwd=str(ROOT))
    return out


def summarise(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("frame_samples") or []
    if not samples:
        return None

    by_stream: dict[int, list] = {}
    for s in samples:
        by_stream.setdefault(int(s["stream_index"]), []).append(s)

    counts, jumps, gaps, det_rates = [], [], 0, []
    for rows in by_stream.values():
        rows.sort(key=lambda r: r["frame_index"])
        prev = None
        for r in rows:
            counts.append(int(r["detection_count"]))
            det_rates.append(float(r.get("detections_per_second") or 0.0))
            if prev is not None:
                jumps.append(abs(int(r["detection_count"]) - prev))
                steps = int(r["frame_index"]) - int(rows[rows.index(r) - 1]["frame_index"])
                if steps > 1:
                    gaps += steps - 1
            prev = int(r["detection_count"])

    return {
        "frames": len(samples),
        "det_mean": statistics.fmean(counts) if counts else 0.0,
        "jump_mean": statistics.fmean(jumps) if jumps else 0.0,
        "jump_max": max(jumps) if jumps else 0,
        "index_gaps": gaps,
        "det_rate": statistics.fmean(det_rates) if det_rates else 0.0,
        "events": len(data.get("events") or []),
    }


def main() -> int:
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    tag = sys.argv[2] if len(sys.argv) > 2 else "run"
    wanted = sys.argv[3:] or list(SCENARIOS)

    print(f"{'scenario':14s} {'frames':>7s} {'det/frame':>10s} {'jump_mean':>10s} "
          f"{'jump_max':>9s} {'index_gaps':>11s} {'DET/s':>8s} {'events':>7s}")
    print("-" * 84)
    rows = {}
    for scenario in wanted:
        result = summarise(run(scenario, seconds, tag))
        rows[scenario] = result
        if not result:
            print(f"{scenario:14s}   (no samples)")
            continue
        print(f"{scenario:14s} {result['frames']:7d} {result['det_mean']:10.2f} "
              f"{result['jump_mean']:10.2f} {result['jump_max']:9d} "
              f"{result['index_gaps']:11d} {result['det_rate']:8.1f} {result['events']:7d}")
    (ROOT / "logs" / f"_qa_{tag}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())