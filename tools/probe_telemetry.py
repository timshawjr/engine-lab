#!/usr/bin/env python3
"""Probe Windows accelerator devices and the PDH ``GPU Engine`` object."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.telemetry.devices_win import (  # noqa: E402
    DeviceInfo,
    enumerate_compute_accelerators,
    enumerate_display_adapters,
    is_intel,
    is_npu,
    select_driver,
)
from app.telemetry.pdh import (  # noqa: E402
    CounterSample,
    GPUEngineCounter,
    PDHError,
    classify_device,
    enumerate_gpu_engine_items,
    parse_instance,
)


def _windows_build() -> str:
    version = sys.getwindowsversion()
    ubr = 0
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
            ) as key:
                ubr = int(winreg.QueryValueEx(key, "UBR")[0])
        except (OSError, ValueError):
            pass
    return f"{version.major}.{version.minor}.{version.build}.{ubr}"


def _print_device(label: str, device: DeviceInfo | None) -> None:
    if device is None:
        print(f"{label}: NOT FOUND")
        return
    print(
        f"{label}: {device.friendly_name} | driver={device.driver_version or 'unknown'} | "
        f"phys_id={device.phys_id if device.phys_id is not None else 'unavailable'} | "
        f"luid={device.luid if device.luid is not None else 'unavailable'} | id={device.device_id}"
    )


def _counter_path_pattern(instance_name: str) -> str:
    return re.sub(r"^pid_\d+", "pid_<pid>", instance_name)


def _sample_once(
    counter: GPUEngineCounter,
    names: list[str],
    npu_ids: set[int],
    gpu_ids: set[int],
    npu_luids: set[int],
    gpu_luids: set[int],
) -> list[CounterSample]:
    samples = counter.sample_utilization()
    by_name = {sample.instance.instance_name: sample for sample in samples}
    measured_at = time.time()
    for name in names:
        sample = by_name.get(name)
        if sample is None:
            print(f"  {name} = NO SAMPLE (counter present but no valid value at {measured_at:.3f})")
        else:
            device = classify_device(sample.instance, npu_ids, npu_luids, gpu_luids, gpu_ids)
            print(
                f"  {name} = {sample.value:.2f}% [{device}/{sample.instance.engtype or 'unknown engine'}]"
            )
    return samples


def _running_fallback(
    counter: GPUEngineCounter,
    names: list[str],
    npu_ids: set[int],
    gpu_ids: set[int],
    npu_luids: set[int],
    gpu_luids: set[int],
) -> list[CounterSample]:
    try:
        first = counter.sample_running_time_delta()
    except PDHError:
        return []
    time.sleep(1.0)
    try:
        samples = counter.sample_running_time_delta()
    except PDHError:
        return []
    del first
    by_name = {sample.instance.instance_name: sample for sample in samples}
    for name in names:
        sample = by_name.get(name)
        if sample is None:
            print(f"  {name} = NO SAMPLE (running-time delta unavailable)")
        else:
            device = classify_device(sample.instance, npu_ids, npu_luids, gpu_luids, gpu_ids)
            print(f"  {name} = {sample.value:.2f}% delta [{device}/{sample.instance.engtype or 'unknown'}]")
    return samples


def build_map(args: argparse.Namespace) -> tuple[dict[str, Any], bool, str]:
    notes: list[str] = []
    try:
        compute_devices = enumerate_compute_accelerators()
    except Exception as exc:  # SetupAPI can fail on non-Windows or policy-restricted hosts.
        compute_devices = []
        notes.append(f"ComputeAccelerator enumeration failed: {type(exc).__name__}: {exc}")
    try:
        display_devices = enumerate_display_adapters()
    except Exception as exc:
        display_devices = []
        notes.append(f"Display adapter enumeration failed: {type(exc).__name__}: {exc}")

    print("Compute-accelerator devices:")
    if compute_devices:
        for device in compute_devices:
            _print_device("  device", device)
    else:
        print("  none")
    print("Display adapters:")
    if display_devices:
        for device in display_devices:
            _print_device("  device", device)
    else:
        print("  none")

    npu_device = select_driver(compute_devices, npu=True)
    gpu_device = select_driver(display_devices, npu=False)
    _print_device("Selected NPU", npu_device)
    _print_device("Selected Intel GPU", gpu_device)

    npu_phys_ids = sorted(
        {
            device.phys_id
            for device in compute_devices
            if is_npu(device) and is_intel(device) and device.phys_id is not None
        }
    )
    gpu_phys_ids = sorted(
        {
            device.phys_id
            for device in display_devices
            if is_intel(device) and device.phys_id is not None
        }
    )
    npu_luids = {
        device.luid
        for device in compute_devices
        if is_npu(device) and is_intel(device) and device.luid is not None
    }
    gpu_luids = {
        device.luid
        for device in display_devices
        if is_intel(device) and device.luid is not None
    }

    if set(npu_phys_ids) & set(gpu_phys_ids):
        notes.append(
            "SetupAPI reports overlapping physical ids for the NPU and GPU; LUID is used as the unambiguous tie-breaker."
        )

    discovered_instances: list[str] = []
    counter_names: list[str] = []
    observed_samples: list[CounterSample] = []
    counter_source = "none"
    pdh_error: str | None = None
    try:
        counter_names, discovered_instances = enumerate_gpu_engine_items()
        counter_source = "pdh_gpu_engine"
        print(f"PDH GPU Engine counters: {', '.join(counter_names) or '(none reported)'}")
        print(f"PDH GPU Engine instances: {len(discovered_instances)}")
        with GPUEngineCounter() as counter:
            counter.collect()
            time.sleep(args.interval)
            deadline = time.monotonic() + args.duration
            while True:
                observed_samples.extend(
                    _sample_once(
                        counter,
                        discovered_instances,
                        set(npu_phys_ids),
                        set(gpu_phys_ids),
                        npu_luids,
                        gpu_luids,
                    )
                )
                if time.monotonic() >= deadline:
                    break
                time.sleep(args.interval)
            if not any(sample.value > 0 for sample in observed_samples):
                print("No non-zero utilization samples; probing the running-time delta fallback.")
                observed_samples.extend(
                    _running_fallback(
                        counter,
                        discovered_instances,
                        set(npu_phys_ids),
                        set(gpu_phys_ids),
                        npu_luids,
                        gpu_luids,
                    )
                )
    except Exception as exc:
        pdh_error = f"{type(exc).__name__}: {exc}"
        notes.append(f"PDH GPU Engine probe failed: {pdh_error}")
        print(f"PDH GPU Engine probe failed: {pdh_error}")

    all_names = set(discovered_instances)
    all_names.update(sample.instance.instance_name for sample in observed_samples)
    npu_instance_names = {
        sample.instance.instance_name
        for sample in observed_samples
        if classify_device(
            sample.instance,
            set(npu_phys_ids),
            npu_luids,
            gpu_luids,
            set(gpu_phys_ids),
        )
        == "NPU"
    }
    all_names.update(npu_instance_names)
    if not gpu_phys_ids:
        observed_gpu_ids = sorted(
            {
                sample.instance.phys_id
                for sample in observed_samples
                if sample.instance.phys_id is not None
                and sample.instance.phys_id not in set(npu_phys_ids)
            }
        )
        if observed_gpu_ids:
            gpu_phys_ids = observed_gpu_ids
            notes.append("GPU physical ids were derived from observed PDH instances because SetupAPI did not expose them.")

    instances: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str | None]] = set()
    unparsed_instance_count = 0
    for name in sorted(all_names):
        sample = next(
            (item for item in observed_samples if item.instance.instance_name == name),
            None,
        )
        parsed = sample.instance if sample is not None else parse_instance(name)
        if parsed.phys_id is None or parsed.luid_node is None:
            unparsed_instance_count += 1
            continue
        engtype = parsed.engtype or None
        device = classify_device(
            parsed,
            set(npu_phys_ids),
            npu_luids,
            gpu_luids,
            set(gpu_phys_ids),
        )
        pattern = _counter_path_pattern(name)
        key = (pattern, device, engtype)
        if key in seen:
            continue
        seen.add(key)
        instances.append({"pattern": pattern, "device": device, "engtype": engtype})

    if unparsed_instance_count:
        notes.append(
            f"{unparsed_instance_count} enumerated instance name(s) could not be parsed safely; "
            "they remain in the raw probe log and were omitted from the generated map."
        )

    if npu_instance_names:
        notes.append(
            "NPU PDH instances were observed using SetupAPI LUID; physical id was used only after checking for an NPU/GPU id collision."
        )
    elif npu_device is not None:
        notes.append(
            "No NPU PDH instance was observed during the probe. The HUD must use the labelled app-measured NPU duty-cycle fallback."
        )
    else:
        notes.append("No Intel NPU device was found; the app-measured fallback cannot prove NPU placement.")
    if pdh_error:
        notes.append("PDH was unavailable, so the telemetry map records counter_source=none.")

    driver_versions = {
        "npu": npu_device.driver_version if npu_device and npu_device.driver_version else None,
        "gpu": gpu_device.driver_version if gpu_device and gpu_device.driver_version else None,
    }
    telemetry_map: dict[str, Any] = {
        "probed_on": datetime.now(timezone.utc).isoformat(),
        "windows_build": _windows_build(),
        "driver_versions": driver_versions,
        "npu_phys_ids": npu_phys_ids,
        "gpu_phys_ids": gpu_phys_ids,
        "counter_source": counter_source,
        "instances": instances,
        "fallback": "npu_duty_cycle",
        "notes": " ".join(notes),
    }
    return telemetry_map, bool(npu_instance_names), counter_source


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=3.0, help="seconds to print current values")
    parser.add_argument("--interval", type=float, default=0.5, help="sampling interval in seconds")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "telemetry_map.json",
        help="telemetry map output path",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="return success when no NPU PDH instance is found and the labelled fallback is selected",
    )
    args = parser.parse_args(argv)
    if args.duration < 0 or args.interval <= 0:
        parser.error("--duration must be non-negative and --interval must be positive")

    telemetry_map, npu_counter_found, counter_source = build_map(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(telemetry_map, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(f"Wrote {output}")
    print(f"Telemetry provider: {counter_source}; NPU PDH instances found: {npu_counter_found}")
    if not npu_counter_found:
        print(
            "This machine did not expose a matching NPU PDH instance during the probe. "
            "The app will use and label its app-measured NPU duty cycle.",
            file=sys.stderr,
        )
        return 0 if args.allow_fallback else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
