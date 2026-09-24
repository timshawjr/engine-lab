# Engine Lab benchmark report

## Phase 0 — machine setup and verification

- **Run date:** 2026-09-24 after driver reboot
- **Machine:** Windows 11 build `26200.7840`
- **Python:** `3.12.10` in `C:\dev\engine-lab\.venv`
- **OpenVINO:** package `2026.4.0`; runtime `2026.4.0-22959-99c81491cc3-releases/2026/4`
- **CPU:** Intel(R) Core(TM) Ultra 9 288V (Lunar Lake)
- **iGPU:** Intel(R) Arc(TM) 140V GPU (16GB), driver `32.0.101.6737`
- **NPU:** Intel(R) AI Boost, driver `32.0.100.5540`
- **Telemetry provider actually observed:** Windows PDH `GPU Engine`, with SetupAPI LUID used to disambiguate the NPU and GPU

## Phase status

**COMPLETE — Phase 0 exit criteria passed.** After installing Intel NPU driver
`32.0.100.5540`, rebooting, and resetting the OpenVINO cache, the final cold-cache preflight
produced **77 PASS, 0 WARN, 0 FAIL**. The local-only self-test also passed 8/8 checks.

## Commands actually run

```text
.venv\Scripts\python.exe tools\verify_sources.py
.venv\Scripts\python.exe tools\download_models.py
.venv\Scripts\python.exe tools\probe_telemetry.py
.venv\Scripts\python.exe tools\preflight.py
.venv\Scripts\python.exe -m app.main --selftest
```

Observed results:

- Source verification: **40 PASS, 0 FAIL**.
- Downloader: **10 model entries, 6 media entries, 0 failures**.
- Telemetry probe: exit **0**; provider `pdh_gpu_engine`; NPU instance found.
- Local-only self-test: **8 PASS, 0 FAIL**, with Python outbound sockets blocked by the test.
- Full cold-cache preflight: **77 PASS, 0 WARN, 0 FAIL** after the driver update and reboot.
- Inference throughput, latency, and booth runtime: **not run**; the Phase 1 application does not exist yet.

## NPU driver installation evidence

- Official package: `https://downloadmirror.intel.com/927127/npu_win_32.0.100.5540.exe`
- SHA-256: `FBA32699B699918A793FCA05CFC6C607C8745772057347476ECD8C4CF7746048`
- Authenticode: `Valid`; signer `Intel Corporation`
- Installer exit code: `1014`
- Post-reboot SetupAPI result: Intel(R) AI Boost driver `32.0.100.5540`
- Cache action: `C:\dev\engine-lab\cache` removed before the final cold preflight

The nonzero installer exit was not treated as success by itself. The installed version was read
again through SetupAPI after reboot and independently passed the preflight version gate.

## Model compilation matrix

These are cold-cache compile timings after the driver update and reboot, **not inference performance
measurements**. Placement is the value returned by the compiled model's `EXECUTION_DEVICES` property.

| Model | NPU | GPU | CPU |
|---|---:|---:|---:|
| `yolo11n-fp16` | PASS, 2126 ms, `NPU` | PASS, 797 ms, `GPU.0` | PASS, 273 ms, `CPU` |
| `yolo11n-int8` | PASS, 2133 ms, `NPU` | PASS, 1741 ms, `GPU.0` | PASS, 481 ms, `CPU` |
| `efficientnet-b0-fp16` | PASS, 1259 ms, `NPU` | PASS, 484 ms, `GPU.0` | PASS, 180 ms, `CPU` |
| `efficientnet-b0-int8` | PASS, 1656 ms, `NPU` | PASS, 1276 ms, `GPU.0` | PASS, 362 ms, `CPU` |
| `person-detection-retail-0013` | PASS, 962 ms, `NPU` | PASS, 555 ms, `GPU.0` | PASS, 225 ms, `CPU` |
| `product-detection-0001` | PASS, 1137 ms, `NPU` | PASS, 421 ms, `GPU.0` | PASS, 184 ms, `CPU` |
| `person-vehicle-bike-detection-crossroad-1016` | PASS, 1110 ms, `NPU` | PASS, 264 ms, `GPU.0` | PASS, 120 ms, `CPU` |
| `vehicle-license-plate-detection-barrier-0106` | PASS, 627 ms, `NPU` | PASS, 485 ms, `GPU.0` | PASS, 140 ms, `CPU` |
| `age-gender-recognition-retail-0013` | PASS, 165 ms, `NPU` | PASS, 111 ms, `GPU.0` | PASS, 38 ms, `CPU` |
| `human-pose-estimation-0001` | PASS, 2613 ms, `NPU` | PASS, 748 ms, `GPU.0` | PASS, 107 ms, `CPU` |

**Compile failures by device:** none observed. Every placement value is derived from
`EXECUTION_DEVICES`; it is not inferred from the requested device string.

## Telemetry probe result

SetupAPI reported `phys_id=0` for both the NPU and iGPU, so physical id alone was ambiguous. Their
post-reboot LUID nodes were distinct (`0x000128A6` NPU, `0x000124A5` iGPU). The probe therefore
uses LUID as the first discriminator and observed this actual NPU instance:

```text
pid_12332_luid_0x00000000_0x000128A6_phys_0_eng_0_engtype_Neural = 0.00% [NPU/Neural]
```

The value was idle at probe time and is not a benchmark. The probe also observed GPU engine strings
`3D`, `Compute`, `Copy`, `VideoDecode`, `VideoProcessing`, and `GSC`. One malformed enumerated
instance name could not be parsed safely; it remains in the raw log and is omitted from the map
rather than guessed. `config/telemetry_map.json` declares `counter_source=pdh_gpu_engine` and
`fallback=npu_duty_cycle`.

## Model configuration correction

The supplied `vehicle-license-plate-detection-barrier-0106` shape `[1,3,256,384]` did not match the
pinned IR. Both OpenVINO and the official OMZ model description report `[1,300,300,3]` (NHWC), so the
single configured value was corrected before the final preflight. No URL or model source changed;
`verify_sources.py` was rerun and returned 40 PASS.

## Final preflight table

| Status | Category | Check | Detail |
|---|---|---|---|
| PASS | system | Windows build | 10.0.26200.7840 (requires 26100+) |
| PASS | dependency | openvino | expected 2026.4.0, found 2026.4.0 |
| PASS | dependency | PySide6 | expected 6.11.2, found 6.11.2 |
| PASS | dependency | psutil | expected 7.2.2, found 7.2.2 |
| PASS | dependency | OpenVINO import | runtime reports 2026.4.0-22959-99c81491cc3-releases/2026/4 |
| PASS | dependency | numpy | 2.5.3 |
| PASS | dependency | opencv-python | 5.0.0.93 |
| PASS | dependency | huggingface_hub | 1.32.0 |
| PASS | dependency | pyyaml | 6.0.3 |
| PASS | runtime | Python 3.12 | 3.12.10 |
| PASS | runtime | repo-local venv | C:\dev\engine-lab\.venv\Scripts\python.exe |
| PASS | runtime | VC++ x64 runtime | v14.50.35710.00 |
| PASS | config | models.json | schema_version=1, verified_on=2026-09-23, models=10 |
| PASS | config | platform profiles | 3 profiles; every displayed peak value has a source |
| PASS | runtime | model cache | C:\dev\engine-lab\cache |
| PASS | device | NPU present | SetupAPI=Intel(R) AI Boost; OpenVINO devices=CPU,GPU,NPU |
| PASS | device | GPU present | Intel Arc 140V; OpenVINO devices=CPU,GPU,NPU |
| PASS | driver | NPU driver version | 32.0.100.5540; minimum 32.0.100.5540 |
| PASS | driver | GPU driver version | 32.0.101.6737 |
| PASS | telemetry | map schema | required schema present |
| PASS | telemetry | NPU counters | 1 NPU PDH pattern, source=pdh_gpu_engine |
| PASS | video | retail_aisle | decoded, 720x404, 59.940 FPS |
| PASS | video | traffic | decoded, 768x432, 12.000 FPS |
| PASS | video | street_cars | decoded, 768x432, 12.500 FPS |
| PASS | video | people_queue | decoded, 768x432, 10.000 FPS |
| PASS | video | walking_people | decoded, 768x432, 12.000 FPS |
| PASS | video | pose_subject | decoded, 768x432, 12.000 FPS |
| PASS | model files/IR/compile | all 10 models | files present; static shapes match; NPU/GPU/CPU compile PASS; placement verified |

The machine-readable form of every row, including explicit remediation text, is generated at
`logs/preflight-latest.json` on the demo machine.

## Driver-update recovery runbook

After any future Intel NPU or graphics driver update:

1. Reboot Windows.
2. Remove `C:\dev\engine-lab\cache` (the NPU blob cache is not guaranteed across driver versions).
3. Rerun source verification and `tools/probe_telemetry.py` so LUIDs are reclassified.
4. Rerun `tools/preflight.py` without `--skip-compile`.
5. Rerun `python -m app.main --selftest`.

Phase 0 passed this sequence after the NPU driver update on 2026-09-24.
