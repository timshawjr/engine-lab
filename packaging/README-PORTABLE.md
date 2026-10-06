# Engine Lab — self-contained booth demo

**Nothing to install. No Python required. No network required.**

## Run it

1. Copy this whole folder to the booth machine.
2. Double-click **`Start Engine Lab.bat`**.

That's it.

Before the show, double-click **`Verify Booth.bat`** once and check that it says
**"0 WARN, 0 FAIL"** and **"0 FAIL"**. It takes about two minutes and needs no
network.

## Controls

| Key | Action |
|---|---|
| `1`–`6` | Pick a scenario (retail, metro, manufacturing, education, health, federal) |
| `R` | Open the document Q&A panel |
| `Esc` or the **Close** button | Put the Q&A panel away |
| `N` / `G` | Toggle the NPU / GPU |
| `C` | Cycle device mode |
| `+` / `-` | Change density (1, 2, 4, 6, 8) |
| `A` | Attract mode |
| `F1` | Operator overlay (devices, `EXECUTION_DEVICES`, fallbacks) |
| `F11` | Fullscreen |
| `Q` | Quit |

## Why there is nothing to install

This folder carries its own Python 3.12 and its own pinned packages
(`openvino 2026.4.0`, `openvino-genai 2026.4.0.0`, PySide6), so:

- it does **not** matter whether Python is installed on this machine;
- it does **not** matter what version is on `PATH`;
- it does **not** need administrator rights, an installer, or the registry.

The models (2.3 GB) and the video files are inside the folder, so the demo runs
completely offline. The document Q&A answers from a copy of NIST SP 800-82r4
built into the bundle.

**You can move or rename the folder.** Every path is relative to the folder
itself, so extracting somewhere different is fine.

## First launch is slow — this is expected

The first double-click takes a few seconds longer while the models load. The
**first press of `R`** also pauses for several seconds while the document Q&A
loads its two pipelines. Neither is a fault. If you are demoing, press `R` once
before visitors arrive.

## What is NOT in here

- `cache/` — the OpenVINO compiled-blob cache. It is **specific to the Intel
  driver version** on the machine that built it, so shipping a copy can fail
  confusingly. Each machine builds its own on first run. That is why the first
  launch is slower than the second.
- Any install of Python, or any package from the internet.

## One honest caveat

Every figure in `bench_report.md` and `SALES-GUIDE.md` — engine utilisation,
detection rates, the ~1.3 s document answers — was measured on the development
machine:

> Core Ultra 9 288V, NPU driver **32.0.100.5540**, GPU driver **32.0.101.6737**

`Verify Booth.bat` prints this machine's driver versions. **If they differ,
re-measure before quoting those numbers to a customer.** Same silicon does not
guarantee the same driver, and an Intel driver update invalidates the NPU blob
cache.

## What the document Q&A can and cannot do

It answers questions about **NIST SP 800-82r4** from passages retrieved on this
machine, and always shows the passages it used so a visitor can judge the answer.

It is **not** a general assistant: it answers only from that one document, and it
says so when the passages do not support an answer. Generation runs on the
**GPU** — the language model does not compile on the NPU — and the panel is
labelled GPU because that is what was measured.

## If something looks wrong

| Symptom | Cause | Fix |
|---|---|---|
| `R` does nothing | An older copy of the folder | Get the current folder again |
| First press of `R` pauses a few seconds | The Q&A pipelines loading | Expected; warm it up before the show |
| Answer takes ~2.5 s with the video running | Video and Q&A share the GPU | Expected; measured |
| "NPU worked yesterday, not today" | An Intel driver update invalidated the blob cache | Delete the `cache` folder next to this one and relaunch |
| Preflight reports a model FAIL | The folder was not copied in full | Copy it again, completely |

**Never** work around a FAIL by disabling a check. The verification exists so the
problem is found before a visitor is standing there.
