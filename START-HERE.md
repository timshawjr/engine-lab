# Engine Lab — booth package

A self-contained copy of the demo for a second machine. No installer, no
administrator rights, no registry changes.

## Run these three, in order

**Prerequisite: Python 3.12** (3.12.x — not 3.11, not 3.13).

Download: https://www.python.org/downloads/release/python-31210/

During install, **tick "Add python.exe to PATH"**. That is the only thing you need
to install by hand; everything else comes from the scripts below. No
administrator rights are required.

The version matters: the application checks it at startup and the pinned OpenVINO
and PySide6 wheels are built for 3.12. `setup-booth-machine.bat` refuses to run on
any other version and tells you so, rather than failing later with a confusing
package error.

| Step | File | What it does | Time |
|---|---|---|---|
| 1 | `setup-booth-machine.bat` | Creates the pinned venv, installs dependencies | ~5 min (needs network) |
| 2 | `download-models.bat` | Fetches the ~2.3 GB of model files | ~10-20 min (needs network) |
| 3 | `verify-booth.bat` | Proves the machine is booth-ready | ~2 min |

Then start the demo:

```
run_demo.bat
```

Press **1**–**6** for scenarios, **R** for document Q&A, **Q** to quit.

**After step 2 the demo never touches the network.** Models, video and the
document index are all on disk.

## What is and is not in here

**Included** — the application, the six video scenarios, all config, the video
files, and the pre-built document index under `models\rag\` (3 MB).

**Not included, by design:**

- `.venv\` — built on your machine in step 1, so the dependency set matches that
  machine's Python exactly.
- `models\` (2.3 GB) — fetched in step 2. Deliberately not copied between
  machines.
- `cache\` — the OpenVINO compiled-blob cache is **driver specific**. Shipping a
  copy risks a stale blob cache, which fails confusingly. This machine builds its
  own on first run.

The document index *is* included, because building it needs the source PDF and a
PDF parser that the runtime does not have. Copying the 3 MB result avoids both.

## Before the show

Run `verify-booth.bat` and read the output. You want:

- preflight: **0 WARN, 0 FAIL**
- self-test: **0 FAIL**

A FAIL is a problem to fix at the venue, not in front of a customer.

## One honest caveat

Every number in `bench_report.md` and the README — engine utilisation, detection
rates, the ~1.2 s document answers — was measured on the development machine:
Core Ultra9 288V, NPU driver 32.0.100.5540, GPU driver 32.0.101.6737.

If `verify-booth.bat` reports different driver versions, **re-measure before
quoting those numbers to a customer.** The same silicon does not guarantee the
same driver, and an Intel driver update invalidates the NPU blob cache (run
`tools\reset_cache.ps1`, or delete `cache\`).

## What the document Q&A can and cannot do

It answers questions about **NIST SP 800-82r4** from passages retrieved on the
machine, and always shows the passages it used.

It is **not** a general assistant: it answers only from that one document, and it
says so when the passages do not support an answer. Generation runs on the
**GPU** — the LLM does not compile on the NPU — and the panel is labelled GPU
because that is what was measured.