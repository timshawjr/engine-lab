# Engine Lab — sales guide

What this demo is, what to say about it, and what to do when a customer asks
the awkward questions. Every number here was measured on the demo machine
(Core Ultra9 288V, NPU driver 32.0.100.5540, GPU driver 32.0.101.6737) and is
reproducible with `verify-booth.bat`.

**Rule for the booth: never quote a number you have not seen on this machine.**
`bench_report.md` is the full record, including the things that did not work.

---

## The one-line pitch

> Everything you are about to see runs on this laptop — detection, classification,
> counting and document question answering. Nothing leaves the machine, and it
> works with the network unplugged.

That last clause is the one that wins deals. Most "AI in the edge" claims mean a
GPU server behind a thin client. This is one SoC.

---

## What is actually running

| | |
|---|---|
| **SoC** | Intel Core Ultra 9 288V — NPU + Arc iGPU + CPU on one package |
| **Scenario video** | Six verticals: retail, metro, manufacturing, education, health, federal |
| **Detection** | YOLO on the **NPU** for one stream, **GPU** for the other (round-robin) |
| **Classification** | CLIP zero-shot on the **GPU**, per crop |
| **Counting / zones** | **CPU** — deliberate, because it is where headroom exists |
| **Document Q&A** | GenAI on the **GPU**: embeddings + Qwen3-1.7B |

The placement bar under the header shows where each stage actually landed. Point
at it — it is the honest bit of the demo.

---

## Running it

```
run_demo.bat          start the demo
1 .. 6                pick a scenario
R                     document Q&A panel
N / G / C             toggle NPU / GPU / cycle device policy
+ / -                 density 1, 2, 4, 6, 8
A                     attract mode
F1                    operator overlay (devices, EXECUTION_DEVICES, fallbacks)
F11                   fullscreen
Q                     quit
```

On a **second machine**: `setup-booth-machine.bat` → `download-models.bat` →
`verify-booth.bat`. See `START-HERE.md`.

---

## Reading the three header numbers

Under the scenario title are three live readouts. They answer three different
questions, in order, and that order is the pitch:

| Readout | Question it answers | What it actually counts |
|---|---|---|
| **FPS** | Is it *live*? | Frames per second the pipeline is processing |
| **DET/s** | Is it doing *real work*? | Detection boxes per second, summed across all active streams |
| **EVENTS** | Does it produce a *business answer*? | Tracked business events in a rolling 60-second window |

### Why DET/s matters

**It is the volume of model work, not a quality score.** One frame with six
people on screen produces six detections; that is six, not one. So DET/s rises
with how busy the scene is and with how many streams run.

It is the number that shows the silicon is genuinely working. A demo that shows
only FPS can be faking it with a video player — 24 FPS proves frames are moving,
not that anything is being inferred. DET/s proves the models ran on those frames.

On the demo (manufacturing, 2 streams): **204–260 DET/s** while FPS holds at 24.
That gap is the point — 24 frames each carrying several objects.

### Why EVENTS matters

**It is the number a customer actually buys.** DET/s is raw model output: every
box, every frame, noisy and repetitive. A person standing in shot for ten seconds
produces *hundreds* of detections and exactly **one** event, because events are:

- **tracked** — the same person across frames is one identity, not 240 detections;
- **zone-gated** — only counts when they enter a defined area;
- **cooldown-suppressed** — one person walking through does not fire repeatedly.

The ratio between the two is the story: **hundreds of detections per second
collapse to dozens of events.** That collapse is the application doing its job,
turning pixels into something an operator can act on. A camera that reports
"12,000 detections" is a science project; one that reports "3 people entered the
restricted zone" is a product.

Events are counted on the **CPU**, deliberately, because that is where headroom
exists — the NPU and GPU are busy inferring.

### The event types on screen

Six, across the scenarios:

| Event | Meaning | Scenarios |
|---|---|---|
| `person_counted` | A person entered a defined zone | retail, metro, manufacturing, education |
| `vehicle_counted` | A vehicle entered a defined zone | metro |
| `object_classified` | A crop was classified (e.g. `ppe worn`, `no ppe`, a product) | retail, manufacturing |
| `object_picked_up` | A tracked object left a shelf | retail |
| `plate_detected` | A licence plate was read | federal |
| `posture_alert` | A fall or unsafe posture was detected | education, health |

Manufacturing on the demo machine shows **54–55 events** in the rolling window —
mostly `person_counted` plus PPE classifications.

### The line to use

> FPS tells you it's live, detections per second tell you the models are really
> running, and events tell you what you'd actually act on. Watch the middle number
> be large and the last one be small — that's the tracking filter doing its job.

## Measured numbers you can quote

Pipeline, manufacturing at density 2 (from `bench_report.md`):

| Metric | Value |
|---|---|
| Detection rate | 204–239 detections/s |
| Inferences/s | 153–180 |
| NPU utilisation | 21–25% |
| GPU utilisation | 34–57% |
| CPU utilisation | 82–90% |
| End-to-end latency p50 | 36–39 ms |

Document Q&A (NIST SP 800-82r4, 321 pages):

| Metric | Value |
|---|---|
| Retrieval | 27–112 ms |
| Generation | ~1,180 ms |
| Total | ~1.3 s |
| Corpus | 595 chunks, 1024-dim embeddings |

**State the machine.** "204 detections/s on a Core Ultra 288V" is a claim.
"204 detections/s" alone is not.

---

## The questions you will get

### "Is that really on the NPU?"

Yes, for one stream, and the placement bar proves it per stage. Say this
precisely:

> Detector 0 is on the NPU, detector 1 on the GPU — that's the round-robin policy
> on screen. Classification is on the GPU and zone counting is on the CPU. Press
> F1 to see the `EXECUTION_DEVICES` the runtime reported for each.

**Do not claim the NPU does everything.** It does not, and the demo shows that
honestly. A customer who checks will respect the accuracy far more than a bigger
claim.

### "How fast does the NPU go?"

NPU peak is **48 TOPS**, GPU peak **67 TOPS** (Intel published). Those are
labelled `peak` on screen with a source. If asked what utilisation the demo
reaches: NPU sits at 21–25% — because at density 1 there simply is not enough
work. **Density is the lever, not migrating CPU work.** Press `+` and watch.

### "Does the document Q&A slow the video down?"

**Honest answer: it does not measurably, and I will not claim otherwise.**
Throughput with Q&A running came out 7% *higher* than without, but the run-to-run
spread was nearly three times that figure. It is noise. What is true:

- the GPU gauge rises about 2 points while the LLM generates;
- the pipeline's own FPS and DET/s readouts stay flat;
- the answer takes ~1.2 s on its own and ~2.5 s with the video running.

If a customer pushes: *"the LLM is on the GPU and so is one detector, so they
share it — that's why the answer takes longer with the video on."* That is
measured and defensible.

### "What happens if I unplug the network?"

Nothing. No network call at runtime. Model files, video and the document index
are all on disk. This is the strongest demo move available — do it live if the
booth allows.

### "What does it do when it doesn't know?"

It shows the passages it used, every time, so the customer can check the answer
against the source. And it declines when the passages do not support one
("NOT IN PROVIDED PAGES"). Two honest limits to know:

- it answers **only** from NIST SP 800-82r4 — ask it something else and it will
  decline or answer thinly;
- occasionally it will phrase a refusal as an ordinary sentence rather than using
  the marker. That is why the passages are always visible.

### "Which model is the LLM?"

Qwen3-1.7B (INT4) for answers, Qwen3-Embedding-0.6B (INT8) for retrieval, both
OpenVINO GenAI exports, both running locally.

---

## Things that did not work — do not demo these

Recorded here because a customer may ask, and because saying "we tried it" beats
being caught out.

| Attempted | Result |
|---|---|
| **Qwen3 on the NPU** | Does not compile (vpux compiler pass failure). The Q&A is GPU-only. |
| **NPU embeddings** | Compiles but **1,170 ms/query vs 16 ms on GPU** — 73× slower, and left padding returns all-zero vectors. |
| **A reranker** (Qwen3-Reranker-0.6B) | Tested and **rejected**: citation accuracy fell from 4/5 to 2/5 and it promoted abbreviation lists over real answers. |
| **Robotics scenario** | Built, measured, then removed — the footage could not be detected honestly. |
| **CLS pooling** (GenAI's default) | Silently returns identical vectors for every question. We pin `LAST_TOKEN`. |

If someone proposes any of these, the measurements are in `bench_report.md`.

---

## If something breaks at the booth

| Symptom | Cause | Fix |
|---|---|---|
| `R` does nothing | Running an old checkout — `main` before the merge | `git pull` |
| First press of `R` waits ~7 s | GenAI pipelines load on first use | Expected; warm with `R` before visitors arrive |
| Answer takes ~2.5 s | Video and LLM share the GPU | Expected; measured |
| Preflight FAIL on a model | Partial download (0-byte `.incomplete`) | Re-run `download-models.bat` |
| NPU "worked yesterday, not today" | Driver update invalidated the blob cache | Delete `cache\` and re-run `verify-booth.bat` |

**Never** work around a FAIL by weakening a check. The preflight table exists so
the problem is found at the venue rather than in front of a customer.

---

## The 30-second version

> One laptop, one SoC. Six live video scenarios — detection on the NPU,
> classification on the GPU, counting on the CPU, all visible on screen — plus
> document question answering over a 321-page federal guide, also local. Ask it
> something and it shows you the page it read. Unplug the network if you like.