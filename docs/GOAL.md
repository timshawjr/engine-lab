# Engine Lab — what this is meant to be (one page)

## The original ask

> "I want an app that is glanceable, that shows the workload like a video and object detection
> happening, and then shows CPU usage and other statistics that would be interesting, and I would
> like to be able to turn on or off the GPU or the NPU to show how the workload increases on the
> CPU." — Intel sales, different workloads per vertical (retail, smart city, medical,
> government/defense), selling Core Ultra Series 2 and 3, on a Windows laptop plugged into a
> monitor at a trade show, with people walking past.

In one sentence: **a stranger walking past must see, within seconds, that the AI work is being
shared across three engines — and watch it move when one is switched off.**

## What a passer-by must be able to say, and when

| by | they should be able to say |
|---|---|
| 5 s | "that's an AI camera doing something, and those three bars are the chips doing the work" |
| 15 s | "the NPU and the GPU are sharing it, the CPU is mostly idle" |
| 30 s | "when he switched the NPU off, the work moved to the GPU and the CPU went up" |

If any of those three fails, the demo has not done its job — regardless of how good the
underlying numbers are.

## "Clearly showing NPU / GPU / CPU usage" means

- **Three bars, one per engine**, each with the engine name and a number readable from two metres.
  At 1080p that is **≥96px for the number** and **≥28px for the bar**. (Those are the original
  spec's numbers; the build currently renders 53-75px numbers and 12px bars, because at 3440×1440
  it picks its "compact" layout.)
- **Colour is the legend.** NPU / GPU / CPU keep the same colour everywhere — gauges, per-stage
  labels, stream tiles — so the eye learns the mapping in one glance.
- **Each bar states its own state in words:** ACTIVE, IDLE, or OFF BY OPERATOR. A switched-off
  engine keeps its last measured number, greyed — never zeroed, because zero claims the engine is
  idle when it is actually switched off.
- **One plain-text line names the split**, e.g. `detector → NPU · classifier → GPU · events → CPU`.
  Three percentages alone never say who is doing what.
- **The toggle has to be dramatic.** Press N and the NPU greys out while the other two visibly
  rise, inside 2 seconds. Today the shift is roughly 13 points on the CPU — real, but weak. The
  cause is arithmetic, not the code: at density 1 with a light model there simply isn't enough
  work per frame to move three bars. **More work per frame (density 2-4, or a heavier model) is
  what makes the three-engine story obvious.**

## "Clearly showing the workload" means

- **The video is the biggest thing on screen** — about 70% of the width — not a small letterboxed
  strip. Today it is capped at 430px tall on this display, so much of the panel is black and the
  action is small.
- **Detection labels are readable from two metres** and say what the model decided, with
  confidence.
- **A strip of the last few classified items** — thumbnail, name, confidence — so a viewer watches
  the app *naming things*, not just drawing boxes. This is the single most persuasive element
  missing today.
- **Each vertical has one obvious on-screen proof:**

| vertical | the proof on screen |
|---|---|
| retail | items named from the shelf/checkout vocabulary, e.g. "Ruffles 91%" |
| smart city | people and vehicles counted per lane, updating live |
| medical | a pose skeleton drawn on a person, with the alert state visible |
| government/defense | a plate read, and people/vehicles tracked at the entry point |

## What is blocking it today

1. **Layout.** At 3440×1440 the app chooses its compact layout, which caps the video panel at 430px
   and shrinks the engine numbers. The scale rule needs to follow the screen the app is actually
   displayed on.
2. **No evidence strip.** Nothing on screen proves a classification happened.
3. **Trust.** A hang (4 runs in 6) and a handful of numbers that can mislead. Fix before any show.

## Definition of done

- A stranger gets the 5-second, 15-second and 30-second readings above, unaided.
- Pressing N and G produces an immediate, obvious change in all three bars.
- The screen names, in words, which engine runs each stage.
- Every vertical shows its proof element.
- No hang, and no number on screen that was not measured.
