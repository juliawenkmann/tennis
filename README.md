# Tennis Tracking Baseline

This is a minimal end-to-end baseline for tennis video analysis.

It produces:

- court coordinates via a simple court detector plus a hold-last-good homography tracker
- player coordinates via YOLO person detection plus light temporal stabilization
- ball coordinates via a TrackNet-style model plus trajectory filtering
- canonical rally / serve / hit / bounce event extraction in `match_events.json`
- XML export in a minimal `match -> point -> event` structure

The code is intentionally small and optimized for testing the full pipeline, not for production robustness.
It now also includes device-aware model selection, so it can automatically use CUDA or MPS when those are available.

On CPU, `ball` and `all` are the slowest commands. On the provided 2-second sample clip, expect roughly 1 to 2 minutes.

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 scripts/download_assets.py
```

That downloads:

- a short broadcast clip for the main demo
- a longer broadcast clip for inspection
- several additional broadcast clips for quick testing in the notebook dropdown
- a few longer broadcast clips for stress-testing the event extraction layer
- an optional extra court-level clip
- TrackNet weights for the ball detector

## Run Each Step

Court only:

```bash
python3 scripts/run_step.py --step court --video data/samples/broadcast_2s.mp4
```

Players only:

```bash
python3 scripts/run_step.py --step players --video data/samples/broadcast_2s.mp4
```

Ball only:

```bash
python3 scripts/run_step.py --step ball --video data/samples/broadcast_2s.mp4
```

Everything:

```bash
python3 scripts/run_step.py --step all --video data/samples/broadcast_2s.mp4
```

By default, person detection runs every 2 frames and the tracker carries the players through the gap frames.
Use `--player-stride 1` if you want full per-frame person detection.

Longer clip stress run:

```bash
python3 scripts/run_long_clip_suite.py
```

That suite runs the full tracking and event extraction on the longer samples and writes a summary JSON without spending time on overlay video rendering.

Outputs are written to `out/` as `court.mp4`, `players.mp4`, `ball.mp4`, `all.mp4` with matching `.json` and `.xml` files.
For `--step all`, the pipeline also writes `all_events.json`, which is the canonical event-layer artifact.

The notebook demo lives at the repo root in `tennis-tracking-overlay-demo.ipynb`.
It now renders the full selected overlay video directly in the notebook from a fast path that prefers CUDA, then MPS, can also target videos under `data/raw/`, uses per-frame player detection for a more stable near-player box, and suppresses non-play camera shots that do not show a valid two-player court view.

The bird's-eye notebook lives at the repo root in `tennis-birdseye-demo.ipynb`.
It uses the same tracked court, player, and ball data, adds cached pose-based player skeletons, and redraws the result on a normalized top-down court with an event-aware ball shadow instead of raw airborne homography.

The event notebook lives at the repo root in `tennis-events-demo.ipynb`.
It renders event-annotated source and bird's-eye videos side by side from the canonical `match_events.json`.

The benchmark notebook lives at the repo root in `tennis-event-benchmark.ipynb`.
It lets you review a clip, edit benchmark labels, and score the current predictions against those labels.

The XML notebook lives at the repo root in `tennis-xml-demo.ipynb`.
It converts the canonical event result into a frame-and-rally XML structure: `match -> court / frame / rally`.

## What The Baseline Assumes

- The full court is visible.
- The broadcast angle is relatively stable.
- There are two main players on court.
- Ball tracking uses a pretrained TrackNet-style model and is the slowest step on CPU.

## Important Limitation

The court homography gives 2D ground-plane coordinates. That means player coordinates are meaningful on the court plane, but airborne ball positions are only projected onto the court, not recovered in true 3D.

## Event Layer

The event model is described in `EVENT_SCHEMA.md`.
The intent is to keep a clean separation between:

- frame-level tracking
- event extraction
- XML export
- post-game analysis

For manual evaluation and tuning, benchmark labels live under `data/benchmark/labels/`.

## Review Note

The current project review and improvement note is in `PROJECT_REVIEW.md`.
