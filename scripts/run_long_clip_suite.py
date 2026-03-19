from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tennis_tracker.events import summarize_match_events
from tennis_tracker.pipeline import run_pipeline


LONG_CLIPS = [
    "broadcast_10s_opening.mp4",
    "broadcast_12s_pressure.mp4",
    "broadcast_15s_exchange.mp4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full pipeline on the longer broadcast clips.")
    parser.add_argument(
        "--weights",
        default=str(ROOT / "models" / "tracknet_weights.pth"),
        help="Path to the TrackNet weights file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "out" / "long_suite"),
        help="Directory for per-clip outputs and the summary JSON.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for faster smoke runs.",
    )
    parser.add_argument(
        "--player-stride",
        type=int,
        default=2,
        help="Run person detection every N frames and carry short gaps temporally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, object]] = {}
    for clip_name in LONG_CLIPS:
        video_path = ROOT / "data" / "samples" / clip_name
        clip_slug = video_path.stem
        clip_output_dir = output_dir / clip_slug

        start_time = time.perf_counter()
        outputs = run_pipeline(
            video_path=video_path,
            output_dir=clip_output_dir,
            step="all",
            weights_path=args.weights,
            max_frames=args.max_frames,
            player_detect_stride=args.player_stride,
            write_video=False,
        )
        elapsed_sec = time.perf_counter() - start_time
        event_summary = summarize_match_events(outputs["events"])
        summary[clip_slug] = {
            "video": str(video_path),
            "elapsed_sec": round(elapsed_sec, 2),
            "events_json": str(outputs["events"]),
            "tracking_json": str(outputs["json"]),
            **event_summary,
        }
        print(f"{clip_slug}: {summary[clip_slug]}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
