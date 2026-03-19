from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tennis_tracker.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one step of the tennis tracking baseline.")
    parser.add_argument(
        "--step",
        required=True,
        choices=["court", "players", "ball", "all"],
        help="Which step to run.",
    )
    parser.add_argument(
        "--video",
        default=str(ROOT / "data" / "samples" / "broadcast_2s.mp4"),
        help="Path to the input video.",
    )
    parser.add_argument(
        "--weights",
        default=str(ROOT / "models" / "tracknet_weights.pth"),
        help="Path to the TrackNet weights file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "out"),
        help="Directory where videos and JSON files will be written.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for faster experiments.",
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
    outputs = run_pipeline(
        video_path=args.video,
        output_dir=Path(args.output_dir),
        step=args.step,
        weights_path=args.weights,
        max_frames=args.max_frames,
        player_detect_stride=args.player_stride,
    )
    print(f"Video: {outputs['video']}")
    print(f"JSON:  {outputs['json']}")
    print(f"XML:   {outputs['xml']}")
    if "events" in outputs:
        print(f"Events:{outputs['events']}")


if __name__ == "__main__":
    main()
