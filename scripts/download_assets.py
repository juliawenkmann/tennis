from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_SAMPLES = ROOT / "data" / "samples"
MODELS = ROOT / "models"

BROADCAST_URL = "https://www.youtube.com/watch?v=EDHq5pQBt84"
COURT_LEVEL_URL = "https://www.youtube.com/watch?v=0625iyVoKm8"
TRACKNET_WEIGHTS_URL = "https://raw.githubusercontent.com/hgupt3/TRACE/main/TrackNet/Weights.pth"

BROADCAST_SAMPLE_SPECS = [
    ("broadcast_2s.mp4", "00:00:30", 2.0),
    ("broadcast_3s_serve.mp4", "00:00:09", 3.0),
    ("broadcast_4s_rally.mp4", "00:00:46", 4.0),
    ("broadcast_5s_baseline.mp4", "00:00:58", 5.0),
    ("broadcast_6s_serve_plus_one.mp4", "00:01:28", 6.0),
    ("broadcast_8s.mp4", "00:00:27", 8.0),
    ("broadcast_10s_opening.mp4", "00:00:07", 10.0),
    ("broadcast_12s_pressure.mp4", "00:02:54", 12.0),
    ("broadcast_15s_exchange.mp4", "00:03:24", 15.0),
    ("broadcast_5s_pressure.mp4", "00:02:58", 5.0),
    ("broadcast_6s_exchange.mp4", "00:03:28", 6.0),
    ("broadcast_5s_tiebreak.mp4", "00:04:28", 5.0),
]


def download_file(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def download_youtube(url: str, destination_stem: Path) -> Path:
    matches = list(destination_stem.parent.glob(f"{destination_stem.name}.*"))
    if matches:
        return matches[0]

    destination_stem.parent.mkdir(parents=True, exist_ok=True)
    command = [
        shutil.which("yt-dlp") or "yt-dlp",
        "-f",
        "b[height<=480]/bv*[height<=480]+ba/b",
        "-o",
        str(destination_stem) + ".%(ext)s",
        url,
    ]
    subprocess.run(command, check=True)
    matches = list(destination_stem.parent.glob(f"{destination_stem.name}.*"))
    if not matches:
        raise RuntimeError(f"yt-dlp finished but no file was created for {url}")
    return matches[0]


def trim_clip(source: Path, destination: Path, start_time: str, duration_seconds: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-y",
        "-ss",
        start_time,
        "-i",
        str(source),
        "-t",
        str(duration_seconds),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(destination),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download tennis sample videos and model weights.")
    parser.add_argument(
        "--skip-court-level",
        action="store_true",
        help="Only download the broadcast sample and the ball weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    broadcast_video = download_youtube(BROADCAST_URL, DATA_RAW / "broadcast_full")
    for filename, start_time, duration_seconds in BROADCAST_SAMPLE_SPECS:
        trim_clip(
            broadcast_video,
            DATA_SAMPLES / filename,
            start_time=start_time,
            duration_seconds=duration_seconds,
        )

    if not args.skip_court_level:
        court_level_video = download_youtube(COURT_LEVEL_URL, DATA_RAW / "court_level_full")
        trim_clip(court_level_video, DATA_SAMPLES / "court_level_3s.mp4", start_time="00:00:18", duration_seconds=3.0)

    download_file(TRACKNET_WEIGHTS_URL, MODELS / "tracknet_weights.pth")

    print("Downloaded assets:")
    print(f"- {broadcast_video}")
    for filename, _, _ in BROADCAST_SAMPLE_SPECS:
        print(f"- {DATA_SAMPLES / filename}")
    if not args.skip_court_level:
        print(f"- {DATA_SAMPLES / 'court_level_3s.mp4'}")
    print(f"- {MODELS / 'tracknet_weights.pth'}")


if __name__ == "__main__":
    main()
