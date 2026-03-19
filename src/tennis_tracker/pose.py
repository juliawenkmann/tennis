from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from .runtime import detect_runtime


def extract_player_poses(
    *,
    video_path: str | Path,
    tracking_json_path: str | Path,
    output_json_path: str | Path,
    model_name: str | Path = "yolo11n-pose.pt",
    device: str | None = None,
    imgsz: int = 256,
    batch_size: int = 16,
) -> Path:
    video_path = Path(video_path)
    tracking_json_path = Path(tracking_json_path)
    output_json_path = Path(output_json_path)

    tracking_data = json.loads(tracking_json_path.read_text())
    frames = tracking_data["frames"]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for pose extraction: {video_path}")

    runtime = detect_runtime()
    model = YOLO(str(model_name))
    device = device or runtime.ultralytics_device
    pose_frames = [
        {"frame_index": frame["frame_index"], "players": {}}
        for frame in frames
    ]
    pose_lookup = {
        int(frame_payload["frame_index"]): frame_payload
        for frame_payload in pose_frames
    }

    pending: list[dict[str, Any]] = []
    for frame_data in frames:
        ok, frame = capture.read()
        if not ok:
            break

        for player in frame_data.get("players", []):
            crop_info = crop_player(frame, player["bbox_xyxy"])
            if crop_info is None:
                continue
            pending.append(
                {
                    "frame_index": frame_data["frame_index"],
                    "label": player["label"],
                    **crop_info,
                }
            )

        if len(pending) >= batch_size:
            run_pose_batch(
                model=model,
                pending=pending,
                pose_lookup=pose_lookup,
                device=device,
                imgsz=imgsz,
            )
            pending = []

    capture.release()

    if pending:
        run_pose_batch(
            model=model,
            pending=pending,
            pose_lookup=pose_lookup,
            device=device,
            imgsz=imgsz,
        )

    output_json_path.write_text(
        json.dumps(
            {
                "video_path": str(video_path),
                "model_name": str(model_name),
                "device": device,
                "imgsz": imgsz,
                "frames": pose_frames,
            },
            indent=2,
        )
    )
    return output_json_path


def run_pose_batch(
    *,
    model: YOLO,
    pending: list[dict[str, Any]],
    pose_lookup: dict[int, dict[str, Any]],
    device: str,
    imgsz: int,
) -> None:
    crops = [item["crop"] for item in pending]
    results = model.predict(
        source=crops,
        device=device,
        verbose=False,
        imgsz=imgsz,
        conf=0.2,
    )

    for item, result in zip(pending, results):
        pose = best_pose_from_result(result, item["offset_x"], item["offset_y"])
        if pose is None:
            continue
        pose_lookup[int(item["frame_index"])]["players"][item["label"]] = pose


def crop_player(frame: np.ndarray, bbox_xyxy: list[float]) -> dict[str, Any] | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]

    pad_x = max(int(round((x2 - x1) * 0.15)), 8)
    pad_y_top = max(int(round((y2 - y1) * 0.18)), 10)
    pad_y_bottom = max(int(round((y2 - y1) * 0.08)), 6)

    crop_x1 = max(int(round(x1)) - pad_x, 0)
    crop_y1 = max(int(round(y1)) - pad_y_top, 0)
    crop_x2 = min(int(round(x2)) + pad_x, width - 1)
    crop_y2 = min(int(round(y2)) + pad_y_bottom, height - 1)

    if crop_x2 - crop_x1 < 24 or crop_y2 - crop_y1 < 48:
        return None

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    return {
        "crop": crop,
        "offset_x": crop_x1,
        "offset_y": crop_y1,
    }


def best_pose_from_result(result: Any, offset_x: int, offset_y: int) -> dict[str, Any] | None:
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return None

    confidences = result.boxes.conf.cpu().numpy()
    best_index = int(np.argmax(confidences))
    keypoints_xy = result.keypoints.xy[best_index].cpu().numpy()
    if result.keypoints.conf is not None:
        keypoints_conf = result.keypoints.conf[best_index].cpu().numpy()
    else:
        keypoints_conf = np.ones(len(keypoints_xy), dtype=np.float32)

    keypoints: list[list[float]] = []
    for (x, y), conf in zip(keypoints_xy, keypoints_conf):
        keypoints.append(
            [
                round(float(x) + offset_x, 2),
                round(float(y) + offset_y, 2),
                round(float(conf), 4),
            ]
        )

    return {
        "keypoints": keypoints,
        "box_confidence": round(float(confidences[best_index]), 4),
    }
