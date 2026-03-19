from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .pipeline import COURT_DRAW_LINES, CourtReference, finalize_video_for_playback


COCO_SKELETON_EDGES = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def render_birdseye_video(
    *,
    tracking_json_path: str | Path,
    output_video_path: str | Path,
    pose_json_path: str | Path | None = None,
    events_json_path: str | Path | None = None,
    pixels_per_meter: int = 34,
    margin_px: int = 48,
    player_style: str = "auto",
    ball_strategy: str = "raw",
) -> Path:
    tracking_json_path = Path(tracking_json_path)
    output_video_path = Path(output_video_path)
    temp_video_path = output_video_path.with_suffix(".opencv.mp4")

    data = json.loads(tracking_json_path.read_text())
    reference = CourtReference()
    resolved_player_style = resolve_player_style(player_style, pose_json_path)
    pose_frames = load_pose_frames(pose_json_path) if resolved_player_style == "skeleton" else {}
    events_data = load_events_data(events_json_path)
    ball_positions = resolve_ball_positions(
        tracking_data=data,
        events_data=events_data,
        reference=reference,
        strategy=ball_strategy,
    )
    canvas_width, canvas_height = canvas_size(reference, pixels_per_meter, margin_px)

    writer = cv2.VideoWriter(
        str(temp_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(data["fps"]),
        (canvas_width, canvas_height),
    )

    ball_history: list[np.ndarray] = []
    for frame, ball_xy in zip(data["frames"], ball_positions):
        frame_pose_data = pose_frames.get(frame["frame_index"], {})
        if ball_xy is None:
            ball_history = []
        else:
            ball_history.append(np.array(ball_xy, dtype=np.float32))
            ball_history = ball_history[-8:]

        birdseye_frame = draw_birdseye_frame(
            frame_data=frame,
            reference=reference,
            pixels_per_meter=pixels_per_meter,
            margin_px=margin_px,
            ball_history=ball_history,
            frame_pose_data=frame_pose_data,
            player_style=resolved_player_style,
            ball_xy=ball_xy,
        )
        writer.write(birdseye_frame)

    writer.release()
    finalize_video_for_playback(temp_video_path, output_video_path)
    return output_video_path


def draw_birdseye_frame(
    *,
    frame_data: dict,
    reference: CourtReference,
    pixels_per_meter: int,
    margin_px: int,
    ball_history: list[np.ndarray],
    frame_pose_data: dict[str, dict],
    player_style: str,
    ball_xy: list[float] | np.ndarray | None,
) -> np.ndarray:
    canvas_width, canvas_height = canvas_size(reference, pixels_per_meter, margin_px)
    frame = np.full((canvas_height, canvas_width, 3), (38, 92, 64), dtype=np.uint8)

    outer_corners = np.array(
        [
            court_to_canvas(point, pixels_per_meter, margin_px)
            for point in reference.outer_corners
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, outer_corners, (145, 82, 37))
    cv2.polylines(frame, [outer_corners], True, (245, 245, 245), 2)

    for start_name, end_name in COURT_DRAW_LINES:
        start = court_to_canvas(reference.keypoints[start_name], pixels_per_meter, margin_px)
        end = court_to_canvas(reference.keypoints[end_name], pixels_per_meter, margin_px)
        cv2.line(frame, start, end, (245, 245, 245), 2)

    for index, point in enumerate(ball_history, start=1):
        px, py = court_to_canvas(point, pixels_per_meter, margin_px)
        radius = 2 + min(index, 4)
        color = (40, 145 + min(index * 12, 80), 255)
        cv2.circle(frame, (px, py), radius, color, -1)

    for player in frame_data.get("players", []):
        court_xy = np.array(player["court_xy_m"], dtype=np.float32)
        px, py = court_to_canvas(court_xy, pixels_per_meter, margin_px)
        color = (64, 113, 255) if player["label"] == "near_player" else (90, 230, 120)
        pose_data = frame_pose_data.get(player["label"])
        if player_style == "skeleton" and pose_data is not None:
            draw_player_skeleton(
                frame=frame,
                pose_data=pose_data,
                player_label=player["label"],
                anchor_court_xy=court_xy,
                pixels_per_meter=pixels_per_meter,
                margin_px=margin_px,
                color=color,
            )
        else:
            cv2.circle(frame, (px, py), 9, color, -1)
            cv2.circle(frame, (px, py), 12, (255, 255, 255), 1)
        label_y = py + 26 if player["label"] == "far_player" else py - 10
        label_y = max(label_y, 22)
        cv2.putText(
            frame,
            player["label"].replace("_", " "),
            (px + 10, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (250, 250, 250),
            1,
        )

    if ball_xy is not None:
        px, py = court_to_canvas(np.array(ball_xy, dtype=np.float32), pixels_per_meter, margin_px)
        cv2.circle(frame, (px, py), 6, (0, 190, 255), -1)
        cv2.circle(frame, (px, py), 10, (255, 255, 255), 1)

    return frame


def resolve_player_style(player_style: str, pose_json_path: str | Path | None) -> str:
    if player_style == "auto":
        return "skeleton" if pose_json_path is not None else "dot"
    if player_style not in {"dot", "skeleton"}:
        raise ValueError(f"Unsupported player_style: {player_style}")
    return player_style


def load_events_data(events_json_path: str | Path | None) -> dict[str, Any] | None:
    if events_json_path is None:
        return None
    events_json_path = Path(events_json_path)
    if not events_json_path.exists():
        return None
    return json.loads(events_json_path.read_text())


def resolve_ball_positions(
    *,
    tracking_data: dict[str, Any],
    events_data: dict[str, Any] | None,
    reference: CourtReference,
    strategy: str,
) -> list[list[float] | None]:
    raw_positions = [
        rounded_ball_point(sanitize_court_point(frame.get("ball", {}).get("court_xy_m"), reference))
        for frame in tracking_data.get("frames", [])
    ]
    if strategy != "event_aware" or events_data is None:
        return raw_positions

    return infer_event_aware_ball_positions(
        tracking_data=tracking_data,
        events_data=events_data,
        reference=reference,
        raw_positions=raw_positions,
    )


def infer_event_aware_ball_positions(
    *,
    tracking_data: dict[str, Any],
    events_data: dict[str, Any],
    reference: CourtReference,
    raw_positions: list[list[float] | None],
) -> list[list[float] | None]:
    frame_count = len(tracking_data.get("frames", []))
    derived_positions: list[np.ndarray | None] = [None] * frame_count
    anchored_rally_frames = np.zeros(frame_count, dtype=bool)

    for rally in events_data.get("rallies", []):
        anchors = build_rally_ball_anchors(rally=rally, reference=reference)
        if len(anchors) < 2:
            continue
        rally_start = max(int(rally.get("start_frame", 0)), 0)
        rally_end = min(int(rally.get("end_frame", frame_count - 1)), frame_count - 1)
        anchored_rally_frames[rally_start : rally_end + 1] = True
        for start_anchor, end_anchor in zip(anchors, anchors[1:]):
            infer_anchor_segment_positions(
                derived_positions=derived_positions,
                start_anchor=start_anchor,
                end_anchor=end_anchor,
                frame_count=frame_count,
                raw_positions=raw_positions,
            )

    resolved_positions: list[list[float] | None] = []
    for index, raw_point in enumerate(raw_positions):
        derived = derived_positions[index]
        if derived is not None:
            resolved_positions.append(rounded_ball_point(derived))
            continue
        if anchored_rally_frames[index]:
            resolved_positions.append(None)
            continue
        resolved_positions.append(raw_point)
    return resolved_positions


def build_rally_ball_anchors(
    *,
    rally: dict[str, Any],
    reference: CourtReference,
) -> list[dict[str, Any]]:
    action_events = [
        event
        for event in rally.get("events", [])
        if event.get("type") in {"serve", "hit", "bounce"}
    ]
    anchors: list[dict[str, Any]] = []
    for index, event in enumerate(action_events):
        next_event = action_events[index + 1] if index + 1 < len(action_events) else None
        point = event_anchor_point(
            event=event,
            next_event=next_event,
            reference=reference,
        )
        if point is None:
            continue
        anchor = {
            "frame": int(event["frame"]),
            "point": point,
            "type": str(event["type"]),
        }
        if anchors and anchors[-1]["frame"] == anchor["frame"]:
            if anchor_priority(anchor["type"]) >= anchor_priority(anchors[-1]["type"]):
                anchors[-1] = anchor
            continue
        anchors.append(anchor)
    return anchors


def event_anchor_point(
    *,
    event: dict[str, Any],
    next_event: dict[str, Any] | None,
    reference: CourtReference,
) -> np.ndarray | None:
    event_type = str(event.get("type"))
    if event_type == "bounce":
        return sanitize_court_point(event.get("ball_court_xy_m"), reference)

    if event_type not in {"serve", "hit"}:
        return None

    player_point = sanitize_court_point(
        event.get("player_court_xy_m"),
        reference,
        margin_m=6.0,
    )
    if player_point is None:
        return None

    target_point = next_anchor_target(
        next_event=next_event,
        player_point=player_point,
        actor=str(event.get("actor", "unknown")),
        reference=reference,
    )
    direction = target_point - player_point
    norm = float(np.linalg.norm(direction))
    if norm < 1e-4:
        direction = default_shot_direction(str(event.get("actor", "unknown")))
        norm = float(np.linalg.norm(direction))
    offset_m = 1.1 if event_type == "serve" else 0.8
    contact_point = player_point + ((direction / max(norm, 1e-4)) * offset_m)
    return clamp_court_point(contact_point, reference, margin_m=6.0)


def next_anchor_target(
    *,
    next_event: dict[str, Any] | None,
    player_point: np.ndarray,
    actor: str,
    reference: CourtReference,
) -> np.ndarray:
    if next_event is not None:
        if next_event.get("type") == "bounce":
            bounce_point = sanitize_court_point(next_event.get("ball_court_xy_m"), reference)
            if bounce_point is not None:
                return bounce_point
        next_player_point = sanitize_court_point(
            next_event.get("player_court_xy_m"),
            reference,
            margin_m=6.0,
        )
        if next_player_point is not None:
            return next_player_point

    fallback = player_point + default_shot_direction(actor)
    return clamp_court_point(fallback, reference, margin_m=6.0)


def default_shot_direction(actor: str) -> np.ndarray:
    if actor == "far_player":
        return np.array([0.0, 9.0], dtype=np.float32)
    if actor == "near_player":
        return np.array([0.0, -9.0], dtype=np.float32)
    return np.array([0.0, 6.0], dtype=np.float32)


def anchor_priority(event_type: str) -> int:
    if event_type == "bounce":
        return 2
    if event_type in {"serve", "hit"}:
        return 1
    return 0


def infer_anchor_segment_positions(
    *,
    derived_positions: list[np.ndarray | None],
    start_anchor: dict[str, Any],
    end_anchor: dict[str, Any],
    frame_count: int,
    raw_positions: list[list[float] | None],
) -> None:
    start_frame = max(int(start_anchor["frame"]), 0)
    end_frame = min(int(end_anchor["frame"]), frame_count - 1)
    if end_frame < start_frame:
        return

    start_point = np.array(start_anchor["point"], dtype=np.float32)
    end_point = np.array(end_anchor["point"], dtype=np.float32)
    direction = end_point - start_point
    length = float(np.linalg.norm(direction))
    if length < 1e-4:
        for frame_index in range(start_frame, end_frame + 1):
            derived_positions[frame_index] = start_point.copy()
        return

    unit_direction = direction / length
    segment_progress = infer_segment_progress(
        raw_positions=raw_positions,
        start_frame=start_frame,
        end_frame=end_frame,
        start_point=start_point,
        unit_direction=unit_direction,
        segment_length=length,
    )
    for local_index, frame_index in enumerate(range(start_frame, end_frame + 1)):
        point = start_point + (unit_direction * segment_progress[local_index])
        derived_positions[frame_index] = point.astype(np.float32)


def infer_segment_progress(
    *,
    raw_positions: list[list[float] | None],
    start_frame: int,
    end_frame: int,
    start_point: np.ndarray,
    unit_direction: np.ndarray,
    segment_length: float,
) -> np.ndarray:
    frame_total = max((end_frame - start_frame) + 1, 1)
    if frame_total == 1:
        return np.array([0.0], dtype=np.float32)

    baseline_progress = np.linspace(0.0, segment_length, frame_total, dtype=np.float32)
    smoothed_progress = baseline_progress.copy()
    raw_weight = 0.18

    for local_index, frame_index in enumerate(range(start_frame, end_frame + 1)):
        raw_point_xy = raw_positions[frame_index]
        if raw_point_xy is None or local_index in {0, frame_total - 1}:
            continue

        raw_point = np.array(raw_point_xy, dtype=np.float32)
        measured_progress = float(np.dot(raw_point - start_point, unit_direction))
        measured_progress = float(np.clip(measured_progress, 0.0, segment_length))
        smoothed_progress[local_index] = (
            ((1.0 - raw_weight) * baseline_progress[local_index]) + (raw_weight * measured_progress)
        )

    return enforce_monotonic_progress(smoothed_progress, segment_length)


def enforce_monotonic_progress(progress: np.ndarray, segment_length: float) -> np.ndarray:
    if progress.size == 0:
        return progress.astype(np.float32)

    clamped = np.clip(progress.astype(np.float32), 0.0, segment_length)
    clamped[0] = 0.0
    clamped[-1] = float(segment_length)

    for index in range(1, clamped.size):
        clamped[index] = max(clamped[index], clamped[index - 1])

    clamped[-1] = float(segment_length)
    for index in range(clamped.size - 2, -1, -1):
        clamped[index] = min(clamped[index], clamped[index + 1])

    clamped[0] = 0.0
    clamped[-1] = float(segment_length)
    return clamped


def sanitize_court_point(
    point_xy_m: list[float] | np.ndarray | None,
    reference: CourtReference,
    *,
    margin_m: float = 4.0,
) -> np.ndarray | None:
    if point_xy_m is None:
        return None
    point = np.array(point_xy_m, dtype=np.float32).reshape(2)
    if not np.all(np.isfinite(point)):
        return None
    if point[0] < -margin_m or point[0] > (reference.width_m + margin_m):
        return None
    if point[1] < -margin_m or point[1] > (reference.length_m + margin_m):
        return None
    return point


def clamp_court_point(point: np.ndarray, reference: CourtReference, *, margin_m: float) -> np.ndarray:
    return np.array(
        [
            np.clip(point[0], -margin_m, reference.width_m + margin_m),
            np.clip(point[1], -margin_m, reference.length_m + margin_m),
        ],
        dtype=np.float32,
    )


def rounded_ball_point(point_xy_m: np.ndarray | None) -> list[float] | None:
    if point_xy_m is None:
        return None
    return [round(float(point_xy_m[0]), 2), round(float(point_xy_m[1]), 2)]


def court_to_canvas(point_xy_m: np.ndarray, pixels_per_meter: int, margin_px: int) -> tuple[int, int]:
    x_pad_m = 2.0
    y_pad_m = 4.0
    x = int(round(margin_px + ((float(point_xy_m[0]) + x_pad_m) * pixels_per_meter)))
    y = int(round(margin_px + ((float(point_xy_m[1]) + y_pad_m) * pixels_per_meter)))
    return x, y


def canvas_size(reference: CourtReference, pixels_per_meter: int, margin_px: int) -> tuple[int, int]:
    x_pad_m = 2.0
    y_pad_m = 4.0
    width_m = reference.width_m + (2.0 * x_pad_m)
    height_m = reference.length_m + (2.0 * y_pad_m)
    canvas_width = int(round((width_m * pixels_per_meter) + (2 * margin_px)))
    canvas_height = int(round((height_m * pixels_per_meter) + (2 * margin_px)))
    return canvas_width, canvas_height


def load_pose_frames(pose_json_path: str | Path | None) -> dict[int, dict[str, dict]]:
    if pose_json_path is None:
        return {}
    pose_json_path = Path(pose_json_path)
    if not pose_json_path.exists():
        return {}

    data = json.loads(pose_json_path.read_text())
    return {
        int(frame["frame_index"]): frame.get("players", {})
        for frame in data.get("frames", [])
    }


def draw_player_skeleton(
    *,
    frame: np.ndarray,
    pose_data: dict,
    player_label: str,
    anchor_court_xy: np.ndarray,
    pixels_per_meter: int,
    margin_px: int,
    color: tuple[int, int, int],
) -> None:
    skeleton_points = skeleton_to_court_points(
        pose_data=pose_data,
        player_label=player_label,
        anchor_court_xy=anchor_court_xy,
    )
    if skeleton_points is None:
        px, py = court_to_canvas(anchor_court_xy, pixels_per_meter, margin_px)
        cv2.circle(frame, (px, py), 9, color, -1)
        cv2.circle(frame, (px, py), 12, (255, 255, 255), 1)
        return

    valid = [point is not None for point in skeleton_points]
    canvas_points = [
        court_to_canvas(point, pixels_per_meter, margin_px) if point is not None else None
        for point in skeleton_points
    ]

    for start_index, end_index in COCO_SKELETON_EDGES:
        if not valid[start_index] or not valid[end_index]:
            continue
        cv2.line(frame, canvas_points[start_index], canvas_points[end_index], color, 2)

    for point in canvas_points:
        if point is None:
            continue
        cv2.circle(frame, point, 3, (255, 255, 255), -1)
        cv2.circle(frame, point, 4, color, 1)


def skeleton_to_court_points(
    *,
    pose_data: dict,
    player_label: str,
    anchor_court_xy: np.ndarray,
) -> list[np.ndarray | None] | None:
    keypoints = np.array(pose_data.get("keypoints", []), dtype=np.float32)
    if keypoints.shape != (17, 3):
        return None

    confidences = keypoints[:, 2]
    if float(np.max(confidences)) < 0.15:
        return None

    anchor_xy = anchor_image_point(keypoints, confidences)
    if anchor_xy is None:
        return None

    top_y = upper_body_top_y(keypoints, confidences)
    pose_height = max(float(anchor_xy[1] - top_y), 35.0)
    scale_m_per_px = 1.7 / pose_height
    y_direction = -1.0 if player_label == "near_player" else 1.0

    court_points: list[np.ndarray | None] = []
    for x, y, conf in keypoints:
        if conf < 0.15:
            court_points.append(None)
            continue

        dx = float(x - anchor_xy[0]) * scale_m_per_px
        dy = float(anchor_xy[1] - y) * scale_m_per_px * y_direction
        dx = float(np.clip(dx, -1.1, 1.1))
        dy = float(np.clip(dy, -0.6, 2.0) if player_label == "far_player" else np.clip(dy, -2.0, 0.6))
        court_points.append(anchor_court_xy + np.array([dx, dy], dtype=np.float32))

    return court_points


def anchor_image_point(keypoints: np.ndarray, confidences: np.ndarray) -> np.ndarray | None:
    ankle_indices = [15, 16]
    ankle_points = [keypoints[index, :2] for index in ankle_indices if confidences[index] >= 0.15]
    if ankle_points:
        return np.mean(ankle_points, axis=0)

    knee_indices = [13, 14]
    knee_points = [keypoints[index, :2] for index in knee_indices if confidences[index] >= 0.15]
    if knee_points:
        return np.mean(knee_points, axis=0)

    hip_indices = [11, 12]
    hip_points = [keypoints[index, :2] for index in hip_indices if confidences[index] >= 0.15]
    if hip_points:
        return np.mean(hip_points, axis=0)
    return None


def upper_body_top_y(keypoints: np.ndarray, confidences: np.ndarray) -> float:
    preferred_indices = [0, 1, 2, 3, 4, 5, 6]
    values = [float(keypoints[index, 1]) for index in preferred_indices if confidences[index] >= 0.15]
    if values:
        return min(values)
    all_values = [float(keypoints[index, 1]) for index in range(len(keypoints)) if confidences[index] >= 0.15]
    return min(all_values) if all_values else float(keypoints[:, 1].min())
