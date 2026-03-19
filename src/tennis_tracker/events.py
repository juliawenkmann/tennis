from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def write_match_events_json(
    *,
    tracking_json_path: str | Path,
    output_json_path: str | Path,
    rally_gap_frames: int = 10,
) -> Path:
    tracking_json_path = Path(tracking_json_path)
    output_json_path = Path(output_json_path)

    tracking_data = json.loads(tracking_json_path.read_text())
    events_payload = extract_match_events(
        tracking_data,
        tracking_source=str(tracking_json_path.resolve()),
        rally_gap_frames=rally_gap_frames,
    )
    output_json_path.write_text(json.dumps(events_payload, indent=2))
    return output_json_path


def extract_match_events(
    tracking_data: dict[str, Any],
    *,
    tracking_source: str | None = None,
    rally_gap_frames: int = 10,
) -> dict[str, Any]:
    frames = tracking_data.get("frames", [])
    rallies: list[dict[str, Any]] = []
    frame_lookup = {int(frame["frame_index"]): frame for frame in frames}
    segments = detect_rally_segments(frames, max_gap_frames=rally_gap_frames)

    for rally_id, (start_frame, end_frame) in enumerate(segments, start=1):
        rally_frames = [frame_lookup[index] for index in range(start_frame, end_frame + 1) if index in frame_lookup]
        if not rally_frames:
            continue
        rally_payload = build_rally_payload(
            rally_id=rally_id,
            rally_frames=rally_frames,
            total_frame_count=len(frames),
            court_length_m=float(tracking_data.get("court_reference_m", {}).get("length", 23.77)),
        )
        if not is_valid_rally_payload(rally_payload):
            continue
        rallies.append(rally_payload)

    return {
        "source_video": Path(tracking_data.get("video_path", "unknown.mp4")).name,
        "fps": float(tracking_data.get("fps", 25.0)),
        "frame_count": len(frames),
        "frame_size": tracking_data.get("frame_size", {"width": 0, "height": 0}),
        "court_reference_m": tracking_data.get("court_reference_m", {}),
        "court": frames[0]["court"] if frames else None,
        "tracking_source": tracking_source,
        "rallies": rallies,
    }


def build_rally_payload(
    *,
    rally_id: int,
    rally_frames: list[dict[str, Any]],
    total_frame_count: int,
    court_length_m: float,
) -> dict[str, Any]:
    start_frame = int(rally_frames[0]["frame_index"])
    end_frame = int(rally_frames[-1]["frame_index"])

    shot_candidates = detect_shot_candidates(rally_frames, court_length_m=court_length_m)
    shot_events = select_shot_events(shot_candidates)
    bounce_events = detect_bounce_events(
        rally_frames=rally_frames,
        shot_events=shot_events,
        court_length_m=court_length_m,
    )

    events: list[dict[str, Any]] = []
    event_id = 1
    first_actor = shot_events[0]["actor"] if shot_events else "unknown"
    events.append(
        {
            "id": event_id,
            "type": "rally_start",
            "frame": start_frame,
            "time_sec": float(rally_frames[0]["timestamp_sec"]),
            "actor": first_actor,
            "confidence": round(rally_confidence(rally_frames, shot_events), 3),
            "source": "ball_presence_rule",
            "extra": {},
        }
    )
    event_id += 1

    for index, shot in enumerate(shot_events):
        event_type = "serve" if index == 0 else "hit"
        event = {
            "id": event_id,
            "type": event_type,
            "frame": int(shot["frame"]),
            "time_sec": float(shot["time_sec"]),
            "actor": shot["actor"],
            "confidence": round(float(shot["confidence"]), 3),
            "source": shot["source"],
            "ball_image_xy": shot["ball_image_xy"],
            "player_image_xy": shot.get("player_image_xy"),
            "player_court_xy_m": shot.get("player_court_xy_m"),
            "extra": {
                "score": round(float(shot["score"]), 3),
            },
        }
        events.append(event)
        event_id += 1

        if index < len(bounce_events):
            bounce = bounce_events[index]
            bounce["id"] = event_id
            events.append(bounce)
            event_id += 1

    events.append(
        {
            "id": event_id,
            "type": "rally_end",
            "frame": end_frame,
            "time_sec": float(rally_frames[-1]["timestamp_sec"]),
            "actor": shot_events[-1]["actor"] if shot_events else "unknown",
            "confidence": round(rally_confidence(rally_frames, shot_events), 3),
            "source": "ball_presence_rule",
            "extra": {
                "reason": rally_end_reason(
                    rally_frames=rally_frames,
                    total_frame_count=total_frame_count,
                    court_length_m=court_length_m,
                )
            },
        }
    )

    events.sort(key=lambda item: (int(item["frame"]), event_type_rank(item["type"]), int(item["id"])))
    for new_id, event in enumerate(events, start=1):
        event["id"] = new_id

    serve_actor = next((event["actor"] for event in events if event["type"] == "serve"), "unknown")
    hit_count = sum(event["type"] == "hit" for event in events)
    bounce_count = sum(event["type"] == "bounce" for event in events)
    ball_detected_frame_count = sum(frame.get("ball", {}).get("image_xy") is not None for frame in rally_frames)
    start_time = float(rally_frames[0]["timestamp_sec"])
    end_time = float(rally_frames[-1]["timestamp_sec"])

    return {
        "id": rally_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_sec": start_time,
        "end_time_sec": end_time,
        "confidence": round(rally_confidence(rally_frames, shot_events), 3),
        "events": events,
        "summary": {
            "event_count": len(events),
            "serve_actor": serve_actor,
            "hit_count": hit_count,
            "bounce_count": bounce_count,
            "duration_sec": round(max(0.0, end_time - start_time), 3),
            "ball_detected_frame_count": ball_detected_frame_count,
        },
    }


def is_valid_rally_payload(rally_payload: dict[str, Any]) -> bool:
    summary = rally_payload.get("summary", {})
    ball_frames = int(summary.get("ball_detected_frame_count", 0))
    duration_sec = float(summary.get("duration_sec", 0.0))
    action_count = sum(
        event["type"] in {"serve", "hit", "bounce"}
        for event in rally_payload.get("events", [])
    )

    if action_count >= 2:
        return True
    if action_count >= 1 and ball_frames >= 8 and duration_sec >= 0.32:
        return True
    if ball_frames >= 12 and duration_sec >= 1.0:
        return True
    return False


def detect_rally_segments(frames: list[dict[str, Any]], *, max_gap_frames: int) -> list[tuple[int, int]]:
    ball_frames = [
        int(frame["frame_index"])
        for frame in frames
        if frame.get("ball", {}).get("image_xy") is not None
    ]
    if not ball_frames:
        return []

    segments: list[tuple[int, int]] = []
    segment_start = ball_frames[0]
    previous = ball_frames[0]
    for frame_index in ball_frames[1:]:
        if frame_index - previous > max_gap_frames:
            segments.append((segment_start, previous))
            segment_start = frame_index
        previous = frame_index
    segments.append((segment_start, previous))
    return segments


def detect_shot_candidates(
    rally_frames: list[dict[str, Any]],
    *,
    court_length_m: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    frame_height = max(float(frame_height_from_rally(rally_frames)), 1.0)

    for index, frame in enumerate(rally_frames):
        ball_image_xy = frame.get("ball", {}).get("image_xy")
        if ball_image_xy is None:
            continue

        best_candidate: dict[str, Any] | None = None
        for actor in ("far_player", "near_player"):
            player = player_by_label(frame, actor)
            if player is None:
                continue

            distance_px = ball_to_contact_distance_px(ball_image_xy, player)
            proximity_score = normalized_inverse(distance_px, 170.0)
            zone_score = contact_zone_score(
                actor=actor,
                ball_court_xy=frame.get("ball", {}).get("court_xy_m"),
                ball_image_xy=ball_image_xy,
                court_length_m=court_length_m,
                frame_height=frame_height,
            )
            direction_score = departure_direction_score(
                rally_frames=rally_frames,
                frame_index=index,
                actor=actor,
            )
            score = (0.55 * proximity_score) + (0.25 * zone_score) + (0.20 * direction_score)
            if score < 0.42 or proximity_score < 0.12:
                continue

            candidate = {
                "frame": int(frame["frame_index"]),
                "time_sec": float(frame["timestamp_sec"]),
                "actor": actor,
                "confidence": max(0.25, min(score, 0.98)),
                "score": score,
                "source": "player_proximity_rule",
                "ball_image_xy": rounded_point(ball_image_xy),
                "player_image_xy": rounded_point(player.get("image_xy")),
                "player_court_xy_m": rounded_point(player.get("court_xy_m")),
            }
            if best_candidate is None or candidate["score"] > best_candidate["score"]:
                best_candidate = candidate

        if best_candidate is not None:
            candidates.append(best_candidate)

    return peak_candidates(candidates)


def peak_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []

    peaks: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        local_window = candidates[max(0, index - 2) : min(len(candidates), index + 3)]
        if any(other["score"] > candidate["score"] for other in local_window if other is not candidate):
            continue
        peaks.append(candidate)
    return peaks


def select_shot_events(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []

    selected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (int(item["frame"]), -float(item["score"]))):
        if not selected:
            selected.append(candidate)
            continue

        previous = selected[-1]
        frame_gap = int(candidate["frame"]) - int(previous["frame"])
        if frame_gap < 6:
            if float(candidate["score"]) > float(previous["score"]):
                selected[-1] = candidate
            continue

        if candidate["actor"] == previous["actor"]:
            if frame_gap < 14:
                if float(candidate["score"]) > float(previous["score"]) + 0.12:
                    selected[-1] = candidate
                continue
            if float(candidate["score"]) < float(previous["score"]) + 0.08:
                continue

        selected.append(candidate)

    return selected


def detect_bounce_events(
    *,
    rally_frames: list[dict[str, Any]],
    shot_events: list[dict[str, Any]],
    court_length_m: float,
) -> list[dict[str, Any]]:
    if len(shot_events) < 2:
        return []

    frame_lookup = {int(frame["frame_index"]): frame for frame in rally_frames}
    net_y = court_length_m / 2.0
    bounce_events: list[dict[str, Any]] = []

    for index in range(len(shot_events) - 1):
        shot = shot_events[index]
        next_shot = shot_events[index + 1]
        receiver = opposite_actor(shot["actor"])

        segment_frames = [
            frame_lookup[frame_index]
            for frame_index in range(int(shot["frame"]) + 1, int(next_shot["frame"]))
            if frame_index in frame_lookup and frame_lookup[frame_index].get("ball", {}).get("image_xy") is not None
        ]
        if len(segment_frames) < 3:
            continue

        receiver_side_frames = []
        for frame in segment_frames:
            court_xy = frame.get("ball", {}).get("court_xy_m")
            if court_xy is None:
                continue
            y_value = float(court_xy[1])
            if receiver == "near_player" and y_value <= net_y + 0.35:
                continue
            if receiver == "far_player" and y_value >= net_y - 0.35:
                continue
            receiver_side_frames.append(frame)

        if len(receiver_side_frames) < 3:
            continue

        best_frame: dict[str, Any] | None = None
        best_score = -1.0
        for inner_index in range(1, len(receiver_side_frames) - 1):
            previous_frame = receiver_side_frames[inner_index - 1]
            frame = receiver_side_frames[inner_index]
            next_frame = receiver_side_frames[inner_index + 1]
            ball_prev = np.array(previous_frame["ball"]["image_xy"], dtype=np.float32)
            ball_curr = np.array(frame["ball"]["image_xy"], dtype=np.float32)
            ball_next = np.array(next_frame["ball"]["image_xy"], dtype=np.float32)

            turn_score = trajectory_turn_score(ball_prev, ball_curr, ball_next)
            inside_score = bounce_zone_score(frame.get("ball", {}).get("court_xy_m"), court_length_m)
            score = (0.75 * turn_score) + (0.25 * inside_score)
            if score > best_score:
                best_score = score
                best_frame = frame

        if best_frame is None:
            continue

        bounce_events.append(
            {
                "type": "bounce",
                "frame": int(best_frame["frame_index"]),
                "time_sec": float(best_frame["timestamp_sec"]),
                "actor": shot["actor"],
                "confidence": round(max(0.2, min(best_score, 0.95)), 3),
                "source": "trajectory_turn_rule",
                "ball_image_xy": rounded_point(best_frame["ball"]["image_xy"]),
                "ball_court_xy_m": rounded_point(best_frame["ball"]["court_xy_m"]),
                "extra": {
                    "receiver": receiver,
                },
            }
        )

    return bounce_events


def rally_confidence(rally_frames: list[dict[str, Any]], shot_events: list[dict[str, Any]]) -> float:
    frame_count = max(len(rally_frames), 1)
    ball_detected_count = sum(frame.get("ball", {}).get("image_xy") is not None for frame in rally_frames)
    coverage_score = ball_detected_count / frame_count
    shot_score = min(len(shot_events) / 4.0, 1.0)
    return max(0.25, min((0.6 * coverage_score) + (0.4 * shot_score), 0.98))


def rally_end_reason(
    *,
    rally_frames: list[dict[str, Any]],
    total_frame_count: int,
    court_length_m: float,
) -> str:
    last_frame = rally_frames[-1]
    if int(last_frame["frame_index"]) >= total_frame_count - 1:
        return "clip_end"

    last_ball_court_xy = last_frame.get("ball", {}).get("court_xy_m")
    if last_ball_court_xy is not None:
        x_value = float(last_ball_court_xy[0])
        y_value = float(last_ball_court_xy[1])
        if x_value < 0.0 or x_value > 10.97 or y_value < 0.0 or y_value > court_length_m:
            return "ball_out_suspected"

    return "ball_missing"


def event_type_rank(event_type: str) -> int:
    ranks = {
        "rally_start": 0,
        "serve": 1,
        "hit": 2,
        "bounce": 3,
        "rally_end": 4,
    }
    return ranks.get(event_type, 99)


def frame_height_from_rally(rally_frames: list[dict[str, Any]]) -> int:
    first_frame = rally_frames[0] if rally_frames else {}
    court = first_frame.get("court", {})
    image_corners = court.get("image_corners")
    if not image_corners:
        return 360
    max_y = max(point[1] for point in image_corners)
    return max(int(round(float(max_y) * 1.25)), 360)


def player_by_label(frame: dict[str, Any], label: str) -> dict[str, Any] | None:
    for player in frame.get("players", []):
        if player.get("label") == label:
            return player
    return None


def ball_to_contact_distance_px(ball_image_xy: list[float], player: dict[str, Any]) -> float:
    bbox = player.get("bbox_xyxy")
    if bbox is None:
        player_xy = np.array(player.get("image_xy", [0.0, 0.0]), dtype=np.float32)
        return float(np.linalg.norm(np.array(ball_image_xy, dtype=np.float32) - player_xy))

    x1, y1, x2, y2 = [float(value) for value in bbox]
    anchor = np.array(
        [
            (x1 + x2) / 2.0,
            y1 + (0.35 * (y2 - y1)),
        ],
        dtype=np.float32,
    )
    return float(np.linalg.norm(np.array(ball_image_xy, dtype=np.float32) - anchor))


def normalized_inverse(value: float, scale: float) -> float:
    return max(0.0, min(1.0 - (value / max(scale, 1.0)), 1.0))


def contact_zone_score(
    *,
    actor: str,
    ball_court_xy: list[float] | None,
    ball_image_xy: list[float],
    court_length_m: float,
    frame_height: float,
) -> float:
    if ball_court_xy is not None:
        y_value = float(ball_court_xy[1])
        target_y = 1.8 if actor == "far_player" else court_length_m - 1.8
        return max(0.0, min(1.0 - (abs(y_value - target_y) / 8.0), 1.0))

    image_y = float(ball_image_xy[1])
    if actor == "far_player":
        return max(0.0, min(1.0 - (image_y / (frame_height * 0.65)), 1.0))
    return max(0.0, min((image_y - (frame_height * 0.35)) / (frame_height * 0.65), 1.0))


def departure_direction_score(
    *,
    rally_frames: list[dict[str, Any]],
    frame_index: int,
    actor: str,
) -> float:
    current_ball = rally_frames[frame_index].get("ball", {}).get("court_xy_m")
    if current_ball is None:
        return 0.0

    current_y = float(current_ball[1])
    future_y: float | None = None
    for future_frame in rally_frames[frame_index + 1 : frame_index + 5]:
        future_ball = future_frame.get("ball", {}).get("court_xy_m")
        if future_ball is None:
            continue
        future_y = float(future_ball[1])
        break
    if future_y is None:
        return 0.0

    delta = future_y - current_y
    if actor == "far_player":
        return max(0.0, min(delta / 3.0, 1.0))
    return max(0.0, min((-delta) / 3.0, 1.0))


def trajectory_turn_score(ball_prev: np.ndarray, ball_curr: np.ndarray, ball_next: np.ndarray) -> float:
    vector_a = ball_curr - ball_prev
    vector_b = ball_next - ball_curr
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a < 1e-3 or norm_b < 1e-3:
        return 0.0

    cosine = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    cosine = max(-1.0, min(1.0, cosine))
    angle = math.acos(cosine)
    return max(0.0, min(angle / math.pi, 1.0))


def bounce_zone_score(ball_court_xy: list[float] | None, court_length_m: float) -> float:
    if ball_court_xy is None:
        return 0.0
    x_value = float(ball_court_xy[0])
    y_value = float(ball_court_xy[1])
    if x_value < 0.0 or x_value > 10.97 or y_value < 0.0 or y_value > court_length_m:
        return 0.0

    center_x_score = 1.0 - min(abs(x_value - (10.97 / 2.0)) / 5.5, 1.0)
    y_margin = min(y_value, court_length_m - y_value)
    depth_score = min(y_margin / 6.0, 1.0)
    return (0.4 * center_x_score) + (0.6 * depth_score)


def opposite_actor(actor: str) -> str:
    if actor == "near_player":
        return "far_player"
    if actor == "far_player":
        return "near_player"
    return "unknown"


def rounded_point(point: list[float] | None) -> list[float] | None:
    if point is None:
        return None
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


def summarize_match_events(event_json_path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(event_json_path).read_text())
    rallies = data.get("rallies", [])
    return {
        "selected_video": data.get("source_video"),
        "rallies": len(rallies),
        "serve_events": sum(event["type"] == "serve" for rally in rallies for event in rally.get("events", [])),
        "hit_events": sum(event["type"] == "hit" for rally in rallies for event in rally.get("events", [])),
        "bounce_events": sum(event["type"] == "bounce" for rally in rallies for event in rally.get("events", [])),
    }


def ensure_match_events_for_video(
    *,
    video_path: str | Path,
    output_dir: str | Path,
    weights_path: str | Path,
    force: bool = False,
) -> dict[str, Path]:
    from tennis_tracker.pipeline import run_pipeline

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = slugify_video_name(video_path.name)
    tracking_json_path = output_dir / f"{slug}_tracking.json"
    tracking_video_path = output_dir / f"{slug}_tracking.mp4"
    events_json_path = output_dir / f"{slug}_match_events.json"

    needs_tracking = (
        force
        or not tracking_json_path.exists()
        or not tracking_video_path.exists()
    )
    if needs_tracking:
        results = run_pipeline(
            video_path=video_path,
            output_dir=output_dir,
            step="all",
            weights_path=weights_path,
        )
        move_artifact(Path(results["json"]), tracking_json_path)
        move_artifact(Path(results["video"]), tracking_video_path)

        if "events" in results:
            move_artifact(Path(results["events"]), events_json_path)

        generic_xml_path = Path(results["xml"])
        if generic_xml_path.exists():
            generic_xml_path.unlink()

        stale_temp_video_path = output_dir / "all.opencv.mp4"
        stale_temp_video_path.unlink(missing_ok=True)

    if force or needs_tracking or not events_json_path.exists():
        write_match_events_json(
            tracking_json_path=tracking_json_path,
            output_json_path=events_json_path,
        )

    return {
        "tracking_json": tracking_json_path,
        "tracking_video": tracking_video_path,
        "events_json": events_json_path,
    }


def slugify_video_name(video_name: str) -> str:
    stem = Path(video_name).stem.lower()
    return "".join(char if char.isalnum() else "_" for char in stem).strip("_")


def move_artifact(source: Path, destination: Path) -> Path:
    source = Path(source)
    destination = Path(destination)

    if not source.exists():
        raise FileNotFoundError(source)

    if source.resolve() == destination.resolve():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    source.replace(destination)
    return destination


def event_map_from_json(events_json_path: str | Path) -> dict[int, list[dict[str, Any]]]:
    data = json.loads(Path(events_json_path).read_text())
    return event_map_from_data(data)


def event_map_from_data(events_data: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    event_map: dict[int, list[dict[str, Any]]] = {}
    for rally in events_data.get("rallies", []):
        for event in rally.get("events", []):
            frame = int(event["frame"])
            event_map.setdefault(frame, []).append(event)
    return event_map


def render_event_annotation_video(
    *,
    input_video_path: str | Path,
    events_json_path: str | Path,
    output_video_path: str | Path,
) -> Path:
    from tennis_tracker.pipeline import finalize_video_for_playback

    input_video_path = Path(input_video_path)
    events_json_path = Path(events_json_path)
    output_video_path = Path(output_video_path)
    temp_video_path = output_video_path.with_suffix(".opencv.mp4")

    event_map = event_map_from_json(events_json_path)

    capture = cv2.VideoCapture(str(input_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(temp_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        annotated = draw_event_labels(frame, event_map.get(frame_index, []))
        writer.write(annotated)
        frame_index += 1

    capture.release()
    writer.release()
    finalize_video_for_playback(temp_video_path, output_video_path)
    return output_video_path


def draw_event_labels(frame: np.ndarray, events: list[dict[str, Any]]) -> np.ndarray:
    if not events:
        return frame

    overlay = frame.copy()
    frame_height, frame_width = overlay.shape[:2]
    cv2.rectangle(overlay, (18, 16), (min(frame_width - 18, 360), 34 + (28 * len(events))), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, dst=overlay)

    for index, event in enumerate(events):
        label = event["type"].replace("_", " ")
        actor = event.get("actor", "unknown").replace("_", " ")
        confidence = float(event.get("confidence", 0.0))
        text = f"{label} | {actor} | {confidence:.2f}"
        cv2.putText(
            overlay,
            text,
            (28, 42 + (28 * index)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            event_color(event["type"]),
            2,
        )

    return overlay


def event_color(event_type: str) -> tuple[int, int, int]:
    colors = {
        "rally_start": (255, 220, 90),
        "serve": (80, 200, 255),
        "hit": (110, 255, 130),
        "bounce": (0, 165, 255),
        "rally_end": (255, 120, 120),
    }
    return colors.get(event_type, (235, 235, 235))
