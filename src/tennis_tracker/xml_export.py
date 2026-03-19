from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .events import ensure_match_events_for_video, slugify_video_name


def write_match_tracking_xml(
    *,
    event_json_path: str | Path,
    output_xml_path: str | Path,
) -> Path:
    event_json_path = Path(event_json_path)
    output_xml_path = Path(output_xml_path)

    events_data = json.loads(event_json_path.read_text())
    tracking_data = load_tracking_data(events_data, base_path=event_json_path.parent)
    frame_lookup = {
        int(frame["frame_index"]): frame
        for frame in tracking_data.get("frames", [])
    }

    root = ET.Element("match")
    initial_court = events_data.get("court")
    if initial_court is not None and initial_court.get("image_corners"):
        root.append(court_element(initial_court))

    current_frame_index = 0
    for rally in events_data.get("rallies", []):
        rally_start = int(rally["start_frame"])
        rally_end = int(rally["end_frame"])

        for frame_index in range(current_frame_index, rally_start):
            frame = frame_lookup.get(frame_index)
            if frame is not None:
                root.append(frame_element(frame, event="none"))

        rally_node = ET.SubElement(root, "rally", {"id": str(rally["id"])})
        for event in coalesce_events_by_frame(rally.get("events", [])):
            frame = frame_lookup.get(int(event["frame"]))
            if frame is None:
                rally_node.append(frame_element_from_event(event))
                continue
            rally_node.append(frame_element(frame, event=event["event"]))

        current_frame_index = rally_end + 1

    for frame_index in range(current_frame_index, int(events_data.get("frame_count", 0))):
        frame = frame_lookup.get(frame_index)
        if frame is not None:
            root.append(frame_element(frame, event="none"))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
    return output_xml_path


def ensure_match_tracking_xml_for_video(
    *,
    video_path: str | Path,
    output_dir: str | Path,
    weights_path: str | Path,
    force: bool = False,
) -> dict[str, Path]:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = ensure_match_events_for_video(
        video_path=video_path,
        output_dir=output_dir,
        weights_path=weights_path,
        force=force,
    )
    xml_path = output_dir / f"{slugify_video_name(video_path.name)}_match.xml"

    events_mtime = artifacts["events_json"].stat().st_mtime
    xml_stale = not xml_path.exists() or xml_path.stat().st_mtime < events_mtime
    if force or xml_stale:
        write_match_tracking_xml(
            event_json_path=artifacts["events_json"],
            output_xml_path=xml_path,
        )

    artifacts["xml"] = xml_path
    return artifacts


def summarize_match_tracking_xml(xml_path: str | Path) -> dict[str, int]:
    root = ET.parse(xml_path).getroot()
    rally_nodes = root.findall("rally")
    rally_frames = [frame for rally in rally_nodes for frame in rally.findall("frame")]
    return {
        "rallies": len(rally_nodes),
        "outside_frames": len(root.findall("frame")),
        "serve_frames": sum("serve" in frame.attrib.get("event", "") for frame in rally_frames),
        "hit_frames": sum("hit" in frame.attrib.get("event", "") for frame in rally_frames),
        "bounce_frames": sum("bounce" in frame.attrib.get("event", "") for frame in rally_frames),
    }


def load_tracking_data(events_data: dict[str, Any], *, base_path: Path) -> dict[str, Any]:
    tracking_source = events_data.get("tracking_source")
    if not tracking_source:
        return {"frames": []}

    tracking_path = Path(tracking_source)
    if not tracking_path.is_absolute():
        candidate = (base_path / tracking_path).resolve()
        tracking_path = candidate if candidate.exists() else tracking_path.resolve()
    if not tracking_path.exists():
        return {"frames": []}
    return json.loads(tracking_path.read_text())


def coalesce_events_by_frame(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault(int(event["frame"]), []).append(event)

    coalesced: list[dict[str, Any]] = []
    for frame in sorted(grouped):
        grouped_events = sorted(
            grouped[frame],
            key=lambda item: (event_priority(item["type"]), int(item["id"])),
        )
        event_names = [event["type"] for event in grouped_events]
        merged_event = grouped_events[0].copy()
        merged_event["event"] = "|".join(event_names)
        coalesced.append(merged_event)
    return coalesced


def event_priority(event_type: str) -> int:
    priorities = {
        "rally_start": 0,
        "serve": 1,
        "hit": 2,
        "bounce": 3,
        "rally_end": 4,
    }
    return priorities.get(event_type, 99)


def court_element(court: dict[str, Any]) -> ET.Element:
    corners = court["image_corners"]
    return ET.Element(
        "court",
        {
            "xLT": integer_string(corners[0][0]),
            "yLT": integer_string(corners[0][1]),
            "xRT": integer_string(corners[1][0]),
            "yRT": integer_string(corners[1][1]),
            "xLB": integer_string(corners[3][0]),
            "yLB": integer_string(corners[3][1]),
            "xRB": integer_string(corners[2][0]),
            "yRB": integer_string(corners[2][1]),
        },
    )


def frame_element(frame: dict[str, Any], *, event: str) -> ET.Element:
    top_player = next((player for player in frame.get("players", []) if player["label"] == "far_player"), None)
    bottom_player = next((player for player in frame.get("players", []) if player["label"] == "near_player"), None)
    ball = frame.get("ball", {})
    ball_xy = ball.get("image_xy")

    attributes = {
        "id": str(frame["frame_index"]),
        "time": time_string(frame["timestamp_sec"]),
        "event": event,
        "xPosTopPl": integer_string(top_player["image_xy"][0] if top_player else 0),
        "yPosTopPl": integer_string(top_player["image_xy"][1] if top_player else 0),
        "xPosBotPl": integer_string(bottom_player["image_xy"][0] if bottom_player else 0),
        "yPosBotPl": integer_string(bottom_player["image_xy"][1] if bottom_player else 0),
        "xPosBall": integer_string(ball_xy[0] if ball_xy else 0),
        "yPosBall": integer_string(ball_xy[1] if ball_xy else 0),
    }
    return ET.Element("frame", attributes)


def frame_element_from_event(event: dict[str, Any]) -> ET.Element:
    ball_xy = event.get("ball_image_xy")
    player_xy = event.get("player_image_xy")
    top_xy = player_xy if event.get("actor") == "far_player" else None
    bottom_xy = player_xy if event.get("actor") == "near_player" else None

    attributes = {
        "id": str(event["frame"]),
        "time": time_string(event["time_sec"]),
        "event": event.get("event", event["type"]),
        "xPosTopPl": integer_string(top_xy[0] if top_xy else 0),
        "yPosTopPl": integer_string(top_xy[1] if top_xy else 0),
        "xPosBotPl": integer_string(bottom_xy[0] if bottom_xy else 0),
        "yPosBotPl": integer_string(bottom_xy[1] if bottom_xy else 0),
        "xPosBall": integer_string(ball_xy[0] if ball_xy else 0),
        "yPosBall": integer_string(ball_xy[1] if ball_xy else 0),
    }
    return ET.Element("frame", attributes)


def integer_string(value: float | int) -> str:
    return str(int(round(float(value))))


def time_string(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")
