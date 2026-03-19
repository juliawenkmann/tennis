from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_FRAME_TOLERANCE = {
    "rally_start": 5,
    "serve": 3,
    "hit": 3,
    "bounce": 4,
    "rally_end": 5,
}


def write_benchmark_label_template(
    *,
    prediction_json_path: str | Path,
    output_label_path: str | Path,
) -> Path:
    prediction_json_path = Path(prediction_json_path)
    output_label_path = Path(output_label_path)

    prediction_data = json.loads(prediction_json_path.read_text())
    label_payload = build_benchmark_label_template(prediction_data)
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    output_label_path.write_text(json.dumps(label_payload, indent=2))
    return output_label_path


def build_benchmark_label_template(prediction_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_video": prediction_data.get("source_video"),
        "prediction_source": prediction_data.get("tracking_source"),
        "annotation_status": "review_needed",
        "notes": "Seeded from the current event predictions. Review and correct manually before using as benchmark truth.",
        "rallies": [
            {
                "id": rally["id"],
                "start_frame": rally["start_frame"],
                "end_frame": rally["end_frame"],
                "events": [
                    {
                        "type": event["type"],
                        "frame": event["frame"],
                        "actor": event.get("actor", "unknown"),
                    }
                    for event in rally.get("events", [])
                ],
            }
            for rally in prediction_data.get("rallies", [])
        ],
    }


def score_event_predictions(
    *,
    prediction_json_path: str | Path,
    label_json_path: str | Path,
    frame_tolerance: dict[str, int] | None = None,
) -> dict[str, Any]:
    prediction_json_path = Path(prediction_json_path)
    label_json_path = Path(label_json_path)

    prediction_data = json.loads(prediction_json_path.read_text())
    label_data = json.loads(label_json_path.read_text())
    results = score_event_predictions_data(
        prediction_data=prediction_data,
        label_data=label_data,
        frame_tolerance=frame_tolerance,
    )
    results["prediction_source"] = str(prediction_json_path)
    results["label_source"] = str(label_json_path)
    return results


def score_event_predictions_data(
    *,
    prediction_data: dict[str, Any],
    label_data: dict[str, Any],
    frame_tolerance: dict[str, int] | None = None,
) -> dict[str, Any]:
    frame_tolerance = frame_tolerance or DEFAULT_FRAME_TOLERANCE
    predicted_events = flatten_events(prediction_data)
    labeled_events = flatten_events(label_data)

    event_types = sorted(
        {
            event["type"]
            for event in predicted_events + labeled_events
        }
    )

    by_type: dict[str, Any] = {}
    total_matches = 0
    total_predicted = len(predicted_events)
    total_labeled = len(labeled_events)

    for event_type in event_types:
        predicted = [event for event in predicted_events if event["type"] == event_type]
        labeled = [event for event in labeled_events if event["type"] == event_type]
        matches, missed, extra = match_events(
            predicted=predicted,
            labeled=labeled,
            frame_tolerance=frame_tolerance.get(event_type, 3),
        )
        total_matches += len(matches)
        by_type[event_type] = {
            "predicted": len(predicted),
            "labeled": len(labeled),
            "matched": len(matches),
            "precision": safe_ratio(len(matches), len(predicted)),
            "recall": safe_ratio(len(matches), len(labeled)),
            "f1": f1_score(len(matches), len(predicted), len(labeled)),
            "missed": missed,
            "extra": extra,
        }

    return {
        "overall": {
            "predicted": total_predicted,
            "labeled": total_labeled,
            "matched": total_matches,
            "precision": safe_ratio(total_matches, total_predicted),
            "recall": safe_ratio(total_matches, total_labeled),
            "f1": f1_score(total_matches, total_predicted, total_labeled),
        },
        "by_type": by_type,
    }


def flatten_events(data: dict[str, Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for rally in data.get("rallies", []):
        rally_id = int(rally["id"])
        for event in rally.get("events", []):
            flattened.append(
                {
                    "rally_id": rally_id,
                    "type": event["type"],
                    "frame": int(event["frame"]),
                    "actor": event.get("actor", "unknown"),
                }
            )
    return flattened


def match_events(
    *,
    predicted: list[dict[str, Any]],
    labeled: list[dict[str, Any]],
    frame_tolerance: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    matches: list[dict[str, Any]] = []
    missed: list[dict[str, Any]] = []
    used_predictions: set[int] = set()

    for label in labeled:
        best_index: int | None = None
        best_distance: int | None = None
        for index, prediction in enumerate(predicted):
            if index in used_predictions:
                continue
            if prediction["rally_id"] != label["rally_id"]:
                continue
            if not actors_match(label.get("actor"), prediction.get("actor")):
                continue

            distance = abs(int(prediction["frame"]) - int(label["frame"]))
            if distance > frame_tolerance:
                continue
            if best_distance is None or distance < best_distance:
                best_index = index
                best_distance = distance

        if best_index is None:
            missed.append(label)
            continue

        used_predictions.add(best_index)
        matches.append(
            {
                "label": label,
                "prediction": predicted[best_index],
                "frame_error": best_distance,
            }
        )

    extra = [
        prediction
        for index, prediction in enumerate(predicted)
        if index not in used_predictions
    ]
    return matches, missed, extra


def actors_match(expected: str | None, observed: str | None) -> bool:
    expected = expected or "unknown"
    observed = observed or "unknown"
    if expected == "unknown" or observed == "unknown":
        return True
    return expected == observed


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 3)


def f1_score(matches: int, predicted: int, labeled: int) -> float:
    precision = safe_ratio(matches, predicted)
    recall = safe_ratio(matches, labeled)
    if precision + recall == 0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 3)
