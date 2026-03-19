from __future__ import annotations

import json
import math
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .events import write_match_events_json
from .runtime import detect_runtime
from .tracknet import BallTrackNet, stack_three_frames


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class CourtReference:
    width_m: float = 10.97
    length_m: float = 23.77
    singles_width_m: float = 8.23
    service_line_from_net_m: float = 6.40

    @property
    def outer_corners(self) -> np.ndarray:
        return np.array(
            [
                [0.0, 0.0],
                [self.width_m, 0.0],
                [self.width_m, self.length_m],
                [0.0, self.length_m],
            ],
            dtype=np.float32,
        )

    @property
    def keypoints(self) -> dict[str, np.ndarray]:
        singles_margin = (self.width_m - self.singles_width_m) / 2.0
        net_y = self.length_m / 2.0
        far_service_y = net_y - self.service_line_from_net_m
        near_service_y = net_y + self.service_line_from_net_m
        return {
            "outer_top_left": np.array([0.0, 0.0], dtype=np.float32),
            "outer_top_right": np.array([self.width_m, 0.0], dtype=np.float32),
            "outer_bottom_right": np.array([self.width_m, self.length_m], dtype=np.float32),
            "outer_bottom_left": np.array([0.0, self.length_m], dtype=np.float32),
            "singles_top_left": np.array([singles_margin, 0.0], dtype=np.float32),
            "singles_top_right": np.array([self.width_m - singles_margin, 0.0], dtype=np.float32),
            "singles_bottom_right": np.array(
                [self.width_m - singles_margin, self.length_m],
                dtype=np.float32,
            ),
            "singles_bottom_left": np.array([singles_margin, self.length_m], dtype=np.float32),
            "far_service_left": np.array([singles_margin, far_service_y], dtype=np.float32),
            "far_service_right": np.array(
                [self.width_m - singles_margin, far_service_y],
                dtype=np.float32,
            ),
            "near_service_left": np.array([singles_margin, near_service_y], dtype=np.float32),
            "near_service_right": np.array(
                [self.width_m - singles_margin, near_service_y],
                dtype=np.float32,
            ),
            "net_left": np.array([0.0, net_y], dtype=np.float32),
            "net_right": np.array([self.width_m, net_y], dtype=np.float32),
            "service_center_top": np.array([self.width_m / 2.0, far_service_y], dtype=np.float32),
            "service_center_bottom": np.array(
                [self.width_m / 2.0, near_service_y],
                dtype=np.float32,
            ),
        }


@dataclass
class CourtDetectionResult:
    image_corners: np.ndarray
    homography_court_to_image: np.ndarray
    homography_image_to_court: np.ndarray
    image_keypoints: dict[str, list[float]]
    source: str
    line_support: float
    shape_score: float
    total_score: float


@dataclass
class PlayerDetectionResult:
    label: str
    bbox_xyxy: list[float]
    image_xy: list[float]
    court_xy_m: list[float]
    confidence: float
    source: str = "detected"


COURT_DRAW_LINES = [
    ("outer_top_left", "outer_top_right"),
    ("outer_top_right", "outer_bottom_right"),
    ("outer_bottom_right", "outer_bottom_left"),
    ("outer_bottom_left", "outer_top_left"),
    ("singles_top_left", "singles_bottom_left"),
    ("singles_top_right", "singles_bottom_right"),
    ("far_service_left", "far_service_right"),
    ("near_service_left", "near_service_right"),
    ("service_center_top", "service_center_bottom"),
    ("net_left", "net_right"),
]

COURT_SUPPORT_LINES = [line for line in COURT_DRAW_LINES if line != ("net_left", "net_right")]


class CourtDetector:
    def __init__(self, reference: CourtReference | None = None) -> None:
        self.reference = reference or CourtReference()

    def detect(self, frame: np.ndarray, line_mask: np.ndarray | None = None) -> CourtDetectionResult:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dominant_hue = self._dominant_hue(hsv)
        mask = self._court_mask(hsv, dominant_hue)
        contour = self._largest_contour(mask)
        corners, source = self._extract_corners(contour)
        return self.build_result_from_corners(
            frame,
            corners,
            source=source,
            line_mask=line_mask,
        )

    def build_result_from_corners(
        self,
        frame: np.ndarray,
        corners: np.ndarray,
        *,
        source: str,
        line_mask: np.ndarray | None = None,
    ) -> CourtDetectionResult:
        ordered_corners = order_points_clockwise(np.asarray(corners, dtype=np.float32))
        if line_mask is None:
            line_mask = self.line_mask(frame)
        refined_corners = self._refine_corners_with_lines(frame, ordered_corners)
        if refined_corners is not None:
            ordered_corners = refined_corners
        homography_court_to_image = cv2.getPerspectiveTransform(
            self.reference.outer_corners,
            ordered_corners.astype(np.float32),
        )
        homography_image_to_court = cv2.getPerspectiveTransform(
            ordered_corners.astype(np.float32),
            self.reference.outer_corners,
        )

        image_keypoints: dict[str, list[float]] = {}
        for name, point in self.reference.keypoints.items():
            projected = project_points(point[None, :], homography_court_to_image)[0]
            image_keypoints[name] = [round(float(projected[0]), 2), round(float(projected[1]), 2)]

        line_support = self._line_support_score(line_mask, image_keypoints)
        shape_score = self._shape_score(ordered_corners, frame.shape)
        total_score = round((0.7 * line_support) + (0.3 * shape_score), 4)

        return CourtDetectionResult(
            image_corners=ordered_corners,
            homography_court_to_image=homography_court_to_image,
            homography_image_to_court=homography_image_to_court,
            image_keypoints=image_keypoints,
            source=source,
            line_support=round(float(line_support), 4),
            shape_score=round(float(shape_score), 4),
            total_score=round(float(total_score), 4),
        )

    def line_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bright_low_sat = ((hsv[..., 1] < 90) & (hsv[..., 2] > 145)).astype(np.uint8) * 255
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            -5,
        )
        mask = cv2.bitwise_and(bright_low_sat, adaptive)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        return mask

    def _dominant_hue(self, hsv: np.ndarray) -> int:
        height, width = hsv.shape[:2]
        crop = hsv[int(height * 0.2) : int(height * 0.9), int(width * 0.15) : int(width * 0.85)]
        valid = (crop[..., 1] > 40) & (crop[..., 2] > 40)
        hues = crop[..., 0][valid]
        if len(hues) == 0:
            raise RuntimeError("Could not estimate a dominant court color from the frame.")
        hist = np.bincount(hues, minlength=180)
        return int(hist.argmax())

    def _court_mask(self, hsv: np.ndarray, dominant_hue: int) -> np.ndarray:
        tolerance = 15
        lower_1 = np.array([max(0, dominant_hue - tolerance), 40, 35], dtype=np.uint8)
        upper_1 = np.array([min(179, dominant_hue + tolerance), 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_1, upper_1)

        if dominant_hue < tolerance:
            lower_2 = np.array([180 - (tolerance - dominant_hue), 40, 35], dtype=np.uint8)
            upper_2 = np.array([179, 255, 255], dtype=np.uint8)
            mask |= cv2.inRange(hsv, lower_2, upper_2)
        elif dominant_hue + tolerance > 179:
            lower_2 = np.array([0, 40, 35], dtype=np.uint8)
            upper_2 = np.array([(dominant_hue + tolerance) - 179, 255, 255], dtype=np.uint8)
            mask |= cv2.inRange(hsv, lower_2, upper_2)

        kernel_small = np.ones((5, 5), np.uint8)
        kernel_large = np.ones((11, 11), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
        return mask

    def _largest_contour(self, mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("Court contour was not found.")
        return max(contours, key=cv2.contourArea)

    def _extract_corners(self, contour: np.ndarray) -> tuple[np.ndarray, str]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True).reshape(-1, 2)
        if len(approx) == 4:
            return order_points_clockwise(approx.astype(np.float32)), "detected_quad"
        if len(approx) > 4:
            outer_quad = self._outer_quad_from_polygon(approx.astype(np.float32))
            if outer_quad is not None:
                return outer_quad, "detected_outer_quad"

        rect = cv2.minAreaRect(contour)
        approx = cv2.boxPoints(rect)
        return order_points_clockwise(approx.astype(np.float32)), "detected_min_area_rect"

    def _outer_quad_from_polygon(self, polygon: np.ndarray) -> np.ndarray | None:
        points = polygon.astype(np.float32)
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1).reshape(-1)
        candidate_points = [
            points[np.argmin(sums)],
            points[np.argmin(diffs)],
            points[np.argmax(sums)],
            points[np.argmax(diffs)],
        ]

        unique_points: list[np.ndarray] = []
        for point in candidate_points:
            if not any(np.linalg.norm(point - existing) < 3.0 for existing in unique_points):
                unique_points.append(point)

        if len(unique_points) != 4:
            return None
        return order_points_clockwise(np.array(unique_points, dtype=np.float32))

    def _line_support_score(
        self,
        line_mask: np.ndarray,
        image_keypoints: dict[str, list[float]],
    ) -> float:
        expanded = cv2.dilate(line_mask, np.ones((5, 5), np.uint8), iterations=1)
        height, width = expanded.shape[:2]
        scores: list[float] = []

        for start_name, end_name in COURT_SUPPORT_LINES:
            start = np.array(image_keypoints[start_name], dtype=np.float32)
            end = np.array(image_keypoints[end_name], dtype=np.float32)
            length = float(np.linalg.norm(end - start))
            sample_count = max(int(length / 10.0), 12)
            hits = 0
            total = 0

            for x, y in np.linspace(start, end, sample_count):
                ix = int(round(float(x)))
                iy = int(round(float(y)))
                if ix < 0 or iy < 0 or ix >= width or iy >= height:
                    continue
                total += 1
                if expanded[iy, ix] > 0:
                    hits += 1

            if total >= 8:
                scores.append(hits / total)

        if not scores:
            return 0.0
        return float(np.mean(scores))

    def _refine_corners_with_lines(
        self,
        frame: np.ndarray,
        corners: np.ndarray,
    ) -> np.ndarray | None:
        edge_mask = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 80, 180)
        height, width = edge_mask.shape[:2]
        polygon_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(polygon_mask, corners.astype(np.int32), 255)
        polygon_mask = cv2.dilate(polygon_mask, np.ones((21, 21), np.uint8), iterations=1)

        edge_specs = [
            (corners[0], corners[1], 18),
            (corners[1], corners[2], 20),
            (corners[3], corners[2], 22),
            (corners[0], corners[3], 20),
        ]

        fitted_lines = [
            self._fit_edge_line(start, end, edge_mask, polygon_mask, band_width)
            for start, end, band_width in edge_specs
        ]
        if any(line is None for line in fitted_lines):
            return None

        top_line, right_line, bottom_line, left_line = fitted_lines
        intersections = [
            self._intersect_lines(top_line, left_line),
            self._intersect_lines(top_line, right_line),
            self._intersect_lines(bottom_line, right_line),
            self._intersect_lines(bottom_line, left_line),
        ]
        if any(point is None for point in intersections):
            return None

        refined = np.array(intersections, dtype=np.float32)
        if not np.isfinite(refined).all():
            return None
        if cv2.contourArea(refined) <= 0:
            return None
        if np.mean(np.linalg.norm(refined - corners, axis=1)) > 30.0:
            return None
        if np.max(np.linalg.norm(refined - corners, axis=1)) > 45.0:
            return None

        x_min = float(refined[:, 0].min())
        x_max = float(refined[:, 0].max())
        y_min = float(refined[:, 1].min())
        y_max = float(refined[:, 1].max())
        if x_min < -(0.1 * width) or x_max > (1.1 * width):
            return None
        if y_min < -(0.1 * height) or y_max > (1.1 * height):
            return None
        return order_points_clockwise(refined)

    def _fit_edge_line(
        self,
        start: np.ndarray,
        end: np.ndarray,
        edge_mask: np.ndarray,
        polygon_mask: np.ndarray,
        band_width: int,
    ) -> np.ndarray | None:
        height, width = edge_mask.shape[:2]
        band_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.line(
            band_mask,
            tuple(int(v) for v in start),
            tuple(int(v) for v in end),
            255,
            thickness=band_width,
        )
        candidate_mask = cv2.bitwise_and(edge_mask, band_mask)
        candidate_mask = cv2.bitwise_and(candidate_mask, polygon_mask)

        target_angle = math.degrees(math.atan2(float(end[1] - start[1]), float(end[0] - start[0])))
        segment_length = float(np.linalg.norm(end - start))
        lines = cv2.HoughLinesP(
            candidate_mask,
            1,
            np.pi / 180.0,
            threshold=15,
            minLineLength=max(int(segment_length * 0.25), 18),
            maxLineGap=10,
        )
        if lines is None:
            return self._line_from_points(start, end)

        best_segment: np.ndarray | None = None
        best_key: tuple[float, float] | None = None
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(value) for value in line]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angle_diff = abs((angle - target_angle + 90.0) % 180.0 - 90.0)
            if angle_diff > 15.0:
                continue

            length = math.hypot(x2 - x1, y2 - y1)
            key = (angle_diff, -length)
            if best_key is None or key < best_key:
                best_key = key
                best_segment = np.array([[x1, y1], [x2, y2]], dtype=np.float32)

        if best_segment is None:
            return self._line_from_points(start, end)
        return self._line_from_points(best_segment[0], best_segment[1])

    def _line_from_points(self, start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        a = float(dy)
        b = float(-dx)
        c = float((dx * start[1]) - (dy * start[0]))
        return np.array([a, b, c], dtype=np.float32)

    def _intersect_lines(self, line_a: np.ndarray | None, line_b: np.ndarray | None) -> np.ndarray | None:
        if line_a is None or line_b is None:
            return None
        matrix = np.array(
            [[line_a[0], line_a[1]], [line_b[0], line_b[1]]],
            dtype=np.float32,
        )
        values = np.array([-line_a[2], -line_b[2]], dtype=np.float32)
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) < 1e-6:
            return None
        intersection = np.linalg.solve(matrix, values)
        return intersection.astype(np.float32)

    def _shape_score(self, corners: np.ndarray, frame_shape: tuple[int, ...]) -> float:
        height, width = frame_shape[:2]
        top = np.linalg.norm(corners[1] - corners[0])
        bottom = np.linalg.norm(corners[2] - corners[3])
        left = np.linalg.norm(corners[3] - corners[0])
        right = np.linalg.norm(corners[2] - corners[1])

        area_fraction = cv2.contourArea(corners) / max(width * height, 1)
        aspect_ratio = ((top + bottom) / 2.0) / max(((left + right) / 2.0), 1e-6)
        border_hits = sum(
            1
            for x, y in corners
            if x <= 3 or y <= 3 or x >= width - 4 or y >= height - 4
        )

        area_score = max(0.0, 1.0 - abs(area_fraction - 0.28) / 0.18)
        aspect_score = max(0.0, 1.0 - abs(aspect_ratio - 1.9) / 1.3)
        border_score = max(0.0, 1.0 - (border_hits / 4.0))
        return float((0.45 * area_score) + (0.35 * aspect_score) + (0.20 * border_score))


class CourtTracker:
    def __init__(self, detector: CourtDetector) -> None:
        self.detector = detector
        self.previous_result: CourtDetectionResult | None = None

    def detect(self, frame: np.ndarray) -> CourtDetectionResult:
        line_mask = self.detector.line_mask(frame)
        detected: CourtDetectionResult | None
        try:
            detected = self.detector.detect(frame, line_mask=line_mask)
        except RuntimeError:
            detected = None

        if self.previous_result is None:
            if detected is None:
                raise RuntimeError("Court could not be detected.")
            self.previous_result = detected
            return detected

        previous_candidate = self.detector.build_result_from_corners(
            frame,
            self.previous_result.image_corners,
            source="previous",
            line_mask=line_mask,
        )
        if detected is None:
            self.previous_result = previous_candidate
            return previous_candidate

        temporal_score = self._temporal_score(detected.image_corners, self.previous_result.image_corners, frame.shape)
        use_previous = False

        if detected.source == "detected_min_area_rect":
            use_previous = temporal_score < 0.78 or detected.total_score <= previous_candidate.total_score + 0.02
        elif detected.total_score + 0.08 < previous_candidate.total_score and temporal_score < 0.45:
            use_previous = True
        elif detected.total_score < 0.20 and previous_candidate.total_score >= detected.total_score:
            use_previous = True

        if use_previous:
            self.previous_result = previous_candidate
            return previous_candidate

        smoothed_corners = self._smooth_corners(detected.image_corners, detected.total_score)
        smoothed = self.detector.build_result_from_corners(
            frame,
            smoothed_corners,
            source=f"{detected.source}_smoothed",
            line_mask=line_mask,
        )
        self.previous_result = smoothed
        return smoothed

    def _temporal_score(
        self,
        candidate_corners: np.ndarray,
        previous_corners: np.ndarray,
        frame_shape: tuple[int, ...],
    ) -> float:
        height, width = frame_shape[:2]
        diagonal = float(np.hypot(width, height))
        mean_distance = float(np.mean(np.linalg.norm(candidate_corners - previous_corners, axis=1)))
        return max(0.0, 1.0 - (mean_distance / max(diagonal * 0.18, 1.0)))

    def _smooth_corners(self, candidate_corners: np.ndarray, candidate_score: float) -> np.ndarray:
        if self.previous_result is None:
            return candidate_corners

        alpha = 0.55 + (0.25 * max(min(candidate_score, 1.0), 0.0))
        alpha = max(0.45, min(alpha, 0.85))
        return (
            ((1.0 - alpha) * self.previous_result.image_corners)
            + (alpha * candidate_corners)
        ).astype(np.float32)


class PlayerDetector:
    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        *,
        device: str | None = None,
        imgsz: int = 640,
        require_cuda: bool = False,
    ) -> None:
        runtime = detect_runtime(require_cuda=require_cuda)
        self.model = YOLO(model_name)
        self.device = device or runtime.ultralytics_device
        self.imgsz = imgsz

    def detect(
        self,
        frame: np.ndarray,
        court: CourtDetectionResult,
        reference: CourtReference | None = None,
    ) -> list[PlayerDetectionResult]:
        reference = reference or CourtReference()
        result = self.model.predict(
            frame,
            classes=[0],
            conf=0.18,
            max_det=8,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        detections: list[PlayerDetectionResult] = []

        for box in result.boxes:
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            bottom_center = np.array([[(x1 + x2) / 2.0, y2]], dtype=np.float32)
            court_point = project_points(bottom_center, court.homography_image_to_court)[0]
            if not self._is_possible_player(court_point, reference):
                continue
            detections.append(
                PlayerDetectionResult(
                    label="candidate",
                    bbox_xyxy=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    image_xy=[round(float(bottom_center[0][0]), 2), round(float(bottom_center[0][1]), 2)],
                    court_xy_m=[round(float(court_point[0]), 2), round(float(court_point[1]), 2)],
                    confidence=round(float(box.conf[0]), 4),
                )
            )

        if not detections:
            return []

        return self._select_relevant_players(detections, reference)

    def _is_possible_player(self, court_point: np.ndarray, reference: CourtReference) -> bool:
        x, y = float(court_point[0]), float(court_point[1])
        return -1.0 <= x <= (reference.width_m + 1.0) and -2.0 <= y <= (reference.length_m + 8.0)

    def _select_relevant_players(
        self,
        detections: list[PlayerDetectionResult],
        reference: CourtReference,
    ) -> list[PlayerDetectionResult]:
        ordered = sorted(detections, key=lambda item: item.court_xy_m[1])
        if len(ordered) == 1:
            ordered[0].label = "player"
            return ordered

        midline_y = reference.length_m / 2.0
        far_candidates = [item for item in ordered if item.court_xy_m[1] <= (midline_y + 1.5)]
        near_candidates = [item for item in ordered if item.court_xy_m[1] >= (midline_y - 1.5)]

        far_player = self._best_candidate(far_candidates) if far_candidates else ordered[0]
        near_player = self._best_candidate(near_candidates) if near_candidates else ordered[-1]

        if far_player is near_player:
            far_player = ordered[0]
            near_player = ordered[-1]

        far_player.label = "far_player"
        near_player.label = "near_player"
        return [far_player, near_player]

    def _best_candidate(self, detections: list[PlayerDetectionResult]) -> PlayerDetectionResult:
        return max(detections, key=self._candidate_score)

    def _candidate_score(self, detection: PlayerDetectionResult) -> float:
        x1, y1, x2, y2 = detection.bbox_xyxy
        area = max((x2 - x1) * (y2 - y1), 1.0)
        area_bonus = min(area / 80000.0, 1.0) * 0.12
        return float(detection.confidence) + area_bonus


@dataclass
class PlayerTrackState:
    detection: PlayerDetectionResult
    missing_frames: int = 0


class PlayerTrackerState:
    def __init__(
        self,
        reference: CourtReference | None = None,
        *,
        max_missing_frames: int = 4,
        smoothing_alpha: float = 0.68,
    ) -> None:
        self.reference = reference or CourtReference()
        self.max_missing_frames = max_missing_frames
        self.smoothing_alpha = smoothing_alpha
        self.tracks: dict[str, PlayerTrackState] = {}

    def update(self, detections: list[PlayerDetectionResult]) -> list[PlayerDetectionResult]:
        assignments = self._assign_labels(detections)
        active_labels: set[str] = set()
        stabilized: list[PlayerDetectionResult] = []

        for label in ("far_player", "near_player"):
            detection = assignments.get(label)
            previous_state = self.tracks.get(label)

            if detection is not None:
                detection.label = label
                smoothed = self._smooth_detection(previous_state.detection if previous_state else None, detection)
                smoothed.source = "detected" if previous_state is None else "smoothed"
                self.tracks[label] = PlayerTrackState(detection=smoothed, missing_frames=0)
                stabilized.append(self._copy_detection(smoothed))
                active_labels.add(label)
                continue

            if previous_state is None or previous_state.missing_frames >= self.max_missing_frames:
                continue

            carried = self._copy_detection(previous_state.detection)
            carried.label = label
            carried.confidence = round(max(previous_state.detection.confidence * 0.88, 0.05), 4)
            carried.source = "carried"
            self.tracks[label] = PlayerTrackState(
                detection=carried,
                missing_frames=previous_state.missing_frames + 1,
            )
            stabilized.append(carried)
            active_labels.add(label)

        for label in list(self.tracks):
            if label not in active_labels:
                self.tracks.pop(label, None)

        stabilized.sort(key=lambda item: item.court_xy_m[1])
        return stabilized

    def _assign_labels(self, detections: list[PlayerDetectionResult]) -> dict[str, PlayerDetectionResult]:
        if not detections:
            return {}

        ordered = sorted(
            (self._copy_detection(item) for item in detections),
            key=lambda item: item.court_xy_m[1],
        )
        if len(ordered) == 1:
            label = self._single_detection_label(ordered[0])
            return {label: ordered[0]}

        default_assignment = {
            "far_player": ordered[0],
            "near_player": ordered[-1],
        }
        if set(self.tracks) >= {"far_player", "near_player"}:
            swapped_assignment = {
                "far_player": ordered[-1],
                "near_player": ordered[0],
            }
            if self._assignment_cost(swapped_assignment) + 0.25 < self._assignment_cost(default_assignment):
                return swapped_assignment
        return default_assignment

    def _single_detection_label(self, detection: PlayerDetectionResult) -> str:
        if set(self.tracks) >= {"far_player", "near_player"}:
            far_distance = self._court_distance(detection, self.tracks["far_player"].detection)
            near_distance = self._court_distance(detection, self.tracks["near_player"].detection)
            return "far_player" if far_distance <= near_distance else "near_player"
        if "far_player" in self.tracks and "near_player" not in self.tracks:
            return "far_player"
        if "near_player" in self.tracks and "far_player" not in self.tracks:
            return "near_player"
        return "near_player" if detection.court_xy_m[1] >= (self.reference.length_m / 2.0) else "far_player"

    def _assignment_cost(self, assignment: dict[str, PlayerDetectionResult]) -> float:
        total = 0.0
        for label, detection in assignment.items():
            previous_state = self.tracks.get(label)
            if previous_state is None:
                continue
            total += self._court_distance(detection, previous_state.detection)
        return total

    def _court_distance(self, first: PlayerDetectionResult, second: PlayerDetectionResult) -> float:
        first_point = np.array(first.court_xy_m, dtype=np.float32)
        second_point = np.array(second.court_xy_m, dtype=np.float32)
        return float(np.linalg.norm(first_point - second_point))

    def _smooth_detection(
        self,
        previous: PlayerDetectionResult | None,
        current: PlayerDetectionResult,
    ) -> PlayerDetectionResult:
        if previous is None:
            return self._copy_detection(current)

        alpha = self.smoothing_alpha
        if current.label == "near_player":
            alpha = min(0.9, alpha + 0.12)
        return PlayerDetectionResult(
            label=current.label,
            bbox_xyxy=self._blend_list(previous.bbox_xyxy, current.bbox_xyxy, alpha),
            image_xy=self._blend_list(previous.image_xy, current.image_xy, alpha),
            court_xy_m=self._blend_list(previous.court_xy_m, current.court_xy_m, alpha),
            confidence=round(float(max(current.confidence, previous.confidence * 0.85)), 4),
            source="smoothed",
        )

    def _blend_list(self, previous: list[float], current: list[float], alpha: float) -> list[float]:
        previous_array = np.array(previous, dtype=np.float32)
        current_array = np.array(current, dtype=np.float32)
        blended = ((1.0 - alpha) * previous_array) + (alpha * current_array)
        return [round(float(value), 2) for value in blended]

    def _copy_detection(self, detection: PlayerDetectionResult) -> PlayerDetectionResult:
        return PlayerDetectionResult(
            label=detection.label,
            bbox_xyxy=list(detection.bbox_xyxy),
            image_xy=list(detection.image_xy),
            court_xy_m=list(detection.court_xy_m),
            confidence=float(detection.confidence),
            source=detection.source,
        )


class TrackNetBallDetector:
    def __init__(
        self,
        weights_path: str | Path,
        *,
        device: str | None = None,
        batch_size: int = 12,
        require_cuda: bool = False,
    ) -> None:
        runtime = detect_runtime(require_cuda=require_cuda)
        self.device = device or runtime.torch_device
        self.use_half = runtime.use_half and self.device.startswith("cuda")
        self.batch_size = batch_size
        self.model = BallTrackNet(out_channels=2)
        checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        if self.use_half:
            self.model.half()
        self.model.eval()
        self.video_width: int | None = None
        self.video_height: int | None = None

    def detect(self, frame: np.ndarray) -> list[float] | None:
        if self.video_width is None:
            self.video_height, self.video_width = frame.shape[:2]

        raise RuntimeError("Use detect_sequence() for ball inference.")

    def detect_sequence(
        self,
        frames: list[np.ndarray],
        *,
        frame_stride: int = 1,
        interpolate: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> list[list[float] | None]:
        if not frames:
            return []

        if self.video_width is None:
            self.video_height, self.video_width = frames[0].shape[:2]

        frame_stride = max(int(frame_stride), 1)
        raw_detections: list[list[float] | None] = [None] * len(frames)
        pending_inputs: list[np.ndarray] = []
        pending_indices: list[int] = []
        target_indices = list(range(2, len(frames), frame_stride))
        total_batches = max(math.ceil(len(target_indices) / max(self.batch_size, 1)), 1)
        completed_batches = 0

        for frame_index in target_indices:
            pending_inputs.append(
                stack_three_frames(
                    frames[frame_index - 2],
                    frames[frame_index - 1],
                    frames[frame_index],
                )
            )
            pending_indices.append(frame_index)

            if len(pending_inputs) >= self.batch_size:
                for detection_index, detection in zip(pending_indices, self._run_batch(pending_inputs)):
                    raw_detections[detection_index] = detection
                pending_inputs = []
                pending_indices = []
                completed_batches += 1
                report_progress(
                    progress_callback,
                    stage="ball_inference",
                    current=completed_batches,
                    total=total_batches,
                    message="Running batched ball inference",
                )

        if pending_inputs:
            for detection_index, detection in zip(pending_indices, self._run_batch(pending_inputs)):
                raw_detections[detection_index] = detection
            completed_batches += 1
            report_progress(
                progress_callback,
                stage="ball_inference",
                current=completed_batches,
                total=total_batches,
                message="Running batched ball inference",
            )

        if interpolate and frame_stride > 1:
            return self._interpolate_sparse_detections(raw_detections, max_gap_frames=frame_stride)
        return raw_detections

    def _run_batch(self, stacked_frames: list[np.ndarray]) -> list[list[float] | None]:
        batch = torch.from_numpy(np.stack(stacked_frames, axis=0)).to(self.device)
        batch = batch.half() if self.use_half else batch.float()
        batch = batch / 255.0
        with torch.inference_mode():
            output = self.model(batch, testing=True)
            output = output.argmax(dim=1).detach().cpu().numpy()
            if self.model.out_channels == 2:
                output *= 255

        detections: list[list[float] | None] = []
        for heatmap in output:
            x, y = self.model._extract_ball_center(heatmap)
            if x is None or y is None:
                detections.append(None)
                continue
            scaled_x = int(x * (self.video_width / 640))
            scaled_y = int(y * (self.video_height / 360))
            detections.append([round(float(scaled_x), 2), round(float(scaled_y), 2)])
        return detections

    def detect_streaming(self, frame_triplet: tuple[np.ndarray, np.ndarray, np.ndarray]) -> list[float] | None:
        if self.video_width is None:
            self.video_height, self.video_width = frame_triplet[-1].shape[:2]

        inputs = stack_three_frames(*frame_triplet)
        tensor = torch.from_numpy(inputs).to(self.device)
        tensor = tensor.half() if self.use_half else tensor.float()
        tensor = tensor / 255.0
        x, y = self.model.inference(tensor)
        if x is None or y is None:
            return None

        x = int(x * (self.video_width / 640))
        y = int(y * (self.video_height / 360))
        return [round(float(x), 2), round(float(y), 2)]

    def _interpolate_sparse_detections(
        self,
        detections: list[list[float] | None],
        *,
        max_gap_frames: int,
    ) -> list[list[float] | None]:
        filled = list(detections)
        previous_index: int | None = None
        previous_point: np.ndarray | None = None

        for index, detection in enumerate(detections):
            if detection is None:
                continue

            point = np.array(detection, dtype=np.float32)
            if (
                previous_index is not None
                and previous_point is not None
                and 1 < (index - previous_index) <= (max_gap_frames + 1)
            ):
                gap = index - previous_index
                distance = float(np.linalg.norm(point - previous_point))
                if distance <= (110.0 * gap):
                    for missing_index in range(previous_index + 1, index):
                        alpha = (missing_index - previous_index) / gap
                        interpolated = ((1.0 - alpha) * previous_point) + (alpha * point)
                        filled[missing_index] = [round(float(interpolated[0]), 2), round(float(interpolated[1]), 2)]

            previous_index = index
            previous_point = point

        return filled


class BallTrajectoryFilter:
    def __init__(
        self,
        reference: CourtReference | None = None,
        *,
        max_prediction_error_px: float = 135.0,
        smoothing_alpha: float = 0.72,
        reset_after_missing: int = 2,
    ) -> None:
        self.reference = reference or CourtReference()
        self.max_prediction_error_px = max_prediction_error_px
        self.smoothing_alpha = smoothing_alpha
        self.reset_after_missing = reset_after_missing
        self.last_image_xy: np.ndarray | None = None
        self.last_velocity_xy: np.ndarray | None = None
        self.missing_frames = 0

    def update(
        self,
        ball_xy: list[float] | None,
        court: CourtDetectionResult,
    ) -> dict[str, Any]:
        if ball_xy is None:
            self.missing_frames += 1
            if self.missing_frames > self.reset_after_missing:
                self.last_image_xy = None
                self.last_velocity_xy = None
            return {
                "image_xy": None,
                "court_xy_m": None,
                "source": "missing",
            }

        candidate_image = np.array(ball_xy, dtype=np.float32)
        candidate_court = project_points(
            candidate_image.reshape(1, 2),
            court.homography_image_to_court,
        )[0]
        if not self._is_reasonable_court_point(candidate_court):
            self.missing_frames += 1
            return {
                "image_xy": None,
                "court_xy_m": None,
                "source": "rejected_outside_court",
            }

        source = "detected"
        if self.last_image_xy is not None:
            predicted = self.last_image_xy
            if self.last_velocity_xy is not None:
                predicted = self.last_image_xy + self.last_velocity_xy
            prediction_error = float(np.linalg.norm(candidate_image - predicted))
            if prediction_error > self.max_prediction_error_px:
                self.missing_frames += 1
                self.last_velocity_xy = None
                return {
                    "image_xy": None,
                    "court_xy_m": None,
                    "source": "rejected_jump",
                }

            source = "smoothed"
            candidate_image = (
                ((1.0 - self.smoothing_alpha) * self.last_image_xy)
                + (self.smoothing_alpha * candidate_image)
            ).astype(np.float32)
            candidate_court = project_points(
                candidate_image.reshape(1, 2),
                court.homography_image_to_court,
            )[0]

        velocity = None if self.last_image_xy is None else candidate_image - self.last_image_xy
        self.last_velocity_xy = velocity
        self.last_image_xy = candidate_image
        self.missing_frames = 0
        return {
            "image_xy": [round(float(candidate_image[0]), 2), round(float(candidate_image[1]), 2)],
            "court_xy_m": [round(float(candidate_court[0]), 2), round(float(candidate_court[1]), 2)],
            "source": source,
        }

    def _is_reasonable_court_point(self, court_point: np.ndarray) -> bool:
        x, y = float(court_point[0]), float(court_point[1])
        return -1.2 <= x <= (self.reference.width_m + 1.2) and -4.0 <= y <= (self.reference.length_m + 4.0)


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    reshaped = points.reshape(-1, 1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(reshaped, homography)
    return projected.reshape(-1, 2)


def order_points_clockwise(points: np.ndarray) -> np.ndarray:
    if len(points) != 4:
        raise ValueError(f"Expected 4 points, got {len(points)}")

    points = points.astype(np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_left = points[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def draw_court_overlay(frame: np.ndarray, court: CourtDetectionResult) -> np.ndarray:
    overlay = frame.copy()
    corners = court.image_corners.astype(int)
    cv2.polylines(overlay, [corners], True, (0, 255, 255), 2)

    for start_name, end_name in COURT_DRAW_LINES:
        start = tuple(int(v) for v in court.image_keypoints[start_name])
        end = tuple(int(v) for v in court.image_keypoints[end_name])
        cv2.line(overlay, start, end, (0, 255, 255), 1)
    return overlay


def draw_players_overlay(frame: np.ndarray, players: list[PlayerDetectionResult]) -> np.ndarray:
    overlay = frame.copy()
    for player in players:
        x1, y1, x2, y2 = [int(v) for v in player.bbox_xyxy]
        px, py = [int(v) for v in player.image_xy]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.circle(overlay, (px, py), 4, (0, 220, 0), -1)
        label = f"{player.label} {player.court_xy_m[0]:.1f},{player.court_xy_m[1]:.1f}m"
        cv2.putText(overlay, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
    return overlay


def draw_ball_overlay(frame: np.ndarray, ball_xy: list[float] | None) -> np.ndarray:
    overlay = frame.copy()
    if ball_xy is None:
        return overlay
    x, y = [int(v) for v in ball_xy]
    cv2.circle(overlay, (x, y), 7, (0, 165, 255), 2)
    cv2.putText(overlay, "ball", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
    return overlay


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def format_number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def report_progress(
    progress_callback: ProgressCallback | None,
    *,
    stage: str,
    current: int,
    total: int,
    message: str,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        {
            "stage": stage,
            "current": int(current),
            "total": max(int(total), 1),
            "message": message,
        }
    )


def write_xml_output(
    *,
    output_path: Path,
    video_path: Path,
    fps: float,
    frame_size: dict[str, int],
    frames_payload: list[dict[str, Any]],
) -> None:
    root = ET.Element(
        "match",
        {
            "source_video": video_path.name,
            "fps": format_number(float(fps)),
            "width": str(frame_size["width"]),
            "height": str(frame_size["height"]),
        },
    )

    end_frame = frames_payload[-1]["frame_index"] if frames_payload else -1
    point = ET.SubElement(
        root,
        "point",
        {
            "id": "1",
            "start_frame": "0",
            "end_frame": str(end_frame),
        },
    )

    if frames_payload:
        first_court = frames_payload[0]["court"]
        court_node = ET.SubElement(
            point,
            "court",
            {
                "frame": str(frames_payload[0]["frame_index"]),
                "source": first_court.get("source", "unknown"),
                "score": format_number(float(first_court.get("score", 0.0))),
            },
        )
        corner_names = [
            "outer_top_left",
            "outer_top_right",
            "outer_bottom_right",
            "outer_bottom_left",
        ]
        for name, (x, y) in zip(corner_names, first_court["image_corners"]):
            ET.SubElement(
                court_node,
                "corner",
                {
                    "name": name,
                    "image_x": format_number(float(x)),
                    "image_y": format_number(float(y)),
                },
            )

    event_id = 1
    for frame in frames_payload:
        frame_index = frame["frame_index"]
        timestamp = frame["timestamp_sec"]
        for player in frame.get("players", []):
            ET.SubElement(
                point,
                "event",
                {
                    "id": str(event_id),
                    "kind": "player_position",
                    "frame": str(frame_index),
                    "time": format_number(float(timestamp)),
                    "player": player["label"],
                    "confidence": format_number(float(player["confidence"])),
                    "image_x": format_number(float(player["image_xy"][0])),
                    "image_y": format_number(float(player["image_xy"][1])),
                    "court_x": format_number(float(player["court_xy_m"][0])),
                    "court_y": format_number(float(player["court_xy_m"][1])),
                },
            )
            event_id += 1

        ball = frame.get("ball")
        if ball is None or ball.get("image_xy") is None:
            continue

        attributes = {
            "id": str(event_id),
            "kind": "ball_observation",
            "frame": str(frame_index),
            "time": format_number(float(timestamp)),
            "image_x": format_number(float(ball["image_xy"][0])),
            "image_y": format_number(float(ball["image_xy"][1])),
        }
        if ball.get("court_xy_m") is not None:
            attributes["court_x"] = format_number(float(ball["court_xy_m"][0]))
            attributes["court_y"] = format_number(float(ball["court_xy_m"][1]))

        ET.SubElement(point, "event", attributes)
        event_id += 1

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def finalize_video_for_playback(temp_video_path: Path, final_video_path: Path) -> None:
    """Re-encode OpenCV output into a broadly compatible H.264 MP4."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        temp_video_path.replace(final_video_path)
        return

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(temp_video_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(final_video_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        temp_video_path.replace(final_video_path)
        return

    temp_video_path.unlink(missing_ok=True)


def run_pipeline(
    *,
    video_path: str | Path,
    output_dir: str | Path,
    step: str,
    weights_path: str | Path,
    max_frames: int | None = None,
    player_detect_stride: int = 2,
    court_detect_stride: int = 1,
    ball_detect_stride: int = 1,
    player_model_imgsz: int = 640,
    player_smoothing_alpha: float = 0.68,
    ball_batch_size: int = 12,
    write_video: bool = True,
    write_xml: bool = True,
    write_events: bool = True,
    output_stem: str | None = None,
    require_cuda: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Path]:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    output_stem = output_stem or step
    court_detect_stride = max(int(court_detect_stride), 1)
    ball_detect_stride = max(int(ball_detect_stride), 1)

    reference = CourtReference()
    court_tracker = CourtTracker(CourtDetector(reference))
    player_detector = (
        PlayerDetector(
            require_cuda=require_cuda,
            imgsz=player_model_imgsz,
        )
        if step in {"players", "all"}
        else None
    )
    player_tracker = (
        PlayerTrackerState(reference, smoothing_alpha=float(player_smoothing_alpha))
        if player_detector is not None
        else None
    )
    ball_detector = (
        TrackNetBallDetector(
            weights_path,
            require_cuda=require_cuda,
            batch_size=ball_batch_size,
        )
        if step in {"ball", "all"}
        else None
    )
    ball_filter = BallTrajectoryFilter(reference) if ball_detector is not None else None

    output_video_path = output_dir / f"{output_stem}.mp4"
    temp_video_path = output_dir / f"{output_stem}.opencv.mp4"
    output_json_path = output_dir / f"{output_stem}.json"
    output_xml_path = output_dir / f"{output_stem}.xml"
    output_events_path = output_dir / f"{output_stem}_events.json"
    writer = None
    if write_video:
        writer = cv2.VideoWriter(
            str(temp_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    frames: list[np.ndarray] = []
    estimated_frame_total = min(source_frame_count, max_frames) if max_frames is not None and source_frame_count else max_frames
    if estimated_frame_total is None:
        estimated_frame_total = max(source_frame_count, 1)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if max_frames is not None and len(frames) >= max_frames:
            break
        frames.append(frame)
        if len(frames) == 1 or (len(frames) % 25) == 0:
            report_progress(
                progress_callback,
                stage="load_frames",
                current=len(frames),
                total=estimated_frame_total,
                message="Loading video frames",
            )

    report_progress(
        progress_callback,
        stage="load_frames",
        current=len(frames),
        total=max(len(frames), 1),
        message="Loading video frames",
    )

    frames_payload: list[dict[str, Any]] = []
    ball_detections = (
        ball_detector.detect_sequence(
            frames,
            frame_stride=ball_detect_stride,
            progress_callback=progress_callback,
        )
        if ball_detector is not None
        else [None] * len(frames)
    )
    last_court: CourtDetectionResult | None = None

    for frame_index, frame in enumerate(frames):
        should_detect_court = (
            last_court is None
            or court_detect_stride <= 1
            or (frame_index % court_detect_stride) == 0
        )
        if should_detect_court:
            last_court = court_tracker.detect(frame)
        court = last_court
        if court is None:
            raise RuntimeError(f"Court detection failed at frame {frame_index}.")
        overlay = None
        if writer is not None:
            overlay = draw_court_overlay(frame, court) if step in {"court", "players", "all"} else frame.copy()

        payload: dict[str, Any] = {
            "frame_index": frame_index,
            "timestamp_sec": round(frame_index / fps, 3),
            "court": {
                "image_corners": [[round(float(x), 2), round(float(y), 2)] for x, y in court.image_corners],
                "image_keypoints": court.image_keypoints,
                "source": court.source,
                "line_support": court.line_support,
                "shape_score": court.shape_score,
                "score": court.total_score,
            },
        }

        if player_detector is not None:
            should_detect_players = (
                frame_index == 0
                or player_tracker is None
                or player_detect_stride <= 1
                or (frame_index % player_detect_stride) == 0
            )
            players = player_detector.detect(frame, court, reference) if should_detect_players else []
            if player_tracker is not None:
                players = player_tracker.update(players)
            if overlay is not None:
                overlay = draw_players_overlay(overlay, players)
            payload["players"] = [player.__dict__ for player in players]

        if ball_detector is not None:
            ball_xy = ball_detections[frame_index]
            ball_payload = ball_filter.update(ball_xy, court) if ball_filter is not None else {
                "image_xy": ball_xy,
                "court_xy_m": None,
                "source": "detected" if ball_xy is not None else "missing",
            }
            if overlay is not None:
                overlay = draw_ball_overlay(overlay, ball_payload["image_xy"])
            payload["ball"] = ball_payload

        if writer is not None:
            writer.write(overlay)
        frames_payload.append(payload)
        if frame_index == 0 or ((frame_index + 1) % 25) == 0 or (frame_index + 1) == len(frames):
            report_progress(
                progress_callback,
                stage="compose_overlay",
                current=frame_index + 1,
                total=max(len(frames), 1),
                message="Composing overlay frames",
            )

    capture.release()
    if writer is not None:
        report_progress(
            progress_callback,
            stage="finalize_video",
            current=0,
            total=1,
            message="Finalizing playback video",
        )
        writer.release()
        finalize_video_for_playback(temp_video_path, output_video_path)
        report_progress(
            progress_callback,
            stage="finalize_video",
            current=1,
            total=1,
            message="Finalizing playback video",
        )

    output_json_path.write_text(
        json.dumps(
            {
                "video_path": str(video_path),
                "fps": fps,
                "frame_size": {"width": width, "height": height},
                "court_reference_m": {
                    "width": reference.width_m,
                    "length": reference.length_m,
                    "singles_width": reference.singles_width_m,
                },
                "frames": frames_payload,
            },
            indent=2,
            default=json_default,
        )
    )

    if write_xml:
        write_xml_output(
            output_path=output_xml_path,
            video_path=video_path,
            fps=float(fps),
            frame_size={"width": width, "height": height},
            frames_payload=frames_payload,
        )

    outputs: dict[str, Path] = {"json": output_json_path}
    if write_video:
        outputs["video"] = output_video_path
    if write_xml:
        outputs["xml"] = output_xml_path
    if step == "all" and write_events:
        write_match_events_json(
            tracking_json_path=output_json_path,
            output_json_path=output_events_path,
        )
        outputs["events"] = output_events_path

    report_progress(
        progress_callback,
        stage="done",
        current=1,
        total=1,
        message="Overlay pipeline finished",
    )

    return outputs
