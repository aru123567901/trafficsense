"""
TrafficSense - Core vehicle detection, tracking and density module.

Fixes vs the original notebook code:
- Density is now based on the CURRENT frame's vehicle count (not a
  cumulative total), so it actually goes up and down like real traffic.
- A lightweight centroid tracker is used so the same vehicle isn't
  counted multiple times as it crosses the line.
- "total_crossed" (cumulative) and "current_count" (live, used for
  density) are tracked separately - both are useful metrics.
- Per-vehicle-type counts (car / motorcycle / bus / truck) are returned
  every frame for dashboard charts.
"""

import cv2
import numpy as np
from ultralytics import YOLO

# COCO class ids -> vehicle types we care about
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class CentroidTracker:
    """Very small IoU-free tracker based on nearest centroid matching.

    Good enough for a short demo video - assigns a stable ID to each
    vehicle across frames so we can detect line-crossing exactly once.
    """

    def __init__(self, max_distance=60, max_disappeared=15):
        self.next_id = 0
        self.objects = {}        # id -> centroid (x, y)
        self.disappeared = {}    # id -> frames since last seen
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    def update(self, centroids):
        """centroids: list of (x, y). Returns dict id -> centroid."""
        if len(centroids) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.disappeared[oid]
            return {}

        input_centroids = np.array(centroids)

        if len(self.objects) == 0:
            assigned = {}
            for c in input_centroids:
                oid = self.next_id
                self.objects[oid] = c
                self.disappeared[oid] = 0
                assigned[oid] = c
                self.next_id += 1
            return assigned

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        D = np.linalg.norm(
            object_centroids[:, None, :] - input_centroids[None, :, :], axis=2
        )

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        assigned = {}

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            oid = object_ids[row]
            self.objects[oid] = input_centroids[col]
            self.disappeared[oid] = 0
            assigned[oid] = input_centroids[col]
            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(D.shape[0])) - used_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                del self.objects[oid]
                del self.disappeared[oid]

        for col in set(range(D.shape[1])) - used_cols:
            oid = self.next_id
            self.objects[oid] = input_centroids[col]
            self.disappeared[oid] = 0
            assigned[oid] = input_centroids[col]
            self.next_id += 1

        return assigned


class TrafficDetector:
    """Wraps YOLOv8 + tracker + density classification for one video stream."""

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.tracker = CentroidTracker()
        self.total_count = 0
        self.crossed_ids = set()
        self.prev_x = {}  # id -> previous x position (to detect crossing)

    @staticmethod
    def classify_density(current_count):
        """Returns (label, BGR color) - matches the thresholds used
        throughout this project: 0-5 LOW, 6-15 MEDIUM, 16+ HIGH."""
        if current_count <= 5:
            return "LOW", (0, 255, 0)
        elif current_count <= 15:
            return "MEDIUM", (0, 255, 255)
        else:
            return "HIGH", (0, 0, 255)

    def process_frame(self, frame, line_x=None):
        """Run detection on a single frame, draw overlays, return
        (annotated_frame, stats_dict)."""
        h, w = frame.shape[:2]
        if line_x is None:
            line_x = w // 2

        results = self.model(frame, verbose=False, conf=self.conf_threshold)

        type_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        centroids = []

        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = VEHICLE_CLASSES[cls]
            type_counts[label] += 1

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            centroids.append((cx, cy))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.0%}", (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

        # update tracker + check line crossings
        tracked = self.tracker.update(centroids)
        for oid, centroid in tracked.items():
            cx = int(centroid[0])
            prev_cx = self.prev_x.get(oid)
            if prev_cx is not None and oid not in self.crossed_ids:
                if (prev_cx < line_x <= cx) or (prev_cx > line_x >= cx):
                    self.total_count += 1
                    self.crossed_ids.add(oid)
            self.prev_x[oid] = cx

        current_count = sum(type_counts.values())
        density, color = self.classify_density(current_count)

        # draw counting line
        cv2.line(frame, (line_x, 0), (line_x, h), (0, 0, 255), 2)
        cv2.putText(frame, "COUNTING LINE", (min(line_x + 10, w - 150), 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # overlay stats
        cv2.putText(frame, f"Current: {current_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Total crossed: {self.total_count}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Density: {density}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        stats = {
            "current_count": current_count,
            "total_count": self.total_count,
            "density": density,
            **type_counts,
        }
        return frame, stats
