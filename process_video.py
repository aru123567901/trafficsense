"""
TrafficSense - batch processing script.

Run on a full video, save an annotated output video and a CSV log
(one row per second) with vehicle counts, density, and per-type counts.

Usage:
    python process_video.py --input traffic.mp4
    python process_video.py --input traffic.mp4 --line 0.5 --conf 0.4
"""

import argparse
import csv
import time

import cv2

from detector import TrafficDetector


def process_video(input_path, output_video="traffic_output.mp4",
                   output_csv="results.csv", line_position=0.5, conf=0.4):
    detector = TrafficDetector(conf_threshold=conf)
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    line_x = int(width * line_position)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    csv_file = open(output_csv, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["time_sec", "current_count", "total_crossed", "density",
                      "car", "motorcycle", "bus", "truck"])

    frame_idx = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, stats = detector.process_frame(frame, line_x=line_x)
        out.write(annotated)

        # log one row per second of video
        if frame_idx % max(int(fps), 1) == 0:
            writer.writerow([
                round(frame_idx / fps, 2),
                stats["current_count"],
                stats["total_count"],
                stats["density"],
                stats["car"], stats["motorcycle"],
                stats["bus"], stats["truck"],
            ])

        frame_idx += 1

    cap.release()
    out.release()
    csv_file.close()

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s | {frame_idx} frames processed")
    print(f"Annotated video -> {output_video}")
    print(f"CSV log         -> {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a traffic video with TrafficSense")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output_video", default="traffic_output.mp4")
    parser.add_argument("--output_csv", default="results.csv")
    parser.add_argument("--line", type=float, default=0.5,
                         help="Counting line position as fraction of frame width (0-1)")
    parser.add_argument("--conf", type=float, default=0.4,
                         help="YOLO detection confidence threshold")
    args = parser.parse_args()

    process_video(args.input, args.output_video, args.output_csv, args.line, args.conf)
