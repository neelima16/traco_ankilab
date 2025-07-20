#!/usr/bin/env python3
import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import ssd300_vgg16

from deep_sort_realtime.deepsort_tracker import DeepSort


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   type=str, required=True, help="Path to your SSD .pth")
    p.add_argument("--video",   type=str, required=True, help="Path to input .mp4")
    p.add_argument("--output",  type=str, required=True, help="Path to save output .mp4")
    p.add_argument("--conf",    type=float, default=0.5, help="Confidence threshold")
    return p.parse_args()


def load_model(path, device, num_classes=5):  # ✅ Corrected: your model expects 5 classes
    print(f"Loading model from {path}")
    model = ssd300_vgg16(weights=None, num_classes=num_classes)

    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    print("Model loaded successfully")
    return model


def detect_frame(model, frame, device, conf_threshold=0.5):
    orig_h, orig_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    resized = F.resize(pil_image, [320, 320])
    tensor = F.to_tensor(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(tensor)

    pred = predictions[0]
    keep = pred["scores"] >= conf_threshold
    if not keep.any():
        return np.empty((0, 4)), np.empty((0,))

    boxes = pred["boxes"][keep].cpu().numpy()
    scores = pred["scores"][keep].cpu().numpy()

    scale_x = orig_w / 320.0
    scale_y = orig_h / 320.0
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    return boxes, scores


def draw_detections(frame, boxes, scores):
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"Hexbug: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
        cv2.putText(frame, text, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
    count_text = f"Detections: {len(boxes)}"
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return frame


def get_color(track_id):
    tid = int(track_id)
    return (
        (tid * 37) % 255,
        (tid * 17) % 255,
        (tid * 29) % 255
    )


def draw_track_lines(frame, track_points):
    for track_id, points in track_points.items():
        color = get_color(track_id)
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], color, thickness=3)
        if points:
            cv2.circle(frame, points[-1], 8, color, -1)
            cv2.putText(frame, f"ID {track_id}", (points[-1][0] + 10, points[-1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.model, device)

    tracker = DeepSort(
        max_age=15,
        n_init=1,
        max_iou_distance=0.95
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {w}x{h}, {fps} FPS, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    frame_count = 0
    total_detections = 0
    track_points = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, scores = detect_frame(model, frame, device, args.conf)

        print(f"Frame {frame_count}: Detected {len(boxes)} boxes")
        if len(boxes) > 0:
            total_detections += len(boxes)
            print(f"Boxes: {boxes}")

        detections = [
            ([float(x1), float(y1), float(x2), float(y2)], float(conf))
            for (x1, y1, x2, y2), conf in zip(boxes, scores)
        ]

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            tid = int(track.track_id)

            if tid not in track_points:
                track_points[tid] = []
            track_points[tid].append((cx, cy))

        frame = draw_track_lines(frame, track_points)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    avg_det = total_detections / float(frame_count) if frame_count > 0 else 0
    print(f"Processed {frame_count} frames")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per frame: {avg_det:.2f}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
