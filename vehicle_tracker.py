#!/usr/bin/env python3
"""
Simple Vehicle Tracking Script for Raspberry Pi
Takes MP4 video as input and tracks vehicles in real-time using YOLOv8
"""

import cv2
import argparse
import json
from ultralytics import YOLO


def main():
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Set up command line arguments
    parser = argparse.ArgumentParser(
        description='Vehicle Tracking with YOLOv8')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input MP4 video file')
    parser.add_argument('--model', type=str, default=config['model'],
                        help=f'YOLOv8 model (default: {config["model"]})')
    parser.add_argument('--conf', type=float, default=config['confidence'],
                        help=f'Confidence threshold (default: {config["confidence"]})')
    args = parser.parse_args()

    # Load YOLOv8 model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Open video file
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} at {fps} FPS, {total_frames} frames")
    print("Press 'q' to quit")

    # Vehicle classes from config
    vehicle_classes = config['vehicle_classes']

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break

        frame_count += 1

        # Run YOLOv8 tracking on the frame
        # persist=True enables tracking across frames
        results = model.track(frame,
                              persist=True,
                              classes=vehicle_classes,
                              conf=args.conf,
                              verbose=False)

        # Draw the results on frame
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Get boxes, track IDs, and classes
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            # Class names from config
            class_names = {int(k): v for k, v in config['class_names'].items()}

            # Draw tracking results
            for i, (box, track_id, cls, conf) in enumerate(zip(boxes, track_ids, classes, confs)):
                x1, y1, x2, y2 = map(int, box)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              config['display']['box_color'], 2)

                # Draw label with track ID
                label = f"{class_names.get(cls, 'vehicle')} ID:{track_id} {conf:.2f}"
                label_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, config['display']['font_scale'], config['display']['font_thickness'])[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                              (x1 + label_size[0], y1), config['display']['box_color'], -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, config['display']['font_scale'], (0, 0, 0), config['display']['font_thickness'])

        # Add frame info
        info_text = f"Frame: {frame_count}/{total_frames}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, config['display']['info_font_scale'], (255, 255, 255), 2)

        # Display frame
        cv2.imshow('Vehicle Tracking', frame)

        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Vehicle tracking completed")


if __name__ == "__main__":
    main()
