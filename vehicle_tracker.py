#!/usr/bin/env python3
"""
Simple Vehicle Tracking Script for Raspberry Pi
Takes MP4 video as input and tracks vehicles in real-time using YOLOv8
"""

import cv2
import argparse
import json
import time
import csv
import os
from datetime import datetime
from ultralytics import YOLO


def log_violation(vehicle_id, speed_kmh, speed_limit, timestamp, frame_count, config):
    """Log speed violation to CSV file"""
    if not config['violation_logging']['log_violations']:
        return

    violation_file = config['violation_logging']['log_file']
    file_exists = os.path.isfile(violation_file)

    with open(violation_file, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'vehicle_id', 'speed_kmh',
                      'speed_limit', 'violation_amount', 'frame_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'timestamp': timestamp,
            'vehicle_id': vehicle_id,
            'speed_kmh': round(speed_kmh, 1),
            'speed_limit': speed_limit,
            'violation_amount': round(speed_kmh - speed_limit, 1),
            'frame_count': frame_count
        })


def save_violation_screenshot(frame, vehicle_id, speed_kmh, timestamp, config):
    """Save screenshot of violation"""
    if not config['violation_logging']['screenshot_violations']:
        return

    screenshot_dir = config['violation_logging']['screenshot_folder']
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)

    filename = f"violation_{vehicle_id}_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
    filepath = os.path.join(screenshot_dir, filename)
    cv2.imwrite(filepath, frame)


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
    print(f"Speed limit: {config['speed_limit']} km/h")
    print("Press 'q' to quit")

    # Vehicle classes from config
    vehicle_classes = config['vehicle_classes']

    # Speed detection setup
    line1_y = int(height * config['speed_detection']['line1_position'])
    line2_y = int(height * config['speed_detection']['line2_position'])
    distance_meters = config['speed_detection']['distance_meters']
    speed_limit = config['speed_limit']

    # DEBUG: Print line positions
    print(
        f"DEBUG: Line1 at y={line1_y}, Line2 at y={line2_y}, Distance={distance_meters}m")

    # Track vehicle crossings
    # {vehicle_id: {'line1_time': time, 'line2_time': time, 'direction': 'forward/reverse'}}
    vehicle_times = {}
    vehicle_speeds = {}  # {vehicle_id: speed_kmh}

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break

        frame_count += 1
        current_time = frame_count / fps

        # Draw speed detection lines
        cv2.line(frame, (0, line1_y), (width, line1_y), (255, 0, 0), 3)
        cv2.line(frame, (0, line2_y), (width, line2_y), (0, 0, 255), 3)
        cv2.putText(frame, "Line1", (10, line1_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Line2", (10, line2_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Run YOLOv8 tracking on the frame
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

            # Draw tracking results and check line crossings
            for i, (box, track_id, cls, conf) in enumerate(zip(boxes, track_ids, classes, confs)):
                x1, y1, x2, y2 = map(int, box)
                center_y = (y1 + y2) // 2

                # DEBUG: Print vehicle center position
                if frame_count % 30 == 0:  # Print every 30 frames to avoid spam
                    print(f"DEBUG: Vehicle {track_id} center_y={center_y}")

                # Initialize vehicle tracking
                if track_id not in vehicle_times:
                    vehicle_times[track_id] = {}

                # Check if vehicle is near lines
                line1_crossed = abs(center_y - line1_y) < 30
                line2_crossed = abs(center_y - line2_y) < 30

                # Skip if vehicle is already processed
                if track_id in vehicle_speeds:
                    pass
                # First crossing - determine direction
                elif 'direction' not in vehicle_times[track_id]:
                    if line1_crossed:
                        vehicle_times[track_id]['line1_time'] = current_time
                        vehicle_times[track_id]['direction'] = 'forward'
                        print(
                            f"DEBUG: Vehicle {track_id} FORWARD direction, crossed LINE1 at {current_time:.2f}s")
                    elif line2_crossed:
                        vehicle_times[track_id]['line2_time'] = current_time
                        vehicle_times[track_id]['direction'] = 'reverse'
                        print(
                            f"DEBUG: Vehicle {track_id} REVERSE direction, crossed LINE2 at {current_time:.2f}s")

                # Second crossing - calculate speed based on direction
                elif vehicle_times[track_id]['direction'] == 'forward':
                    if line2_crossed and 'line2_time' not in vehicle_times[track_id]:
                        vehicle_times[track_id]['line2_time'] = current_time
                        time_diff = vehicle_times[track_id]['line2_time'] - \
                            vehicle_times[track_id]['line1_time']
                        print(
                            f"DEBUG: Vehicle {track_id} FORWARD completed, crossed LINE2 at {current_time:.2f}s")
                        print(f"DEBUG: Time difference: {time_diff:.2f}s")

                        if time_diff > 0:
                            speed_ms = distance_meters / time_diff
                            speed_kmh = speed_ms * 3.6
                            vehicle_speeds[track_id] = speed_kmh
                            print(
                                f"DEBUG: Vehicle {track_id} speed calculated: {speed_kmh:.1f} km/h")

                            # Check for speed violation and log
                            if speed_kmh > speed_limit:
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                log_violation(
                                    track_id, speed_kmh, speed_limit, timestamp, frame_count, config)
                                save_violation_screenshot(
                                    frame, track_id, speed_kmh, timestamp, config)
                                print(
                                    f"VIOLATION: Vehicle {track_id} speeding at {speed_kmh:.1f} km/h (limit: {speed_limit} km/h)")

                elif vehicle_times[track_id]['direction'] == 'reverse':
                    if line1_crossed and 'line1_time' not in vehicle_times[track_id]:
                        vehicle_times[track_id]['line1_time'] = current_time
                        time_diff = vehicle_times[track_id]['line1_time'] - \
                            vehicle_times[track_id]['line2_time']
                        print(
                            f"DEBUG: Vehicle {track_id} REVERSE completed, crossed LINE1 at {current_time:.2f}s")
                        print(f"DEBUG: Time difference: {time_diff:.2f}s")

                        if time_diff > 0:
                            speed_ms = distance_meters / time_diff
                            speed_kmh = speed_ms * 3.6
                            vehicle_speeds[track_id] = speed_kmh
                            print(
                                f"DEBUG: Vehicle {track_id} speed calculated: {speed_kmh:.1f} km/h")

                            # Check for speed violation and log
                            if speed_kmh > speed_limit:
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                log_violation(
                                    track_id, speed_kmh, speed_limit, timestamp, frame_count, config)
                                save_violation_screenshot(
                                    frame, track_id, speed_kmh, timestamp, config)
                                print(
                                    f"VIOLATION: Vehicle {track_id} speeding at {speed_kmh:.1f} km/h (limit: {speed_limit} km/h)")

                # Draw bounding box - use red for violations, green for normal
                box_color = (
                    0, 0, 255) if track_id in vehicle_speeds and vehicle_speeds[track_id] > speed_limit else config['display']['box_color']
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # Draw label with track ID and speed
                speed_text = f" {vehicle_speeds.get(track_id, 0):.1f} km/h" if track_id in vehicle_speeds else ""
                violation_text = " SPEEDING!" if track_id in vehicle_speeds and vehicle_speeds[
                    track_id] > speed_limit else ""
                label = f"{class_names.get(cls, 'vehicle')} ID:{track_id}{speed_text}{violation_text}"
                label_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, config['display']['font_scale'], config['display']['font_thickness'])[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                              (x1 + label_size[0], y1), box_color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, config['display']['font_scale'], (255, 255, 255), config['display']['font_thickness'])

        # Add frame info and speed limit
        info_text = f"Frame: {frame_count}/{total_frames} | Speed Limit: {speed_limit} km/h"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, config['display']['info_font_scale'], (255, 255, 255), 2)

        # Display frame
        cv2.imshow('Vehicle Tracking', frame)

        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # DEBUG: Print final summary
    print(f"DEBUG: Final vehicle_times: {vehicle_times}")
    print(f"DEBUG: Final vehicle_speeds: {vehicle_speeds}")

    # Print violation summary
    if config['violation_logging']['log_violations']:
        violation_count = sum(
            1 for speed in vehicle_speeds.values() if speed > speed_limit)
        print(f"\nVIOLATION SUMMARY:")
        print(f"Total vehicles tracked: {len(vehicle_speeds)}")
        print(f"Violations detected: {violation_count}")
        print(
            f"Violations logged to: {config['violation_logging']['log_file']}")

        if config['violation_logging']['screenshot_violations'] and violation_count > 0:
            print(
                f"Screenshots saved to: {config['violation_logging']['screenshot_folder']}/")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Vehicle tracking completed")


if __name__ == "__main__":
    main()
