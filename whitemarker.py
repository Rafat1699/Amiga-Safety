from __future__ import annotations

import argparse
import asyncio
import cv2
import numpy as np
from pathlib import Path

from farm_ng.canbus.canbus_pb2 import Twist2d
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file

# --- YOLO Setup for Person Detection ---
YOLO_CONFIG = "yolov4-tiny.cfg"
YOLO_WEIGHTS = "yolov4-tiny.weights"
YOLO_CLASSES = "coco.names"

net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
with open(YOLO_CLASSES, 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]

# --- Control Parameters ---
TARGET_ROW_CENTER = 320  # Assuming 640px wide camera image
FORWARD_SPEED = 0.1  # Constant forward speed (m/s)
MAX_ANGULAR_SPEED = 0.2  # Reduce max turning speed (was 0.3)
STEERING_GAIN = 0.0007  # Reduce gain to avoid oversteering


def detect_person(frame) -> bool:
    """Detect person using YOLOv4-tiny."""
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and CLASSES[class_id] == "person":
                print(f"🚶 Person detected with confidence {confidence:.2f}")
                return True
    return False


def detect_crop_rows(frame):
    """Detects white circular markers (lids) and finds their centerline."""
    
    # Convert to grayscale (white will have high intensity)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a threshold to detect bright white objects
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)  # Adjusted for better detection

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    white_x_positions = []

    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out small objects (remove noise)
        if w > 30 and h > 30:  # Adjust based on marker size
            center_x = x + w // 2
            
            # ✅ Ignore detections that are out of range (possible noise)
            if 0 <= center_x <= 640:  
                white_x_positions.append(center_x)

    # Debugging: Print only valid detections
    print(f"Detected {len(white_x_positions)} markers. Valid X-positions: {white_x_positions}")

    if len(white_x_positions) == 0:
        return None  # No markers detected

    # Compute the centerline as the average x-position of all detected white markers
    row_center = int(np.mean(white_x_positions))

    # Ensure row center stays within a valid range
    row_center = max(min(row_center, 640), 0)

    # Optional: Show the detected mask
    cv2.imshow("White Marker Mask", mask)

    return row_center


async def main(canbus_config_path: Path, camera_config_path: Path):
    # Load service configs
    canbus_config = proto_from_json_file(canbus_config_path, EventServiceConfig())
    camera_config = proto_from_json_file(camera_config_path, EventServiceConfig())

    canbus_client = EventClient(canbus_config)
    camera_client = EventClient(camera_config)

    # Open display window for camera feed
    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

    twist = Twist2d()
    person_detected = False

    # Initial forward speed (robot always moving unless person detected)
    twist.linear_velocity_x = FORWARD_SPEED

    async for event, message in camera_client.subscribe(camera_config.subscriptions[0], decode=True):
        # Decode image from camera
        frame = cv2.imdecode(np.frombuffer(message.image_data, dtype="uint8"), cv2.IMREAD_UNCHANGED)
        cv2.imshow("Camera Feed", frame)

        if detect_person(frame):
            if not person_detected:
                print("⚠️ Person detected! Stopping.")
                twist.linear_velocity_x = 0.0
                twist.angular_velocity = 0.0
                await canbus_client.request_reply("/twist", twist)
                person_detected = True
        else:
            if person_detected:
                print("✅ Person left. Resuming row following at 0.1 m/s.")
                twist.linear_velocity_x = FORWARD_SPEED
                twist.angular_velocity = 0.0
                await canbus_client.request_reply("/twist", twist)
                person_detected = False

            # Process crop rows if no person detected
            row_center = detect_crop_rows(frame)

            if row_center is None:
                print("❓ No white markers detected. Going straight.")
                twist.angular_velocity = 0.0
            else:
                # Compute offset and clamp extreme values
                offset = TARGET_ROW_CENTER - row_center
                offset = max(min(offset, 200), -200)  # ✅ Limit offset to prevent excessive turning
                angular_correction = STEERING_GAIN * offset
                twist.angular_velocity = np.clip(angular_correction, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)

                print(f"⚪ Row Center: {row_center}, Offset: {offset}, Steering: {twist.angular_velocity:.3f}")

            await canbus_client.request_reply("/twist", twist)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting.")
            break

        await asyncio.sleep(0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amiga Lab Test - White Lid Row Following + Person Detection.")
    parser.add_argument("--canbus-config", type=Path, required=True, help="Path to canbus config.")
    parser.add_argument("--camera-config", type=Path, required=True, help="Path to camera config.")
    args = parser.parse_args()

    asyncio.run(main(args.canbus_config, args.camera_config))
