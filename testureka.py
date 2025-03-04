

import argparse
import asyncio
from pathlib import Path
import cv2
import numpy as np
from farm_ng.canbus.canbus_pb2 import Twist2d
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file

# YOLOv4-tiny model files
YOLO_CONFIG = "yolov4-tiny.cfg"
YOLO_WEIGHTS = "yolov4-tiny.weights"
YOLO_CLASSES = "coco.names"

# Load YOLO model
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)

with open(YOLO_CLASSES, 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]


def detect_person(frame) -> bool:
    """Detect if a person is present using YOLOv4-tiny."""
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
                print(f"👤 Person detected with confidence: {confidence:.2f}")
                return True

    return False


async def main(canbus_config_path: Path, camera_config_path: Path) -> None:
    # Load configs
    canbus_config = proto_from_json_file(canbus_config_path, EventServiceConfig())
    camera_config = proto_from_json_file(camera_config_path, EventServiceConfig())

    canbus_client = EventClient(canbus_config)
    camera_client = EventClient(camera_config)

    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

    # Control state
    person_detected = False

    async for event, message in camera_client.subscribe(camera_config.subscriptions[0], decode=True):
        frame = cv2.imdecode(np.frombuffer(message.image_data, dtype="uint8"), cv2.IMREAD_UNCHANGED)

        cv2.imshow("Camera Feed", frame)

        if detect_person(frame):
            if not person_detected:
                print("⚠️ Person detected! Stopping robot.")
                twist = Twist2d(linear_velocity_x=0.0, angular_velocity=0.0)
                await canbus_client.request_reply("/twist", twist)
                person_detected = True
        else:
            if person_detected:
                print("✅ Person left! Resuming forward at 0.1 m/s.")
                twist = Twist2d(linear_velocity_x=0.1, angular_velocity=0.0)
                await canbus_client.request_reply("/twist", twist)
                person_detected = False

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting program.")
            break

        await asyncio.sleep(0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amiga robot automatic stop/resume based on YOLO person detection.")
    parser.add_argument("--canbus-config", type=Path, required=True, help="Path to canbus service config.")
    parser.add_argument("--camera-config", type=Path, required=True, help="Path to camera service config.")
    args = parser.parse_args()

    asyncio.run(main(args.canbus_config, args.camera_config))
