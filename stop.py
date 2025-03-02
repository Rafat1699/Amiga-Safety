import argparse
import asyncio
import cv2
import numpy as np
from pathlib import Path

from farm_ng.canbus.canbus_pb2 import Twist2d
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file

# YOLO Model Paths
YOLO_CONFIG = "yolov4-tiny.cfg"
YOLO_WEIGHTS = "yolov4-tiny.weights"
YOLO_CLASSES = "coco.names"

# Load YOLO
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
with open(YOLO_CLASSES, 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]

def detect_person_yolo(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    detections = net.forward(output_layers)

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and CLASSES[class_id] == "person":
                return True
    return False

async def main(camera_config_path: Path, canbus_config_path: Path) -> None:
    # Read configs
    camera_config = proto_from_json_file(camera_config_path, EventServiceConfig())
    canbus_config = proto_from_json_file(canbus_config_path, EventServiceConfig())

    # Create clients
    camera_client = EventClient(camera_config)
    canbus_client = EventClient(canbus_config)

    # Initialize twist message
    twist = Twist2d()

    cv2.namedWindow("Camera Stream", cv2.WINDOW_NORMAL)

    async for event, message in camera_client.subscribe(camera_config.subscriptions[0], decode=True):
        frame = cv2.imdecode(np.frombuffer(message.image_data, dtype="uint8"), cv2.IMREAD_UNCHANGED)

        if detect_person_yolo(frame):
            print("⚠️ Person detected! Stopping robot.")
            twist.linear_velocity_x = 0.0
            twist.angular_velocity = 0.0
            await canbus_client.request_reply("/twist", twist)
            break

        cv2.imshow("Camera Stream", frame)
        cv2.waitKey(1)
        await asyncio.sleep(0.05)

if name == "main":
    parser = argparse.ArgumentParser(description="Amiga person detection using YOLO.")
    parser.add_argument("--camera-config", type=Path, required=True, help="Path to camera config JSON.")
    parser.add_argument("--canbus-config", type=Path, required=True, help="Path to canbus config JSON.")
    args = parser.parse_args()

    asyncio.run(main(args.camera_config, args.canbus_config))
