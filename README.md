# Stop Feature

This repository implements the **Stop Feature** which integrates object detection with vehicle and CAN bus data processing. The feature leverages a camera client and CAN bus service to detect objects (such as vehicles and persons) using a YOLOv4-tiny model, and it is designed for automotive or robotic applications where quick response to dynamic scenarios is critical.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation & Setup](#installation--setup)
- [Configuration Files](#configuration-files)
- [YOLO Model Setup](#yolo-model-setup)
- [Running the Application](#running-the-application)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

The Stop Feature monitors real-time video input via a camera client while simultaneously processing CAN bus signals. By applying the YOLOv4-tiny object detection model, the system can detect specified entities (e.g., vehicles, persons) in the environment. This information can then be used to influence vehicle behavior or trigger control sequences.

## Requirements

- **Python 3.x** – Ensure you have an appropriate version of Python installed.
- **Virtual Environment** – Using Python’s `venv` to manage dependencies.
- **wget** – The `wget` utility must be installed for downloading the YOLO model files.
- **Additional Python packages** – Refer to `requirements.txt` (if provided) for other dependencies.
Note: you have to access the brain using your local pc as the windows will pop up. Also connect the farm-ng-ssh connection for comunicating with the brain.
## Installation & Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-github-username/farm-ng-amiga.git
   cd farm-ng-amiga

2. **Setup the enviorment:**

   ```bash
   cd farm-ng-amiga
   source venv/bin/activate
   cd py/examples/camera_client
   
3. **Install Dependencies:**

   ```bash
   pip install opencv-python numpy

4. **Preparing the YOLO MODEL Files:**

   ```bash
   wget https://github.com/AlexeyAB/darknet/raw/master/cfg/yolov4-tiny.cfg
   wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights
   wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

5. **Running the application **
   ```bash
   python3 stop.py --camera-config camera_config.json --canbus-config canbus_config.json


   
