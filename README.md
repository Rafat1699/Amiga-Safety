Stop Feature:
First Cd into farm-ng-amiga
Then source venv/bin/activate
Then go to cd py/examples/camera_client
Then create two different service config file one for camera client another for the canbus
For the yolo model run this in the same directory
wget https://github.com/AlexeyAB/darknet/raw/master/cfg/yolov4-tiny.cfg
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
run the code using this
the code is in the github repo.
python3 vehicle_twist_person_detect.py --canbus-config canbus_config.json --camera-config camera_config.json

