# Multi Object Tracker with Visualization
Use Deep SORT algorithm to track objects in a video, then visualize the resulting tracks with simple streamlit web app.

## Installation:
- Create python environment (python 3.9) and install the dependencies `pip install -r requirements.txt`
- Download YOLOv4 weights and put them in the model folder:
    - https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights


## Usage
- Run Object tracker `python object_tracking.py` on a video (default video is cars.mp4 in the data folder). New Database with tracking info should be created.
- To change the target video, put the video in the data folder and use the `--video` command line argument to specify the video path
- Run `streamlit run visualize.py` and open the web interface

## Repo Overview:
- `object_tracking.py`: Track multiple objects in a video using deep sort algorithm. Specify video path with --video command line argument or leave it out for default `data/cars.mp4` video.
- `database.py`: Features a simple interface to a sqlite database, used to write and read tracking information.
- `mobilenet.py`: Embed an image to a single dimensional vector, used for feature comparison in the deep sort algorithm.
- `visualize.py`: Visualize the resulting tracks in a simple streamlit web app. Select the video and trackID to visualize the whole trajectory or view a single frame and the matching bounding box, along with the detection confidence.
- `data` folder: Holds the videos and the class names for YOLO object detector.
- `model_cfg` folder: Holds the configuration files for YOLO models.
- `model` folder: Holds the weights for YOLO models.