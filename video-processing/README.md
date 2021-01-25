## Attribution
Originally forked from [https://github.com/ZQPei/deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch) with modifications to analyze traffic footage.


## Dependencies
- python 3 (python2 not sure)
- numpy
- scipy
- opencv-python
- sklearn
- torch >= 0.4
- torchvision >= 0.1
- pillow
- vizer
- edict

## Installation and usage
0. Check all dependencies installed
```bash
pip install -r requirements.txt
```
for user in china, you can specify pypi source to accelerate install like:
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

1. Clone this repository
```
git clone git@github.com:LeonardoViotti/cv_traffic_analysis.git
```

2. Download YOLOv3 parameters
```
cd detector/YOLOv3/weight/
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
cd ../../../
```

3. Download deepsort parameters ckpt.t7
```
cd deep_sort/deep/checkpoint
# download ckpt.t7 from
https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6 to this folder
cd ../../../
```  

4. Compile nms module
```bash
cd detector/YOLOv3/nms
sh build.sh
cd ../../..
```

5. Run demo
```
usage: python video_processing.py VIDEO_PATH
                                [--help]
                                [--frame_interval FRAME_INTERVAL]
                                [--config_detection CONFIG_DETECTION]
                                [--config_deepsort CONFIG_DEEPSORT]
                                [--display]
                                [--display_width DISPLAY_WIDTH]
                                [--display_height DISPLAY_HEIGHT]
                                [--save_path SAVE_PATH]          
                                [--cpu]          

# yolov3_tiny + deepsort
python video_processing.py [VIDEO_PATH] --config_detection ./configs/yolov3_tiny.yaml

# yolov3 + deepsort on webcam
python3 video_processing.py /dev/video0 --camera 0

# yolov3_tiny + deepsort on webcam
python3 video_processing.py /dev/video0 --config_detection ./configs/yolov3_tiny.yaml --camera 0
```
Use `--display` to enable display.

Results will be saved to `../output/`.
