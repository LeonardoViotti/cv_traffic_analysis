# Traffic analysis with computer vision (Under development)

Open-source tool for processing and analyzing traffic footage for users without a expertise in Deep Learning or computer vision. 

Current focus is on near collisions. But the objective is to have a tool flexible enough so it can be easily adapted for other purposes.

## Video processing
Uses yolov3 and DeepSort to process video input and outputs an annotated video file and a csv containing bounding boxes and classes per frame. 

This was originally forked from https://github.com/ZQPei/deep_sort_pytorch with minor modifications to process traffic footage.

## Data analysis
Processes outputted data to identify events of interest.

 1. Identify objects trajectories intersections
 2. Homography - re-project trajectories from pixels to lat-long.
 3. Calculate speeds.
 4. Determine conflict severity. 
