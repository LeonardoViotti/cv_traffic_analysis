#-------------------------------------------------------------------------------------
# Annotate video with a csv file
#-------------------------------------------------------------------------------------

# TODO
# - Cut video with to only show frames contained in df


export = True

#-------------------------------------------------------------------------------------
# Settings
import cv2
import pandas as pd
import numpy as np

from modules.utils import *
from modules.draw import *

#-------------------------------------------------------------------------------------
# Load data

# CSV annotation data
df = pd.read_csv('data/2-sample-30min.csv')

# Video capture object
cap = cv2.VideoCapture('data/2-sample-30min.mp4')

#-------------------------------------------------------------------------------------
# Data frame processing

class_dict = {0 : 'Person', 
              1 : 'Bicycle', 
              2 : 'Car',
              3: 'Motorbike', 
              5: 'Bus', 
              7: 'Truck'}


df = df.replace({"class": class_dict})

# Select a subset of the df to display video
df = df[df['obj_id'] == 15]


#-------------------------------------------------------------------------------------
# Video meta data and exporting 

# Retrieve video frame properties.
frame_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = int(cap.get(cv2.CAP_PROP_FPS))


# Specify the value for fourcc
# fourcc_avi = cv2.VideoWriter_fourcc('M','J','P','G')
fourcc_mp4 = cv2.VideoWriter_fourcc(*'XVID')

# Specify the video output filenames.
# file_out_avi = 'video_out.avi'
file_out_mp4 = 'video_out.mp4'

# Create the video writer objects.
# out_avi = cv2.VideoWriter(file_out_avi, fourcc_avi, frame_fps, (frame_w,frame_h))
out_mp4 = cv2.VideoWriter(file_out_mp4, fourcc_mp4, frame_fps, (frame_w,frame_h))

#-------------------------------------------------------------------------------------
# 

# Clip video parameters
start_frame = df['frame'].min()
end_frame = df['frame'].max()

# Add a little margin
end_frame = end_frame + 10
if start_frame < 10:
    start_frame = 0
else:
    start_frame = start_frame -10

# Set initial frame position 
# First parameter is 1: CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
cap.set(1,start_frame)


# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

# Base variables
idx_frame = 0

# Read until video is completed
# while(cap.isOpened()):

# Read until last frame from csv plus a margin
while(idx_frame < end_frame):
    
    idx_frame += 1
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        
        # Anotate frame with df
        frame_df = df[df['frame'] == idx_frame]
        draw_boxes(frame, frame_df)
        
        # Display the resulting frame
        cv2.imshow('Frame',frame)
        key = cv2.waitKey(1) 
        
        if export:
            out_mp4.write(frame)
        
        # Press Q on keyboard to  exit
        if key == ord('q'):
            break
            
    # Break the loop
    else: 
        break

# When everything done, release the video capture object
cap.release()
# out_avi.release()
out_mp4.release()

# Closes all the frames
cv2.destroyAllWindows()
