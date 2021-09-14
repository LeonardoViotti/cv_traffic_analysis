#-------------------------------------------------------------------------------------
# Annotate video with a csv file
#-------------------------------------------------------------------------------------

# TODO
# - Cut video with to only show frames contained in df


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
              4: 'Bus', 
              5: 'Truck'}


df = df.replace({"class": class_dict})


#-------------------------------------------------------------------------------------
# 

# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

# Base variables
idx_frame = 0

# Read until video is completed
while(cap.isOpened()):
    
    idx_frame += 1
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        
        # Anotate frame with df
        frame_df = df[df['frame'] == idx_frame]
        draw_boxes(frame, frame_df)
        
        # Display the resulting frame
        cv2.imshow('Frame',frame)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        
        print(idx_frame)
    
    # Break the loop
    else: 
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
