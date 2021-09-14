

#-------------------------------------------------------------------------------------
# Settings
import cv2
import pandas as pd
import numpy as np

#-------------------------------------------------------------------------------------
# Load data

# CSV annotation data
df = pd.read_csv('data/2-sample-30min.csv')

# Video capture object
cap = cv2.VideoCapture('data/2-sample-30min.mp4')


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
        # Display the resulting frame
        cv2.imshow('Frame',frame)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        print(idx_frame)
    
    # Break the loop
    else: 
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()