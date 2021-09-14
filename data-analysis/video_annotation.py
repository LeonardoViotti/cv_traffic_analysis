

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

#-------------------------------------------------------------------------------------
# DRAFT


bar = draw_box(frame, foo, 'Person')
ishow(bar)

# Draw single box function
def draw_box(org_img, bbox_df, class_id, color_dict = color_dict, imutable = True):
    x1,y1,x2,y2 = bbox_df['xi'], bbox_df['yi'], bbox_df['xj'], bbox_df['yj']
    
    # Create another image
    if imutable:
        img = cp.deepcopy(org_img)
    else:
        img = org_img
    # Coolor and lable
    # color = (0,255,0)
    color  = color_dict[class_id]
    label = class_id
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    # Boox rectangle
    cv2.rectangle(img,(x1, y1),(x2,y2),color,1)
    # Text rectangle
    cv2.rectangle(img,(x1, y1),(x1+t_size[0],y1+t_size[1]+1), color,-1)
    cv2.putText(img,label,(x1,y1+t_size[1]+1), cv2.FONT_HERSHEY_PLAIN, .75, [255,255,255], 1)
    
    if imutable:
        return img

def draw_boxes(img, df, imutable = False):
    for index, row in df.iterrows():
        draw_box(img, row, row['class'], imutable = imutable)

draw_boxes(frame, frame_df)


frame_df = df[df['frame'] == idx_frame]

# for index, row in frame_df.iterrows():
#     draw_box(frame, row, row['class'], imutable = False)

# draw_box(frame, foo, 'Person', imutable = False)
ishow(frame)