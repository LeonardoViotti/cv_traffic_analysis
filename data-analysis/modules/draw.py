
import os
import copy as cp
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#------------------------------------------------------------------------------
# Color to class definition

color_dict = {'Person' : (0,255,255), 
              'Bicycle' : (255,0, 255), 
              'Car': (0, 0, 255),
              'Motorbike' : (0,255,0), 
              'Bus' : (255,0,0), 
              'Truck' : (255,255,0)}
            


#------------------------------------------------------------------------------
# Single object annotations

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


# Draw multiple boxes
def draw_boxes(img, df, imutable = False):
    for index, row in df.iterrows():
        draw_box(img, row, row['class'], imutable = imutable)

# Draw single point
def draw_centroid(org_img, cent_df):
    img = cp.deepcopy(org_img)
    # Anotate
    cv2.circle(img, (cent_df['cx'], cent_df['cy']), 2, (255, 255, 255), -1)
    
    return img
    
def draw_trajectory(org_img, trajectory_df, color_dict = color_dict):
    """
    This draws a trakjectorie line on a frame
    img :  has to be an image converted into numpy.ndarray
    trajectory_df : pandas DataFrame containig pixel coordinates
                    and class of a SINGLE object. 
    color_dict : a dictionary containt {class_id : (B,G,R)}
    """
    img = cp.deepcopy(org_img)
    
#     obj_class = trajectory_df['class'].iloc[0]
    
    trajectory_df = trajectory_df[['cx', 'cy']]
    for p in range(1, len(trajectory_df)):
        # print(p)
        cv2.line(img, tuple(trajectory_df.iloc[p-1]), tuple(trajectory_df.iloc[p]), color_dict[obj_class], 2)
    return(img)
