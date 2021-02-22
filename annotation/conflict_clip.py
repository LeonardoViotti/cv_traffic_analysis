#------------------------------------------------------------------------------
# Create video of a single conlict
#------------------------------------------------------------------------------

# This assumes the video is already just a shor clip of the conflict

#------------------------------------------------------------------------------
# Settings and globals

import numpy as np
import pandas as pd
import copy as cp

from modules.utils import *
from modules.draw import *
from modules.homography import PixelMapper

#------------------------------------------------------------------------------
# Load video

# video_file_name = '../data/2-sample-2020.mp4'
video_file_name = os.path.join('../data/2-sample-2020.mp4')


#------------------------------------------------------------------------------
# Load data
df = pd.read_csv('../output/2-sample-2020.csv')

# Replace Coco class ids with str labels
class_dict = {0: 'Person', 
              1: 'Bicycle', 
              2: 'Car', 
              3: 'Motorbyke', 
              5: 'Bus', 
              7: 'Truck'}
df['class_label'] = df['class'].map(class_dict)

# Set colors
# COLORS = np.random.uniform(0, 255, size=(len(class_dict), 3))

# class_dict.keys()

# Create centroids 

df['cx'] =  round(df['xi'] + (df['xj'] - df['xi'])/2).astype(int)
df['cy'] =  round(df['yi'] + (df['yj'] - df['yi'])/2).astype(int)


#------------------------------------------------------------------------------
# MOVE THIS OUT

# Draw single box function
def draw_box(org_img, bbox_df, color_dict = color_dict):
    # Create another image
    img = cp.deepcopy(org_img)
    # Get bbox elements
    x1,y1,x2,y2 = bbox_df['xi'], bbox_df['yi'], bbox_df['xj'], bbox_df['yj']
    obj_class = bbox_df['class_label'].item()
    # obj_class = 'Person'
    # Formating locals
    color  = color_dict[obj_class]
    label = obj_class
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    
    # Anotate bbox rectangle
    cv2.rectangle(img,(x1, y1),(x2,y2),color,1)
    # Anotate text rectangle
    cv2.rectangle(img,(x1, y1),(x1+t_size[0],y1+t_size[1]+1), color,-1)
    cv2.putText(img,label,(x1,y1+t_size[1]+1), cv2.FONT_HERSHEY_PLAIN, .75, [255,255,255], 1)
    
    return img

#------------------------------------------------------------------------------
# Subset df to conflict only

df = df[df['obj_id'].isin([21,22])]

#------------------------------------------------------------------------------
# Process video


# Load video
cap = cv2.VideoCapture(video_file_name)
_, img_0 = cap.read()

im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('../output/2-sample-2020-conf.mp4', fourcc, 20, (im_width, im_height))


# For now simpling the df
# df = df[df['obj_id'].isin([1,2])]


idx_frame = 0
while(cap.isOpened()):
    idx_frame += 1
    ret, frame = cap.read()
    if ret==True:
        # Draw bbox from df
        if idx_frame in df['frame'].unique():
            
            # Subset df to only contain active frame
            df_frame = df[df['frame'] == idx_frame]
            
            # Loop over each object in frame
            for obj_id in df_frame['obj_id'].unique():
                df_frame_obj = df_frame[df_frame['obj_id'] == obj_id]
                frame = draw_box(frame, df_frame_obj)
        
        # Save video frame by frame
        writer.write(frame)
        
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
writer.release()
cv2.destroyAllWindows()


#------------------------------------------------------------------------------
# Draft draw box 

# Complete dataframe with number of frames?
# Match frame number?

# # Grab second frame
# cap.set(cv2.CAP_PROP_POS_FRAMES, 2)
# _, img_2 = cap.read()

# # Grab one bbox int the 2nd frame
# bbox_df = df.loc[0]
