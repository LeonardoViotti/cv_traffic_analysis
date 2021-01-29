# https://www.youtube.com/watch?v=1FJWXOO1SRI

# Settings
from utils import *

import copy as cp
import pandas as pd


video_file_name = '../data/11-sample-short.mp4'

# Load video
cap = cv2.VideoCapture(video_file_name)

# Load tracking csv
data = pd.read_csv('../output/11-sample-short.csv')

# Convert all columns to integer for now
convert_dict = {'frame': int, 
                'x_i': int,
                'y_i': int,
                'x_j': int,
                'y_j': int,
                'obj_id': int,
                'class': int
                # 'conf': float,
               } 
  
data = data.astype(convert_dict) 

#------------------------------------------------------------------------------
# Process data

# Replace Coco class ids with str labels
class_dict = {0: 'Person', 
              1: 'Bicycle', 
              2: 'Car', 
              3: 'Motorbyke', 
              5: 'Bus', 
              7: 'Truck'}
data['class'] = data['class'].map(class_dict)

# Set colors
# COLORS = np.random.uniform(0, 255, size=(len(class_dict), 3))

# class_dict.keys()
#------------------------------------------------------------------------------
# Select first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
_, img_0 = cap.read()
# stable_show(img_0)

#------------------------------------------------------------------------------
# Static annotations


COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# Draw single box function
def draw_box(org_img, bbox_df, class_id):
    x1,y1,x2,y2 = bbox_df['x_i'], bbox_df['y_i'], bbox_df['x_j'], bbox_df['y_j']
    # Create another image
    img = cp.deepcopy(org_img)
    
    # Coolor and lable
    color = (0,255,0)
    label = class_id
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    # Boox rectangle
    cv2.rectangle(img,(x1, y1),(x2,y2),color,1)
    # Text rectangle
    cv2.rectangle(img,(x1, y1),(x1+t_size[0],y1+t_size[1]+1), color,-1)
    cv2.putText(img,label,(x1,y1+t_size[1]+1), cv2.FONT_HERSHEY_PLAIN, .75, [255,255,255], 1)
    
    return img

# Draw single point
def draw_centroid(org_img, bbox_df):
    x1,y1,x2,y2 = bbox_df['x_i'], bbox_df['y_i'], bbox_df['x_j'], bbox_df['y_j']
    # Create another image
    img = cp.deepcopy(org_img)
    # Calculate centroid
    cx = round(x1 + (x2 - x1)/2)
    cy = round(y1 + (y2 - y1)/2)
    # Anotate
    cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1)
    
    return img


def draw_trajectory(img, trajectory_df, class_id):
    pass

# Draw a dataframe of boxes NOT WORKING
def draw_boxes(or_img, data):
    # img = cp.deepcopy(or_img)
    # for index,row in data.iterrows():
    #     draw_box(or_img, row, row['class'])
    # return img
    pass

# very simlified version of the data
data_i = data[data['frame'] == 2][['x_i','y_i', 'x_j', 'y_j', 'class']]
data_i = data_i.iloc[0:3,]

bbox_df = data_i

data_i.loc[0]

foo = draw_box(img_0, data_i.loc[0], data_i.loc[0]['class'])
foo = draw_centroid(img_0, data_i.loc[0])
# foo = draw_boxes(img_0, data_i)


# img_show(img_0)
# img_show(foo)

#------------------------------------------------------------------------------
# Process video frame by frame

# # Video loop
# while True:
    
#     # timer for frames per section
#     timer = cv2.getTickCount()
    
#     # Read each frame
#     success, img_i = cap.read()
    
#     # Add fps to display
#     # fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
#     # cv2.putText(img_i,
#     #             str(int(fps)),
#     #             (50,50),
#     #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    
#     # # Add tracked centroids to array
#     # ct_i = create_centroids(bboxes_i)
#     # ct_tracked = np.concatenate([ct_tracked,ct_i], axis = 1)
    
#     # # draw tracked objects bbox
#     # for i, new_ct in enumerate(ct_i):
#     #     drawCentroid(img_i, 
#     #                  new_ct[0],# new_ct has an extra dimention for the concatenation to work
#     #                  class_id= classes[i]) 
    
#     # Show video
#     cv2.imshow("Video", img_i)
#     # Break out by pressing 'q' when window is selected
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Make sure there are no open graphics devices
# cv2.destroyAllWindows()



