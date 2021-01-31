# https://www.youtube.com/watch?v=1FJWXOO1SRI

# Settings
from utils import *

import copy as cp
import pandas as pd


video_file_name = '../data/2-sample-2020.mp4'

# Load video
cap = cv2.VideoCapture(video_file_name)

# Load tracking csv
data = pd.read_csv('../output/2-sample-2020.csv')



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

# Create centroids 

data['cx'] =  round(data['xi'] + (data['xj'] - data['xi'])/2).astype(int)
data['cy'] =  round(data['yi'] + (data['yj'] - data['yi'])/2).astype(int)

]
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
    x1,y1,x2,y2 = bbox_df['xi'], bbox_df['yi'], bbox_df['xj'], bbox_df['yj']
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
def draw_centroid(org_img, cent_df):
    img = cp.deepcopy(org_img)
    # Anotate
    cv2.circle(img, (cent_df['cx'], cent_df['cy']), 2, (255, 255, 255), -1)
    
    return img
    
def draw_trajectory(org_img, trajectory_df):
    """
    This draws a trakjectorie line on a frame
    img -  has to be an image converted into numpy.ndarray
    trajectory_df 
    """
    img = cp.deepcopy(org_img)
    for p in range(1, len(trajectory_df)):
        # print(p)
        cv2.line(img, tuple(trajectory_df.iloc[p-1]), tuple(trajectory_df.iloc[p]), (0, 0, 255), 2)
    return(img)


df = data[data['obj_id'] == 1][['cx', 'cy']]
foo = cp.deepcopy(img_0)

trajectory_df = df



for idx, row in df.iterrows():
    print(df[idx])

tuple(df.iloc[0])


# foo = draw_boxes(img_0, data_i)


# img_show(img_0)
img_show(foo)

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



