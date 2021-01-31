# https://www.youtube.com/watch?v=1FJWXOO1SRI

# Settings
from utils import *
from modules.draw import *

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


#------------------------------------------------------------------------------
# Select first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
_, img_0 = cap.read()
# stable_show(img_0)

# Grab post conflict frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 143-1)
_, img_c = cap.read()


#------------------------------------------------------------------------------
# Colors

df_conf = data[data['obj_id'].isin([21,22])]
df1 = df_conf[df_conf['obj_id'] == 21]
df2 = df_conf[df_conf['obj_id'] == 22]


# Loop over different object ids in the df
img = cp.deepcopy(img_c)
for oid in trajectories_df['obj_id'].unique():
    df_i = trajectories_df[trajectories_df['obj_id'] == oid]
    # print(df_i.head(5))
    img = draw_trajectory(img, df_i)


ishow(img)


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



