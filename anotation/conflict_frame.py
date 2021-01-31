#------------------------------------------------------------------------------
# Frame annotation for a single conflict
#------------------------------------------------------------------------------

# Ref
# https://www.youtube.com/watch?v=1FJWXOO1SRI

# Settings
from utils import *
from modules.draw import *

import copy as cp
import pandas as pd

#------------------------------------------------------------------------------
# Import data

filename = '2-sample-2020'

RAW_DATA_PATH = '../data'
OUT_PATH = '../output'

# video_file_name = '../data/2-sample-2020.mp4'
video_file_name = os.path.join(RAW_DATA_PATH, filename + '.mp4')

# Load video
cap = cv2.VideoCapture(video_file_name)

# Load tracking data csv
data = pd.read_csv(os.path.join(OUT_PATH, filename + '.csv'))

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
# Subset just conflict into one df per tracked object

trajectories_df = data[data['obj_id'].isin([21,22])]
# df1 = trajectories_df[trajectories_df['obj_id'] == 21]
# df2 = trajectories_df[trajectories_df['obj_id'] == 22]

# Loop over different object ids in the df
img = cp.deepcopy(img_c)
for oid in trajectories_df['obj_id'].unique():
    df_i = trajectories_df[trajectories_df['obj_id'] == oid]
    # print(df_i.head(5))
    img = draw_trajectory(img, df_i)

ishow(img)

# Export image
cv2.imwrite(os.path.join(OUT_PATH, filename + '-cnflct_1.png'), img)


