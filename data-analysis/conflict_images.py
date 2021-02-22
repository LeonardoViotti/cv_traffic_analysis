#------------------------------------------------------------------------------
# Create side by side trajectories on video frame and sat image
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Settings and globals

import numpy as np
import pandas as pd
import copy as cp

from modules.utils import *
from modules.draw import *
from modules.homography import PixelMapper


# Load data
df = pd.read_csv('../output/2-sample-2020.csv')

# Load conflict frame
img_f = cv2.imread('../output/2-sample-2020-cnflct_frame.png') 

# Load satellite image
img_s = cv2.imread('../data/2-sat.jpg')

#------------------------------------------------------------------------------
# Create conflict frame

# # video_file_name = '../data/2-sample-2020.mp4'
# video_file_name = os.path.join(RAW_DATA_PATH, filename + '.mp4')

# # Load video
# cap = cv2.VideoCapture(video_file_name)

# # Grab post conflict frame
# cap.set(cv2.CAP_PROP_POS_FRAMES, 143-1)
# _, img_c = cap.read()



#------------------------------------------------------------------------------
# Process data

#------------------------------------------------------------------------------
# Process data

# Replace Coco class ids with str labels
class_dict = {0: 'Person', 
              1: 'Bicycle', 
              2: 'Car', 
              3: 'Motorbyke', 
              5: 'Bus', 
              7: 'Truck'}
df['class'] = df['class'].map(class_dict)

# Set colors
# COLORS = np.random.uniform(0, 255, size=(len(class_dict), 3))

# class_dict.keys()

# Create centroids 

df['cx'] =  round(df['xi'] + (df['xj'] - df['xi'])/2).astype(int)
df['cy'] =  round(df['yi'] + (df['yj'] - df['yi'])/2).astype(int)


#------------------------------------------------------------------------------
# Coordinates mapping 

# Create one instance of PixelMapper to convert video frames to coordinates
quad_coords = {
    # Check these!
    "lonlat": np.array([
        [9.020958, 38.511327], #  top right
        [9.020882, 38.511203], #  top left
        [9.020798, 38.511146], # bottom left
        [9.020819, 38.511240] #  bottom right
    ]),
    "pixel": np.array([
        [225, 40], # top right
        [88, 78], #  top left
        [382, 327], #  bottom left
        [386, 108] # bottom right
    ]),
    "pixel_sat": np.array([
        [274, 79], # top right
        [146, 138], #  top left
        [91, 201], #  bottom left
        [170, 179] # bottom right
    ])
}

# Create pixel maper instance to convert from video to lat long (and vice versa)
pm = PixelMapper(quad_coords["pixel"], quad_coords["lonlat"])

# Create pixel maper instance to convert from sat image to lat long (and vice versa)
pm_sat = PixelMapper(quad_coords["pixel_sat"], quad_coords["lonlat"])


#------------------------------------------------------------------------------
# Test a trajectory on video frame and sat image and longlat

# REWRITE THIS!
def draw_trajectory(img, trajectory_array, color):
    img_cp = cp.deepcopy(img)
    for p in range(1, len(trajectory_array)):
            cv2.line(img_cp, tuple(trajectory_array[p-1]), tuple(trajectory_array[p]), color, 2)
    return img_cp


# Create trajectory df
car_df = df[df['obj_id'] == 21]
per_df = df[df['obj_id'] == 22]

t_car = car_df[['cx', 'cy']].to_numpy()
t_per = per_df[['cx', 'cy']].to_numpy()

# Anotate trajectory on initial video frame
img_cf = img_f.copy()
img_cf = draw_trajectory(img_cf, t_car, (0, 0, 255))
img_cf = draw_trajectory(img_cf, t_per, (0, 255, 255))

ishow(img_cf)

# Transform trajectories to long lat
t_car_ll = pm.pixel_to_lonlat(t_car) # t_car created in draft-intersections.py
t_per_ll = pm.pixel_to_lonlat(t_per) # t_car created in draft-intersections.py


# Transform lat long trajectory into pixels of sat image
t_car_s = pm_sat.lonlat_to_pixel(t_car_ll).astype(int)
t_per_s = pm_sat.lonlat_to_pixel(t_per_ll).astype(int)


# Anotate trajectory on sat image
img_cs = img_s.copy()
img_cs = draw_trajectory(img_cs, t_car_s, (0, 0, 255))
img_cs = draw_trajectory(img_cs, t_per_s, (0, 255, 255))

# t_per_s
ishow(img_cs)


res = hstack_images(img_cf, img_cs)

ishow(res)

# Export
cv2.imwrite('../output/2-sample-2020-cnflct_frame_and_sat.png', res)

#----------------------
# Draf test homography

# Function to compare points on both images
# def show_hom_points(img, img_sat points_vdframe, points_sat ):
#     """
#     img and img_sat : numpy.ndarray of pixels
#     points_vdframe and points_sat : two dictionaries with the structure below:5
#     - points should go anti-clock-wise
#     {
#     "lonlat": np.array([
#         [x1, y1], #  top right
#         [x2, y2], #  top left
#         [x3, y3], # bottom left
#         [x4, y4] #  bottom right]),
#     "pixel": np.array([
#         [x1, y1], #  top right
#         [x2, y2], #  top left
#         [x3, y3], # bottom left
#         [x4, y4] #  bottom right
#     ])}
#     """
#     return 0

def draw_point(img, point, color, label = None):
    img_cp = cp.deepcopy(img)
    pcoords = tuple(point)
    cv2.circle(img_cp, pcoords, 3, color, -1)
    if label is not None:
        tcoords = tuple(point + 5)
        cv2.putText(img_cp, label, tcoords,  cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    # ishow(img_cp)
    return img_cp

def draw_hom_points(img, points_array):
    img_cp = cp.deepcopy(img)
    # Loop over points
    i = 0
    for p in points_array:
        i += 1
        label = 'p' + str(i)
        img_cp = draw_point(img_cp, p, (0, 0, 255), label)
    return img_cp

img_s_points = draw_hom_points(img_s, quad_coords['pixel_sat'])
img_f_points = draw_hom_points(img_f, quad_coords['pixel'])


vis = hstack_images(img_f_points, img_s_points)

# pm_sat = PixelMapper(quad_coords_sat["pixel"], quad_coords_sat["lonlat"])

# Create anothe pixel mapper instance to convert from lonlat to sat image
pm_sat = PixelMapper(quad_coords["pixel_sat"], quad_coords["lonlat"])