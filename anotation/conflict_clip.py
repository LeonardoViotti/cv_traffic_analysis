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
# Load data
df = pd.read_csv('../output/2-sample-2020.csv')


#------------------------------------------------------------------------------
# Load video

# video_file_name = '../data/2-sample-2020.mp4'
video_file_name = os.path.join(RAW_DATA_PATH, filename + '.mp4')

# Load video
cap = cv2.VideoCapture(video_file_name)
