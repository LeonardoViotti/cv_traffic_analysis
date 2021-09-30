#-------------------------------------------------------------------------------------
# Clean tabular trajectory data
#-------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import os
import copy as cp

import matplotlib.pyplot as plt
import plotly.express as px

#-------------------------------------------------------------------------------------
# IDEAS

"""

 - Should I do angles this before or after interpolation?
 - Sliding angle (every 10 points or so)?

Normally, points that have a sharp angle are distant from each other. Lost tracking for
a moment and picked-up some frames later on a diferent object.

Maybe a rule if skipped X frames and has a sharp angle? 

"""


#-------------------------------------------------------------------------------------
# Load data

df = pd.read_csv('data/2-sample-30min.csv')

#-------------------------------------------------------------------------------------
# Create columns


# Calculate centroids
df['cx'] =  round(df['xi'] + (df['xj'] - df['xi'])/2).astype(int)
df['cy'] =  round(df['yi'] + (df['yj'] - df['yi'])/2).astype(int)

# Lag points
df = df.sort_values(['obj_id', 'frame'])

df['cx_l'] = df.groupby(['obj_id'])['cx'].shift(20)
df['cy_l'] = df.groupby(['obj_id'])['cy'].shift(20)

df['cx_l2'] = df.groupby(['obj_id'])['cx'].shift(40)
df['cy_l2'] = df.groupby(['obj_id'])['cy'].shift(40)


# Test if there is movement beteween 3 points
df['stopepd'] = (df['cx'] == df['cx_l']) & (df['cy'] == df['cy_l']) | (df['cx_l'] == df['cx_l2']) & (df['cy_l'] == df['cy_l2'])
df['stopepd'] = df['stopepd'].astype(int)


#-------------------------------------------------------------------------------------
# Clean sharp angles in trajectories

"""
Split trajectories with sharp angles. This is useful to clean out problems with the
tracking algorithm when tracking jumps from one object to the other.

For example, if tracking jumps from one car going north to another going south

"""



def get_angle(df,
              p1_cols = ['cx', 'cy'],
              p2_cols = ['cx_l', 'cy_l'],
              p3_cols = ['cx_l2', 'cy_l2']):
    """
    Calculate angle between 3 points vectorially.
    
    Formula source: 
    https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
    """
    
    # Get point columns in array format
    a = df[p1_cols].to_numpy()
    b = df[p2_cols].to_numpy()
    c = df[p3_cols].to_numpy()
    
    # Differences
    ba = a - b
    bc = c - b
    
    # Row-wise dot multiplication
    # https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy
    row_wise_dot = np.sum(ba*bc, axis=1)
    
    # Row-wise L2 norm
    # https://stackoverflow.com/questions/7741878/how-to-apply-numpy-linalg-norm-to-each-row-of-a-matrix
    norm_ba = np.sum(np.abs(ba)**2,axis=-1)**(1./2)
    norm_bc = np.sum(np.abs(bc)**2,axis=-1)**(1./2)
    
    # Consine calc
    cosine_angle = row_wise_dot /(norm_ba * norm_bc)
    
    # Get angle from arc cosine
    angle = np.arccos(cosine_angle)
    
    # Convert to degrees 
    angle_degrees = np.degrees(angle)
    
    return angle_degrees

# foo = df[df['obj_id'] == 16]
# foo = df[df['obj_id'] == 197]
# foo = df[df['obj_id'] == 53]

# foo['angle'] = get_angle(foo)

# fig = px.scatter(foo, x="cx", y="cy", text="frame", log_x=True, size_max=60)
# fig.show()

# foo.to_csv('temp.csv', index = False)