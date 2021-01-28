import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# Functions
def img_show(img, max_t = 10000):
    # Kill window if Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        v2.destroyAllWindows()
    
    # Show image
    cv2.imshow('Preview',img)
    
    # Otherwise kill after max_t, default 10s
    cv2.waitKey(max_t)
    cv2.destroyAllWindows()

def stable_show_vid(cap):
    while True:
        sucess, img_i = cap.read()
        cv2.imshow("Video", img_i)
        # Break out by pressing 'q' when window is selected
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # When everything done, release the capture
    # cap.release()
    cv2.destroyAllWindows() 

#------------------------------------------------------------------------------
# Image annotation functions

def draw_trajectories(img, trajectory_array):
    """
    This draws a trakjectorie line on a frame
    img -  has to be an image converted into numpy.ndarray
    trajectory_array - a single (p,2) numpy.ndarray or a (n,p,2) 
     numpy.ndarray, where n is the number of tracked objects, p is
     the number of frames each object was tracked for and 2 are
     the X and Y coordinates in relation to img.
    """
    # If just a single trajectory is passed
    if len(trajectory_array.shape) < 3:
        for p in range(1, len(trajectory_array)):
            cv2.line(img, tuple(trajectory_array[p-1]), tuple(trajectory_array[p]), (0, 0, 255), 2)
    else:
        # Otherwise loop over trajectories array to draw trajectories lines
        for t in range(0, len(trajectory_array)):
            # print(t)
            trajectory_t = trajectory_array[t]
            # Loop over each point in each trajectory
            for p in range(1, len(trajectory_t)):
                cv2.line(img, tuple(trajectory_t[p-1]), tuple(trajectory_t[p]), (0, 0, 255), 2)