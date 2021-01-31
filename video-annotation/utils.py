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
