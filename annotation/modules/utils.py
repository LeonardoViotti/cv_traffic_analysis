import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# Functions
def ishow(img, max_t = 10000):
    # Kill window if Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        v2.destroyAllWindows()
    
    # Show image
    cv2.imshow('Preview',img)
    
    # Otherwise kill after max_t, default 10s
    cv2.waitKey(max_t)
    cv2.destroyAllWindows()

def vshow(cap):
    while True:
        sucess, img_i = cap.read()
        cv2.imshow("Video", img_i)
        # Break out by pressing 'q' when window is selected
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # When everything done, release the capture
    # cap.release()
    cv2.destroyAllWindows() 

def hstack_images(img1, img2):
    #https://stackoverflow.com/a/24522170/8692138
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    #create empty matrix
    vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
    
    #combine 2 images
    vis[:h1, :w1,:3] = img1
    vis[:h2, w1:w1+w2,:3] = img2
    return vis
