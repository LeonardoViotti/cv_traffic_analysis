# Show a bit more stable
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Functions
def stable_show(img, t = 5000):
    cv2.startWindowThread()
    cv2.namedWindow("preview")
    cv2.imshow('preview', img)
    cv2.waitKey(t)
    cv2.destroyAllWindows()