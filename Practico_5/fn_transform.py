#! /usr/bin/env python


import cv2
import numpy as np


def euclidean(image, angle,tx=0, ty=0):
    (h, w) = image.shape[:2]
    angle = np.radians(angle)   #Giro antihorario

    E = np.float32([[np.cos(angle), np.sin(angle), tx],
                    [-np.sin(angle), np.cos(angle), ty],
                    ])


    rotated = cv2.warpAffine(image, E, (w, h))
    return rotated
