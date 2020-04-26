#! /usr/bin/env python


import cv2
import numpy as np


def euclidean(image, angle,tx=0, ty=0):
    (h, w) = image.shape[:2]
    angle = np.radians(angle)   #Giro antihorario

    E = np.float32([[np.cos(angle), np.sin(angle), tx],
                    [-np.sin(angle), np.cos(angle), ty],
                    ])


    euclidean = cv2.warpAffine(image, E, (w, h))
    return euclidean

def similarity(image, angle,tx=0, ty=0, scale=1.0):
    (h, w) = image.shape[:2]
    angle = np.radians(angle)   #Giro antihorario

    S = np.float32([[scale*np.cos(angle),  scale*np.sin(angle), tx],
                    [-scale*np.sin(angle), scale*np.cos(angle), ty],
                    ])


    similarity = cv2.warpAffine(image, S, (w, h))
    return similarity

def affine(image, angle,tx=0, ty=0, scale=1.0):
    (h, w) = image.shape[:2]
    angle = np.radians(angle)   #Giro antihorario

    S = np.float32([[scale*np.cos(angle),  scale*np.sin(angle), tx],
                    [-scale*np.sin(angle), scale*np.cos(angle), ty],
                    ])


    affine = cv2.warpAffine(image, S, (w, h))
    return affine
