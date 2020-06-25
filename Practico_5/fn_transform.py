#! /usr/bin/env python


import cv2
import numpy as np


def euclidean(image, angle,tx=0, ty=0):  #imagen , angulo , traslacion en x , translacion y 
    (h, w) = image.shape[:2]  #diemncion de la matriz 
    angle = np.radians(angle)   #Giro antihorario paso a radianes por que la mtris que utlixo recibe angle en radianres

    E = np.float32([[np.cos(angle), np.sin(angle), tx],                # genero la matriz esta 
                    [-np.sin(angle), np.cos(angle), ty],            
                    ])


    euclidean = cv2.warpAffine(image, E, (w, h))  #imgaen , la matriz transofrmacion , tamaño de la imgane de slaida 
    return euclidean
