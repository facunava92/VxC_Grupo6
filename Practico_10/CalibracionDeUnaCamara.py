#!/usr/bin/python
#Utilizando el template dado calibrar la camara usando las imagenes del patron ya capturadas.
#Corroborar la correcta calibracion usando el resultado de calibracion obtenido para eliminar la
#distorsion de las imagenes dadas.

import cv2
import numpy as np

camera_matrix = np.load('camera_mat.npy')
dist_coefs = np.load('dist_coefs.npy')
pattern_size = (11, 8)

img = cv2.imread('distortioned.JPG')
img = cv2.resize(img, (800, 600))
img_show = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

res, corners = cv2.findChessboardCorners(img_show, pattern_size)              # Se obtiene una aprox de las coordenadas de las esquinas 
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 1e-3)     # Aumento las precision usando "criteria" y "corners"
corners = cv2.cornerSubPix(gray, corners, (10, 10), (-1, -1), criteria)
    #corners (img a tratar, coordenadas aprox, dim de la zona interes, zona ignorada (-1,-1) no indica zonas) 
h_corners = cv2.undistortPoints(corners, camera_matrix, dist_coefs) # ELimina distorsion
h_corners = np.c_[h_corners.squeeze(), np.ones(len(h_corners))]

img_pts, _ = cv2.projectPoints(h_corners, (0, 0, 0), (0, 0, 0), camera_matrix, None) #Agregamos la coordenada z para proyectar 3D

for c in corners:
    cv2.circle(img_show, tuple(c[0]), 10, (0, 255, 0), 2)

for c in img_pts.squeeze().astype(np.float32):
    cv2.circle(img_show, tuple(c), 5, (0, 0, 255), 2)

cv2.imshow('undistorted corners', img_show)
cv2.waitKey()
cv2.destroyAllWindows()

img_pts, _ = cv2.projectPoints(h_corners, (0, 0, 0), (0, 0, 0), camera_matrix, dist_coefs)

for c in img_pts.squeeze().astype(np.float32):
    cv2.circle(img_show, tuple(c), 2, (255, 255, 0), 2)

cv2.imshow('reprojected corners', img_show)
cv2.waitKey()
cv2.destroyAllWindows()

ud_img = cv2.undistort(img, camera_matrix, dist_coefs)
cv2.imwrite('undistortioned.jpg', ud_img)

horizontal_concat = np.concatenate((img, ud_img), axis=1)

cv2.imshow('result', horizontal_concat)
cv2.waitKey()

cv2.destroyAllWindows()

