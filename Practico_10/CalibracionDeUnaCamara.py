#!/usr/bin/python
#Utilizando el template dado calibrar la camara usando las imagenes del patron ya capturadas.
#Corroborar la correcta calibracion usando el resultado de calibracion obtenido para eliminar la
#distorsion de las imagenes dadas.

import cv2
import numpy as np

camera_matrix = np.load('camera_mat.npy')        #carga una matriz guradada en un archivo (MATRIZ DE CAMARA)
dist_coefs = np.load('dist_coefs.npy')              #parametros de distorcionk k p1 p2 
pattern_size = (11, 8)                          #numero de esquinas internas 

img = cv2.imread('distortioned.JPG')            #leo la iamgen distortioned
img = cv2.resize(img, (800, 600))                   # re-dimenciono
img_show = img.copy()                   #HAGO UNA COPIA
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #paso a escala de grises

res, corners = cv2.findChessboardCorners(img_show, pattern_size)       #Encuentra las posiciones de las esquinas internas , paso la imagen y los  número de esquinas internas por fila y columna
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 1e-3)  # determina la diferencia en la posición de esquina anterior y siguiente , siempre que sea menor que 1e-3 o se haya ejecutado  30 iteracciones
corners = cv2.cornerSubPix(gray, corners, (10, 10), (-1, -1), criteria) #refinamos nivel subpixel #imagen , coordenadas iniciales de las esquinas de entrada , la mitad de la longitud lateral de la ventana de búsqueda ,se usa una ventana de búsqueda.
#Criterios para la terminación del proceso iterativo de refinamiento de esquina 
h_corners = cv2.undistortPoints(corners, camera_matrix, dist_coefs)  # calcula ptos ideales con respeto a los meiddos (pixeles), adimencional
h_corners = np.c_[h_corners.squeeze(), np.ones(len(h_corners))]  #aca saco una dimencion 

img_pts, _ = cv2.projectPoints(h_corners, (0, 0, 0), (0, 0, 0), camera_matrix, None) #Proyecta puntos 3D a un plano de imagen. matriz de puntos de objeto , Vector de rotación ,  Vector de traducción.
#la matriZ a 
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

ud_img = cv2.undistort(img, camera_matrix, dist_coefs) #elimina la distorsión de las lentes de la imagen.
cv2.imwrite('undistortioned.jpg', ud_img)

horizontal_concat = np.concatenate((img, ud_img), axis=1)

cv2.imshow('result', horizontal_concat)
cv2.waitKey()

cv2.destroyAllWindows()

