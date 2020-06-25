#!/usr/bin/python
import cv2
import glob
import numpy as np

pattern_size = (11, 8)  # 11 esquinas por 8 esquinas
samples = []

images = glob.glob('tmp/*.JPG')   # paquete glob que me permite todaas las imagenes que estan en un directorio

for fname in images:   #recoremos las listas de iamgenes 
    img = cv2.imread(fname)  #leo y lo guardo 
    img = cv2.resize(img, (800, 600))  #redimenciono 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #paso a escala de grises
    res, corners = cv2.findChessboardCorners(img, pattern_size, None)  #encontramos las esquinas 
    img_show = img.copy()  #copio en img_show
    cv2.drawChessboardCorners(img_show, pattern_size, corners, res)  #dibumaos
    cv2.putText(img_show, 'Muestra numero: %d' % (len(samples)+1), (0, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow('chessboard', img_show) #muestro 
    wait_time = 0 if res else 30 #espero 30 
    k = cv2.waitKey(wait_time)
    if k == ord('s') and res:
        samples.append((gray, corners))  #  agregamos los ptos que genermaos en los ptos 3d los agramos en SAMPLE
    elif k == 27:
        break

cv2.destroyAllWindows()  #CUANDO SE RECORRIENRON TODAS LAS IMAGENES TERMIN

criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 1e-3) 

for i in range(len(samples)):                 # recooremos los ptos de Sanple                           
    img, corners =samples[i]                    # lo guaramos en img y corners 
    corners = cv2.cornerSubPix(img, corners, (10, 10), (-1, -1), criteria)   # resolucion de subpixel
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)   # definimos los ptos en 3d .. 9 x 7 esquinas con 3 elemtos  , guarmamos acad ubicacion de la esquinas en el mundo 3d y lo incialiaos en cero  
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)    # genera combinaciones (pares de ptos (que la primer esquina esta en 00 , la segundo en 01 )) y eso lo guardamos en las primeras 2 compoenntes del pto objeto
images, corners = zip(*samples)        #los agrega en una tupla y lo devuelve.                                         #
pattern_points = [pattern_points]*len(corners)             #
rms, camera_matrix, dist_coefs, rvecs, tvecs =\                                             
    cv2.calibrateCamera(pattern_points, corners, images[0].shape, None, None)  # LLAMAMOS AL METODO DE CALIBRACION DE LA CAMARA     

np.save('camera_mat.npy', camera_matrix)
np.save('dist_coefs.npy', dist_coefs)
