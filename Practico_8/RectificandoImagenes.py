#!/usr/bin/python
#Teniendo en cuenta que:
#una homografía se representa con una matriz de 3 × 3 (pero tiene sólo 8 grados delibertad) y puede ser recuperada con 4 puntos no colineales.
#   A. Crear un método que compute la homografía entre los 4 puntos seleccionados y las esquinas de la segunda imagen de m × n pixeles.
#   B. Usando como base el programa anterior, escriba un programa que
#     * con la letra “h” permita seleccionar con el mouse 4 puntos no colineales en una imagen y transforme (rectifique) la selección en 
#        una nueva imagen rectangular.
#   Ayuda
#           cv2.getPerspectiveTransform
#           cv2.warpPerspective

import cv2
import numpy as np


selected_points = []


def perspective(image, src_pts, dst_pts):

    (h, w) = image.shape[:2]
    
    P = cv2.getPerspectiveTransform(src_pts, dst_pts)
    rectified = cv2.warpPerspective(img, P, (w, h))

    return rectified 


def mouse_callback (event, x, y, flags, param):
    global selected_points, show_img

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append([x, y])
        cv2.circle(show_img, (x, y), 3, (0, 255, 0), -1)


def select_points(image, points_num):
    global selected_points
    selected_points = []
    cv2.namedWindow('Seleccione_4_Puntos')
    cv2.setMouseCallback('Seleccione_4_Puntos', mouse_callback)

    while True:
        cv2.imshow('Seleccione_4_Puntos', image)
        k = cv2.waitKey(1)
        if len(selected_points) == points_num:
            break
    cv2.destroyAllWindows()

    return np.array(selected_points, dtype=np.float32)


img = cv2.imread('tarjeta.jpg', cv2.IMREAD_COLOR)
backup = img.copy()
(h, w) = img.shape[:2]

while (True):
    cv2.imshow('Perspective', img)
    option = cv2.waitKey(1) & 0b11111111  # Enmascaro con una AND

    if option == ord('h'):
        cv2.destroyAllWindows()
        show_img = backup.copy()
        img = backup.copy()
        src_pts = select_points(show_img, 4)
        print (src_pts)
        dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        img = perspective(img, src_pts, dst_pts)

    elif option == ord('g'):
        cv2.imwrite('rectified.png', img)

    elif option == ord('q'):
        break

