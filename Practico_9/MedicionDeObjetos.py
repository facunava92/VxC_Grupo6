



import cv2
import numpy as np
import math

selected_points = []
patron=0.01990049751        # patron  de referencia en cm/pixeles

def mouse_callback (event, x, y, flags, param):
    global selected_points, show_img

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append([x, y])
        cv2.circle(show_img, (x, y), 4, (0, 255, 0), -1)

def select_points(image, points_num):
    global selected_points
    selected_points = []                    # tupla de dos elementos (puntos)
    cv2.namedWindow('Seleccione_2_Puntos')
    cv2.setMouseCallback('Seleccione_2_Puntos', mouse_callback)

    while True:
        cv2.imshow('Seleccione_2_Puntos', image)
        k = cv2.waitKey(1)
        if len(selected_points) == points_num:
            break
    cv2.destroyAllWindows()

    return np.array(selected_points, dtype=np.float32)

def distancia (ptos):
    a= ptos [0]
    b=ptos [1]
    dst = np.linalg.norm(a-b)  #calcula la distancia 
    return dst

img = cv2.imread('card.jpg', cv2.IMREAD_COLOR) 
backup = img.copy()


while (True):
    cv2.imshow('tecla "h" habilita la seleccion de puntos  ', img)
    option = cv2.waitKey(1) & 0b11111111  # Enmascaro con una AND


    if option == ord('h'):
        cv2.destroyAllWindows()
        show_img = backup.copy()
        puntos = select_points(show_img, 2)
        distan=distancia(puntos)
        distanc=round(distan)
        dist=distanc*patron   
        cv2.putText(show_img,"Medicion: {:.1f} cm".format(dist),(400,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1, cv2.LINE_AA)
        cv2.imshow('medicion  ', show_img)

    elif option == ord('q'):
        break

h