
import cv2
import numpy as np
import math


selected_points = []


def mouse_callback (event, x, y, flags, param):
    global selected_points, show_img

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append([x, y])
        cv2.circle(show_img, (x, y), 3, (0, 255, 0), -1)



def select_points(image, points_num):
    global selected_points
    selected_points = []                    #voy a saber el tamaño de tupla (osea cunado tenga 2 ptos)
    cv2.namedWindow('Seleccione_2_Puntos')
    cv2.setMouseCallback('Seleccione_2_Puntos', mouse_callback)

    while True:
        cv2.imshow('Seleccione_2_Puntos', image)
        k = cv2.waitKey(1)
        if len(selected_points) == points_num:
            break
    cv2.destroyAllWindows()

    return np.array(selected_points, dtype=np.float32)

def distiancia (ptos):
    a= ptos [0]
    b=ptos [1]
    dst = np.linalg.norm(a-b)  #calcula la distancia 
    return dst

img = cv2.imread('card.jpg', cv2.IMREAD_COLOR) #leo la imagen
backup = img.copy()


while (True):
    cv2.imshow('Perspective', img)
    option = cv2.waitKey(1) & 0b11111111  # Enmascaro con una AND


    if option == ord('h'):
        cv2.destroyAllWindows()
        show_img = backup.copy()
        puntos = select_points(show_img, 2)
        distan=distiancia(puntos)
        distanc=round(distan)
        dist=distanc*0.01538461538   # patron 
        #print ("la distancia entre los ptos son",distanc, "cm" )
        print ("la distancia entre los ptos son", "{:.1f}".format(dist) , "cm")

    elif option == ord('q'):
        break

