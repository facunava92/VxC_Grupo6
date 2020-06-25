#!/usr/bin/python
#Teniendo en cuenta que, una transformación afín se representa con una matriz de 2 × 3 (tiene 6 grados de libertad y
#puede ser recuperada con 3 puntos no colineales.
#   A. Crear una método que compute la transformación afín entre los 3 puntos seleccionados
#   B. Usando como base el programa anterior, escriba un programa que
#    con la letra “a” permita seleccionar con el mouse 3 puntos no colineales en una imagen e incruste entre estos puntos seleccionados una segunda imagen.
#   Ayuda
#           cv2.getAffineTransform
#           cv2.warpAffine
#   Generar una máscara para insertar una imagen en otra

import cv2
import numpy as np

selected_points = []
def affine(image, src_pts, dst_pts):
    (h, w) = image.shape[:2]
    A = cv2.getAffineTransform(src_pts, dst_pts)
    affine= cv2.warpAffine(img, A, (w, h), borderValue=(255, 255, 255))
    
    return affine 
def mouse_callback (event, x, y, flags, param):
    global selected_points, show_img
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append([x, y])
        cv2.circle(show_img, (x, y), 3, (200, 200, 0), -1)

def select_points(image, points_num):
    global selected_points
    selected_points = []
    cv2.namedWindow('Seleccione_3_Puntos')
    cv2.setMouseCallback('Seleccione_3_Puntos', mouse_callback)

    while True:
        cv2.imshow('Seleccione_3_Puntos', image)
        k = cv2.waitKey(1)
        if len(selected_points) == points_num:
            break
    cv2.destroyAllWindows()

    return np.array(selected_points, dtype=np.float32)

img = cv2.imread('opencv-logo.png', cv2.IMREAD_COLOR)
img_dst = cv2.imread('ojo.jpg', cv2.IMREAD_COLOR)
img_dst = cv2.resize(img_dst, img.shape[1::-1])  # redimenciono inicio: fin: paso  invierte los 

backup_src = img.copy()  #python
backup_dst = img_dst.copy() # ojo
(h, w) = img.shape[:2]


while (True):
    cv2.imshow('Affine', img)
    option = cv2.waitKey(1) & 0b11111111  # Enmascaro con una AND

    if option == ord('a'):
        cv2.destroyAllWindows()
        show_img = backup_src.copy()  
        src_pts = select_points(show_img, 3)  #le pmando la imagen python , deveulve el array con los 3 ptos 
        show_img = img_dst.copy()  # imgaen ojo  dimencionada y lo guardo a show_img
        dst_pts = select_points(show_img, 3)   #le mando la imagen ojo   ,  deveulve el array con los 3 ptos 
        
        img = backup_src.copy()   # python
        img_dst = backup_dst.copy()  #ojo

        img =  affine(img, src_pts, dst_pts) #ojo , ptos python , ptos ojo

        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #python es img2gray
        #cv2.imshow('A',img2gray)
        for row in range(h):
            for col in range(w):
                if img2gray[row, col] <= 20:  #menor o igual a 20 umbral
                    img2gray[row, col] = 255      #coloco a 255 (blanco )  # obtendrá blanco y negro
         #generamos mascara
        #cv2.imshow('A',img2gray)
        # El primer parámetro aquí es la imagen. El siguiente parámetro es el umbral, estamos eligiendo 200. El siguiente es el valor máximo, que estamos eligiendo como 255. Luego y finalmente tenemos el tipo de umbral, que hemos elegido como THRESH_BINARY
        ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY ) #umbraizacion
        #cv2.imshow('A',mask)  #blanco y negro la imagen de ptthon 
        mask_inv = cv2.bitwise_not(mask)
        #cv2.imshow('A',mask_inv)  #imagen inversa lo que era blanco ahora e engro 
        #cv2.imshow('A',img_dst)
       # cv2.imshow('r',mask)
        img1_bg = cv2.bitwise_and(img_dst, img_dst, mask=mask)  #2 imagens ojo y 1 (blanco y negro) ptyon con la mascara del combinacuon lineal de las imagenes  #python combnacion lineal
        #cv2.imshow('A',img1_bg) # me aparece el ojo con el python negro
        img2_fg = cv2.bitwise_and(img, img, mask=mask_inv)  #ojo mascara inversa
      #  cv2.imshow('A',img2_fg)  me aparece todo negro y el pthon a color 
        img2_fg = cv2.blur(img2_fg, (20,20))
        #cv2.imshow('A',img2_fg) desenfocado
        img = cv2.add(img1_bg, img2_fg) #sumo las 2 imagenes  (hace una combinacion lineal . hacer uq el igane este superpuesta sobre la otra)

    elif option == ord('g'):
        cv2.imwrite('Affine_eye.png', img)

    elif option == ord('q'):
        break

