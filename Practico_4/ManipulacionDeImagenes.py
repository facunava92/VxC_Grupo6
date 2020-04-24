#!/usr/bin/python
#Usando como base el programa anterior, escribir un programa que permita seleccionar un rectángulo de una imagen, luego
#   “g” guarda la imagen dentro del rectángulo en el disco,
#   “r” restaura la imagen original y permite realizar nuevamente la selección,
#   “q” finaliza.

import cv2


xi, yi = -1, -1
xf, yf = -1, -1
width = 2
drawing = False

def crop_image(event, x, y, flags, param):
    global xi, yi, xf, yf, drawing, img, width
    if (event == cv2.EVENT_LBUTTONDOWN):
        xi, yi = x, y
        print('xi= ', xi, 'yi= ', yi)
        drawing = True
    elif (event == cv2.EVENT_MOUSEMOVE):
        if (drawing):
            img = roi.copy()
            cv2.rectangle(img, (xi, yi), (x,y), (255, 0, 0), width)
    elif (event == cv2.EVENT_LBUTTONUP):
        drawing = False
        if(x < 0):
            x = 0 
        if(y < 0):
            y = 0
        xf, yf = x, y   


img = cv2.imread('lenna.png', cv2.IMREAD_COLOR)
backup = img.copy()
roi = img.copy()
cv2.namedWindow('Lenna')
cv2.setMouseCallback('Lenna', crop_image, img)

while(True):
    cv2.imshow('Lenna', img)
    option = cv2.waitKey(1) & 0b11111111    #Enmascaro con una AND
    if option == ord('r'):
        img = backup.copy()
        roi = backup.copy()

    elif option == ord('g'):
        if(xf<0 or yf<0):
            cv2.imwrite('Lenna_crop.png', roi)
        else:
            if (xi > xf):
                xi, xf = xf, xi
            if (yi > yf):
                yi, yf = yf, yi

            roi = backup[yi:yf, xi:xf]
            img = roi.copy()
            cv2.imwrite('Lenna_crop.png', roi)
            xf, yf = -1, -1
    
    elif option == ord('q'):
        break
