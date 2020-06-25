#!/usr/bin/python
#A. Crear un método nuevo que aplique una transformación euclidiana, recibiendo los siguientes parámetros:
#Parámetros
#    * angle: Ángulo
#    * tx: traslación en x
#    * ty: traslación en y
#    Recordar que la transformación euclidiana tiene la siguiente forma:
#        [cos(angle)  sin(angle) tx]
#        [-sin(angle) cos(angle) ty]
#B. Usando como base el programa anterior, escriba un programa que:
#permita seleccionar un rectángulo de una imagen y con la letra “e” aplique una transformación 
#euclidiana a la imagen dentro del rectángulo y la guarde en el disco y sale.


import cv2
from fn_transform import euclidean
from fn_transform import similarity

xi, yi, xf, yf = 0, 0, 0, 0
angle, tx, ty = 45, 80, 80
scale = 2.0
drawing = False

def crop_image(event, x, y, flags, param):
    w = 2
    global xi, yi, xf, yf, drawing, img, width
    if (event == cv2.EVENT_LBUTTONDOWN):
        xi, yi = x, y
        drawing = True

    elif (event == cv2.EVENT_MOUSEMOVE):
        if (drawing):
            img = roi.copy()
            cv2.rectangle(img, (xi, yi), (x,y), (255, 0, 0), w)

    elif (event == cv2.EVENT_LBUTTONUP):
        drawing = False
        if(x < 0): x = 0 
        if(y < 0): y = 0
        xf, yf = x, y   


img = cv2.imread('python.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img,(800, 600))
xi, yi = 0, 0
xf, yf = img.shape[1], img.shape[0]
backup = img.copy()
roi = img.copy()

cv2.namedWindow('Practico 6')
cv2.setMouseCallback('Practico 6', crop_image, img)


while(True):
    cv2.imshow('Practico 6', img)
    option = cv2.waitKey(1) & 0b11111111    #Enmascaro con una AND
    if option == ord('r'):
        img = backup.copy()
        roi = backup.copy()
        xi, yi = 0, 0
        xf, yf = img.shape[1], img.shape[0]

    elif option == ord('g'):
        xi, xf = min(xi, xf), max(xi, xf)
        yi, yf = min(yi, yf), max(yi, yf)
        roi = backup[yi:yf, xi:xf]
        img = roi.copy()
        cv2.imwrite('python_crop.png', roi)
    
    elif option == ord('e'):
        xi, xf = min(xi, xf), max(xi, xf)
        yi, yf = min(yi, yf), max(yi, yf)
        roi = backup[yi:yf, xi:xf]
        roi = euclidean(roi, angle, tx, ty)
        img = roi.copy()
        cv2.imwrite('gnu_logo_euclidean.png', roi)

    elif option == ord('s'):
        xi, xf = min(xi, xf), max(xi, xf)
        yi, yf = min(yi, yf), max(yi, yf)
        roi = backup[yi:yf, xi:xf]
        roi = similarity(roi, angle, tx=40, ty=40, scale=1.8)
        img = roi.copy()
        cv2.imwrite('python_similarity.png', roi)
    
    elif option == ord('q'):
        break


