#!/usr/bin/python
#Escribir un programa en python que lea una imagen y realice un umbralizado binario,guardando el resultado en otra imagen.
#   NOTA: No usar ninguna función de las OpenCV, excepto para leer y guardar la imagen

import cv2


img = cv2.imread ('hoja.png', cv2.IMREAD_GRAYSCALE)

#image[0, 0] el primero indica y o row(fila) y el segundo x o column(columna).
#image[0, 0, 0] igual, agrego el canal de color BGR respectivamente. 
#image[0,0] upper-left corner
umbral = int(input('Introduzca el valor del umbral entre 0-255: ')) 
x, y= img.shape

for row in range(x):
    for col in range(y):
        if (img[row, col]  <= umbral):
            img[row, col] = 0

cv2.imwrite('resultado.png', img)
cv2.imshow('Imagen Umbralizada', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
