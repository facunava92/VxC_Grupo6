#!/usr/bin/python
#Escribir un programa en python que lea una imagen y realice un umbralizado binario,guardando el resultado en otra imagen.
#   NOTA: No usar ninguna función de las OpenCV, excepto para leer y guardar la imagen
#la idea es generar una mascara , donde cada valor va represetnar la intecidad de las imagenes , todo lo que superer ese umbral se represente en blanco y todo lo que represente en negro
import cv2


img = cv2.imread ('hoja.png', cv2.IMREAD_GRAYSCALE) #leer en escala de grises los pixeles q no sean blanco y negro van a estar entre 0 y 255

#image[0, 0] el primero indica y o row(fila) y el segundo x o column(columna).
#image[0, 0, 0] igual, agrego el canal de color BGR respectivamente. 
#image[0,0] upper-left corner
umbral = int(input('Introduzca el valor del umbral entre 0-255: ')) 
x , y= img.shape  #dimencione de la imagen 
#utilizamos 2 for anidados , donde voy a recorrer las filas 
for row in range(x):    # trae todas las filas
    for col in range(y):   # trae cada uno de los elementos de esa fila 
        if (img[row, col]  <= umbral): #si el elemento de la iamgen es menor al umbral 
            img[row, col] = 0

cv2.imwrite('resultado.png', img)
cv2.imshow('Imagen Umbralizada', img)
cv2.waitKey(0) #espera que el usuario preciono una tecla
cv2.destroyAllWindows() 
