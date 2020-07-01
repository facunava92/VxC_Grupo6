#!/usr/bin/python
#Usando como base el programa anterior, escribir un programa que permita seleccionar un rectángulo de una imagen, luego
#   “g” guarda la imagen dentro del rectángulo en el disco,
#   “r” restaura la imagen original y permite realizar nuevamente la selección,
#   “q” finaliza.

import cv2

xi, yi, xf, yf = 0, 0, 0, 0   # inicializamos (harcodeamos las coroendas xi , yf en cero)
drawing = False

def crop_image(event, x, y, flags, param): #evento que sucedio #banderas #pos x e y posicion del mouse  , 
    w = 2
    global xi, yi, xf, yf, drawing, img, width    # 
    if (event == cv2.EVENT_LBUTTONDOWN): # si el evento aprete el boton izqitod y lo baje 
        xi, yi = x, y                   #asigno las variables xi , yi al x y del mouse (coordenadas )
        drawing = True                    #yo tengo el boton apretado del mause

    elif (event == cv2.EVENT_MOUSEMOVE): #el puntero del mouse se movio ?
        if (drawing):
            img = roi.copy()
            cv2.rectangle(img, (xi, yi), (x,y), (255, 0, 0), w) # dibujo el ractangulo , pto inicial final y en la posicion actual donde se movio el mouse
    elif (event == cv2.EVENT_LBUTTONUP):              #si  suelto el boton
        drawing = False                               #Registrar las coordenadas finales (x, y) e indicar que
                                                      # la operación de recorte ha finalizado
        if(x < 0):                     
             x = 0 
        if(y < 0):
             y = 0
        xf, yf = x, y                                #inicializo las coordenadas finales con las ultimas coordenadas del mouse


img = cv2.imread('lenna.png', cv2.IMREAD_COLOR)   #leo la imagen (cargo la imagen )
xi, yi = 0, 0                               # harcodeo las coordenadas iniciales 
xf, yf = img.shape[0], img.shape[1]          #me devuelve el tamaño de la imagen y lo guardo en xf , yf
backup = img.copy()
roi = img.copy()


img = cv2.imread('lenna.png', cv2.IMREAD_COLOR)
xi, yi = 0, 0     
xf, yf = img.shape[0], img.shape[1] 
backup = img.copy()
roi = img.copy()

cv2.namedWindow('Lenna')
cv2.setMouseCallback('Lenna', crop_image, img) 

cv2.namedWindow('Lenna')                #creamos una ventana de nombre lenna
cv2.setMouseCallback('Lenna', crop_image, img)    #y a esa ventana le agregamos un mousecallback , pasamos como parametro la funcion y la imagen  
# cuando se produzca un evento del Mouse  en esa ventana se llama a la funcion , y esa funcion necesita la imagen que cargo

while(True):
    cv2.imshow('Lenna', img)
    option = cv2.waitKey(1) & 0b11111111    #Enmascaro con una AND   
    if option == ord('r'):        # restaurar la imagen colocando sus estados iniciales 
        img = backup.copy()
        roi = backup.copy()
        xi, yi = 0, 0
        xf, yf = img.shape[0], img.shape[1]


    elif option == ord('g'):
        xi, xf = min(xi, xf), max(xi, xf)  #   coordenada mini y maxi de la variable X
        yi, yf = min(yi, yf), max(yi, yf)       #lo mismo en la varbiale Y 
        roi = backup[yi:yf, xi:xf]             # darle esoss ptos (guardo en roi) en mi imgane roi  pegarlas en IMG       
        img = roi.copy()
        cv2.imwrite('Lenna_crop.png', roi)

    
    elif option == ord('q'):
        break


