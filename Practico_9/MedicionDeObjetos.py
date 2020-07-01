#!/usr/bin/python 
import cv2
import numpy as np
 
patron=0.04926108374    # proporcion de medicion con respecto a la medida en cm/pixeles    (203 pixeles(de la base ) / 10 cm   = 20.3       20.3*x = 1 cm    x= 0.04926108374 )


def perspective(image, src_pts, dst_pts):
    (h, w) = image.shape[:2]
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    rectified = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
    return rectified

def g_contour(image,ud_img):
    #El operador laplaciano también es un operador derivado que se usa para encontrar bordes en una imagen. Es una máscara derivada de segundo orden. En esta máscara tenemos dos clasificaciones adicionales, una es Operador laplaciano positivo y otra es Operador laplaciano negativo.

    #A diferencia de otros operadores, Laplacian no sacó bordes en ninguna dirección en particular
    edges = cv2.Laplacian(gray_img, cv2.CV_8U, gray_img, ksize=5) # 5 kernel , iamgen destino salida  ,tipo de datos cv2.CV_8U Entero sin signo de 1 byte 
    #l algoritmo Canny pretende satisfacer tres criterios principales:Baja tasa de error: lo que significa una buena detección de solo los bordes existentes.
        #Buena localización: la distancia entre los píxeles de borde detectados y los píxeles de borde reales deben minimizarse.

#Baja tasa de error: lo que significa una buena detección de solo los bordes existentes.
#Buena localización: la distancia entre los píxeles de borde detectados y los píxeles de borde reales deben minimizarse.
#Respuesta mínima: solo una respuesta del detector por borde.
    edges = cv2.Canny(edges, 100, 300)   #detector de borde ....umbralizdo 100  Dado que la detección de bordes es susceptible al ruido en la imagen, el primer paso es eliminar el ruido en la imagen con un filtro gaussiano de 5x5. #100 umbral menor #300 el humbral mayor
    contours, hierachy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Método de aproximación de contorno
    #almacena absolutamente todos los puntos de contorno.
    #cv2.RETR_EXTERNAL recupera solo los contornos exteriores extremos.
    # cv2.CHAIN_APPROX_SIMPLE solo necesitamos dos puntos finales de esa línea
    #contornos : contornos detectados. Cada contorno se almacena como un vector de puntos.
    #vector de salida opcional, que contiene información sobre la topología de la imagen
    #Tiene tantos elementos como el número de contornos
    for c in contours:
        if cv2.contourArea(c) < 300:  #El área de contorno sea menor al umbral 300 umbral mayor 
            continue
        x, y, w, h = cv2.boundingRect(c) #dimenciones de los contornos Sea (x, y) la coordenada superior izquierda del rectángulo y (w, h) su ancho y alto.
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        #print (w,h)
        base=round (w*patron ,1)
        altura=round (h*patron , 1)
        if base == 2.4 or base==2.5:
            radio = base/2
            cv2.putText(img, "Rad: {:.2f}  ".format(radio), (x-1, y+70), cv2.FONT_HERSHEY_COMPLEX,0.29,(0, 0, 255), 1)   #texto , ubicacion , fuente , factor de escala color y el ancho 
            cv2.putText(img, "{:.1f} x {:.1f} ".format(base,altura), (x-9 , y-7 ), cv2.FONT_HERSHEY_COMPLEX,0.29,(0, 0, 255), 1)
        else:
            cv2.putText(img, "{:.1f} x {:.1f} cm ".format(base,altura), (x+9, y-10 ), cv2.FONT_HERSHEY_COMPLEX,0.4,(0, 0, 255), 1)

    horizontal_concat = np.concatenate((ud_img, img), axis=1)
    cv2.imshow('Resultado', horizontal_concat)
    cv2.imwrite('resultado.png', horizontal_concat)

img = cv2.imread('a_medir.jpg')
bkup = img.copy()
cv2.imshow('Perspective', img)

while True:
    option = cv2.waitKey(1) & 0b11111111  # enmascaro con una and
    if option == ord('h'):   #cuando apreto h 
        img = bkup.copy()
        m = bkup.copy()
        cv2.destroyAllWindows()
        dst_pts = np.array([[53, 105], [253, 105], [253, 305], [53, 305]], dtype=np.float32) #  dst_pts (ptos destinos ) harcodeo para que esten en la misma linea y que sean equidistante  
        selected_points = ([[55,  105], [248,  136], [246, 326], [28, 310]])     # harcodeo los selected_point (pto 8 ) 
        
        src_pts = np.array(selected_points, dtype=np.float32) # lo convierto en variable tipo float 
        img = perspective(img, src_pts, dst_pts)  #y los envio a la funcion perspective para rectificar
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #paso a escala de grises
        gray_img = cv2.GaussianBlur(gray_img, (5,5), 11) #11 como una mascarita (es un bloque que se mueve ))(combolucion) sobre la imagen ,   El filtro decomvolucion gaussiano es un filtro de paso bajo que elimina los componentes de alta frecuencia se reducen.  La altura y el ancho deben ser impares y pueden tener valores diferente (osea el kernel) , Desviación estándar x , y  del kernel , 
        g_contour(gray_img,m)  # envio  gray_img  y "m" lo envio solo para concatenar (pegar) las imagenes 
    elif option == ord('g'):
        cv2.imwrite('rectificado.png', img)  #escribo la imgaen como rectificado.png

    elif option == ord('q'):
        break

