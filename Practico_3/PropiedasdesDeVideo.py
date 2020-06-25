#!/usr/bin/python
#A. ¿Cómo obtener el frame rate o fps usando las OpenCV? Usarlo para no tener que harcodear el delay del waitKey.
#B. ¿Cómo obtener el ancho y alto de las imágenes capturadas usando las OpenCV? Usarlo para no tener que harcodear el frameSize del video generado

import cv2 


videoCapture = cv2.VideoCapture('tierra.avi')  #si le pasamos un archivo , creamos un objeto de video captura   objero de tipo videcpature

fourcc_XVID = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D') #.avi    # fourcc_XVID es la variable para codificacion de video ,  cramos el codigo de 4 caracteres , en este caso xvid
#fourcc_I420 = cv2.VideoWriter_fourcc('I', '4', '2', '0') #.avi
#fourcc_PIM1 = cv2.VideoWriter_fourcc('P', 'I', 'M', '1') #.avi
#fourcc_MP4V = cv2.VideoWriter_fourcc('M', 'P', '4', 'V') #.mp4
#fourcc_X264 = cv2.VideoWriter_fourcc('X', '2', '6', '4') #.mp4
#fourcc_THEO = cv2.VideoWriter_fourcc('T', 'H', 'E', 'O') #.ogv
#fourcc_FLV1 = cv2.VideoWriter_fourcc('F', 'L', 'V', '1') #.flv
fps = videoCapture.get(cv2.CAP_PROP_FPS)                                #usamos para obtener el FPS del video para no harkodearlo        
framesize = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),          #obtener el tamaño exacto   ancho y alto  de la fram que va a capturar           
             int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
delay = int(1000/fps)                         # dividir un segundo por la cantidad de frame               
success, frame = videoCapture.read()   # leemos un cuadro (frame) , sucess es un boleano si es si se leyo o no  el frame
#(framesize )= frame.shape [:2]  # otra forma de obtener el ancho y alto 
videoWriter = cv2.VideoWriter('output_XVID.avi', fourcc_XVID, fps, framesize, isColor = False) #sirve para escribir video es una clase video , nos sirve para escrbir video ...IS COLOR  indica si el video es en escala de grises o en color
while success:   # si leyo las frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    #convierto a escala de  grises  
    videoWriter.write(gray) # lo escribo 
    cv2.imshow('Image gray', gray)  # lo muestra
    if (cv2.waitKey(delay) & 0b11111111) == ord('q'):        # se queda esperando un tiempo , preugntando si es usuario apreto la q  o si el tiempoq ue hace falta para leer la proxima imagen                                
        break
    success, frame = videoCapture.read()   # si no apreto la q leemos el otro cuadro 

videoCapture.release()  #libera recursos 
videoWriter.release()    # libera recursos 
cv2.destroyAllWindows()  # destryte la ventana 

