import cv2
import numpy as np

a,x1,y1,x2,y2,x3,y3,x4,y4=0,0,0,0,0,0,0,0,0

dst=np.float32([[0,480],[0,0],[640,0],[640,480]])
scr=np.float32([[0,0],[0,0],[0,0],[0,0]])

#Función para seleccionar los puntos
def puntos(event,x,y,flag,params):
	global a,x1,y1,x2,y2,x3,y3,scr
	if event==cv2.EVENT_LBUTTONDOWN:
		cv2.circle(tarjeta,(x,y),3,(255,0,0),-1)

	if event==cv2.EVENT_LBUTTONUP:
		if a==0:
			(x1,y1)=(x,y)
		if a==1:
			(x2,y2)=(x,y)
		if a==2:
			(x3,y3)=(x,y)
		if a==3:
			(x4,y4)=(x,y)
			scr=np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
		a=a+1

#Función para hacer la transformación afin
def homografia(tarjeta,scr,dst):
	M=cv2.getPerspectiveTransform(scr,dst)
	rect=cv2.warpPerspective(tarjeta,M,(640,480))
	return rect

#Leo la imagen del Tarjeta y selecciono los puntos
tarjeta=cv2.imread('amex.jpg')
cv2.namedWindow('Imagen')
cv2.setMouseCallback('Imagen',puntos)

while(1):
	cv2.imshow('Imagen',tarjeta)
	g=cv2.waitKey(1)&0xFF
	if a == 4:
		cv2.destroyAllWindows()
		break
			
imagen=homografia(tarjeta,scr,dst)

#Muestro la imagen transformada
while(1):
	cv2.imshow('Imagen2',imagen)
	k = cv2.waitKey(1) & 0xFF
	if k == ord('s'):
		break	

cv2.destroyAllWindows()

