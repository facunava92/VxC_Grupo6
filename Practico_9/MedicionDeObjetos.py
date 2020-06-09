#!/usr/bin/python 
import cv2
import numpy as np

patron=0.04926108374


def perspective(image, src_pts, dst_pts):
    (h, w) = image.shape[:2]
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    rectified = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
    return rectified

def g_contour(image,ud_img):
    edges = cv2.Laplacian(image, cv2.CV_8U, gray_img, ksize=5)
    edges = cv2.Canny(edges, 100, 300)
    contours, hierachy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        if cv2.contourArea(c) < 300:
            continue

        x, y, w, h = cv2.boundingRect(c)
        print (w,h)

        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        
        base=round (w*patron ,1)
        altura=round (h*patron , 1)
        if base == 2.4 or base==2.5:
            radio = base/2
            cv2.putText(img, "Rad: {:.2f}  ".format(radio), (x-1, y+70), cv2.FONT_HERSHEY_COMPLEX,0.29,(0, 0, 255), 1)
            cv2.putText(img, "{:.1f} x {:.1f} ".format(base,altura), (x-9 , y-7 ), cv2.FONT_HERSHEY_COMPLEX,0.29,(0, 0, 255), 1)
        else:
            cv2.putText(img, "{:.1f} x {:.1f} cm ".format(base,altura), (x+9, y-10 ), cv2.FONT_HERSHEY_COMPLEX,0.4,(0, 0, 255), 1)

    horizontal_concat = np.concatenate((ud_img, img), axis=1)
<<<<<<< HEAD:Practico_9/practico9.py
    cv2.imshow('Resultado', horizontal_concat)
    cv2.imwrite('RESULTADO.png', horizontal_concat)
=======
    cv2.imshow('resultado', horizontal_concat)
    cv2.imwrite('resultado.png', horizontal_concat)
>>>>>>> b81608253497f3242f2d250e165c5606f285f732:Practico_9/MedicionDeObjetos.py

img = cv2.imread('a_medir.jpg')
bkup = img.copy()
cv2.imshow('Perspective', img)

while True:
    option = cv2.waitKey(1) & 0b11111111  # enmascaro con una and
    if option == ord('h'):
        img = bkup.copy()
        m = bkup.copy()
        cv2.destroyAllWindows()
        dst_pts = np.array([[53, 105], [253, 105], [253, 305], [53, 305]], dtype=np.float32) 
        selected_points = ([[55,  105], [248,  136], [246, 326], [28, 310]]) 
        src_pts = np.array(selected_points, dtype=np.float32)
        img = perspective(img, src_pts, dst_pts)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.GaussianBlur(gray_img, (5, 5), 11)
        g_contour(gray_img,m)
    elif option == ord('g'):
        cv2.imwrite('rectificado.png', img)

    elif option == ord('q'):
        break

