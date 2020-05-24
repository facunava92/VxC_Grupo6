#!/usr/bin/python 
import cv2
import numpy as np

patron=0.04926108374


def perspective(image, src_pts, dst_pts):
    (h, w) = image.shape[:2]
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    rectified = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
    return rectified

def g_contour(image):
    edges = cv2.Laplacian(gray_img, cv2.CV_8U, gray_img, ksize=5)
    edges = cv2.Canny(edges, 100, 300)
    cv2.imshow('canny', edges)

    contours, hierachy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        if cv2.contourArea(c) < 300:
            continue

        x, y, w, h = cv2.boundingRect(c)
        
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        base=w*patron
        altura=h*patron
        cv2.putText(img, "{:.1f} x {:.1f} ".format(base,altura), (x-4 , y-5 ), cv2.FONT_HERSHEY_COMPLEX,
                            0.29,(0, 0, 255), 1)


img = cv2.imread('prueba2.JPG')
bkup = img.copy()


while True:
    cv2.imshow('perspective', img)
    option = cv2.waitKey(1) & 0b11111111  # enmascaro con una and

    if option == ord('h'):
        img = bkup.copy()
        cv2.destroyAllWindows()
        dst_pts = np.array([[53, 105], [253, 105], [253, 305], [53, 305]], dtype=np.float32)
        selected_points = ([[55,  105], [248,  136], [246, 326], [28, 310]])
        src_pts = np.array(selected_points, dtype=np.float32)
        img = perspective(img, src_pts, dst_pts)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.GaussianBlur(gray_img, (5, 5), 11)
        g_contour(gray_img)

    elif option == ord('g'):
        cv2.imwrite('rectificado.png', img)

    elif option == ord('q'):
        break
