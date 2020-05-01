#!/usr/bin/python
# Usando como base el programa anterior, escribir un programa que permita seleccionar un rectángulo de una imagen, luego
#   “g” guarda la imagen dentro del rectángulo en el disco,
#   “r” restaura la imagen original y permite realizar nuevamente la selección,
#   “q” finaliza.

import cv2
import numpy as np

selected_points = []


def mouse_callback (event, x, y, flags, param):
    global selected_points, show_img

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append([x, y])
        cv2.circle(show_img, (x, y), 3, (0, 255, 0), -1)


def select_points(image, points_num):
    global selected_points
    selected_points = []
    cv2.namedWindow('Seleccione_3_Puntos')
    cv2.setMouseCallback('Seleccione_3_Puntos', mouse_callback)

    while True:
        cv2.imshow('Seleccione_3_Puntos', image)
        k = cv2.waitKey(1)
        if len(selected_points) == points_num:
            break
    cv2.destroyAllWindows()

    return np.array(selected_points, dtype=np.float32)


img = cv2.imread('opencv-logo.png', cv2.IMREAD_COLOR)
img_dst = cv2.imread('ojo.jpg', cv2.IMREAD_COLOR)
img_dst = cv2.resize(img_dst, img.shape[1::-1])

backup_src = img.copy()
backup_dst = img_dst.copy()
(h, w) = img.shape[:2]

while (True):
    cv2.imshow('Affine', img)
    option = cv2.waitKey(1) & 0b11111111  # Enmascaro con una AND

    if option == ord('a'):
        cv2.destroyAllWindows()
        show_img = backup_src.copy()
        src_pts = select_points(show_img, 3)
        show_img = img_dst.copy()
        dst_pts = select_points(show_img, 3)
        img = backup_src.copy()
        img_dst = backup_dst.copy()
        affine_m = cv2.getAffineTransform(src_pts, dst_pts)
        img = cv2.warpAffine(img, affine_m, (w, h), borderValue=(255, 255, 255))

        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for row in range(h):
            for col in range(w):
                if img2gray[row, col] <= 20:
                    img2gray[row, col] = 255

        ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY )
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(img_dst, img_dst, mask=mask)
        img2_fg = cv2.bitwise_and(img, img, mask=mask_inv)
        img2_fg = cv2.blur(img2_fg, (20,20))
        img = cv2.add(img1_bg, img2_fg)

    elif option == ord('g'):
        cv2.imwrite('Affine_eye.png', img)

    elif option == ord('q'):
        break
