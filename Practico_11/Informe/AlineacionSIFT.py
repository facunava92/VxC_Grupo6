#! /usr/bin/python

import numpy as np
import cv2

MIN_MATCH_COUNT = 10

img1 = cv2.imread('img1.png')
img2 = cv2.imread('img2.png')
kp_img1 = img1.copy()
kp_img2 = img2.copy()

dscr = cv2.xfeatures2d.SIFT_create(100)
kp1, des1 = dscr.detectAndCompute(img1, None)
kp2, des2 = dscr.detectAndCompute(img2, None)

cv2.drawKeypoints(img1, kp1, kp_img1,(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.drawKeypoints(img2, kp2, kp_img2,(0, 0 ,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

horizontal_concat = np.concatenate((kp_img1, kp_img2), axis=1)
cv2.imwrite('sift_kp.png', horizontal_concat)
cv2.imshow('key points', horizontal_concat)
cv2.waitKey(0)
cv2.destroyAllWindows()

matcher = cv2.BFMatcher(cv2.NORM_L2)
matches_01 = matcher.knnMatch(des1, des2, k=2)
matches_10 = matcher.knnMatch(des2, des1, k=2)

# Guardamos los buenos matches usando el test de razón de Lowe
def ratio_test(matches, ratio_thr):
    good_matches = []
    for m in matches:
        ratio = m[0].distance / m[1].distance
        if ratio < ratio_thr:
            good_matches.append(m[0])
    return good_matches

RATIO_THR = 0.7 #0.7 LOWE
good_matches01 = ratio_test(matches_01, RATIO_THR)
good_matches10 = ratio_test(matches_10, RATIO_THR)

good_matches10_ = {(m.trainIdx, m.queryIdx) for m in good_matches10}
final_matches = [m for m in good_matches01 if (m.queryIdx, m.trainIdx) in good_matches10_]

img_show = cv2.drawMatches(img1, kp1, img2, kp2, final_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite('sift_mtch.png', img_show)
cv2.imshow('matches', img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()

if(len(final_matches) > MIN_MATCH_COUNT):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in final_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in final_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0) # Computamos la homografía con RANSAC

wimg2 = cv2.warpPerspective(img2, H, img2.shape[:2][::-1])

# Mezclamos ambas imagenes
alpha = 0.5
blend = np.array(wimg2 * alpha + img1 * (1 - alpha), dtype=np.uint8)


cv2.imshow('final', blend)
cv2.imwrite('imgSIFT.jpg', blend)

while(True):
    option = cv2.waitKey(1) & 0b11111111  # Enmascaro con una AND

    if option == ord('q'):
        cv2.destroyAllWindows()
        break

