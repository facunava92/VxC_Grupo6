#!/usr/bin/python

import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('saved_model/my_model')

# Read the input image 
img = cv2.imread('digits.jpg', cv2.IMREAD_COLOR)
backup = img.copy() 

imgBlur = cv2.GaussianBlur(img, (3,3),1)  # 3 1 1 
imgCanny = cv2.Canny(img,100,210)  # 80 210 
kernel = np.ones((13,13),np.uint8)  # 3 3 
imgDil = cv2.dilate(imgCanny, kernel, iterations=1)


contours, hierachy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result= []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    xf, yf = x+w, y+h
    cv2.rectangle(img, (x,y), (xf, yf), (0, 255, 0), 3)
    roi = backup[y-15:yf+15, x-15:xf+15]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(roi,(5,5),0)
    ret, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    th = cv2.resize(th,(28, 28))
    image_n = th[:,:].reshape(1,28,28,1)
    image_n = image_n.astype('float32')
    image_n /= 255.
    predict = model.predict(image_n)
    preds = np.argmax(predict, axis=1)  
    result.append(int(preds))

# Busca los contornos de la imagen
contours, hierachy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
i = 0
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.putText(img, str(int(result[i])), (x, y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    i+=1


cv2.imwrite('resultado_tf.png', img)
cv2.imshow("Resulting Image with Rectangular ROIs", img)
cv2.waitKey()

