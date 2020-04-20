#!/usr/bin/python
#A. ¿Cómo obtener el frame rate o fps usando las OpenCV? Usarlo para no tener que harcodear el delay del waitKey.
#B. ¿Cómo obtener el ancho y alto de las imágenes capturadas usando las OpenCV? Usarlo para no tener que harcodear el frameSize del video generado

import cv2 

videoCapture = cv2.VideoCapture('tierra.avi')

fourcc_XVID = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D') #.avi
#fourcc_I420 = cv2.VideoWriter_fourcc('I', '4', '2', '0') #.avi
#fourcc_PIM1 = cv2.VideoWriter_fourcc('P', 'I', 'M', '1') #.avi
#fourcc_MP4V = cv2.VideoWriter_fourcc('M', 'P', '4', 'V') #.mp4
#fourcc_X264 = cv2.VideoWriter_fourcc('X', '2', '6', '4') #.mp4
#fourcc_THEO = cv2.VideoWriter_fourcc('T', 'H', 'E', 'O') #.ogv
#fourcc_FLV1 = cv2.VideoWriter_fourcc('F', 'L', 'V', '1') #.flv
fps = videoCapture.get(cv2.CAP_PROP_FPS)
framesize = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
delay = int(1000/fps)
success, frame = videoCapture.read()
#(framesize )= frame.shape [:2]  # otra forma de obtener el ancho y alto 
videoWriter = cv2.VideoWriter('output_XVID.avi', fourcc_XVID, fps, framesize, isColor = False)
while success:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    videoWriter.write(gray)
    cv2.imshow('Image gray', gray)
    if (cv2.waitKey(delay) & 0b11111111) == ord('q'):                                       
        break
    success, frame = videoCapture.read()

videoCapture.release()
videoWriter.release()
cv2.destroyAllWindows()

