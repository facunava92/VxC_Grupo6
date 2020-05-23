import cv2
import numpy as np

patronbase=0.04405286344
patronaltura=0.03968253968
selected_points = []
def getContours(img,imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        x , y , w, h = cv2.boundingRect(cnt)
        rect=cv2.minAreaRect(cnt)
        box= cv2.boxPoints(rect)
        box=np.int0(box)
        cv2.drawContours(imgContour, [box], 0, (0,255, 0), 4)
        base=w*patronbase
        altura=h*patronaltura
        cv2.putText(imgContour, "{:.1f} X {:.1f} cm".format(base,altura), (x+50 , y-5 ), cv2.FONT_HERSHEY_COMPLEX,0.4,(0, 0, 255), 1)
    return (imgContour)
def perspective(image, src_pts, dst_pts):
    (h, w) = image.shape[:2]
    P = cv2.getPerspectiveTransform(src_pts, dst_pts)
    rectified = cv2.warpPerspective(img, P, (w, h))
    return rectified 
img = cv2.imread('prueba2.jpg', cv2.IMREAD_COLOR)
backup = img.copy()
(h, w) = img.shape[:2]
while (True):
    cv2.imshow('Perspective', img)
    option = cv2.waitKey(1) & 0b11111111  # Enmascaro con una AND
    if option == ord('h'):
        cv2.destroyAllWindows()
        show_img = backup.copy()
        img = backup.copy() 
        cv2.imshow("medicion de objetos", show_img)
        dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32) 
        selected_points=([[ 53 ,  20] , [377 ,  76] ,[381 , 479] , [  1 , 462]])
        src_pts=np.array(selected_points, dtype=np.float32)
        img = perspective(img, src_pts, dst_pts)
        imgContour = img.copy()
        smooth_image_bf=cv2.bilateralFilter (img ,20,35 ,35)    
        imgBlur = cv2.GaussianBlur(smooth_image_bf, (3, 1), 1)   #  bien 3 1 1 
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        imgCanny = cv2.Canny(imgGray,90,200)  
        kernel = np.ones((3,3),np.uint8) 
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
        imgStack=getContours(imgDil,imgContour)
        cv2.imshow("Result", imgStack)
    elif option == ord('g'):
        cv2.imwrite('rectificado.png', img)

    elif option == ord('q'):
        break



