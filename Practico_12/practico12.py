import cv2
import numpy as np

# Constants:
SIZE_IMAGE = 20
NUMBER_CLASSES = 10

def deskew(img):
    """Pre-processing of the images"""

    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SIZE_IMAGE * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SIZE_IMAGE, SIZE_IMAGE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def get_accuracy(predictions, labels):
    """Returns the accuracy based on the coincidences between predictions and labels"""

    accuracy = (np.squeeze(predictions) == labels).mean()
    return accuracy * 100


def get_hog():
    """ Get hog descriptor """
    hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (8, 8), (4, 4), (8, 8), 9, 1, -1, 0, 0.2, 1, 64, True)
    print("hog descriptor size: '{}'".format(hog.getDescriptorSize()))
    return hog


def raw_pixels(img):
    """Return raw pixels as feature from the image"""

    return img.flatten()

#Load the kNN Model
with np.load('knn_data.npz') as data:
    print(data.files)
    train = data['train']
    train_labels = data['train_labels']# Load the classifier

# Create KNN:
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

# Read the input image 

img = cv2.imread('MMM.jpg', cv2.IMREAD_COLOR)

imgBlur = cv2.GaussianBlur(img, (3,3),1)  # 3 1 1 
imgCanny = cv2.Canny(img,100,210)  # 80 210 
kernel = np.ones((13,13),np.uint8)  # 3 3 
imgDil = cv2.dilate(imgCanny, kernel, iterations=1)


# Find descriptor de funciones
contours, hierachy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# HoG feature descriptor:
hog = get_hog()

digits= []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    xf, yf = x+w, y+h
    cv2.rectangle(img, (x,y), (xf, yf), (0, 255, 0), 3)
    roi = imgDil[y-15:yf+15, x-15:xf+15]
    roi = cv2.resize(roi, (20, 20))
    digits.append(roi)
digits = np.array(digits)

# HoG feature descriptor:
hog = get_hog()

#calcula el descriptor HoG
hog_descriptors = []
for digit in digits:
    hog_descriptors.append(hog.compute(deskew(digit)))
hog_descriptors = np.squeeze(hog_descriptors)


# Create KNN:
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
ret, result, neighbours, dist = knn.findNearest(hog_descriptors, k=3)
print(result)
cv2.imshow("Resulting Image with Rectangular ROIs", img)

# Busca los contornos de la imagen
contours, hierachy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
i = 0
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.putText(img, str(int(result[i])), (x, y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    i+=1

cv2.imshow("Resulting Image with Rectangular ROIs", img)
cv2.waitKey()

