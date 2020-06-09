#!/usr/bin/python
#Este programa entrena la red segun los mejores valores calculados en el script de k y porcentajes de train/test varaibles.
#Al finalizar guarda los coeficientes en "knn_data.npz" 

#Import required packages:
import cv2
import numpy as np

#Constants:
SIZE_IMG = 20           #Tamaño en pixeles de cada numero
NUMBER_CLASSES = 10     #Tamaño de los números a explorar 0-9


def load_digits_and_labels(big_image):
    """Devuelve todas los dígitos de la imagen 'grande' y crea las etiquetas correspondientes para cada imagen"""
    
    # Cargamos la imagen:
    digits_img = cv2.imread(big_image, cv2.IMREAD_GRAYSCALE)
    y, x = digits_img.shape[:2]
    
    # Algoritmo para obtener cada celda con los números de la imagen original
    number_cols = (x/SIZE_IMG)
    rows = np.vsplit(digits_img, y/SIZE_IMG)
    
    digits = []
    for row in rows:
        row_cells = np.hsplit(row, number_cols)
        for digit in row_cells:
            digits.append(digit)
    digits = np.array(digits)

    # Crea las etiquetas para cada imagen
    labels = np.repeat(np.arange(NUMBER_CLASSES), (len(digits)/NUMBER_CLASSES))
    return digits, labels


def deskew(img):
    """Pre-processing of the images"""

    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SIZE_IMG * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SIZE_IMG, SIZE_IMG), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def get_accuracy(predictions, labels):
    """Retorna la precision basado en las coincidencias entre las predicciones y las etiquetas"""

    accuracy = (np.squeeze(predictions) == labels).mean()
    return accuracy * 100


def get_hog():
    """ Get hog descriptor """

    # cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
    # L2HysThreshold, gammaCorrection, nlevels, signedGradient)
    hog = cv2.HOGDescriptor((SIZE_IMG, SIZE_IMG), (8, 8), (4, 4), (8, 8), 9, 1, -1, 0, 0.2, 1, 64, True)
    print("hog descriptor size: '{}'".format(hog.getDescriptorSize()))
    return hog


def raw_pixels(img):
    """Retorna los pixeles en bruto"""
    return img.flatten()


#cargo la imagen, la subdivido y genero etiquetas de cada uno
digits, labels = load_digits_and_labels('digits.png')

# Mezclando datos
# Construimos un generador de números aleatorios
rand = np.random.RandomState(1234)
# Realizamos el mezclado aleatorio
shuffle = rand.permutation(len(digits))
digits, labels = digits[shuffle], labels[shuffle]

# HoG descriptor:
hog = get_hog()

# Computamos los descriptores para todas las imagenes.
# En este caso, los decriptores de HoG es calculado
hog_descriptors = []
for img in digits:
    hog_descriptors.append(hog.compute(deskew(img)))
hog_descriptors = np.squeeze(hog_descriptors)


#Creamos KNN:
knn = cv2.ml.KNearest_create()

split_value = 0.9
partition = int(split_value * len(hog_descriptors))
hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [partition])
labels_train, labels_test = np.split(labels, [partition])

# Entrenamos modelo KNN:
print('Entrenando modelo KNN - descriptor HoG')
knn.train(hog_descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)

# Guardamos la presicion segun arroja el testeo 
k = 2
ret, result, neighbours, dist = knn.findNearest(hog_descriptors_test, k)

# Computamos la presicion
acc = get_accuracy(result, labels_test)
print(" {}".format("%.2f" % acc))

# Salvamos modelo KNN
np.savez('knn_data.npz',train=hog_descriptors_train, train_labels=labels_train)
