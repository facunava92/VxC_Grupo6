#!/usr/bin/python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#DATADIR= "D:\dataset\PetImages"
#categorias = ["Dog","Cat"]

train_datagen = ImageDataGenerator( rescale = 1./255, 
                                    rotation_range=40,
                                    shear_range = 0.2,
                                    zoom_range = 0.2, 
                                    width_shift_range = 0.2,
                                    height_shift_range=0.2,                                    
                                    horizontal_flip = True)

test_datagen  = ImageDataGenerator(rescale = 1./255)

#Train
x_train = train_datagen.flow_from_directory('./dataset/train',  
                                            target_size = (128,128),
                                            batch_size=32,
                                            class_mode='binary')
#Test
x_test  = test_datagen.flow_from_directory( './dataset/test',    
                                            target_size = (128 ,128),
                                            batch_size=32,
                                            class_mode='binary')

model = Sequential()
model.add(Conv2D(4, (5,5), activation = 'relu', input_shape = (128, 128, 3)))
model.add(Conv2D(12,(5,5), strides=(2,2),activation='relu'))
model.add(Conv2D(24,(4,4), strides=(2,2),activation='relu'))
model.add(Flatten())
model.add(Dropout(rate=.25))
model.add(Dense(200, activation='relu'))
model.add(Dense(10, activation='softmax'))

#Optimizador es el metodo que usamos para minimizar la funcion de error
optimizer = Adam(decay=.0001)
model.compile(  optimizer= optimizer,  
                #funcion de perdida (crossentropia categorica) y sparse es el one-hot 
                loss = 'sparse_categorical_crossentropy',  
                #exactitud 
                metrics= ['accuracy'])  



#Entrenamos el modelo. Le vamos a mandar los datos de entrenamiento, las entradas y las etiquetas y las epocas 
#epocas = cuantas veces le mostramos los datos de entrenamiento
model.fit(x_train , epochs=30)   
#model.evaluate (x_test , y_test)    # evaluamos sobre el "conjunto testigo "
model.evaluate(x_test)

#for category in categorias: 
#    path= os.path.join(DATADIR , category)  #serian las clases
#    for img in os.listdir(path):
#        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

#        break
#    break

#tam=50
#new_array=cv2.resize(img_array , (tam , tam))
#plt.imshow(new_array,cmap='gray')
#plt.show()
#print (new_array.shape)







#training_data =[]
#def create_training_data():
#    for category in categorias:
#        path= os.path.join(DATADIR ,category)  #serian las clases
#        class_num=categorias.index(category)
#        for img in os.listdir(path):
 #           try:
 #               img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
 #               new_array=cv2.resize(img_array , (50 , 50))
 #               training_data.append([new_array, class_num])
 #           except Exception as e:
 #               pass
#create_training_data()
#print(len(training_data))


