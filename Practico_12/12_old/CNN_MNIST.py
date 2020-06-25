#!/usr/bin/python

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam

#Load data, dos tuplas, x=imgs y=labels
(x_train , y_train) , (x_test , y_test)=mnist.load_data()  

#Redimensionamos a [samples][widht][height][channels]
ts = x_train.shape
tt = x_test.shape
x_train = x_train.reshape(ts[0], ts[1], ts[2], 1)
x_test  = x_test.reshape (tt[0], tt[1], tt[2], 1)

#Normalizamos entradas de [0-255] a [0-1]
x_train = x_train/ 255.0
x_test  = x_test / 255.0 

model = Sequential()
model.add(Conv2D(4, (5,5), activation = 'relu', input_shape = (28, 28, 1)))
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
model.fit(x_train , y_train , epochs=10)   
model.evaluate(x_test, y_test)
model.summary()

#Guardamos el modelo, para no tener q volver a ejecutar el entrenamiento.
model.save('saved_model/my_model') 
