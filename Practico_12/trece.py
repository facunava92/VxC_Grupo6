import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#DATADIR= "D:\dataset\PetImages"
#categorias = ["Dog","Cat"]
train_datagen=ImageDataGenerator(rescale= 1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

x_train = train_datagen.flow_from_directory('D:/dataset/train',   #train
                                                    target_size = (64 ,64),
                                                    batch_size=32,
                                                    class_mode='binary')
x_test = test_datagen.flow_from_directory('D:/dataset/test',   #test 
                                                    target_size = (64 ,64),
                                                    batch_size=32,
                                                    class_mode='binary')

model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(64,64,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


optimizer = tf.keras.optimizers.Adam(decay=.0001)

model.compile(optimizer= 'adam' ,  #optimizador es el metodo que usamos para minimizar la funcion de error  ... 
              loss = 'sparse_categorical_crossentropy',  # funcionde perdida (crossentropia  categorica )  y sparse es el one-hot 
              metrics= ['accuracy'] )    # la exactitud 

model.fit(x_train , epochs=5)   # entrenamos el modelo    ( le vamos a mandar los datos de entremsoiento , la entradas y las etiquetas y las epocas = cuantas veces le mostramos los datos de entrenamiento )

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


