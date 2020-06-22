import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

mnist = tf.keras.datasets.mnist #traemos el dataset 

(x_train , y_train) , (x_test , y_test)=mnist.load_data()  #nos trae dos tuplas  1ra datos de entramiento  y la segnda con la de texto   ..... x =  imagenes    y = etiquetas 

x_train , x_test = x_train / 255.0 , x_test / 255.0   #intesidades de las imagenes 0 y 255 

ts=x_train.shape
x_train=x_train.reshape(ts[0] , ts[1] ,ts[2] ,1  )
tt=x_test.shape
x_test= x_test.reshape(tt[0] , tt[1] ,tt[2] ,1  )

model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(4, [5,5],activation='relu',
                        input_shape=(28,28,1)),

  tf.keras.layers.Conv2D(12,(5,5), strides=(2,2),activation='relu'),
  tf.keras.layers.Conv2D(24,(4,4), strides=(2,2),activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(rate=.25),
  tf.keras.layers.Dense(200, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])                                          #quitar una neurona .. parametrizando una neuroan y quietar otras ...elimina el 25% de las neuronas en forma aleatorias (en esa capa )

optimizer = tf.keras.optimizers.Adam(decay=.0001)

model.compile(optimizer= optimizer ,  #optimizador es el metodo que usamos para minimizar la funcion de error  ... 
              loss = 'sparse_categorical_crossentropy',  # funcionde perdida (crossentropia  categorica )  y sparse es el one-hot 
              metrics= ['accuracy'] )    # la exactitud 

model.fit(x_train , y_train , epochs=5)   # entrenamos el modelo    ( le vamos a mandar los datos de entremsoiento , la entradas y las etiquetas y las epocas = cuantas veces le mostramos los datos de entrenamiento )

model.evaluate(x_test, y_test)

image = cv2.imread('novee.png',cv2.IMREAD_COLOR)
image  = cv2.resize(image,(28, 28))
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_n = cv2.bitwise_not(grayImage)
image_n = image_n[:,:].reshape(1,28,28,1)
image_n = image_n.astype('float32')
image_n /= 255.
predict=model.predict(image_n)
preds = np.argmax(predict, axis=1)  

#for print image to 28x28 
new_image=image_n.reshape(28,28)
fig, ax = plt.subplots()
fig.subplots_adjust()
ax.imshow(new_image, cmap='binary')
xlabel = "Predic: {}".format(preds)
ax.set_xlabel(xlabel)
plt.show()

