#!/usr/bin/python

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model('saved_model/my_model')

image_width=128
image_height=128
image_size=(image_width,image_height)
image_channels=3
batch_size = 50

test_datagen = ImageDataGenerator(  rescale=1./255) # Note that validation data should not be augmented

test_dir = 'test'
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = image_size,
    batch_size = batch_size,
    class_mode='categorical')

test_loss, test_acc = model.evaluate(test_generator)
print('test acc:', test_acc)

plt.figure(figsize=(10, 10))
#
for i in range(10):
    img = cv2.imread("pets/" + str(i+1) + ".jpg", cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    img = img[:,:].reshape(1, 128, 128, 3)
    img = img.astype('float32')
    img /= 255.
    predict = model.predict(img)
    print(predict)
    result =np.argmax(predict, axis=-1) 
    print(result)
    #
    if result:
        text_label = "PERRO"
    else:
        text_label = "GATO"
    #
    img = load_img("pets/" + str(i+1) + ".jpg", (128, 128))
    plt.subplot(5, 2, i+1)
    plt.imshow(img)
    plt.xlabel(text_label)
plt.tight_layout()
plt.show()
