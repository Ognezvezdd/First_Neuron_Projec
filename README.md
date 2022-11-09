# First_Neuron_Projec
import os
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers import Normalization
from keras.optimizers import Adam
from keras import utils
from keras.preprocessing import image

import matplotlib.pyplot as plt
from keras.utils import image_dataset_from_directory
from tensorflow.python import keras
from tensorflow.python.client import device_lib

test_dataset = image_dataset_from_directory('Olimp_train',
                                            shuffle=False,
                                            label_mode='int',
                                            batch_size=128,
                                            image_size=(250, 250))

model = tf.keras.models.load_model('corr_model.h5')

prediction = model.predict(test_dataset)

# for i in range(9):  # prediction.size
#     if prediction[i][0] >= 0.5:
#         print('profile')
#     else:
#         print('anfas')


filenames = []
for root, dirs, files in os.walk('Olimp_train/test'):
    for filename in files:
        filenames.append(filename)

for i in filenames:
    print(i)
f = open('text.txt', 'w')
sdv = 9 * 0
plt.figure(figsize=(10, 10))
for images, labels in test_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i + sdv].numpy().astype("uint8"))
        plt.title("profile" if prediction[i + sdv][0] <= 0.5 else "anfas")
        plt.axis("off")
plt.show()

for images, labels in test_dataset.take(1):
    for i in range(prediction.size):
        # s = str(i + 1)
        # s += "."
        s = ""
        temp = filenames[i]
        temp = temp[1:len(temp)]
        s += temp
        # s = s[0:len(s)-4]
        s+=","
        x = ("1" if prediction[i][0] >= 0.5 else "0")
        s += x
        f.write(s + '\n')

f.close()
