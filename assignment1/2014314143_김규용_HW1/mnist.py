# %%

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

(x_train,y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=100)

model.evaluate(x_test,y_test)

# %%

img_dir = '/Users/nimjk/OneDrive/Desktop/Ryzen/mltest'
img1_name = 'mine.jpg'
img1_path = os.path.join(img_dir,img1_name)


img1 = tf.keras.utils.load_img(img1_path,target_size=(28,28))
img1 = img1.convert('L')
img_tensor = tf.keras.utils.img_to_array(img1)

img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

plt.rcParams['figure.figsize'] = (10,10)
plt.imshow(img_tensor[0],cmap='gray')
plt.show()

# %%
prediction = model.predict(img_tensor)
score = tf.nn.softmax(prediction[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(np.argmax(score), 100 * np.max(score))
)

n = 0
while n < 10: 
    print("{} : {:.2f}%".format(n, 100*np.max(score[n])))
    n += 1
# %%
img2_name = 'dog.jpg'
img2_path = os.path.join(img_dir,img2_name)


img2 = tf.keras.utils.load_img(img2_path,target_size=(28,28))
img2 = img2.convert('L')
img2_tensor = tf.keras.utils.img_to_array(img2)

img2_tensor = np.expand_dims(img2_tensor, axis=0)
img2_tensor /= 255.

plt.rcParams['figure.figsize'] = (10,10)
plt.imshow(img2_tensor[0],cmap='gray')
plt.show()

# %%
prediction2 = model.predict(img2_tensor)
score2 = tf.nn.softmax(prediction2[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(np.argmax(score2), 100 * np.max(score2))
)

n = 0
while n < 10: 
    print("{} : {:.2f}%".format(n, 100*np.max(score2[n])))
    n += 1
# %%
