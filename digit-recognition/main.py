import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# importing datset
mnist = tf.keras.datasets.mnist
# loading already splitted data in variable touples
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# NORAMLIZE the pixels of the data 
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# Creating neural network model
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) 
# model.add(tf.keras.layers.Dense(128, activation = 'relu'))
# model.add(tf.keras.layers.Dense(128, activation = 'relu'))
# model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

# model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# model.fit(x_train, y_train, epochs=5)

# model.save('Digi_recognition.keras')

model = tf.keras.models.load_model('Digi_recognition.keras')

loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)

image_number = 0
while os.path.isfile(f"./sample_digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f'./sample_digits/digit{image_number}.png')[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f'This digit is probably a {np.argmax(prediction)}')
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number+=1