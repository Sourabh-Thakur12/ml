import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# importing datset
mnist = tf.keras.datasets.mnist
# loading already splitted data in variable touples
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# NORAMLIZE the pixels of the data 
X_train = tf.keras.utils.normalize(X_train, axis = 1)
X_test = tf.keras.utils.normalize(X_test, axis = 1)

# Creating neural network model
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) 
# model.add(tf.keras.layers.Dense(128, activation = 'relu'))
# model.add(tf.keras.layers.Dense(128, activation = 'relu'))
# model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

# model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# model.fit(X_train, y_train, epochs=20)

# model.save('Digi_recognition.keras')

# model = tf.keras.models.load_model('D:/CODING/ML/Projects-Ai-Ml/Handwritten Recognisation/Handwritten_Recognisation.keras')
model = tf.keras.models.load_model("D:/CODING/ML/digit-recognition/Digi_recognition.keras")

score = model.evaluate(X_test, y_test)
print(score[0])
print(score[1])

image_number = 0
while os.path.isfile(f"D:/CODING/ML/digit-recognition/sample_digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f'D:/CODING/ML/digit-recognition/sample_digits/digit{image_number}.png')[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f'This digit is probably a {np.argmax(prediction)}')
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number+=1