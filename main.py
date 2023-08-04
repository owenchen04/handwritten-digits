import os
# cv2 used for computer vision - load/process images
import cv2
# numpy important for working with numpy erase
import numpy as np
# matplotlib used for visualization of digits
import matplotlib.pyplot as plt
# tensorflow used for ML
import tensorflow as tf

"""
# Load MNIST dataset from tensorflow
mnist = tf.keras.datasets.mnist
# Labeled data: we already know what the digits are -- split into training data
# (used to train the model) and testing data (used to assess the model)
# x: pixel data, y: classification
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixels (not the digit classifications):
#   scale values down so every value is b/w 0 and 1
#   makes it easy for neural network to do calculations
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
"""


"""
# Create NN model itself - basic sequential neural network
model = tf.keras.models.Sequential()
# Add flatten layer - turns grid into a single line of pixels
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# Add dense layer - basic NN layer where each neuron is connected to each
# other neuron of the other layers
# Activation function: 'relu' = rectify linear unit
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
# Output dense layer - 10 units represent the individual digits
# Activation function: 'softmax' makes sure all the outputs add up to 1; can
# interpret as a confidence/probability for each digit to be the right answer
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model (train the model) by passing in training data
model.fit(x_train, y_train, epochs=10)

model.save('handwritten.model')
"""


# Same as running above code and proceeding with model, but instead of training
# it everytime, we can just load it because we've saved it
model = tf.keras.models.load_model('handwritten.model')


"""
# Evaluate the model - aim for low loss, high accuracy (b/w 0 - 1)
loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)
"""


# Read all digit files
image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        # Put image in a list
        # Turn into numpy array to be able to pass into neural network
        # Invert image to white on black
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        # argmax() gives us the INDEX of the field with the highest number, i.e.
        # which neuron has the highest activation.
        # No add. formatting needed b/c index 0 represents digit 0, etc.
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1
