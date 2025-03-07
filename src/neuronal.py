import numpy as np
from keras.models import Sequential #type: ignore
from keras.layers import Dense, Input #type: ignore
from keras.utils import to_categorical #type: ignore
from keras.datasets import mnist #type: ignore
import matplotlib.pyplot as plt

def entrenar_modelo_mnist():

    # Load MNIST dataset for training and testing
    (train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()

    # Print the shape of the training data and the label of the first training example
    print(train_data_x.shape)
    print(train_labels_y[0])

    ### Convert Matplotlib figure to OpenCV format
    ### export DISPLAY=:0
    #fig, ax = plt.subplots()
    #fig.canvas.draw()
    #image = train_data_x[0]
    #cv2.imshow("Matplotlib Image", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Define the architecture of the neural network
    model = Sequential([
        Input(shape=(28*28,)),  # Input layer with 28x28 pixels flattened
        Dense(512, activation='relu'),  # Hidden layer with 512 units and ReLU activation
        Dense(10, activation='softmax')  # Output layer with 10 units (for 10 classes) and softmax activation
    ])

    # Compile the model with RMSprop optimizer, categorical crossentropy loss, and accuracy metric
    model.compile(
        optimizer='rmsprop', 
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # Print a summary of the model architecture
    model.summary()

    # Define the architecture of the neural network (repeated for clarity)
    model = Sequential([
        Input(shape=(28*28,)),  # Input layer with 28x28 pixels flattened
        Dense(512, activation='relu'),  # Hidden layer with 512 units and ReLU activation
        Dense(10, activation='softmax')  # Output layer with 10 units (for 10 classes) and softmax activation
    ])

    # Compile the model with RMSprop optimizer, categorical crossentropy loss, and accuracy metric
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # Print a summary of the model architecture
    model.summary()

    # Normalize the training data: reshape, convert to float32, and scale to [0, 1]
    x_train = train_data_x.reshape(60000, 28*28)
    x_train = x_train.astype('float32')/255
    y_train = to_categorical(train_labels_y)  # Convert labels to one-hot encoding

    # Normalize the test data: reshape, convert to float32, and scale to [0, 1]
    x_test = test_data_x.reshape(10000, 28*28)
    x_test = x_test.astype('float32')/255
    y_test = to_categorical(test_labels_y)  # Convert labels to one-hot encoding

    # Train the model on the training data for 8 epochs with a batch size of 128
    model.fit(x_train, y_train, epochs=8, batch_size=128)

    # Evaluate the model on the test data
    model.evaluate(x_test, y_test)