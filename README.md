# MNIST Model Training Project

This project demonstrates how to train a simple neural network model using the MNIST dataset. The code is divided into two main scripts:

1. **`neuronal.py`**: Contains the implementation of the neural network model and the training process.
2. **`main.py`**: The entry point of the program, which calls the training function from `neuronal.py`.

## Overview of the Code

### 1. `neuronal.py`

This script defines a function `entrenar_modelo_mnist()` that performs the following steps:

1. **Load the MNIST Dataset**:
   - The MNIST dataset is loaded using Keras's `mnist.load_data()` function.
   - The dataset is split into training and testing data, including images (`train_data_x`, `test_data_x`) and labels (`train_labels_y`, `test_labels_y`).

2. **Preprocess the Data**:
   - The training and testing images are reshaped from 28x28 matrices into 784-dimensional vectors.
   - The pixel values are normalized to the range `[0, 1]` by dividing by 255.
   - The labels are converted to one-hot encoded vectors using `to_categorical()`.

3. **Define the Neural Network Architecture**:
   - A sequential model is created with:
     - An input layer of size 784 (flattened 28x28 image).
     - A hidden dense layer with 512 units and ReLU activation.
     - An output dense layer with 10 units (for the 10 digit classes) and softmax activation.

4. **Compile the Model**:
   - The model is compiled using the RMSprop optimizer, categorical cross-entropy loss, and accuracy as the evaluation metric.

5. **Train the Model**:
   - The model is trained on the training data for 8 epochs with a batch size of 128.

6. **Evaluate the Model**:
   - The model's performance is evaluated on the test data.

### 2. `main.py`

This script serves as the entry point for the program. It performs the following steps:

1. **Import the Training Function**:
   - The `entrenar_modelo_mnist()` function is imported from the `neuronal` module.

2. **Execute the Training Process**:
   - A message is printed to indicate the start of the training process.
   - The `entrenar_modelo_mnist()` function is called to train the model.
   - A message is printed to indicate the completion of the training.

3. **Run the Program**:
   - The `main()` function is executed when the script is run directly.

---

## How to Use

1. Clone the repository or download the source code.
2. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
3. Run the main.py file:
   ```bash
   python main.py
4. Expected Output:
    - The script will print the shape of the training data and the label of the first training example.
    - It will display a summary of the model architecture.
    - It will train the model and evaluate its performance on the test data.
    - Finally, it will print a message indicating the completion of the training.

