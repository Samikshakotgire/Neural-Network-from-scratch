Neural Network from Scratch
Overview
This project implements a neural network from scratch in Python, without using any high-level libraries like TensorFlow or PyTorch. The goal is to gain a deep understanding of the underlying mechanisms of neural networks, including forward and backward propagation, gradient descent, and activation functions.

Features
Fully connected neural network implementation
Support for multiple hidden layers
Activation functions: Sigmoid, ReLU, and Tanh
Forward and backward propagation
Training using gradient descent
Performance evaluation using accuracy
Installation
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/Samikshakotgire/neural-network-from-scratch.git
cd neural-network-from-scratch
Ensure you have Python installed. You can install the required dependencies using:

bash
Copy code
pip install -r requirements.txt
Usage
To train the neural network, you can run the following command:

bash
Copy code
python train.py
Example
Here's an example of how to create and train a neural network using the provided code:

python
Copy code
from neural_network import NeuralNetwork

# Example dataset (replace with your own data)
X_train, y_train = ...
X_test, y_test = ...

# Define the architecture
nn = NeuralNetwork(layers=[784, 128, 64, 10], activation='relu')

# Train the network
nn.train(X_train, y_train, epochs=1000, learning_rate=0.01)

# Evaluate the network
accuracy = nn.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}%')
Project Structure
neural_network.py: Contains the implementation of the neural network class.
train.py: Script to train the neural network using a dataset.
utils.py: Utility functions for data preprocessing and other helper functions.
requirements.txt: List of dependencies required to run the project.
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes or enhancements.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Samiksha kotgire
Inspired by various online tutorials and courses on neural networks
