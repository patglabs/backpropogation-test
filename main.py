# simple backpropagation implementation for a single layer neural network
# we will train it to recognise two classes of data points in a 2D space

import numpy as np
import matplotlib.pyplot as plt

x1 = 0.5 # input neuron 1
x2 = 0.5 # input neuron 2

input_layer = [x1, x2] # 2 input neurons

x1weight = 0.1 # weight for input neuron 1
x2weight = 0.1 # weight for input neuron 2

output_layer = [0, [x1weight, x2weight]] # 1 output neuron

# NEW activation function
def tanh_activation(x):
    # Using numpy's built-in tanh function
    return np.tanh(x)

loss = 1 # loss function

def inference(x1, x2):
    # calculate the output of the network using tanh
    output = tanh_activation(x1weight * x1 + x2weight * x2)
    return output

def plot_inference():
    # plot the decision boundary of the network
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Z using the new tanh function
    Z = tanh_activation(x1weight * X + x2weight * Y)

    # The contour plot will automatically adjust its scale to range from -1 to 1
    plt.contourf(X, Y, Z, levels=50, cmap='RdBu')
    plt.colorbar() # Pay attention to the numbers on this colorbar when you run it!
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Decision Boundary using Tanh (-1 to 1)')
    plt.show()

print(f"Test inference output: {inference(5, 5)}") # test the inference function
plot_inference() # plot the decision boundary