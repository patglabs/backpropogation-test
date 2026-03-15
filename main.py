# simple backpropagation implementation for a single layer neural network
# we will train it to recognise two classes of data points in a 2D space

# 2 arraays one for each layer

x1 = 0 # input neuron 1
x2 = 0 # input neuron 2

input_layer = [x1, x2] # 2 input neurons


x1weight = 0.5 # weight for input neuron 1
x2weight = 0.5 # weight for input neuron 2
output_layer = [0 [x1weight, x2weight]] # 1 output neuron

# activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

loss = 100 # loss function