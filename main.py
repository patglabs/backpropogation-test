# simple backpropagation implementation for a single layer neural network
# we will train it to recognise two classes of data points in a 2D space

import numpy as np
import matplotlib.pyplot as plt

# --- Network Setup ---
x1weight = 0.5 # current guess for weight 1
x2weight = -0.5 # current guess for weight 2

# NEW activation function
def tanh_activation(x):
    return np.tanh(x)

def inference(x1, x2):
    return tanh_activation(x1weight * x1 + x2weight * x2)

# --- Plotting ---
def plot_inference(title='Decision Boundary vs Actual Data', save_path=None):
    try:
        data = np.loadtxt('data.txt', delimiter=',')
        x1_data = data[:, 0]
        x2_data = data[:, 1]
        labels = data[:, 2]
    except FileNotFoundError:
        print("Error: Could not find 'data.txt'. Please ensure it's in the same folder.")
        return

    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    
    Z = tanh_activation(x1weight * X + x2weight * Y)

    plt.figure() # Create a new figure
    plt.contourf(X, Y, Z, levels=50, cmap='RdBu', alpha=0.8)
    plt.colorbar()

    colors = ["#5D9FD1" if label == 1 else '#F4A460' for label in labels]
    plt.scatter(x1_data, x2_data, c=colors, edgecolors='white', s=40, zorder=2)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    
    # Save before showing, otherwise the figure clears and saves a blank image!
    if save_path:
        plt.savefig(save_path)
        
    plt.show()

# Show the initial untrained state (you must close the window to start training!)
print("Showing initial untrained boundary. Close the plot window to begin training...")
plot_inference(title='Before Training')

def load_data():
    with open('data.txt', 'r') as f:
        data = []
        for line in f:
            x1, x2, label = line.strip().split(',')
            data.append((float(x1), float(x2), int(label)))
    return data 

EPOCHS = 5

def train_network():
    # TELL PYTHON TO USE THE GLOBAL VARIABLES
    global x1weight, x2weight 
    
    data = load_data()
    
    # Renamed loop variable to 'epoch'
    for epoch in range(EPOCHS):
        np.random.shuffle(data)
        split_index = int(0.8 * len(data))
        train_data = data[:split_index]
        test_data = data[split_index:]

        batch_size = 10
        learning_rate = 0.01
        
        # Renamed loop variable to 'batch_start' to avoid overwriting 'epoch'
        for batch_start in range(0, len(train_data), batch_size):
            batch = train_data[batch_start:batch_start+batch_size]
            error_sum = 0
            
            for x1, x2, label in batch:
                output = inference(x1, x2)
                error = (output - label) ** 2
                
                d_error_d_output = 2 * (output - label)
                d_output_d_x1weight = x1 * (1 - output ** 2) 
                d_output_d_x2weight = x2 * (1 - output ** 2) 

                x1weight -= learning_rate * d_error_d_output * d_output_d_x1weight
                x2weight -= learning_rate * d_error_d_output * d_output_d_x2weight

                error_sum += error
            
            # Print batch error
            print(f"Batch {batch_start//batch_size + 1}: Average Error = {error_sum / len(batch):.4f}")
            
        print(f"--- Epoch {epoch+1}/{EPOCHS} completed ---")

train_network()

# after training, plot again to see the changes
print(f"Weights after training: x1={x1weight:.4f}, x2={x2weight:.4f}")
plot_inference(title='After Training', save_path='inference_after_training.png')