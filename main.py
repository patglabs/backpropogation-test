# simple backpropagation implementation for a single layer neural network
# we will train it to recognise two classes of data points in a 2D space

import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Load Data & Setup Grid ONCE ---
try:
    data_array = np.loadtxt('data.txt', delimiter=',')
    x1_data = data_array[:, 0]
    x2_data = data_array[:, 1]
    labels = data_array[:, 2]
    colors = ["#5D9FD1" if label == 1 else '#F4A460' for label in labels]
except FileNotFoundError:
    print("Error: Could not find 'data.txt'.")
    exit()

x_grid = np.linspace(-6, 6, 100)
y_grid = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x_grid, y_grid)

# --- 2. Network Setup ---
x1weight = 0.5 
x2weight = -0.5 

def tanh_activation(x):
    return np.tanh(x)

def inference(x1, x2):
    return tanh_activation(x1weight * x1 + x2weight * x2)

if not os.path.exists('output_frames'):
    os.makedirs('output_frames')

# --- 3. Real-Time Plotting Function ---
frame_count = 0

# NEW: Added 'current_loss' as a parameter
def update_plot(epoch, batch_num=None, current_loss=None):
    global frame_count
    
    plt.clf() 
    
    Z = tanh_activation(x1weight * X + x2weight * Y)

    plt.contourf(X, Y, Z, levels=50, cmap='RdBu', alpha=0.8)
    plt.scatter(x1_data, x2_data, c=colors, edgecolors='white', s=40, zorder=2)

    plt.xlabel('x1')
    plt.ylabel('x2')
    
    # NEW: Multi-line title dashboard showing Epoch, Loss, and Weights
    if batch_num is not None and current_loss is not None:
        dashboard_text = (f"Epoch: {epoch+1} | Batch: {batch_num} | Loss: {current_loss:.4f}\n"
                          f"Weight 1: {x1weight:.4f}  |  Weight 2: {x2weight:.4f}")
    else:
        dashboard_text = (f"Final Decision Boundary\n"
                          f"Weight 1: {x1weight:.4f}  |  Weight 2: {x2weight:.4f}")
        
    plt.title(dashboard_text, fontsize=11, fontweight='bold', pad=10)
    
    plt.savefig(f"output_frames/frame_{frame_count:03d}.png")
    frame_count += 1
    
    plt.pause(0.01) 

# --- 4. Training Loop ---
EPOCHS = 5
learning_rate = 0.01
batch_size = 10

plt.ion() 
fig = plt.figure(figsize=(7, 6))

print("Starting training visualization...")

def train_network():
    global x1weight, x2weight 
    
    train_data = [(row[0], row[1], int(row[2])) for row in data_array]
    
    # Initial plot (passing 0.0 for the starting loss)
    update_plot(epoch=0, batch_num=0, current_loss=0.0)

    for epoch in range(EPOCHS):
        np.random.shuffle(train_data)
        
        for batch_start in range(0, len(train_data), batch_size):
            batch = train_data[batch_start:batch_start+batch_size]
            error_sum = 0 # Track error for the dashboard
            
            for x1, x2, label in batch:
                output = inference(x1, x2)
                
                # Calculate Mean Squared Error
                error = (output - label) ** 2
                error_sum += error
                
                # Gradients
                d_error_d_output = 2 * (output - label)
                d_output_d_x1weight = x1 * (1 - output ** 2) 
                d_output_d_x2weight = x2 * (1 - output ** 2) 

                # Update weights
                x1weight -= learning_rate * d_error_d_output * d_output_d_x1weight
                x2weight -= learning_rate * d_error_d_output * d_output_d_x2weight
            
            # NEW: Calculate the average loss for this specific batch
            avg_batch_loss = error_sum / len(batch)
            
            # Pass the calculated loss into the plot function!
            update_plot(epoch, batch_start // batch_size + 1, avg_batch_loss)
            
        print(f"--- Epoch {epoch+1}/{EPOCHS} completed ---")

train_network()

plt.ioff() 
update_plot(EPOCHS) 
print(f"Training complete! Weights: x1={x1weight:.4f}, x2={x2weight:.4f}")
plt.show()