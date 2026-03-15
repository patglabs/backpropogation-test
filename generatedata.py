import random
import matplotlib.pyplot as plt

def generate_data(num_points):
    data = []
    
    # Generate a blob-shaped cluster in the top right (Quadrant 1)
    for i in range(num_points // 2):
        # random.gauss(mean, standard_deviation)
        x1 = random.gauss(2, 1)  # Centered at x=2
        x2 = random.gauss(2, 1)  # Centered at y=2
        label = 1 # class 1 (Blue)
        data.append((x1, x2, label))
        
    # Generate a blob-shaped cluster in the bottom left (Quadrant 3)
    for i in range(num_points // 2):
        x1 = random.gauss(-2, 1) # Centered at x=-2
        x2 = random.gauss(-2, 1) # Centered at y=-2
        label = -1 # class 0 (Orange)
        data.append((x1, x2, label))    

    return data

# Generate 200 data points (increased to better match the density in the image)
data = generate_data(200)

# Save the data to a file
with open('data.txt', 'w') as f:
    for point in data:
        f.write(f"{point[0]},{point[1]},{point[2]}\n")

# Visualize the data
x1 = [point[0] for point in data]
x2 = [point[1] for point in data]
labels = [point[2] for point in data]

# Map labels to the specific colors in your reference image
colors = ['#F4A460' if label == -1 else '#5D9FD1' for label in labels]

# Plot with white edgecolors to match the styling
plt.scatter(x1, x2, c=colors, edgecolors='white', s=50)

# Set axes limits to match the reference image
plt.xlim(-6, 6)
plt.ylim(-6, 6)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Generated Gaussian Blobs')

# Save and show
plt.savefig('generated_data.png')
plt.show()