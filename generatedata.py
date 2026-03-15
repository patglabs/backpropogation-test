# generate an array with two clusters in a 2D space one in the first quadrant and one in the third quadrant
import random

def generate_data(num_points):
    data = []
    for _ in range(num_points):
        # generate a random point in the first quadrant
        x1 = random.uniform(2, 10)
        x2 = random.uniform(2, 10)
        data.append((x1, x2, 1)) # label 1 for first quadrant

        # generate a random point in the third quadrant
        x1 = random.uniform(-10, -2)
        x2 = random.uniform(-10, -2)
        data.append((x1, x2, 0)) # label 0 for third quadrant

    return data

# generate 100 data points
data = generate_data(100)

# save the data to a file
with open('data.txt', 'w') as f:
    for point in data:
        f.write(f"{point[0]},{point[1]},{point[2]}\n")

# visualize the data
import matplotlib.pyplot as plt
x1 = [point[0] for point in data]
x2 = [point[1] for point in data]
labels = [point[2] for point in data]
plt.scatter(x1, x2, c=labels, cmap='bwr')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Generated Data')
plt.show()

# save the image
plt.savefig('generated_data.png')