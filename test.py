import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the image
image_path = "your_image.jpg"  # Replace with your image file path
#image = Image.open(r"C:\Users\jdidk\Downloads\colorwheel.jpg")
image = Image.open(r"C:\TriggerBot\images\2024-08-09_14-53-58.png")

# Resize image for faster plotting (optional)
image = image.resize((100, 100))  # Adjust resolution as needed

# Convert image to numpy array
img_array = np.array(image)

# Get dimensions
height, width, _ = img_array.shape

# Create x, y coordinates
x = np.arange(width)
y = np.arange(height)
x, y = np.meshgrid(x, y)

# Compute z as the sum of RGB values for simplicity (normalize to 0-1 range)
z = np.mean(img_array, axis=2)  # Use average RGB for height

# Plot the 3D graph
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Normalize color values to plot
colors = img_array / 255.0

# Flatten arrays for scatter plot
x_flat, y_flat, z_flat = x.ravel(), y.ravel(), z.ravel()
colors_flat = colors.reshape(-1, 3)

# Use scatter for a pixel-like visualization
ax.scatter(x_flat, y_flat, z_flat, c=colors_flat, marker='.')

# Set labels
ax.set_xlabel('Width (X)')
ax.set_ylabel('Height (Y)')
ax.set_zlabel('RGB Intensity (Z)')

plt.show()
