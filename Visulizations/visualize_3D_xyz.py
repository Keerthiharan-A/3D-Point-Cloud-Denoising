import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to load .xyz file
def load_xyz(file_path):
    points = np.loadtxt(file_path)
    return points

# Function to plot 3D point cloud
def plot_point_cloud(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract the X, Y, Z coordinates from the points
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

# Load the .xyz point cloud file
file_path = 'punet_data/punet_d/50000_ground_truth/armadillo.xyz'  # Replace with your file path
points = load_xyz(file_path)
plot_point_cloud(points)