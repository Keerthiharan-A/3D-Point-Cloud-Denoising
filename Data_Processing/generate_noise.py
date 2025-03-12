import os
import numpy as np

# Ensure reproducibility
np.random.seed(2025)

# Define noise functions
def uniform_noise(pointcloud, severity):
    N, C = pointcloud.shape
    c = [0.01, 0.02, 0.03, 0.04, 0.05][severity-1]
    jitter = np.random.uniform(-c, c, (N, C))
    return (pointcloud + jitter).astype('float32')

def gaussian_noise(pointcloud, severity):
    N, C = pointcloud.shape
    c = [0.01, 0.015, 0.02, 0.025, 0.03][severity-1]
    jitter = np.random.normal(size=(N, C)) * c
    return np.clip(pointcloud + jitter, -1, 1).astype('float32')

def background_noise(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N//45, N//40, N//35, N//30, N//20][severity-1]
    jitter = np.random.uniform(-1, 1, (c, C))
    return np.concatenate((pointcloud, jitter), axis=0).astype('float32')

def impulse_noise(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N//30, N//25, N//20, N//15, N//10][severity-1]
    index = np.random.choice(N, c, replace=False)
    pointcloud[index] += np.random.choice([-1, 1], size=(c, C)) * 0.1
    return pointcloud.astype('float32')

MAP = {
    'uniform': uniform_noise,
    'gaussian': gaussian_noise,
    'background': background_noise,
    'impulse': impulse_noise
}

INPUT_FOLDER = "3D_Data/pcnet_clean"
OUTPUT_FOLDER = "3D_Data/pcnet_noisy"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_xyz(file_path):
    return np.loadtxt(file_path, delimiter=' ')

def save_xyz(file_path, pointcloud):
    np.savetxt(file_path, pointcloud, fmt='%.6f', delimiter=' ')

if __name__ == "__main__":
    xyz_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".xyz")]
    
    for xyz_file in xyz_files:
        file_path = os.path.join(INPUT_FOLDER, xyz_file)
        pointcloud = load_xyz(file_path)
        base_filename = os.path.splitext(xyz_file)[0]

        for corruption, func in MAP.items():
            for severity in range(1, 6):
                noisy_pc = func(pointcloud, severity)
                output_file = os.path.join(OUTPUT_FOLDER, f"{base_filename}_{corruption}_{severity}.xyz")
                save_xyz(output_file, noisy_pc)
                print(f"Saved: {output_file}")
    
    print("Noise generation complete!")
    
    # print('[', end="")
    # for corruption, func in MAP.items():
    #         for severity in range(1, 6):
    #             print(f"{corruption}_{severity}, ", end="")
    # print(']')