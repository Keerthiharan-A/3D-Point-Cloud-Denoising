import numpy as np
import torch
from scipy.spatial import Delaunay
import config
import os
import multiprocessing
from torch_geometric.data import Data

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_xyz(file_path):
    """Load XYZ point cloud data from a file with error handling."""
    try:
        data = np.loadtxt(file_path, usecols=(0, 1, 2))
        return data
    except Exception as e:
        print(f"Warning: Skipping file {file_path} due to error: {e}")
        return None

def generate_input(point_cloud):
    """Generate edge indices and attributes for the point cloud graph."""
    try:
        tri = Delaunay(point_cloud)
        edges = set()
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    edges.add(tuple(sorted([simplex[i], simplex[j]])))

        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous().to(device)
        distances = np.linalg.norm(point_cloud[edge_index[0].cpu().numpy()] - point_cloud[edge_index[1].cpu().numpy()], axis=1)
        edge_attr = torch.tensor(distances, dtype=torch.float).view(-1, 1).to(device)

        return edge_index, edge_attr
    
    except Exception as e:
        print(f"Warning: Error in generating input graph. Skipping this point cloud. Error: {e}")
        return None, None

def get_label(file_name):
    """Extract the noise label from the filename."""
    base_name = os.path.splitext(file_name)[0]
    
    for noise_level in config.noise_levels:
        if base_name.endswith(noise_level):
            parts = noise_level.split("_")
            noise, s = parts[0], int(parts[-1])
            severity = "low" if s < 3 else "high"
            return config.label_map.get(f"{noise}_{severity}", 0)

    return 0  # Default label if no match found

def process_file(file_path):
    """Process a single .xyz file."""
    data_array = load_xyz(file_path)
    if data_array is None:
        return None  # Skip corrupt files

    edge_index, edge_attr = generate_input(data_array)
    if edge_index is None or edge_attr is None:
        return None  # Skip files where graph creation failed

    file_name = os.path.basename(file_path)
    label = get_label(file_name)

    try:
        data = Data(
            x=torch.tensor(data_array, dtype=torch.float).to(device), 
            edge_index=edge_index, 
            edge_attr=edge_attr,
            y=torch.tensor(label, dtype=torch.long).to(device)
        )
        print("Processed ", file_path)
        return data
    
    except Exception as e:
        print(f"Warning: Skipping file {file_path} due to data creation error: {e}")
        return None

def generate_data(folder_path, output_file, num_workers=4):
    print(device)
    # Get all .xyz files including subdirectories
    count = 0
    xyz_files = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith('.xyz'):
                xyz_files.append(os.path.join(root, f))
                count += 1
    print(f"Found {count} .xyz files")

    # Use multiprocessing to process files in parallel
    with multiprocessing.Pool(num_workers) as pool:
        data_list = pool.map(process_file, xyz_files)

    # Remove None values (failed files)
    data_list = [data for data in data_list if data is not None]

    if not data_list:
        print("Error: No valid data processed. Check input files.")
        return

    # Save processed dataset
    torch.save(data_list, output_file)
    print(f"Processed dataset saved to {output_file}")