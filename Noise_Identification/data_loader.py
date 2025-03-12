import numpy as np
import torch
from scipy.spatial import Delaunay
import config
import os
from torch_geometric.data import Data

def load_xyz(file_path):
    data = np.loadtxt(file_path, usecols=(0, 1, 2))
    return data

def generate_input(point_cloud):

    tri = Delaunay(point_cloud)
    # Extract the edges (pairs of points connected in the triangulation)
    edges = set()
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                edges.add(tuple(sorted([simplex[i], simplex[j]])))

    # Convert the edge list to a torch tensor (edge_index)
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()

    # Calculate the distances (edge weights) based on Euclidean distance
    distances = np.linalg.norm(point_cloud[edge_index[0].numpy()] - point_cloud[edge_index[1].numpy()], axis=1)
    edge_attr = torch.tensor(distances, dtype=torch.float).view(-1, 1)

    return edge_index, edge_attr

def get_label(file_name):
    base_name = os.path.splitext(file_name)[0]
    
    # Check if any noise level matches the end of the base file name
    for noise_level in config.noise_levels:
        if base_name.endswith(noise_level):
            parts = noise_level.split("_")
            noise, s = parts[0], int(parts[-1])
            if s<3: severity = "low"
            else: severity = "high"
            return config.label_map[f"{noise}_{severity}"]
            
    return 0

def generate_data(folder_path, output_file):
    # Load all .xyz files from the folder
    xyz_files = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith('.xyz'):
                xyz_files.append(os.path.join(root, f))
    print("Processing ", len(xyz_files), " .xyz files")

    # Prepare the data list to hold all graph data
    i = 0
    data_list = []

    for file_path in xyz_files:
        data_array = load_xyz(file_path)
        edge_index, edge_attr = generate_input(data_array)
        file_name = os.path.basename(file_path)
        label = get_label(file_name)

        data = Data(
            x=torch.tensor(data_array, dtype=torch.float), 
            edge_index=edge_index, 
            edge_attr=edge_attr,
            y=torch.tensor(label, dtype=torch.long)
        )
        data_list.append(data)
        print("File : ", i)
        i += 1
    
    all_x = torch.cat([data.x for data in data_list], dim=0)
    all_edge_index = torch.cat([data.edge_index for data in data_list], dim=1)
    all_edge_attr = torch.cat([data.edge_attr for data in data_list], dim=0) if data_list[0].edge_attr is not None else None
    all_labels = torch.cat([data.y.unsqueeze(0) for data in data_list], dim=0)  # Combine labels

    # Create a single batched Data object
    batched_data = Data(x=all_x, edge_index=all_edge_index, edge_attr=all_edge_attr, y=all_labels)

    # Save the processed data
    # output_file = '3D-Point-Cloud-Denoising/Noise_Identification/processed_data.pt'
    torch.save(batched_data, output_file)
    print(f"Processed dataset saved to {output_file}")