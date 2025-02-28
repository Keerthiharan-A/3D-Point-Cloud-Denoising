import numpy as np
import torch
from scipy.spatial import Delaunay

def generate_input(point_cloud):
    # Perform Delaunay triangulation
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