import torch.optim as optim
import config
from gcn_autoencoder import GraphAutoencoder
import torch
import numpy as np
from data_loader import generate_input
from sklearn.cluster import KMeans
from torch_geometric.data import Data

def train(model, data, criterion, optimizer, epochs=100):
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        x, edge_index = data.x, data.edge_index
        
        # Forward pass: Get the reconstructed input and latent representation
        reconstructed, latent = model(x, edge_index)
        
        loss = criterion(reconstructed, x)
        
        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

def get_latent_features(model, data):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        latent_features = model.encode(x, edge_index, edge_attr)
    
    return latent_features

def load_xyz(file_path):
    data = np.loadtxt(file_path, skiprows=2, usecols=(1, 2, 3)) # Skip 2 rows cause of metadata
    return data

# Initialize the model
input_dim = config.input_dim
hidden_dim = config.hidden_dim
latent_dim = config.latent_dim
model = GraphAutoencoder(input_dim, hidden_dim, latent_dim)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Load the data from the .xyz file
file_path = '' # Specify the path to the .xyz file
data_array = load_xyz(file_path)
edge_index, edge_attr = generate_input(data_array)
data = Data(x=torch.tensor(data_array, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)

# Train the model
train(model, data, criterion, optimizer, epochs=100)

# Extract latent features after training
latent_features = get_latent_features(model, data)
kmeans = KMeans(n_clusters=2)  # Assuming you want to cluster into 2 categories (low vs high noise)
noise_labels = kmeans.fit_predict(latent_features.detach().numpy())
print("Clustered Noise Labels:", noise_labels)