import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GraphAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GraphAutoencoder, self).__init__()
        
        # Encoder: GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)  # First GCN layer
        self.conv2 = GCNConv(hidden_dim, latent_dim)  # Latent representation layer
        
        # Decoder: GCN layers to reconstruct the graph
        self.deconv1 = GCNConv(latent_dim, hidden_dim)  # First deconvolution layer
        self.deconv2 = GCNConv(hidden_dim, input_dim)  # Output layer, reconstruct input features
    
    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))  # Apply first GCN layer with ReLU activation
        x = F.dropout(x, p=0.5, training=self.training)  # Dropout for regularization
        latent = self.conv2(x, edge_index)  # Latent space representation
        return latent
    
    def decode(self, latent, edge_index):
        x = F.relu(self.deconv1(latent, edge_index))  # Apply first deconvolution layer
        x = F.dropout(x, p=0.5, training=self.training)  # Dropout for regularization
        reconstructed = self.deconv2(x, edge_index)  # Reconstructed graph features
        return reconstructed
    
    def forward(self, x, edge_index):
        latent = self.encode(x, edge_index)  # Encode the input data to latent space
        reconstructed = self.decode(latent, edge_index)  # Decode the latent space back to input space
        return reconstructed, latent
