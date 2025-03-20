import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool

class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes):
        super(GraphAutoencoder, self).__init__()
        
        # Encoder
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, latent_dim)  # Bottleneck layer
        
        # Decoder
        self.deconv1 = GATConv(latent_dim, hidden_dim)
        self.deconv2 = GATConv(hidden_dim, input_dim)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),  # First fully connected layer
            nn.ReLU(),  # Activation function
            nn.Linear(latent_dim // 2, num_classes),  # Output layer (num_classes = 10)
        )

    def encode(self, x, edge_index, edge_attr=None):
        # Use edge_attr in the convolution layer if provided
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)
        latent = self.conv2(x, edge_index, edge_attr=edge_attr)
        return latent

    def decode(self, latent, edge_index, edge_attr=None):
        # Use edge_attr in the convolution layer if provided
        x = F.relu(self.deconv1(latent, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)
        reconstructed = self.deconv2(x, edge_index, edge_attr=edge_attr)
        return reconstructed

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        latent = self.encode(x, edge_index, edge_attr)
        reconstructed = self.decode(latent, edge_index, edge_attr)

        if batch is not None:
            batch_size = batch.max().item() + 1  # Total graphs in batch
            pooled_latent = torch.zeros(batch_size, latent.shape[1], device=latent.device)  

            for graph_idx in range(batch_size):
                pooled_latent[graph_idx] = latent[batch == graph_idx].mean(dim=0)  # Aggregate per graph

        else:
            pooled_latent = latent.mean(dim=0, keepdim=True)

        noise_pred = self.classifier(pooled_latent)  # Shape: (batch_size, num_classes)
        return reconstructed, latent, noise_pred


def combined_loss(reconstructed, original, noise_pred, noise_label):
    recon_loss = F.mse_loss(reconstructed, original)
    class_loss = F.cross_entropy(noise_pred, noise_label)
    return recon_loss, class_loss

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss + self.min_delta < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience