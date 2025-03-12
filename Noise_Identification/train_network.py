import torch
import torch.optim as optim
from sklearn.cluster import KMeans
from torch_geometric.data import Data, DataLoader
from gcn_autoencoder import GraphAutoencoder, combined_loss, EarlyStopping
from data_loader import generate_data
import config

def get_latent_features(model, data):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        latent_features = model.encode(x, edge_index, edge_attr)
    
    return latent_features

output_file = '3D-Point-Cloud-Denoising/Noise_Identification/processed_data.pt'
generate_data("3D_Data", output_file) # To be run again only when datset is changed
batched_data = torch.load(output_file)
print("Dataset loaded successfully!")

# Print the shapes to confirm
print(f"Feature shape: {batched_data.x.shape}")
print(f"Edge Index shape: {batched_data.edge_index.shape}")
print(f"Labels shape: {batched_data.y.shape}")

# Get total number of samples
num_samples = batched_data.x.shape[0]

# Define split sizes
train_size = int(0.6 * num_samples)
val_size = int(0.2 * num_samples)
test_size = num_samples - train_size - val_size

# Shuffle indices
indices = torch.randperm(num_samples)
train_idx, val_idx, test_idx = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]

# Split features and labels into train, val, and test sets
train_data = Data(
    x=batched_data.x[train_idx], 
    edge_index=batched_data.edge_index, 
    edge_attr=batched_data.edge_attr,
    y=batched_data.y[train_idx]
)

val_data = Data(
    x=batched_data.x[val_idx], 
    edge_index=batched_data.edge_index, 
    edge_attr=batched_data.edge_attr,
    y=batched_data.y[val_idx]
)

test_data = Data(
    x=batched_data.x[test_idx], 
    edge_index=batched_data.edge_index, 
    edge_attr=batched_data.edge_attr,
    y=batched_data.y[test_idx]
)

print("Data successfully split into Train, Validation, and Test sets!")
print(f"Train samples: {train_size}, Validation samples: {val_size}, Test samples: {test_size}")

train_loader = DataLoader([train_data], batch_size=32, shuffle=True)
val_loader = DataLoader([val_data], batch_size=32, shuffle=False)
test_loader = DataLoader([test_data], batch_size=32, shuffle=False)


def train(model, train_loader, val_loader, optimizer, device, alpha=1.0, num_epochs=50, patience=5):
    early_stopping = EarlyStopping(patience=patience)
    model.to(device)

    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0

        # Training loop
        for batch in train_loader:
            batch = batch.to(device)  # Move batch to device
            optimizer.zero_grad()

            # Forward pass
            reconstructed, latent, noise_pred = model(batch.x, batch.edge_index)
            
            # Compute loss
            loss = combined_loss(reconstructed, batch.x, noise_pred, batch.y, alpha)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)  # Move batch to device
                reconstructed, latent, noise_pred = model(batch.x, batch.edge_index)

                val_loss = combined_loss(reconstructed, batch.x, noise_pred, batch.y, alpha)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model based on validation loss
        if avg_val_loss <= best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"best_model_epoch{epoch+1}.pt")  # Save best model
            print("Model improved, saving checkpoint")

        # Check early stopping
        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print("Training completed, Best model saved as 'best_model.pt'")

# Initialize the model
input_dim = config.input_dim
hidden_dim = config.hidden_dim
latent_dim = config.latent_dim
num_classes = config.num_classes
model = GraphAutoencoder(input_dim, hidden_dim, latent_dim, num_classes)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, optimizer, and train
train(model, train_loader, val_loader, optimizer, device, alpha=1.0, num_epochs=50, patience=5)

# Load the best model before testing
model.load_state_dict(torch.load("best_model.pt"))
model.to(device)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        out = model(batch.x, batch.edge_index)
        predictions = out.argmax(dim=1)
        correct += (predictions == batch.y).sum().item()
        total += batch.y.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")


## FOR CLUSTERING

# After training, extract the latent features from the batched data
# latent_features = get_latent_features(model, batched_data)

# # Perform K-means clustering on the latent features
# noise_types = 2  # Assuming 2 categories: low vs. high noise
# kmeans = KMeans(n_clusters=noise_types)
# noise_labels = kmeans.fit_predict(latent_features.detach().numpy())

# print("Clustered Noise Labels:", noise_labels)