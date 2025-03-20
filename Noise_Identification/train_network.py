import torch
import torch.optim as optim
from sklearn.cluster import KMeans
from torch_geometric.loader import DataLoader
from gcn_autoencoder import GraphAutoencoder, combined_loss, EarlyStopping
from data_loader import generate_data
import config
import os
import glob

def get_latent_features(model, data):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        latent_features = model.encode(x, edge_index, edge_attr)
    
    return latent_features

''' Change the file location accordingly'''
output_file = '3D-Point-Cloud-Denoising/Noise_Identification/processed_data.pt'

generate_data("3D_Data", output_file) # To be run again only when datset is changed
data_list = torch.load(output_file, weights_only=False)
print("Dataset loaded successfully!")

# Get total number of samples
num_samples = len(data_list)

# Define split sizes
train_size = int(0.6 * num_samples)
val_size = int(0.2 * num_samples)
test_size = num_samples - train_size - val_size

# Shuffle indices
indices = torch.randperm(num_samples)
train_idx, val_idx, test_idx = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]

# Step 2: Create the split datasets
train_data = [data_list[i] for i in train_idx]
val_data = [data_list[i] for i in val_idx]
test_data = [data_list[i] for i in test_idx]

print("Data successfully split into Train, Validation, and Test sets!")
print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")

# Step 3: Create DataLoader instances for each split (to feed into the model)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

best_epoch = 0

def train(model, train_loader, val_loader, optimizer, device, num_epochs=50, patience=5):
    global best_epoch
    early_stopping = EarlyStopping(patience=patience)
    model.to(device)

    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_train_class_loss, total_train_recon_loss = 0, 0

        # Training loop
        for batch in train_loader:
            batch = batch.to(device)  # Move batch to device
            optimizer.zero_grad()

            # Forward pass
            reconstructed, latent, noise_pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            # Compute loss
            recon_loss, class_loss = combined_loss(reconstructed, batch.x, noise_pred, batch.y)
            loss = recon_loss + class_loss
            loss.backward()
            optimizer.step()

            total_train_class_loss += class_loss.item()
            total_train_recon_loss += recon_loss.item()
        
        avg_train_class_loss = total_train_class_loss / len(train_loader)
        avg_train_recon_loss = total_train_recon_loss / len(train_loader)

        # Validation loop
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)  # Move batch to device
                reconstructed, latent, noise_pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                recon_loss, class_loss = combined_loss(reconstructed, batch.x, noise_pred, batch.y)
                val_loss = recon_loss + class_loss
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Class Loss: {avg_train_class_loss:.4f}, Train Reconstruction Loss: {avg_train_recon_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model based on validation loss
        if avg_val_loss <= best_val_loss:
            prev_checkpoint = f"best_model_epoch{best_epoch}.pt"
            if os.path.exists(prev_checkpoint):
                os.remove(prev_checkpoint)
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"best_model_epoch{epoch+1}.pt")  # Save best model
            print("Model improved, saving checkpoint")
            best_epoch = epoch+1

        # Check early stopping
        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training completed, Best model saved as 'best_model_epoch{best_epoch}.pt'")

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
train(model, train_loader, val_loader, optimizer, device, num_epochs=config.num_epochs, patience=5)

# Load the best model before testing
model.load_state_dict(torch.load(f"best_model_epoch{best_epoch}.pt"))
model.to(device)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        predictions = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)[-1]
        predicted_labels = predictions.argmax(dim=1)
        print("Predicted Labels:", predicted_labels)
        print("True Labels:", batch.y)
        correct += (predicted_labels == batch.y).sum().item()
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