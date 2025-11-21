import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

# --- CONFIGURATION ---
DATA_DIR = "../../data"        # <-- UPDATED
RESULTS_DIR = "../../results/GAN"   # <-- UPDATED
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001

os.makedirs(RESULTS_DIR, exist_ok=True)


# --- 1. MODEL ARCHITECTURE ---
class Generator(nn.Module):
    def __init__(self, time_steps):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_steps, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, time_steps),
            nn.Sigmoid()   # NDVI/EVI2 stays between 0–1
        )
    def forward(self, x):
        return self.net(x)


# --- 2. DATA PREPARATION ---
def load_data():
    print("Loading data...")

    # Your dataset: gapped_h.npy, original_h.npy
    y_gap = np.load(f"{DATA_DIR}/gapped_h.npy")
    y_true = np.load(f"{DATA_DIR}/original_h.npy")

    T, H, W = y_gap.shape     # (365,50,60)

    # Flatten: (T,H,W) → (H*W, T) = (3000,365)
    X_train = y_gap.reshape(T, -1).T
    Y_train = y_true.reshape(T, -1).T

    return X_train, Y_train, T, H, W


# --- 3. TRAINING LOOP ---
def train():
    X, Y, T, H, W = load_data()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator(T)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(generator.parameters(), lr=LR)

    print(f"Training on {len(X)} pixels for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            reconstruction = generator(batch_x)
            loss = criterion(reconstruction, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(loader):.6f}")

    # --- 4. INFERENCE ---
    print("Generating GAN output (full resolution)...")
    generator.eval()
    with torch.no_grad():
        gan_output_flat = generator(X_tensor).numpy()

    gan_output_3d = gan_output_flat.T.reshape(T, H, W)

    np.save(f"{RESULTS_DIR}/gan_output_h.npy", gan_output_3d)

    print(f"✅ GAN output saved → {RESULTS_DIR}/gan_output_h.npy")
    print(f"Shape: {gan_output_3d.shape}")


if __name__ == "__main__":
    train()
