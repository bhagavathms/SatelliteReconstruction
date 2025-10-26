import os, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

DATADIR = "../../data/gan_dataset"
OUTDIR  = "../../results/GAN"; os.makedirs(OUTDIR, exist_ok=True)

# --- Tiny 1D GAN (temporal only) ---
class Gen(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(T, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, T)
        )
    def forward(self, x): return self.net(x)

class Disc(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(T, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x)

# Load train/val
Xtr = np.load(f"{DATADIR}/X_train.npy").astype("float32")
Ytr = np.load(f"{DATADIR}/Y_train.npy").astype("float32")
Xva = np.load(f"{DATADIR}/X_val.npy").astype("float32")
Yva = np.load(f"{DATADIR}/Y_val.npy").astype("float32")
T = Xtr.shape[1]

device = torch.device("cpu")
G, D = Gen(T).to(device), Disc(T).to(device)
optG = torch.optim.Adam(G.parameters(), lr=2e-4)
optD = torch.optim.Adam(D.parameters(), lr=2e-4)
bce  = nn.BCEWithLogitsLoss()
l1   = nn.L1Loss()

train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
val_ds   = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva))
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=256)

EPOCHS = 30  # keep modest for CPU
for ep in range(1, EPOCHS+1):
    G.train(); D.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)

        # --- Train D: real vs fake ---
        optD.zero_grad()
        real_logit = D(yb)
        with torch.no_grad():
            fake_y = G(xb)
        fake_logit = D(fake_y.detach())
        d_loss = bce(real_logit, torch.ones_like(real_logit)) + \
                 bce(fake_logit, torch.zeros_like(fake_logit))
        d_loss.backward(); optD.step()

        # --- Train G: fool D + L1 to target (mask-agnostic here) ---
        optG.zero_grad()
        gen_y = G(xb)
        g_adv = bce(D(gen_y), torch.ones_like(real_logit))
        g_rec = l1(gen_y, yb)
        g_loss = g_adv*0.3 + g_rec*0.7
        g_loss.backward(); optG.step()

    # simple val L1
    G.eval()
    with torch.no_grad():
        vloss = 0.0; n=0
        for xv, yv in val_dl:
            xv, yv = xv.to(device), yv.to(device)
            pr = G(xv); vloss += l1(pr,yv).item()*xv.size(0); n+=xv.size(0)
        vloss/=n
    print(f"Epoch {ep:02d} | val L1: {vloss:.4f}")

# --- Apply to full image ---
Y_FULL = "../../data/original.npy"; Y_GAP = "../../data/gapped.npy"
y_gap = np.load(Y_GAP).astype("float32")   # (T,H,W)
T,H,W = y_gap.shape
Xall = y_gap.transpose(1,2,0).reshape(H*W, T)
Xall = np.nan_to_num(Xall, nan=0.0).astype("float32")

G.eval()
with torch.no_grad():
    preds = []
    BS=512
    for i in tqdm(range(0, Xall.shape[0], BS), desc="Reconstructing"):
        xb = torch.from_numpy(Xall[i:i+BS]).to(device)
        pr = G(xb).cpu().numpy()
        preds.append(pr)
    preds = np.concatenate(preds, axis=0)

recon = preds.reshape(H, W, T).transpose(2,0,1)  # (T,H,W)
np.save(f"{OUTDIR}/recon.npy", recon)
print("✅ GAN reconstruction saved →", f"{OUTDIR}/recon.npy")
