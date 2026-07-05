import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("wine.csv") 

features = ["volatile acidity", "citric acid", "sulphates", "alcohol"]
target   = "quality"

X = df[features].values
y = df[target].values.reshape(-1, 1)

# ─────────────────────────────────────────────
#  SCALER
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")
print("✅ scaler.pkl saved")

# ─────────────────────────────────────────────
#  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.float32)

# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),
        )
    def forward(self, x):
        return self.model(x)

model = ANN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test), y_test)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

# ─────────────────────────────────────────────
#  SAVE MODEL
# ─────────────────────────────────────────────
torch.save(model.state_dict(), "model.pkl")
print("✅ model.pkl saved")
print("Done! Both files ready 🍷")
