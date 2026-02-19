import torch
import torch.nn as nn
import torch.optim as optim
import geoopt
import matplotlib.pyplot as plt
import numpy as np
import random

# --- SEED CONTROL ---
# Copy your custom or randomly generated seed here.
# The seed for the visual used in the README.me is 137
USE_LOCKED_SEED = True
LOCKED_SEED = 137

if USE_LOCKED_SEED:
    current_seed = LOCKED_SEED
else:
    # Pick a random manageable number
    current_seed = random.randint(1000, 99999)

print("\n" + "="*60)
print(f"SEED: {current_seed}")
print("="*60 + "\n")

torch.manual_seed(current_seed)

# --- 1. The Model (Stamina Test Version) ---
class DynamicCurvatureNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # SST Gate
        self.sst_gate = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(), 
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        # Dendrite
        self.manifold = geoopt.PoincareBall(c=1.0)
        self.weight = geoopt.ManifoldParameter(
            torch.randn(hidden_dim, input_dim) * 0.01,
            manifold=self.manifold
        )
        self.bias = geoopt.ManifoldParameter(
            torch.zeros(hidden_dim), 
            manifold=self.manifold
        )
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gamma = self.sst_gate(x).mean()
        c = torch.clamp(gamma, min=1e-4, max=5.0)
        
        temp_manifold = geoopt.PoincareBall(c=c)
        x_hyp = temp_manifold.expmap0(x)
        hidden = temp_manifold.mobius_matvec(self.weight, x_hyp)
        hidden = temp_manifold.mobius_add(hidden, self.bias)
        hidden_out = temp_manifold.logmap0(hidden)
        
        return self.classifier(hidden_out), gamma

# --- 2. Training Helper ---
x_data = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
y_data = torch.tensor([[0.], [1.], [1.], [0.]])

def train_and_return_model(label, tax_rate):
    print(f"Training {label} (Tax={tax_rate})...")
    
    # Using hidden_dim=2 to allow solution IF curved
    model = DynamicCurvatureNet(input_dim=2, hidden_dim=2, output_dim=1)
    
    # Initialize "Open"
    nn.init.constant_(model.sst_gate[2].bias, 2.0)
    
    # Optimizer with Weight Decay (The Stamina Trap)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.1)
    mse = nn.MSELoss()
    
    # Train for 1500 epochs to ensure convergence
    for epoch in range(1501):
        optimizer.zero_grad()
        pred, gamma = model(x_data)
        
        task_loss = mse(pred, y_data)
        
        # Fixed: Added '*' for multiply and '**' for power
        tax_loss = tax_rate * (gamma ** 2)
        
        # Fixed: Added '*' for multiply
        total_loss = (task_loss * 20.0) + tax_loss
        
        total_loss.backward()
        optimizer.step()
        
    # Fixed: Formatting syntax
    print(f" > Final Gamma: {gamma.item():.4f}")
    return model

# --- 3. Execute Training ---
healthy_model = train_and_return_model("Healthy", tax_rate=0.001)
alzheimers_model = train_and_return_model("Alzheimer's", tax_rate=5.0)

# --- 4. Generate Visualization Grid ---
# Create a grid of points from -0.5 to 1.5
x_range = np.linspace(-0.5, 1.5, 100)
y_range = np.linspace(-0.5, 1.5, 100)
xx, yy = np.meshgrid(x_range, y_range)

# Flatten for batch processing
grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

# Get predictions
with torch.no_grad(): 
    # Healthy Predictions
    h_pred, h_gamma = healthy_model(grid_tensor)
    h_z = h_pred.reshape(xx.shape).numpy()
    
    # Alzheimer's Predictions
    a_pred, a_gamma = alzheimers_model(grid_tensor)
    a_z = a_pred.reshape(xx.shape).numpy()

# --- 5. Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def plot_decision_boundary(ax, z_data, title, gamma_val):
    contour = ax.contourf(xx, yy, z_data, levels=20, cmap="RdBu", alpha=0.6, vmin=0, vmax=1)
    
    # Data Points
    ax.scatter([0, 1], [0, 1], c='darkred', s=200, edgecolor='white', linewidth=2, label="Class 0")
    ax.scatter([0, 1], [1, 0], c='darkblue', s=200, edgecolor='white', linewidth=2, label="Class 1")
    
    ax.set_title(f"{title}\nCurvature $\gamma \\approx {gamma_val:.2f}$")
    
    ax.grid(alpha=0.3)
    ax.legend()
plot_decision_boundary(ax1, h_z, "Healthy Brain (Warped Space)", h_gamma.mean().item())
plot_decision_boundary(ax2, a_z, "Pathological Brain (Flat Space)", a_gamma.mean().item())
plt.tight_layout()
plt.savefig("decision_boundary_comparison.png")
print("Saved visualization to 'decision_boundary_comparison.png'")
plt.show()