import torch
import torch.nn as nn
import torch.optim as optim
import geoopt
import matplotlib.pyplot as plt
import numpy as np
import random

# --- SEED CONTROL ---
# Try your own seeds, or turn off for random ones.
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
print("If you want to explore this seed's Decision Boundary Comparison, copy it over to visualize_decision_boundary.py")
print("="*60 + "\n")

torch.manual_seed(current_seed)

# --- The "Stamina" Model ---
class DynamicCurvatureNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # 1. SST Shunting Gate
        self.sst_gate = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # 2. The Dendrite (Manifold)
        self.manifold = geoopt.PoincareBall(c=1.0)
        
        # Initialize weights VERY small to force reliance on geometry or growth
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
        
        # CLAMP: Ensure we don't hit exact zero, but get close
        c = torch.clamp(gamma, min=1e-4, max=5.0)
        
        # Dynamic Manifold
        temp_manifold = geoopt.PoincareBall(c=c)
        
        # Hyperbolic Pass
        x_hyp = temp_manifold.expmap0(x)
        hidden = temp_manifold.mobius_matvec(self.weight, x_hyp)
        hidden = temp_manifold.mobius_add(hidden, self.bias)
        hidden_out = temp_manifold.logmap0(hidden)
        
        return self.classifier(hidden_out), gamma

# --- Simulation Logic ---
x_data = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
y_data = torch.tensor([[0.], [1.], [1.], [0.]])

def run_stamina_test(label, tax_rate):
    print(f"Running Stamina Test: {label}")
    
    # hidden_dim=2 is the "possible but hard" boundary
    model = DynamicCurvatureNet(input_dim=2, hidden_dim=2, output_dim=1)
    
    # Bias the brain to start Hyperbolic
    # This lets the Healthy brain "taste" success before the tax hits.
    nn.init.constant_(model.sst_gate[2].bias, 2.0)
    
    # Weight Decay (0.1)
    # This prevents the Euclidean brain from "Brute Forcing" the solution with large weights.
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.1)
    
    mse = nn.MSELoss()
    history = {"gamma": [], "error": []}
    
    for epoch in range(1500):
        optimizer.zero_grad()
        pred, gamma = model(x_data)
        
        # 1. Task Loss (Predator Pressure)
        task_loss = mse(pred, y_data)
        
        # 2. Metabolic Tax (Starvation Pressure)
        tax_loss = tax_rate * (gamma ** 2)
        
        # Total Loss (Survival Pressure)
        total_loss = (task_loss * 20.0) + tax_loss
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            history["gamma"].append(gamma.item())
            history["error"].append(task_loss.item())
            
    return history

# --- Run Experiments ---

healthy = run_stamina_test("Healthy", tax_rate=0.001)

pathology = run_stamina_test("Alzheimer's", tax_rate=5.0)

# --- Advanced Visualization ---
plt.figure(figsize=(14, 6))

epochs_x = np.arange(0, 1500, 10)

# Plot A: (Gamma)
plt.subplot(1, 2, 1)
plt.plot(epochs_x, healthy["gamma"], 'g', lw=3, label="Healthy (Maintained Structure)")
plt.plot(epochs_x, pathology["gamma"], 'r--', lw=3, label="Alzheimer's (Structural Collapse)")
plt.title("Dendritic Curvature ($\gamma$)")
plt.xlabel("Epochs")
plt.ylabel("Curvature")
plt.legend()
plt.grid(alpha=0.3)

# Plot B: (Error)
plt.subplot(1, 2, 2)
plt.plot(epochs_x, healthy["error"], 'g', lw=3, label="Healthy (Solved)")
plt.plot(epochs_x, pathology["error"], 'r--', lw=3, label="Alzheimer's (Cognitive Failure)")
plt.axhline(0.25, color='k', linestyle=':', label="Random Guess Limit")

# Force the 0.25 line to the exact center
plt.ylim(-0.02, 0.52)

plt.title("Cognitive Performance (XOR)")
plt.xlabel("Epochs")
plt.ylabel("MSE Error")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("brain_sim.png")
print("Saved visualization to 'brain_sim.png'")
plt.show()