import torch
import torch.nn as nn
import geoopt

class HyperbolicLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.manifold = geoopt.PoincareBall(c=1.0)
        self.weight = geoopt.ManifoldParameter(torch.randn(out_features, in_features) * 0.2)
        self.bias = geoopt.ManifoldParameter(torch.zeros(out_features))

    def forward(self, x, current_c):
        # Ensure c is positive and stable
        c = torch.clamp(current_c, min=1e-4, max=5.0)
        temp_manifold = geoopt.PoincareBall(c=c)
        
        # Hyperbolic Transform
        x_hyp = temp_manifold.expmap0(x)
        output_hyp = temp_manifold.mobius_matvec(self.weight, x_hyp)
        output_hyp = temp_manifold.mobius_add(output_hyp, self.bias)
        
        return temp_manifold.logmap0(output_hyp)

class SST_Gate(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.sense = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Sigmoid(),
            nn.Linear(8, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.sense(x)).mean()

class DynamicCurvatureNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.sst_neuron = SST_Gate(input_dim)
        self.dendrite = HyperbolicLayer(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gamma = self.sst_neuron(x)
        hidden = self.dendrite(x, gamma)
        out = self.classifier(hidden)
        return out, gamma