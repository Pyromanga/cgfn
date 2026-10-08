import torch
import torch.nn as nn


class ContinuousRNN(nn.Module):
    """dz/dt = -z/tau + W tanh(z) + U x(t), RK2-integriert."""

        def __init__(self, input_dim=1, hidden_dim=32, tau=1.0, dt=0.1, substeps=5):
                super().__init__()
                        self.hidden_dim = hidden_dim
                                self.tau, self.dt, self.substeps = tau, dt, substeps
                                        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.3)
                                                self.U = nn.Linear(input_dim, hidden_dim)
                                                        self.readout = nn.Linear(hidden_dim, input_dim)

                                                            def dynamics(self, z, x):
                                                                    return -z / self.tau + torch.tanh(z) @ self.W.T + self.U(x)

                                                                        def forward(self, x_seq, control=None):
                                                                                B, T, _ = x_seq.shape
                                                                                        z = torch.zeros(B, self.hidden_dim, device=x_seq.device)
                                                                                                states = []
                                                                                                        for t in range(T):
                                                                                                                    x = x_seq[:, t]
                                                                                                                                for _ in range(self.substeps):
                                                                                                                                                k1 = self.dynamics(z, x)
                                                                                                                                                                k2 = self.dynamics(z + 0.5 * self.dt * k1, x)
                                                                                                                                                                                z = z + self.dt * k2
                                                                                                                                                                                                if control is not None:
                                                                                                                                                                                                                    z = z + self.dt * control(z, t)
                                                                                                                                                                                                                                states.append(z)
                                                                                                                                                                                                                                        states = torch.stack(states, dim=1)
                                                                                                                                                                                                                                                return self.readout(states), states