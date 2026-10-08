import torch
import torch.nn as nn


class ContinuousRNN(nn.Module):
    """dz/dt = -z/tau + W tanh(z) + U [x; task_vec], RK2-integriert."""

    def __init__(self, input_dim=1, task_dim=0, hidden_dim=32,
                 tau=1.0, dt=0.1, substeps=5, task_scale=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_dim = task_dim
        self.task_scale = task_scale
        self.tau, self.dt, self.substeps = tau, dt, substeps
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.3)
        self.U = nn.Linear(input_dim + task_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, input_dim)

    def dynamics(self, z, x, task_vec):
        if self.task_dim > 0:
            u = torch.cat([x, task_vec * self.task_scale], dim=-1)
        else:
            u = x
        return -z / self.tau + torch.tanh(z) @ self.W.T + self.U(u)

    def forward(self, x_seq, task_vec=None):
        B, T, _ = x_seq.shape
        if task_vec is None:
            task_vec = torch.zeros(B, 0, device=x_seq.device)
        z = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        states = []
        for t in range(T):
            x = x_seq[:, t]
            for _ in range(self.substeps):
                k1 = self.dynamics(z, x, task_vec)
                k2 = self.dynamics(z + 0.5 * self.dt * k1, x, task_vec)
                z = z + self.dt * k2
            states.append(z)
        states = torch.stack(states, dim=1)
        return self.readout(states), states
