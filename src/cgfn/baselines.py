"""
Standard-RNN-Baselines mit gleicher Parameterzahl wie ContinuousRNN.
Entscheidet, ob die ODE-Struktur ueberhaupt eine Rolle spielt.
"""
import torch
import torch.nn as nn


class GRUBaseline(nn.Module):
    def __init__(self, input_dim=1, task_dim=0, hidden_dim=32):
        super().__init__()
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim + task_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_seq, task_vec=None):
        B, T, _ = x_seq.shape
        if task_vec is None and self.task_dim > 0:
            task_vec = torch.zeros(B, self.task_dim, device=x_seq.device)
        elif task_vec is not None and self.task_dim > 0:
            task_vec = task_vec.unsqueeze(1).expand(-1, T, -1)
            x_seq = torch.cat([x_seq, task_vec], dim=-1)
        out, _ = self.gru(x_seq)
        return self.readout(out), out.unsqueeze(0)


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim=1, task_dim=0, hidden_dim=32):
        super().__init__()
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim + task_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_seq, task_vec=None):
        B, T, _ = x_seq.shape
        if task_vec is None and self.task_dim > 0:
            task_vec = torch.zeros(B, self.task_dim, device=x_seq.device)
        elif task_vec is not None and self.task_dim > 0:
            task_vec = task_vec.unsqueeze(1).expand(-1, T, -1)
            x_seq = torch.cat([x_seq, task_vec], dim=-1)
        out, _ = self.lstm(x_seq)
        return self.readout(out), out.unsqueeze(0)


class VanillaRNNBaseline(nn.Module):
    def __init__(self, input_dim=1, task_dim=0, hidden_dim=32):
        super().__init__()
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_dim + task_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_seq, task_vec=None):
        B, T, _ = x_seq.shape
        if task_vec is None and self.task_dim > 0:
            task_vec = torch.zeros(B, self.task_dim, device=x_seq.device)
        elif task_vec is not None and self.task_dim > 0:
            task_vec = task_vec.unsqueeze(1).expand(-1, T, -1)
            x_seq = torch.cat([x_seq, task_vec], dim=-1)
        out, _ = self.rnn(x_seq)
        return self.readout(out), out.unsqueeze(0)
