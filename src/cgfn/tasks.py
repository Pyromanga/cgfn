import torch

TASKS = ["copy", "reverse", "parity", "mod3", "delayed_xor"]


def make_batch(task, batch_size=32, seq_len=16, device="cpu"):
    x = torch.randint(0, 2, (batch_size, seq_len), device=device).float()
    if task == "copy":
        y = x
    elif task == "reverse":
        y = x.flip(1)
    elif task == "parity":
        y = torch.cumsum(x, dim=1) % 2
    elif task == "mod3":
        z = torch.randint(0, 3, (batch_size, seq_len), device=device).float()
        y = torch.cumsum(z, dim=1) % 3 / 2.0
        return z.unsqueeze(-1) / 2.0, y.unsqueeze(-1)
    elif task == "delayed_xor":
        B, T = x.shape
        y = torch.zeros_like(x)
        y[:, 2:] = (x[:, 2:] + x[:, :-2]) % 2
        y[:, :2] = x[:, :2]
    else:
        raise ValueError(task)
    return x.unsqueeze(-1), y.unsqueeze(-1)
