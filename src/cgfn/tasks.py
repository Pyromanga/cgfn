import torch

TASKS = ["copy", "reverse", "parity"]


def make_batch(task, batch_size=32, seq_len=16, device="cpu"):
    x = torch.randint(0, 2, (batch_size, seq_len), device=device).float()
    if task == "copy":
        y = x
    elif task == "reverse":
        y = x.flip(1)
    elif task == "parity":
        y = torch.cumsum(x, dim=1) % 2
    else:
        raise ValueError(task)
    return x.unsqueeze(-1), y.unsqueeze(-1)
