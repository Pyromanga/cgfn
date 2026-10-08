import torch
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch

device = "cpu"


def one_hot(task_idx, batch_size, task_dim):
    v = torch.zeros(batch_size, task_dim, device=device)
    v[:, task_idx] = 1.0
    return v


def train_with_insd(seed, task, reference_states=None, steps=4000,
                    lr=1e-3, seq_len=8, hidden_dim=32,
                    task_dim=len(TASKS), lambda_insd=0.5):
    torch.manual_seed(seed)
    task_idx = TASKS.index(task)
    model = ContinuousRNN(task_dim=task_dim, hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()

    if reference_states is None:
        reference_states = []

    if len(reference_states) > 0:
        R = torch.cat(reference_states, dim=0)
        # Orthonormale Basis der Referenzrichtungen, H x k
        Q, _ = torch.linalg.qr(R.T)
    else:
        Q = None

    for step in range(steps):
        x, y = make_batch(task, seq_len=seq_len, device=device)
        tv = one_hot(task_idx, x.shape[0], task_dim)
        out, states = model(x, tv)
        task_loss = lossfn(out, y)

        if Q is not None:
            B, T, H = states.shape
            Z = states.reshape(B * T, H)
            # Q ist (H, k), Projektion Z @ Q ist (B*T, k)
            proj = Z @ Q
            insd_penalty = (proj ** 2).mean()
            loss = task_loss + lambda_insd * insd_penalty
        else:
            loss = task_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    return model, task_loss.item()


@torch.no_grad()
def collect_states_for_ref(model, task, seq_len=8, batch=32):
    task_idx = TASKS.index(task)
    x, _ = make_batch(task, batch_size=batch, seq_len=seq_len, device=device)
    tv = one_hot(task_idx, batch, len(TASKS))
    _, states = model(x, tv)
    B, T_len, H = states.shape
    return states.reshape(B * T_len, H)
