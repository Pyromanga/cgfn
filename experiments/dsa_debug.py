import torch
from cgfn.model import ContinuousRNN
from cgfn.baselines import GRUBaseline, LSTMBaseline, VanillaRNNBaseline
from cgfn.tasks import TASKS, make_batch
from cgfn.dsa import dsa_debug

device = "cpu"
task_dim = len(TASKS)

def train(ModelClass, seed, task, steps=2000, lr=1e-3, hidden_dim=32):
    torch.manual_seed(seed)
    m = ModelClass(task_dim=task_dim, hidden_dim=hidden_dim)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    idx = TASKS.index(task)
    for _ in range(steps):
        x, y = make_batch(task, seq_len=8, device=device)
        tv = torch.zeros(x.shape[0], task_dim, device=device); tv[:, idx] = 1.0
        out, _ = m(x, tv)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return m

for name, MC in [("CT-RNN", ContinuousRNN), ("GRU", GRUBaseline), ("LSTM", LSTMBaseline), ("Vanilla", VanillaRNNBaseline)]:
    print(f"\n--- {name} ---")
    m1 = train(MC, 0, "copy")
    m2 = train(MC, 1, "copy")
    d = dsa_debug(m1, m2, "copy")
    print(f"  cross-seed copy: err_self={d['err_self']:.4f} err_cross={d['err_cross']:.4f} raw={d['raw_score']:.4f}")
    m3 = train(MC, 0, "parity")
    d = dsa_debug(m1, m3, "copy")
    print(f"  copy vs parity: err_self={d['err_self']:.4f} err_cross={d['err_cross']:.4f} raw={d['raw_score']:.4f}")
