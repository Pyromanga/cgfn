import torch
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch
from cgfn.criticality import jacobian_spectral_radius, branching_ratio
from cgfn.geometry import mean_state_signature, signature_distance

torch.manual_seed(0)
device = "cpu"


def train_on(task, steps=2000, lr=1e-3, seq_len=8, hidden_dim=64):
    model = ContinuousRNN(hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        x, y = make_batch(task, seq_len=seq_len, device=device)
        out, _ = model(x)
        loss = lossfn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, loss.item()


def main():
    print(f"{'task':10s} {'loss':>8s} {'spec_rad':>10s} {'branch':>8s}")
    signatures = {}
    for task in TASKS:
        model, loss = train_on(task)
        x, _ = make_batch(task, batch_size=64, seq_len=8, device=device)
        with torch.no_grad():
            _, states = model(x)
        sr = jacobian_spectral_radius(model, states[:, -1]).mean().item()
        br = branching_ratio(states).mean().item()
        signatures[task] = mean_state_signature(states)
        print(f"{task:10s} {loss:8.4f} {sr:10.4f} {br:8.4f}")

    print("\nSignatur-Distanzen (Mittelwert im Feature-Raum):")
    keys = list(signatures)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = signature_distance(signatures[keys[i]], signatures[keys[j]])
            print(f"  {keys[i]:8s} vs {keys[j]:8s}: {d:.4f}")

    print("\nSignatur-Normen:")
    for k, v in signatures.items():
        print(f"  {k:8s}: {v.norm().item():.4f}")


if __name__ == "__main__":
    main()
