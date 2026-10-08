import torch
from cgfn.model import ContinuousRNN
from cgfn.tasks import TASKS, make_batch
from cgfn.criticality import jacobian_spectral_radius, branching_ratio
from cgfn.geometry import svd_signature

torch.manual_seed(0)
device = "cpu"


def train_on(task, steps=300, lr=3e-3):
    model = ContinuousRNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        x, y = make_batch(task, device=device)
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
        x, _ = make_batch(task, batch_size=64, device=device)
        with torch.no_grad():
            _, states = model(x)
        z_final = states[:, -1]
        sr = jacobian_spectral_radius(model, z_final).mean().item()
        br = branching_ratio(states).mean().item()
        sig, _ = svd_signature(states)
        signatures[task] = sig.mean(dim=0)
        print(f"{task:10s} {loss:8.4f} {sr:10.4f} {br:8.4f}")

    print("\nPaarweise Signatur-Distanzen:")
    keys = list(signatures)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = (signatures[keys[i]] - signatures[keys[j]]).norm().item()
            print(f"  {keys[i]:8s} vs {keys[j]:8s}: {d:.4f}")


if __name__ == "__main__":
    main()
