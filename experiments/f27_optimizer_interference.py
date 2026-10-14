'''
F27 — OPTIMIERER-RESSOURCE-INTERFERENZ: Ist der kappa(W)-Drift (F16)
ein Artefakt der Nicht-Normalitaet des OPTIMIERER-Operators?

These (Tiefenanalyse VI, E3b): Adam und SGD+Momentum sind generisch
nicht-normale Update-Operatoren (arXiv:2605.23476); ihre transiente
Verstaerkung wirkt selektiv in Gewichtsrichtungen, die Schur-
Stoerungen aufbauen — d.h. der Optimierer selbst driftet die
Nicht-Normalitaet kappa(W) der Netz-Dynamik, unabhaengig vom Task.

VORHERSAGEN (scharf, widerlegbar):
  P1: kappa(W) driftet unter Adam staerker als unter SGD ohne
      Momentum (kappa(V_opt) von Adam ist hoeher).
  P2: Der Drift korreliert positiv mit der gemessenen Nicht-
      Normalitaet des linearisierten Update-Operators kappa(V_opt).
  P3: Der Drift ist Task-schwach: er bleibt sichtbar, wenn auf
      Rauschen statt auf Task gelernet wird (kein Task-Verlust).

DESIGN:
  - Gleiche Anfangsnetze (designed, festes lam, festes s) pro Seed.
  - 3 Optimierer: SGD (lr 0.01), SGD-Momentum (0.9, lr 0.01),
    Adam (lr 2e-3). 400 Steps auf sinus_memory-Batches.
  - Alle 20 Steps: kappa(W) (spektral/|eig|max), lam(W), Drift-Delta.
  - kappa(V_opt): linearisierter Update-Operator des laufenden
    Optimierers auf die W-Fluierung — approximiert per diagonal-
    preconditioning-matrix D und momentum-ubertrag:
      SGD:  J = I - lr*H_diag          (H_diag ~ diag der quadra-
      MGD:  J = (1-m)(I-lr*D) + m*I    tischen Verlustkraemmung,
      Adam: J ~ I - lr_eff * D         approximiert durch EMA der
    Wir schaetzen die diagonale Kraemmung als Varianz der letzten
    Gradienten (Billig-Approximation) und berechnen kappa(V) des
    so gebauten Operators analytisch (2x2-Block-Approximation der
    dominanten Gewichtsrichtung ist zu teuer — wir nutzen die
    Bedingungszahl der Eigenvektoren von J als Proxy).
  - Rauschen-Kontrolle: derselbe Lauf mit y = 0 (reine Regularis-
    ierung durch Nicht-Task-Gewichte) prueft P3.

ENTSCHEIDUNG:
  P1+P2 halten => F16 hat einen MECHANISMUS: der Optimierer frisst
  die Ressource. Homeostat (F21/F22) ist dann keine Spielerei,
  sondern die Antwort auf ein reales, von der Industrie nicht
  gesehenes Kontrollproblem.
  P1 faellt => kappa-Drift ist Task-getrieben, E3b trifft nicht
  auf unser Regime zu — auch gut zu wissen, bevor wir es im
  Papier behaupten.
'''
import argparse
import os
from pathlib import Path

import torch

from cgfn.model import ContinuousRNN
from cgfn.continuous import make_continuous_batch
from cgfn.design import make_designed_model


def kappa_of(W):
    sv = torch.linalg.svdvals(W)[0].item()
    lam = torch.linalg.eigvals(W).abs().max().item()
    return sv / max(lam, 1e-9), lam


def update_operator_condition(optimizer_name, lr, momentum, grad_sq_ema, step):
    '''Billig-Proxy: kappa(V) des approximierten diagonalen Update-
    Operators. Fuer diagonale D ist J symmetrisch-stoergbar; die
    Nicht-Normalitaet kommt beim MGD aus den gekoppelten Block-
    (w, v)-Zustaenden:  J_block = [[(1-m)(1-lr*d),  m ],[ (1-lr*d), 0]]
    mit d = diag-Kraemmung. Wir bauen den 2x2-Block und messen
    seine Eigenvektor-Konditionszahl — der Moment-Zustand ist die
    Quelle der Nicht-Normalitaet.'''
    d = grad_sq_ema.clamp(1e-8, 1e2)
    m = momentum if optimizer_name != 'sgd' else 0.0
    a = (1.0 - m) * (1.0 - lr * d)
    J = torch.tensor([[a, m], [1.0, 0.0]])
    eig = torch.linalg.eigvals(J)
    lam_max = eig.abs().max().clamp(1e-9)
    sv_max = torch.linalg.svdvals(J)[0]
    return (sv_max / lam_max).item()


def run_training(opt_name, seed, steps, noise_only, lr_map):
    torch.manual_seed(seed)
    m = make_designed_model(32, 1.0, 1.0, 0.8, seed, task_dim=0)
    if opt_name == 'sgd':
        opt = torch.optim.SGD(m.parameters(), lr=lr_map['sgd'])
        momentum = 0.0
    elif opt_name == 'mgd':
        opt = torch.optim.SGD(m.parameters(), lr=lr_map['mgd'], momentum=0.9)
        momentum = 0.9
    else:
        opt = torch.optim.Adam(m.parameters(), lr=lr_map['adam'])
        momentum = 0.9  # beta1
    lossfn = torch.nn.MSELoss()
    g_sq_ema = None
    traj = []
    k0, lam0 = kappa_of(m.W.detach())
    for step in range(steps):
        x, y = make_continuous_batch('sinus_memory', 16, 16)
        if noise_only:
            y = torch.randn_like(y) * 0.1
        out, _ = m(x)
        loss = lossfn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            g = m.W.grad
        g_sq = (g ** 2).mean()
        g_sq_ema = g_sq if g_sq_ema is None else 0.9 * g_sq_ema + 0.1 * g_sq
        if (step + 1) % 20 == 0:
            kv = update_operator_condition(opt_name, lr_map[opt_name], momentum,
                                           torch.tensor([g_sq_ema]), step)
            k, lam = kappa_of(m.W.detach())
            traj.append((step + 1, k, lam, kv, float(loss.item())))
    return traj


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--seeds', default='0,1,2,3')
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--out', default='results/f27_optimizer_interference_results.txt')
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    lr_map = {'sgd': 0.01, 'mgd': 0.01, 'adam': 2e-3}
    L = []
    L.append('=' * 110)
    L.append('F27 OPTIMIERER-RESSOURCE-INTERFERENZ: kappa(W)-Drift vs Nicht-Normalitaet des Optimierers')
    L.append(f'seeds={seeds} steps={args.steps} lr={lr_map}')
    L.append('P1: adam > mgd > sgd (Drift) | P2: Drift ~ kappa(V_opt) | P3: Drift bleibt bei Rausch-Targets')
    L.append('=' * 110)

    results = {}
    for noise_only in (False, True):
        tag = 'RAUSCHEN' if noise_only else 'TASK'
        L.append('')
        L.append(f'########## Modus: {tag} ##########')
        L.append(f"{'opt':>5s} {'seed':>4s} {'kappa0':>7s} {'kappaE':>7s} {'dkappa':>7s} {'lam0':>6s} {'lamE':>6s} {'kV_end':>7s} {'loss_end':>9s}")
        for opt_name in ('sgd', 'mgd', 'adam'):
            for seed in seeds:
                traj = run_training(opt_name, seed, args.steps, noise_only, lr_map)
                k0, kE = traj[0][1], traj[-1][1]
                lam0, lamE = traj[0][2], traj[-1][2]
                kv_end = traj[-1][3]
                loss_end = traj[-1][4]
                L.append(f'{opt_name:>5s} {seed:4d} {k0:7.2f} {kE:7.2f} {kE-k0:+7.2f} {lam0:6.3f} {lamE:6.3f} {kv_end:7.2f} {loss_end:9.4f}')
                results.setdefault((tag, opt_name), []).append((kE - k0, kv_end))

    L.append('')
    L.append('--- Aggregate (Seed-Mittel) ---')
    for tag in ('TASK', 'RAUSCHEN'):
        row = []
        for opt_name in ('sgd', 'mgd', 'adam'):
            sub = results[(tag, opt_name)]
            dk = sum(v[0] for v in sub) / len(sub)
            kv = sum(v[1] for v in sub) / len(sub)
            row.append((opt_name, dk, kv))
            L.append(f'[{tag}] {opt_name}: dkappa={dk:+.3f} | kappa(V_opt)={kv:.3f}')
        # P1-Check: adam > mgd > sgd?
        by = {o: d for o, d, _ in row}
        p1 = by['adam'] > by['mgd'] > by['sgd']
        L.append(f'[{tag}] P1 (adam>mgd>sgd): {"JA" if p1 else "NEIN"}')

    # P2: Pearson dkappa vs kappa(V_opt) ueber alle Runs (beide Modi)
    dks = torch.tensor([v[0] for sub in results.values() for v in sub])
    kvs = torch.tensor([v[1] for sub in results.values() for v in sub])
    if dks.std() > 1e-6 and kvs.std() > 1e-6:
        corr = torch.corrcoef(torch.stack([dks, kvs]))[0, 1].item()
        L.append(f'P2: corr(dkappa, kappa(V_opt)) ueber alle Runs = {corr:+.3f}')

    # P3: Drift bleibt unter Rauschen sichtbar?
    drift_task = sum(v[0] for v in results[('TASK', 'adam')]) / len(results[('TASK', 'adam')])
    drift_noise = sum(v[0] for v in results[('RAUSCHEN', 'adam')]) / len(results[('RAUSCHEN', 'adam')])
    L.append(f'P3: adam dkappa TASK {drift_task:+.3f} vs RAUSCHEN {drift_noise:+.3f} '
             f'=> {"Drift task-unabhaengig" if abs(drift_noise) > 0.3 * max(abs(drift_task), 1e-9) else "Drift task-getrieben"}')

    L.append('')
    L.append('LESEAT: P1+P2 => Optimierer frisst die Ressource (E3b). P3 entscheidet,')
    L.append('ob es Interferenz (task-unabhaengig) oder veraenderte Task-Geometrie ist.')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} geschrieben.')


if __name__ == '__main__':
    main()
