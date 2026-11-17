#!/usr/bin/env python
'''B1 — BASIS-ABHAENGIGKEIT SPEKTRALER THEORIEN v1 (neue B-Serie,
registriert 2026-11-18 VOR dem Lauf, Nachtrag 58).
FRAGE: Praediziert eine basisnae Groesse Gedaechtnis-Lernen, wo
spektraler Radius (unter Konjugation EXAKT konstant) und DM es
nicht tun (T26: DM praediziert Ebenen, nicht Seeds)?
KANDIDATEN: K1 Kommutator-Norm ||[W,D]||_F / (||W||_F ||D||_F),
D = diag(Mittel sech^2 der getriebenen Trajektorie); K2 Participa-
tion Ratio der Singulaerwerte von J = -I + W D; K3 log10 Konditions-
zahl der Kovarianz erreichbarer Zustaende.
TEST: 5 Dosen a {0, 0.25, 0.5, 0.75, 1.0}, je 20 Seeds, T23-MEM
Produkt-Task, 600 Steps. Pro Seed: gelernt (acc >= 0.99) + Kandi-
daten aus dem UNTRAINIERTEN W0 + deterministischem Drive (777).
Auswertung: gepoolter Punkt-bisektionaler r je Kandidat vs r_DM.
KILL: kein Kandidat >= |r_DM| + 0.05. PASS: einer >= 0.30 UND
>= |r_DM| + 0.10. GEMISCHT: dazwischen. UNENTSCHEIDEN: Sanity.
Sanity: pf(a=0) >= 0.5, pf(a=1) <= 0.30.
SEKUNDAER: historische Stein-Bilanz (Lueckenliste ohne Schaetzung).
Bekannt-Neu-Trennung: siehe N58.'''
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import torch

torch.set_default_dtype(torch.float64)

import lab_kernel
from contracts.b1 import contract_b1

C = contract_b1()
P = C['params']
FIXED_P0 = P['fixed_p0']
TEST_SEED = P['test_seed']
T_SEQ = P['t_seq']
TAU = P['tau']
DT = P['dt']
SUBSTEPS = P['substeps']
STEPS = P['steps']
BATCH = P['batch']
LR = P['lr']
LAM_MAX = P['lam_max']
WB = P['wb']
Q_SEED = P['q_seed']
A_LO, A_HI = P['a_lo'], P['a_hi']
B_LO, B_HI = P['b_lo'], P['b_hi']
HOLD = P['hold']
DELAY = P['delay']
TEACHER0 = P['teacher_seed0']
T_HIDDEN = P['t_hidden']


def make_inputs(batch, gen):
    x = torch.zeros(batch, T_SEQ, 2)
    for i in range(batch):
        pa = int(torch.randint(A_LO, A_HI, (1,), generator=gen).item())
        pb = int(torch.randint(B_LO, B_HI, (1,), generator=gen).item())
        a = float(2.0 * torch.randint(0, 2, (1,), generator=gen).item() - 1)
        b = float(2.0 * torch.randint(0, 2, (1,), generator=gen).item() - 1)
        x[i, pa:pa + HOLD, 0] = a
        x[i, pb:pb + HOLD, 1] = b
    return x


def make_teacher(seed):
    tgen = torch.Generator().manual_seed(TEACHER0 + seed)
    A1 = torch.randn(T_HIDDEN, 2, generator=tgen)
    a2 = torch.randn(T_HIDDEN, generator=tgen)
    a2 = a2 / a2.norm()
    return A1, a2


def make_task(seed, batch, gen):
    x = make_inputs(batch, gen)
    y = torch.zeros(batch, dtype=torch.long)
    for i in range(batch):
        a = float(x[i, :, 0][x[i, :, 0] != 0][0])
        b = float(x[i, :, 1][x[i, :, 1] != 0][0])
        y[i] = 0 if (a * b > 0) else 1
    return x, y


def eigenbasis(H, seed, wb):
    gen = torch.Generator().manual_seed(seed)
    genP = torch.Generator().manual_seed(FIXED_P0)
    P0, _ = torch.linalg.qr(torch.randn(H, H, generator=genP))
    th = wb * torch.rand(H, generator=gen)
    M = torch.diag(torch.cos(th)) + torch.diag(torch.sin(th)) @ P0
    R, _ = torch.linalg.qr(M)
    return R


def init_W(H, seed):
    genS = torch.Generator().manual_seed(9999 + H)
    eigs = torch.cat([torch.tensor([LAM_MAX]),
                      0.9 * LAM_MAX * (2.0 * torch.rand(H - 1, generator=genS) - 1.0)])
    R = eigenbasis(H, seed, WB)
    return R @ torch.diag(eigs) @ R.T


def conjugate(W, a, H):
    if a == 0:
        return W
    genQ = torch.Generator().manual_seed(Q_SEED)
    G = torch.randn(H, H, generator=genQ)
    Q, _ = torch.linalg.qr(torch.eye(H) + a * G)
    return Q @ W @ Q.T


def make_Uw(H, gen):
    s = 2.2
    v0 = torch.randn(H, generator=gen)
    v1 = torch.randn(H, generator=gen)
    v0 = v0 / v0.norm() * s
    v1 = v1 / v1.norm() * s
    return torch.stack([v0, v1], dim=1)


def dyn(z, W, b, uin):
    return -z / TAU + torch.tanh(z + b) @ W.T + uin


def rollout_states(W, b, Uw, x):
    B = x.shape[0]
    H = W.shape[0]
    zs = torch.zeros(B, T_SEQ, H)
    z = torch.zeros(B, H)
    for t in range(T_SEQ):
        xt = x[:, t]
        for _s in range(SUBSTEPS):
            k1 = dyn(z, W, b, xt @ Uw.T)
            k2 = dyn(z + 0.5 * DT * k1, W, b, xt @ Uw.T)
            z = z + DT * k2
        zs[:, t, :] = z
    return zs


def compute_invariants(W0, b, Uw):
    # deterministischer Drive (Seed 777, 200 Schritte, gleiche
    # Input-Verteilung wie der Task) -> D aus sech^2-Mittel
    gen = torch.Generator().manual_seed(777)
    H = W0.shape[0]
    z = torch.zeros(H)
    Z = torch.zeros(200, H)
    dsum = torch.zeros(H)
    for t in range(200):
        u2 = torch.zeros(2)
        u2[0] = float(torch.randint(0, 2, (1,), generator=gen).item()) * 2 - 1
        u2[1] = float(torch.randint(0, 2, (1,), generator=gen).item()) * 2 - 1
        u = u2 @ Uw.T
        for _s in range(SUBSTEPS):
            k1 = dyn(z, W0, b, u)
            k2 = dyn(z + 0.5 * DT * k1, W0, b, u)
            z = z + DT * k2
        Z[t] = z
        dsum += 1.0 - torch.tanh(z + b) ** 2
    D = torch.diag(dsum / 200.0)
    k1 = float((W0 @ D - D @ W0).norm() / (W0.norm() * D.norm() + 1e-30))
    J = -torch.eye(H) / TAU + W0 @ D
    sv = torch.linalg.svdvals(J)
    k2 = float((sv.sum() ** 2) / ((sv ** 2).sum() + 1e-30))
    Zc = Z - Z.mean(dim=0, keepdim=True)
    cov = Zc.T @ Zc / (200 - 1)
    ev = torch.linalg.eigvalsh(cov)
    ev = torch.clamp(ev, min=1e-12)
    k3 = math.log10(float(ev.max() / ev.min()))
    dm = float(W0.diag().abs().mean() / W0.abs().mean())
    spec = float(torch.linalg.eigvals(W0).abs().max())
    return dict(k1=k1, k2=k2, k3=k3, dm=dm, spec=spec)


def train_one(seed, arm, H):
    a = float(arm.get('conj_a', 0.0))
    gen = torch.Generator().manual_seed(seed)
    W0 = conjugate(init_W(H, seed), a, H)
    Uw = make_Uw(H, gen)
    colnorm = float(Uw.norm(dim=0).mean())
    b = 0.1 * torch.randn(H, generator=gen)
    inv = compute_invariants(W0, b, Uw)
    W = W0.clone().requires_grad_(True)
    head_w = torch.zeros(2, H, requires_grad=True)
    head_b = torch.zeros(2, requires_grad=True)
    opt = torch.optim.Adam([W, head_w, head_b], lr=LR)
    train_gen = torch.Generator().manual_seed(1000 + seed)
    for _ in range(STEPS):
        x, y = make_task(seed, BATCH, train_gen)
        zs = rollout_states(W, b, Uw, x)
        z = zs[:, -1, :]
        logits = z @ head_w.T + head_b
        loss = torch.nn.functional.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            return None, colnorm, inv
        opt.zero_grad()
        loss.backward()
        opt.step()
    test_gen = torch.Generator().manual_seed(TEST_SEED)
    xT, yT = make_task(seed, 1024, test_gen)
    with torch.no_grad():
        zs = rollout_states(W, b, Uw, xT)
        z = zs[:, -1, :]
        pred = (z @ head_w.T + head_b).argmax(dim=1)
        acc = (pred == yT).double().mean().item()
    return acc, colnorm, inv


def fmt(v):
    return '%.4f' % v if v == v else 'nan'


def main():
    out_path = 'results/b1_basisinvariante_results.txt'
    ms_path = 'results/b1_measurements.json'
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    L = []
    L.append('=' * 118)
    L.append('B1 — BASIS-ABHAENGIGKEIT SPEKTRALER THEORIEN (neue B-Serie; 3 Kandidaten gegen Spektrum + DM)')
    L.append('FRAGE: Praediziert eine basisnae Groesse Gedaechtnis-Lernen, wo spektraler Radius (EXAKT konstant')
    L.append('unter Konjugation) und DM es nicht tun? K1 Kommutator-Norm | K2 Jacobian-Effektivrang | K3 Zustands-Kond.')
    L.append('5 Dosen a {0,0.25,0.5,0.75,1.0}, je 20 Seeds, T23-MEM, 600 Steps. Bloecke 1510..1709.')
    L.append('Verdikt generiert | Contract: contracts/b1.py')
    L.append('=' * 118)

    errs = lab_kernel.preflight_gate(C)
    if errs:
        L.append('K-GATE VERWEIGERT VOR DEM LAUF — keine Runner-Minuten verbrannt:')
        for e in errs:
            L.append('  ALARM: %s' % e)
        out = chr(10).join(L) + chr(10)
        Path(out_path).write_text(out, encoding='utf-8')
        print(out)
        sys.exit(1)
    L.append('K-PRE-FLIGHT: OK — Contract strukturell feuerbar.')
    Path(out_path).write_text(chr(10).join(L) + chr(10), encoding='utf-8')

    per_seed = []
    arms_out = {}
    for arm in C['arms']:
        name, H = arm['name'], arm['H']
        seeds = arm['seeds']
        pvals = []
        ndiv = 0
        for seed in seeds:
            acc, cn, inv = train_one(seed, arm, H)
            if acc is None or acc != acc:
                ndiv += 1
                acc = float('nan')
            pvals.append(acc)
            per_seed.append(dict(arm=name, dose=float(arm.get('conj_a', 0.0)),
                                 seed=seed, acc=acc, **inv))
        ok = [v for v in pvals if v == v]
        pf = sum(1 for v in ok if v >= 0.99) / len(ok) if ok else float('nan')
        arms_out[name] = dict(pf=pf, ndiv=ndiv,
                              dose=float(arm.get('conj_a', 0.0)))
        L.append('')
        L.append('Arm %s (a=%s, seeds %d..%d): n_ok=%d/%d | P_lernt=%.2f | median acc=%s | div=%d'
                 % (name, arm.get('conj_a'), seeds[0], seeds[-1], len(ok),
                    len(pvals), pf,
                    fmt(float(torch.tensor(ok).median())) if ok else 'nan',
                    ndiv))
        Path(out_path).write_text(chr(10).join(L) + chr(10), encoding='utf-8')

    used = [r for r in per_seed if r['acc'] == r['acc'] and r['dm'] == r['dm']]
    ys = torch.tensor([1.0 if r['acc'] >= 0.99 else 0.0 for r in used])
    rs = {}
    if len(used) > 2 and ys.std() > 0:
        for key in ['k1', 'k2', 'k3', 'dm', 'spec']:
            xs = torch.tensor([r[key] for r in used])
            if xs.std() == 0:
                rs[key] = float('nan')
                continue
            xs = (xs - xs.mean()) / (xs.std() + 1e-30)
            yy = (ys - ys.mean()) / (ys.std() + 1e-30)
            rs[key] = float((xs * yy).mean())

    pf_a0 = arms_out.get('DOSE_A0', {}).get('pf', float('nan'))
    pf_a1 = arms_out.get('DOSE_A10', {}).get('pf', float('nan'))
    sanity_ok = (pf_a0 >= 0.5 and pf_a1 <= 0.30)
    best_key = max([k for k in ['k1', 'k2', 'k3'] if k in rs and rs[k] == rs[k]],
                   key=lambda k: abs(rs[k]), default=None)
    rdm = rs.get('dm', float('nan'))
    if not sanity_ok or best_key is None:
        verdict_txt = 'UNENTSCHEIDEN — SANITY/DATEN GERICHT'
    elif abs(rs[best_key]) >= 0.30 and abs(rs[best_key]) >= abs(rdm) + 0.10:
        verdict_txt = 'PASS — BASISINVARIANTE-TRAEGER (%s)' % best_key
    elif abs(rs[best_key]) >= abs(rdm) + 0.05:
        verdict_txt = 'GEMISCHT — %s ueber DM, unter der PASS-Schwelle' % best_key
    else:
        verdict_txt = 'KILL — KEINE BASISINVARIANTE GROESSE'
    L.append('')
    L.append('--- B1-AUSWERTUNG (generiert, nicht handgeschrieben) ---')
    L.append('P-Stufen: pf(a=0)=%.2f | pf(a=0.5)=%.2f | pf(a=1)=%.2f'
             % (pf_a0, arms_out.get('DOSE_A05', {}).get('pf', float('nan')), pf_a1))
    L.append('SANITY: pf(a=0) >= 0.5: %s | pf(a=1) <= 0.30: %s'
             % (pf_a0 >= 0.5, pf_a1 <= 0.30))
    L.append('Gepoolte Korrelationen (frische Bloecke, n=%d):' % len(used))
    for key in ['k1', 'k2', 'k3', 'dm', 'spec']:
        L.append('  r_%s = %s' % (key, fmt(rs.get(key, float('nan')))))
    L.append('Spektralradius unter Konjugation konstant (0.6): kann die Dosis-Antwort prinzipiell nicht tragen.')
    L.append('')
    L.append('VERDICT: %s' % verdict_txt)
    L.append('=== FERTIG ===')
    out = chr(10).join(L) + chr(10)
    Path(out_path).write_text(out, encoding='utf-8')
    Path(ms_path).write_text(json.dumps(dict(
        per_seed=per_seed, arms=arms_out, correlations=rs,
        verdict=verdict_txt), indent=1), encoding='utf-8')
    print(out)
    print('[i] Laufzeit: %.1f min' % ((time.time() - t0) / 60.0))


if __name__ == '__main__':
    main()
