'''
F33 — FTLE-SPEKTRUM ALS BETRIEBSZUSTAND (registrierter Neustart, XIV).

Vorgeschichte: F30.4 KILL — MC als Task-Praediktor fiel beim Transfer
(H=256 + NARMA-10). Forensik: F30.3s +0.748 hing am Chaos/Tod-Split;
innerhalb des stabilen Regimes war die MC-Task-Relation NEGATIV.
Robust ueberlebte nur: Stabilitaet (ben < 0) praediziert Task.

Literatur-Konvergenz (Tiefenanalyse XIV): Carroll (arXiv:2012.01409)
— Instabilitaet kommt von positiven LOKALEN Exponenten bei negativem
globalen; "Beyond the Edge of Chaos" (arXiv:2607.17909) — das Ziel
wird von einem breiten Band Modi getragen, stabil im Mittel, stark
fluktuierend; EPJ B s10051-026-01204-4 — FTLE als Werkzeug. Unsere
eigene F18c ("Regime B: lokal expansiv, asymptotisch stabil") war
dieses Fenster schon.

FRAGE: Ist die FORM des FTLE-Spektrums der Betriebszustand, der
Aufgabenleistung vorhersagt — staerker als der Skalar ben_max?

INSTRUMENT: benettin_spectrum (F11-Validierung) liefert das volle
Spektrum; benettin_ftle_max (der F30.4-Messpfad) warf es auf
spectrum[0] away. F33 liest die Form aus.

DESIGN (stability-gematcht, der Konfund ist KONTROLLIERT):
  - Zellen: (H=32, sinus), (H=32, NARMA-10), (H=256, sinus),
    (H=256, NARMA-10).
  - H=32: lam-Bisektion auf ben-Ziel -0.30 (F24-Muster), 12 Iter.
  - H=256: lam-Sweep {0.3,0.5,0.7,0.9} + Selektion auf Band
    ben1 in [-0.45,-0.15] (Benettin bei H=256 ist teuer; Selektion
    statt Bisektion, registriert als Design).
  - Grid je Zelle: s in {0.0,0.8,1.6} x seeds {0,1,2,3}.

REGISTRIERTE MERKMALE (vor dem Lauf festgenagelt, kein Post-hoc-Zoo):
  PRIMAER:  top25  = Mittel der obersten 25% der Exponenten
            (Carroll/2607.17909: die schnelle Bande).
  Deskriptiv (geloggt, zaehlen NICHT fuer das Verdikt):
            mean_spec, std_spec, frac_edge = Anteil der
            Exponenten > -0.05 (lokal expansive Modi).
  Kontroll-Skalare: ben1 (skalarer Stabilitaet), MC.

VORAB FESTGENAGELT (keine Nachverhandlung):
  Zelle gueltig: n_band >= 15 UND TASK_R2-std > 0.01 (Deckel-Check).
  Zellen-Treffer: corr(top25, TASK_R2) >= +0.4 UND
                  corr(top25, TASK_R2) > corr(ben1, TASK_R2).
  VERDICT: >= 3 gueltige Zellen: PASS falls >= 3 Treffer,
           sonst KILL (FTLE-Form traegt nicht ueber Stabilitaet hinaus).
           < 3 gueltige Zellen: UNENTSCHEIDEN (Design, nicht Schwelle).
  Konsequenz KILL: Betriebspunkt-These wird Stabilitaets-Trivialitaet;
           Projekt ggf. auf reine Antagonisten-Phaenomenologie (F27)
           zurueckgestuft.
Freeze-Regel: ein Ergebnis pro Stein. Keine Nachrutsch-Laefue.
'''
import argparse
import math
import os
from pathlib import Path

import torch

from cgfn.benettin import benettin_spectrum
from cgfn.design import make_designed_model
from cgfn.field import drive_input
from cgfn.memory_capacity import memory_capacity

DELAY = 3
BEN_TARGET = -0.30
BAND = (-0.45, -0.15)
FEDGE = -0.05


@torch.no_grad()
def sinus_readout_r2(m, seed, n_train=64, n_test=32, seq_len=48, ridge=1e-2):
    '''Identisch zu F30.3/F30.4 (sinus_memory, delay 3, geschlossener Ridge).'''
    def collect(n, sd):
        gen = torch.Generator().manual_seed(sd)
        feats, targs = [], []
        for _ in range(n):
            t = torch.arange(seq_len, dtype=torch.float32)
            freq = 0.25 + 1.75 * torch.rand(1, 1, generator=gen)
            phase = 2 * math.pi * torch.rand(1, 1, generator=gen)
            x = torch.sin(freq * t.unsqueeze(0) + phase)
            y = x.clone()
            y[:, DELAY:] = x[:, :-DELAY]
            _, states = m(x.unsqueeze(-1), None)
            feats.append(states[0, DELAY:])
            targs.append(y[0, DELAY:])
        return torch.cat(feats, 0), torch.cat(targs, 0)

    Ftr, Ytr = collect(n_train, seed + 555)
    Fte, Yte = collect(n_test, seed + 777)
    A = Ftr.T @ Ftr + ridge * torch.eye(Ftr.shape[1])
    W = torch.linalg.solve(A, Ftr.T @ Ytr.unsqueeze(1))
    pred = (Fte @ W).squeeze(1)
    denom = ((Yte - Yte.mean()) ** 2).mean().clamp_min(1e-9)
    return (1.0 - ((Yte - pred) ** 2).mean() / denom).item()


@torch.no_grad()
def narma10_readout_r2(m, seed, n_train=50, n_test=25, seq_len=160,
                       washout=40, ridge=1e-2):
    '''NARMA-10, identisch zu F30.4 Bedingung B.'''
    def collect(n, sd):
        gen = torch.Generator().manual_seed(sd)
        feats, targs = [], []
        for _ in range(n):
            u = torch.rand(seq_len, generator=gen) * 0.5
            y = torch.zeros(seq_len)
            yh = torch.zeros(seq_len + 10)
            for t in range(seq_len - 1):
                prod = 1.0
                for i in range(10):
                    prod = prod * yh[10 + t - i]
                un = u[t - 9] if t >= 9 else u[0]
                y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * prod + 1.5 * un * u[t] + 0.1
                yh[10 + t + 1] = y[t + 1]
            _, states = m(u.unsqueeze(0).unsqueeze(-1), None)
            feats.append(states[0, washout:seq_len - 1])
            targs.append(y[washout + 1:])
        return torch.cat(feats, 0), torch.cat(targs, 0)

    Ftr, Ytr = collect(n_train, seed + 555)
    Fte, Yte = collect(n_test, seed + 777)
    A = Ftr.T @ Ftr + ridge * torch.eye(Ftr.shape[1])
    W = torch.linalg.solve(A, Ftr.T @ Ytr.unsqueeze(1))
    pred = (Fte @ W).squeeze(1)
    denom = ((Yte - Yte.mean()) ** 2).mean().clamp_min(1e-9)
    return (1.0 - ((Yte - pred) ** 2).mean() / denom).item()


def ftle_features(spec):
    '''Registrierte Merkmale aus dem vollen Benettin-Spektrum.'''
    sp = spec.sort(descending=True).values
    k = max(1, sp.numel() // 4)
    return dict(
        top25=float(sp[:k].mean()),
        mean_spec=float(sp.mean()),
        std_spec=float(sp.std()),
        frac_edge=float((sp > FEDGE).float().mean()),
        ben1=float(sp[0]),
    )


def tune_lam(hidden, s, seed, target, xdr, lo=0.1, hi=2.5, iters=12):
    '''lam-Bisektion auf ben-Ziel (F24-Muster).'''
    lam = 0.5 * (lo + hi)
    for _ in range(iters):
        lam = 0.5 * (lo + hi)
        torch.manual_seed(seed)
        m = make_designed_model(hidden, 1.0, lam, s, seed, task_dim=0)
        spec, _ = benettin_spectrum(m, xdr)
        ben = float(spec[0])
        if ben < target:
            lo = lam
        else:
            hi = lam
    return lam, m, spec


def corr2(a, b):
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float('nan')
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def eval_cell(tag, rows, L):
    '''Aggregat + registrierte Zellen-Entscheidung.'''
    n = len(rows)
    tasks = torch.tensor([r['task'] for r in rows], dtype=torch.float64)
    deckel = tasks.std().item() > 0.01 if n else False
    valid = n >= 15 and deckel
    res = dict(tag=tag, n=n, deckel=deckel, valid=valid)
    L.append(f'--- Zelle {tag}: n={n}, deckelfrei={"JA" if deckel else "NEIN"}, '
             f'gueltig={"JA" if valid else "NEIN (n>=15 und Deckel noetig)"}')
    if n < 8:
        L.append(f'Zelle {tag}: zu wenige Konfigs fuer Korrelationen.')
        res['hit'] = False
        res['c_top'] = float('nan')
        return res
    c_top = corr2([r['f']['top25'] for r in rows], [r['task'] for r in rows])
    c_ben = corr2([r['f']['ben1'] for r in rows], [r['task'] for r in rows])
    res['c_top'] = c_top
    res['c_ben'] = c_ben
    L.append(f'Zelle {tag}: PRIMAER corr(top25, TASK_R2) = {c_top:+.3f}   (>= +0.4 noetig)')
    L.append(f'Zelle {tag}: Kontrolle corr(ben1, TASK_R2)  = {c_ben:+.3f}   (top25 muss schlagen)')
    for key in ('mean_spec', 'std_spec', 'frac_edge'):
        c = corr2([r['f'][key] for r in rows], [r['task'] for r in rows])
        L.append(f'Zelle {tag}: deskriptiv corr({key}, TASK_R2) = {c:+.3f}')
    c_mc = corr2([r['mc'] for r in rows], [r['task'] for r in rows])
    L.append(f'Zelle {tag}: Kontrolle corr(MC, TASK_R2)     = {c_mc:+.3f}')
    res['hit'] = (valid and c_top == c_top and c_top >= 0.4 and c_top > c_ben)
    L.append(f'Zelle {tag}: Treffer = {"JA" if res["hit"] else "NEIN"}')
    return res


def main():
    p = argparse.ArgumentParser(description='F33 FTLE spectrum operating state')
    p.add_argument('--seeds', default='0,1,2,3')
    p.add_argument('--s-values', default='0.0,0.8,1.6')
    p.add_argument('--out', default='results/f33_ftle_spectrum_results.txt')
    args = p.parse_args()
    seeds = [int(v) for v in args.seeds.split(',')]
    s_vals = [float(v) for v in args.s_values.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    xdr = drive_input(4, 48, 0.5)
    L = []
    L.append('=' * 118)
    L.append('F33 FTLE-SPEKTRUM ALS BETRIEBSZUSTAND (Tiefenanalyse XIV, registrierter Neustart)')
    L.append(f'PRIMAER: top25 (obere 25% der Exponenten) | Kontrolle: ben1, MC | Freeze: keine Nachverhandlung')
    L.append(f'H=32: Bisektion auf ben={BEN_TARGET} | H=256: Sweep + Band-Selektion {BAND}')
    L.append(f'Verdikt: >=3 gueltige Zellen -> PASS falls >=3 Treffer (corr >= +0.4 UND > ben1), sonst KILL')
    L.append('=' * 118)

    results = []
    tasks = {'sinus': sinus_readout_r2, 'narma': narma10_readout_r2}

    for task_name, task_fn in tasks.items():
        # --- Zelle H=32 (Bisektion) ---
        rows = []
        L.append('')
        L.append(f'########## Zelle H=32 {task_name} (Bisektion ben={BEN_TARGET}) ##########')
        L.append(f"{'s':>4s} {'seed':>4s} {'lam*':>6s} {'ben1':>7s} {'top25':>7s} {'mean':>7s} {'std':>6s} {'fedge':>6s} {'MC':>5s} {'TASK_R2':>8s}")
        for s_nn in s_vals:
            for seed in seeds:
                lam, m, spec = tune_lam(32, s_nn, seed, BEN_TARGET, xdr)
                f = ftle_features(spec)
                torch.manual_seed(seed + 9000)
                mc, _ = memory_capacity(m, max_lag=16, seq_len=64, input_scale=0.5)
                t2 = task_fn(m, seed)
                L.append(f"{s_nn:4.1f} {seed:4d} {lam:6.3f} {f['ben1']:+7.3f} {f['top25']:+7.3f} "
                         f"{f['mean_spec']:+7.3f} {f['std_spec']:6.3f} {f['frac_edge']:6.3f} {mc:5.2f} {t2:+8.3f}")
                rows.append(dict(f=f, mc=mc, task=t2, lam=lam, s=s_nn, seed=seed))
        results.append(eval_cell(f'H=32 {task_name}', rows, L))
        Path(args.out).write_text('\n'.join(L) + '\n', encoding='utf-8')

        # --- Zelle H=256 (Sweep + Band-Selektion) ---
        rows = []
        L.append('')
        L.append(f'########## Zelle H=256 {task_name} (Sweep, Band {BAND}) ##########')
        L.append(f"{'lam':>5s} {'s':>4s} {'seed':>4s} {'ben1':>7s} {'top25':>7s} {'mean':>7s} {'std':>6s} {'fedge':>6s} {'MC':>5s} {'TASK_R2':>8s}")
        for lam in (0.3, 0.5, 0.7, 0.9):
            for s_nn in s_vals:
                for seed in seeds:
                    m = make_designed_model(256, 1.0, lam, s_nn, seed, task_dim=0)
                    spec, _ = benettin_spectrum(m, xdr)
                    f = ftle_features(spec)
                    if not (BAND[0] <= f['ben1'] <= BAND[1]):
                        continue
                    torch.manual_seed(seed + 9000)
                    mc, _ = memory_capacity(m, max_lag=16, seq_len=64, input_scale=0.5)
                    t2 = task_fn(m, seed)
                    L.append(f"{lam:5.2f} {s_nn:4.1f} {seed:4d} {f['ben1']:+7.3f} {f['top25']:+7.3f} "
                             f"{f['mean_spec']:+7.3f} {f['std_spec']:6.3f} {f['frac_edge']:6.3f} {mc:5.2f} {t2:+8.3f}")
                    rows.append(dict(f=f, mc=mc, task=t2, lam=lam, s=s_nn, seed=seed))
        results.append(eval_cell(f'H=256 {task_name}', rows, L))
        Path(args.out).write_text('\n'.join(L) + '\n', encoding='utf-8')

    valid = [r for r in results if r['valid']]
    hits = [r for r in valid if r['hit']]
    L.append('')
    L.append('--- F33-VERDICT (vorab genagelt, keine Nachverhandlung) ---')
    for r in results:
        c = r.get('c_top')
        cstr = f"{c:+.3f}" if c == c else '  n/a'
        L.append(f"Zelle {r['tag']}: n={r['n']}, gueltig={'JA' if r['valid'] else 'NEIN'}, "
                 f"corr(top25)={cstr}, Treffer={'JA' if r.get('hit') else 'NEIN'}")
    if len(valid) < 3:
        verdict = ('UNENTSCHEIDEN — weniger als 3 gueltige Zellen (Design-Problem: '
                   'Band zu eng / Deckel). Schwelle bleibt, Anpassung nur am Design.')
    elif len(hits) >= 3:
        verdict = ('PASS — FTLE-Spektrum-Form (top25) praediziert Aufgabenleistung '
                   'ueber Stabilitaet hinaus, in >= 3 von 4 Transfer-Zellen. '
                   'Betriebszustand = Spektrum-Form. Basis fuer Homeostat/Telemetrie '
                   'erneut offen (naechster Stein: Regler auf top25 statt Skalar).')
    else:
        verdict = ('KILL — top25 traegt in < 3 gueltigen Zellen. Betriebspunkt-These '
                   'wird Stabilitaets-Trivialitaet; Rueckstufung des Projekts auf '
                   'Antagonisten-Phaenomenologie (F27-Linie, deskriptiv).')
    L.append('')
    L.append(f'VERDICT: {verdict}')
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print(f'[i] Ergebnisse nach {args.out} geschrieben.')


if __name__ == '__main__':
    main()
