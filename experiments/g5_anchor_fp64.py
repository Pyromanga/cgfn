#!/usr/bin/env python
'''G5 — INSTRUMENT-REPARATUR UND ANKER-BESTAETIGUNG IN FLOAT64.
(registriert 2026-11-06, VOR dem Lauf)

VORGESCHICHTE (G4: UNENTSCHEIDEN, registriert und stehengelassen):
  G4s Instrument-Gate fiel durch — Delta(lokal vs. Bibliothek) bis
  0.29. Ursache am Code verifiziert: model.U ist ein nn.Linear MIT
  BIAS. Die Bibliothek rechnet U(u) = u@W.T + b, die lokale
  Version nur u@W.T. Der Bias ist ZUFAELLIG pro Konstruktion
  (globale RNG, U(-1,1)). Konsequenzen:
  (a) G2/G3 fuehrten pro Seed einen unkontrollierten zufaelligen
      Zusatzantrieb (Bias lottert mit) — ein Confound in den
      Serien G2 und G3.
  (b) G4s NC-Arm kollabierte trotzdem EXAKT in 3/4 Zellen
      (std 0.0000/0.0005/0.0000, 10/10 Treffer) gegen echtes
      N-Lotto (0.0245/0.2195/0.0109/0.0253) — die Anker-These
      hat innerhalb G4s bestanden, nur das Gate war blind.
  (c) Einzige Ausnahme H=128 s=0.8 (std_NC=0.031): Hypothese
      fp32-Rauschfuss des Benettin unter extremer transienter
      Vergroesserung. In float64 muss dieser Fuss verschwinden.

G5-FRAGEN (alle registriert VOR dem Lauf):
  (1) ANKER EXAKT: Kollabiert NC in float64 in ALLEN 4 Zellen
      auf Rundungsniveau? (Dann war Zelle 4 fp32-Fuss.)
  (2) BIAS-LOTTO: Wie gross ist der versteckte Bias-Lotto
      allein (nur U.bias variiert, W fix)?
  (3) G2-REPLAY BIAS-FREI: ueberlebt das "Basis-Schicksal"
      (nur-Q-Rotation) den Bias-Entzug, oder war es teilweise
      Antriebs-Artefakt?

INSTRUMENT (repariert): float64 global (set_default_dtype),
lokaler Benettin 1:1 wie cgfn/benettin, antriebsseitig OHNE
Bias. Instrument-Check: Bibliothek (Bias genullt) vs. lokal auf
Arm N, Seed 0 — |Delta| <= 1e-6, sonst UNENTSCHEIDEN.

ARME (Spektrum fix wo vermerkt, Q rotiert wo vermerkt,
Antrieb bei NC/N = Q[:,0], bias-frei):
  NC custom:  tanh mitfahrend (Q-System). Erwartung: exakt.
  N  custom:  tanh absolut. Erwartung: Lotto.
  BIAS:       W fix (Seed 0), U.weight fix (geseedet 4242), NUR
              U.bias variiert (geseedert, U(-1,1)), Bibliotheks-
              Benettin. Misst den versteckten Lotto direkt.
  A_clean:    G2-Arm A Replay (nur Q rotiert), Bibliotheks-
              Benettin, Bias genullt, U.weight FIX (4242).
  C_clean:    G2-Arm C Replay (Q+Spektrum+Rauschen rotieren),
              sonst wie A_clean.

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  Zellen: H in {32,128} x s in {0.0, 0.8}.
  PRIMAER (Anker, alle 4 Zellen, float64):
    (i)   std_NC <= 1e-4
    UND (ii)  std_N >= 0.005
    UND (iii) std_N >= 100 * std_NC
    UND (iv)  Instrument-Check |Delta| <= 1e-6.
    PASS -> "Anker-These exakt bestaetigt: Das Seed-Lotto des
      getriebenen Reservoirs ist ein Koordinaten-Artefakt der
      elementweisen Nichtlinearitaet; mitfahrende Nichtlin-
      earitaet kollabiert es auf Rundungsniveau. G4-Zelle 4
      war fp32-Transienten-Fuss."
    KILL: Sonst.
    UNENTSCHEIDEN: Instrument-Check faellt ODER >30% ungueltig.
  SEKUNDAER (Bericht mit registrierter Schwelle): Bias-Audit
    (std_BIAS) und G2-Replay in den s=0.8-Zellen: Falls
    std_A_clean <= 0.5 * std_C_clean in >= 2/2 Audit-Zellen ->
    "G2-KILL bleibt registriert, aber das Basis-Schicksal war
    teilweise Antriebs-/Bias-Artefakt". Sonst -> "Basis-
    Schicksal ueberlebt den Bias-Entzug". Beides nur Bericht,
    aendert kein eingefrorenes Verdikt.
  Kalibrierung: pro (Zelle, Arm) EINMAL lam aus 10-Punkt-Raster
  (0.2..2.0) auf Seed 0 (BIAS: lam=lam(N), keine Kalibrierung).
  Bewertung: 10 Seeds (0..9). Ziel ben=-0.30+-0.05 (nur Bericht).
'''
import argparse
import os
import sys
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f34_exploration import IS
from cgfn.benettin import benettin_spectrum
from cgfn.field import drive_input
from cgfn.model import ContinuousRNN

BEN_TARGET = -0.30
BEN_TOL = 0.05
KNOBS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
FIXED = 9999
UVEC_SEED = 4242


def factored_WQT(H, lam_max, s_nn, q_seed, spec_seed, noise_seed):
    genQ = torch.Generator().manual_seed(q_seed)
    A = torch.randn(H, H, generator=genQ)
    Q, _ = torch.linalg.qr(A)
    genS = torch.Generator().manual_seed(spec_seed)
    others = lam_max * 0.9 * (2.0 * torch.rand(H - 1, generator=genS) - 1.0)
    eigs = torch.cat([torch.tensor([lam_max]), others])
    T = torch.diag(eigs)
    if s_nn > 0:
        genN = torch.Generator().manual_seed(noise_seed)
        upper = torch.triu(torch.randn(H, H, generator=genN), diagonal=1)
        T = T + s_nn * lam_max * upper
    return Q @ T @ Q.T, Q, T


def fixed_uvec(H):
    gen = torch.Generator().manual_seed(UVEC_SEED)
    u = torch.randn(H, 1, generator=gen)
    return u / u.norm()


def benettin_custom(dyn, x_seq, H, dt=0.1, substeps=5, tau=1.0):
    B, T_, _ = x_seq.shape
    z = torch.zeros(B, H)
    dz = torch.eye(H).unsqueeze(0).expand(B, H, H).contiguous()
    log_sum = torch.zeros(H)
    t_total = 0.0
    for t in range(T_):
        x_t = x_seq[:, t]
        for _ in range(substeps):
            f1, g1 = dyn(z, dz, x_t)
            z2 = z + 0.5 * dt * f1
            dz2 = dz + 0.5 * dt * g1
            f2, g2 = dyn(z2, dz2, x_t)
            z = z + dt * f2
            dz = dz + dt * g2
            t_total += dt
            Q_, R_ = torch.linalg.qr(dz)
            diag_r = torch.diagonal(R_, dim1=-2, dim2=-1).abs()
            log_sum += torch.log(diag_r.clamp_min(1e-30)).mean(dim=0)
            dz = Q_
    spectrum = (log_sum / max(t_total, 1e-9)).sort(descending=True).values
    return float(spectrum[0])


def make_dyn(H, tau, W, Q, T, Uw, arm):
    def sech2(y):
        return 1.0 - torch.tanh(y).pow(2)

    if arm == 'N':
        def dyn(z, dz, x_t):
            f = -z / tau + torch.tanh(z) @ W.T + x_t @ Uw.T
            g = -dz / tau + torch.einsum('ij,bjk->bik', W,
                                         sech2(z).unsqueeze(-1) * dz)
            return f, g
        return dyn
    if arm == 'NC':
        def dyn(z, dz, x_t):
            y = z @ Q
            t = torch.tanh(y)
            f = -z / tau + (t @ T.T) @ Q.T + x_t @ Uw.T
            inner = sech2(y).unsqueeze(-1) * (dz @ Q)
            w = torch.einsum('ij,bjk->bik', T, inner)
            g = -dz / tau + w @ Q.T
            return f, g
        return dyn
    raise ValueError(arm)


def make_lib_model(H, s_nn, seed, arm, lam, bias_mode):
    '''G2-/Bias-Audit-Arme ueber die Bibliothek.'''
    if arm == 'A':
        W, Q, _ = factored_WQT(H, lam, s_nn, seed, FIXED, FIXED)
    elif arm == 'C':
        W, Q, _ = factored_WQT(H, lam, s_nn, seed, seed, seed)
    elif arm == 'BIAS':
        W, Q, _ = factored_WQT(H, lam, s_nn, 0, 0, 0)
    else:
        raise ValueError(arm)
    model = ContinuousRNN(input_dim=1, task_dim=0, hidden_dim=H, tau=1.0)
    with torch.no_grad():
        model.W.copy_(W)
        model.U.weight.copy_(fixed_uvec(H))
        if bias_mode == 'zero':
            model.U.bias.zero_()
        elif bias_mode == 'seeded':
            gen = torch.Generator().manual_seed(31000 + seed)
            model.U.bias.copy_(2.0 * torch.rand(H, generator=gen) - 1.0)
    model.W.requires_grad_(False)
    return model


def ben_lib(model, xdr):
    return float(benettin_spectrum(model, xdr)[0][0])


def stats(bens):
    ok = [b for b in bens if b == b and abs(b) < 1e6]
    st = float(torch.tensor(ok, dtype=torch.float64).std()) if len(ok) > 1 else float('nan')
    hits = sum(1 for b in ok if abs(b - BEN_TARGET) <= BEN_TOL)
    return st, hits, len(bens) - len(ok)


def main():
    p = argparse.ArgumentParser(description='G5 instrument repair + anchor fp64')
    p.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    p.add_argument('--out', default='results/g5_anchor_fp64_results.txt')
    p.add_argument('--csv', default='results/g5_anchor_fp64_table.csv')
    args = p.parse_args()
    seeds = [int(v) for v in args.seeds.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    xdr = drive_input(4, 48, IS)
    L = []
    L.append('=' * 118)
    L.append('G5 — INSTRUMENT-REPARATUR + ANKER IN FLOAT64 (registriert: PASS nur wenn in')
    L.append('ALLEN 4 Zellen std_NC<=1e-4 UND std_N>=0.005 UND std_N>=100*std_NC UND')
    L.append('Instrument |Delta|<=1e-6; Bias-Audit + G2-Replay nur sekundaer)')
    L.append('=' * 118)

    rows = []
    cell_stats = {}
    audit_stats = {}
    inst_ok = True
    for H in (32, 128):
        for s_nn in (0.0, 0.8):
            L.append('')
            L.append('########## H=' + str(H) + ' s=' + str(s_nn) + ' ##########')
            # Instrument-Check: Bibliothek (Bias genullt) vs. lokal, Arm N, Seed 0
            W, Q, T = factored_WQT(H, 1.0, s_nn, 0, FIXED, FIXED)
            Uw = Q[:, 0:1].contiguous()
            dyn_loc = make_dyn(H, 1.0, W, Q, T, Uw, 'N')
            mchk = ContinuousRNN(input_dim=1, task_dim=0, hidden_dim=H, tau=1.0)
            with torch.no_grad():
                mchk.W.copy_(W)
                mchk.U.weight.copy_(Uw)
                mchk.U.bias.zero_()
            b_lib = ben_lib(mchk, xdr)
            b_loc = benettin_custom(dyn_loc, xdr, H)
            d_inst = abs(b_lib - b_loc)
            L.append('Instrument-Check N(seed0, lam=1.0, bias=0): Bib=%.9f lokal=%.9f Delta=%.3g'
                     % (b_lib, b_loc, d_inst))
            if d_inst > 1e-6:
                inst_ok = False

            # Primaer-Arme: NC und N (custom, bias-frei)
            block = {}
            for arm in ('NC', 'N'):
                best_k, best_ben, best_gap = None, None, 1e9
                for k in KNOBS:
                    Wk, Qk, Tk = factored_WQT(H, k, s_nn, 0, FIXED, FIXED)
                    Uwk = Qk[:, 0:1].contiguous()
                    b = benettin_custom(make_dyn(H, 1.0, Wk, Qk, Tk, Uwk, arm), xdr, H)
                    gap = abs(b - BEN_TARGET)
                    if b == b and gap < best_gap:
                        best_k, best_ben, best_gap = k, b, gap
                bens = []
                for seed in seeds:
                    Ws, Qs, Ts = factored_WQT(H, best_k, s_nn, seed, FIXED, FIXED)
                    Uws = Qs[:, 0:1].contiguous()
                    bens.append(benettin_custom(
                        make_dyn(H, 1.0, Ws, Qs, Ts, Uws, arm), xdr, H))
                st, hits, bad = stats(bens)
                ok = [b for b in bens if b == b]
                block[arm] = dict(std=st, hits=hits, lam=best_k, bad=bad, n=len(bens),
                                  mean=float(torch.tensor(ok).mean()) if ok else float('nan'))
                L.append('Arm %s (lam=%s): std=%.6g Treffer=%d/%d mean=%+.4f ungueltig=%d'
                         % (arm, best_k, st, hits, len(bens), block[arm]['mean'], bad))
                rows.append(dict(H=H, s=s_nn, arm=arm, lam=best_k, std=st,
                                 hits=hits, n=len(bens), bad=bad))
            cell_stats[(H, s_nn)] = block

            # Sekundaer: Bias-Audit + G2-Replay nur in s=0.8-Zellen
            if s_nn > 0:
                aud = {}
                lam_ref = block['N']['lam']
                for arm in ('A', 'C'):
                    best_k, best_ben, best_gap = None, None, 1e9
                    for k in KNOBS:
                        b = ben_lib(make_lib_model(H, s_nn, 0, arm, k, 'zero'), xdr)
                        gap = abs(b - BEN_TARGET)
                        if b == b and gap < best_gap:
                            best_k, best_ben, best_gap = k, b, gap
                    bens = [ben_lib(make_lib_model(H, s_nn, sd, arm, best_k, 'zero'), xdr)
                            for sd in seeds]
                    st, hits, bad = stats(bens)
                    aud[arm + '_clean'] = dict(std=st, lam=best_k, hits=hits, bad=bad)
                    L.append('Audit %s_clean (lam=%s, bias=0): std=%.4f Treffer=%d/%d ungueltig=%d'
                             % (arm, best_k, st, hits, len(bens), bad))
                    rows.append(dict(H=H, s=s_nn, arm=arm + '_clean', lam=best_k, std=st,
                                     hits=hits, n=len(bens), bad=bad))
                bbens = [ben_lib(make_lib_model(H, s_nn, sd, 'BIAS', lam_ref, 'seeded'), xdr)
                         for sd in seeds]
                st, hits, bad = stats(bbens)
                aud['BIAS'] = dict(std=st, lam=lam_ref, hits=hits, bad=bad)
                L.append('Audit BIAS (lam=%s, nur U.bias lottert): std=%.4f ungueltig=%d'
                         % (lam_ref, st, bad))
                rows.append(dict(H=H, s=s_nn, arm='BIAS', lam=lam_ref, std=st,
                                 hits=hits, n=len(bbens), bad=bad))
                audit_stats[H] = aud
            Path(args.out).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(args.csv, 'w', encoding='utf-8') as fh:
        fh.write('H,s,arm,lam,std_ben1,hits,n,bad\n')
        for r in rows:
            fh.write('%d,%s,%s,%s,%.6g,%d,%d,%d\n'
                     % (r['H'], r['s'], r['arm'], r['lam'], r['std'],
                        r['hits'], r['n'], r['bad']))

    L.append('')
    L.append('--- G5-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    pass_all = True
    for (H, s_nn), b in sorted(cell_stats.items()):
        stdNC, stdN = b['NC']['std'], b['N']['std']
        c = (stdNC == stdNC and stdN == stdN and stdNC <= 1e-4
             and stdN >= 0.005 and stdN >= 100 * stdNC)
        L.append('Zelle H=%d s=%s: std_NC=%.3g std_N=%.4f (Faktor %.0fx) -> %s'
                 % (H, s_nn, stdNC, stdN,
                    (stdN / stdNC if stdNC > 0 else float('inf')),
                    'ERFUELLT' if c else 'NICHT ERFUELLT'))
        pass_all = pass_all and c

    fate_cells = 0
    for H, aud in sorted(audit_stats.items()):
        aC, cC = aud['A_clean']['std'], aud['C_clean']['std']
        ok = (aC == aC and cC == cC and aC <= 0.5 * cC)
        fate_cells += 1 if ok else 0
        L.append('SEKUNDAER H=%d: std_A_clean=%.4f std_C_clean=%.4f std_BIAS=%.4f -> %s'
                 % (H, aC, cC, aud['BIAS']['std'],
                    'BIAS-ARTEFAKT-VERDACHT (A<=0.5*C)' if ok else 'BASIS-SCHICKSAL UEBERLEBT'))
    if fate_cells >= 2:
        fate = ('G2-KILL bleibt registriert, aber das Basis-Schicksal war teilweise '
                'Antriebs-/Bias-Artefakt (bias-frei schrumpft es unter die Haelfte).')
    else:
        fate = ('Basis-Schicksal ueberlebt den Bias-Entzug — G2 bleibt auch '
                'sauber eine Geometrie-Aussage.')

    gate_bad = any(r['bad'] > 0.3 * r['n'] for r in rows)
    if not inst_ok:
        verdict = 'UNENTSCHEIDEN — Instrument-Check (float64, bias-gefeit) gefallen.'
    elif gate_bad:
        verdict = 'UNENTSCHEIDEN — zu viele ungueltige Punkte.'
    elif pass_all:
        verdict = ('PASS — Anker-These exakt bestaetigt in float64: Das Seed-Lotto '
                   'des getriebenen Reservoirs ist ein Koordinaten-Artefakt der '
                   'elementweisen Nichtlinearitaet. Mitfahrende Nichtlinearitaet '
                   'kollabiert es auf Rundungsniveau; G4-Zelle H=128 s=0.8 war '
                   'fp32-Transienten-Fuss. Konstruktions-Knopf: rotations-'
                   'aquivariante Nichtlinearitaet. ' + fate)
    else:
        verdict = ('KILL — Auch in float64 kollabiert NC nicht auf Rundungsniveau '
                   'in allen Zellen: Anker-These faellt. ' + fate)
    L.append('')
    L.append('VERDICT: ' + verdict)
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(args.out).write_text(out, encoding='utf-8')
    print(out)
    print('[i] Ergebnisse nach ' + args.out + ' und ' + args.csv + ' geschrieben.')


if __name__ == '__main__':
    main()
