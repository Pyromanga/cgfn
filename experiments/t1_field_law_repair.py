#!/usr/bin/env python
'''T1 — FELDGESETZ-REPARATUR: IST DAS SEED-LOTTO BERECHENBAR?
Das cgfn-Feldgesetz (F17/F18) bekommt den Rahmen.
(registriert 2026-11-06, VOR dem Lauf)

TIEFGANG-PROGRAMM (nach G1-G10: wir haben Aggregate gemessen,
nie die Maschine geoeffnet): Das Feldgesetz sagt
    ben1 = max Re eig( -I/tau + W diag(d) ),  d = <sech^2(z)>
mit selbstkonsistenter Gain-Iteration (cgfn/field.py). Das ist
die vor-G5-Theorie — und sie ignoriert den Nichtlinearitaets-
Rahmen. Die WAHRE Linearisierung des getriebenen Systems ist
    J(t) = -I/tau + W F diag(sech^2(F^T z(t))) F^T,
Zeitmittel:  J = -I/tau + W F diag(d_y) F^T mit
    d_y = <sech^2(y_i)>,  y = F^T z (Rahmen-Koordinaten).
Repariertes Gesetz: gleiche Iteration, aber mit Rahmen-
Sandwich. T1-FRAGE: Trifft das reparierte Gesetz den gemessenen
Benettin-Exponenten — und reproduziert es sogar den SEED-LOTTO
(Spread UND Rangfolge ueber Seeds) ohne eine Benettin-Rechnung?

DAS WAERE DER TIEFGANG: Aus G2s "irreduzibles Schicksal" wird
ein berechenbares Schicksal — Vorhersage aus einer 5-Zeilen-
Eigenwertrechnung statt einer Tangenten-Integration.

ARME:
  N  (F=I): Feld-Standard — hier war das alte Gesetz zu Hause,
     hier lebt der Seed-Lotto (G2/G5). Altes und neues Gesetz
     stimmen hier UEBEREIN (F=I) — Kontrollarm fuer beide.
  NC (F=Q): mitfahrend (G5, exakt seed-invariant) — altes
     Gesetz falsch (ignoriert Q), neues sollte sitzen.
  (Das ist der scharfe Kontrast: bei NC ist alt!=neu per
  Konstruktion, bei N ist alt==neu. Der Test misst, ob das
  Sandwich die NC-Exponenten trifft, ohne bei N schlechter
  zu werden.)

PUNKTE: H in {32,128} x s in {0.0,0.8} x lam in
  {0.6, 1.0, 1.4} x Seeds 0..9 x Arm {N, NC}.
  Gemessen: ben1 via registriertes Benettin (G5-Instrument,
  korrekte Tangenten beider Rahmen, float64).
  Vorhergesagt: ben1_alt (F=I-Gesetz) und ben1_neu
  (Rahmen-Sandwich), beide mit derselben Gain-Iteration
  (d aus der getriebenen Trajektorie, 1 Durchlauf reicht —
  registriert: KEINE Iteration, Einmal-Trajektorie, d aus
  derselben benutzten Drive-Sequenz wie beim Benettin).

REGISTRIERUNG (GENAGELT, KEINE NACHVERHANDLUNG):
  PRIMAER, PASS genau dann wenn:
    (P1) Arm NC: median |ben1_neu - ben1_gemessen| <= 0.05
         ueber alle (H, s, lam, seed) — das Sandwich trifft
         den gerahmten Fall.
    UND (P2) Arm N: Spearman(ben1_neu, ben1_gemessen) ueber
         alle Punkte einer Zelle >= 0.8 in >= 3/4 Zellen —
         das Gesetz reproduziert die Rangfolge des Lottos.
    UND (P3) Arm NC: median |ben1_alt - ben1_gemessen| >=
         2 * median |ben1_neu - ben1_gemessen| — die Reparatur
         ist noetig, nicht kosmetisch.
  PASS -> "Das Seed-Lotto ist berechenbar: Ein gerahmtes
     Feldgesetz trifft den getriebenen Exponenten (und seinen
     Seed-Lotto) aus Eigenwerten plus einer Gain-Messung. Das
     Feld hat seit F17 am falschen Jacobianer gerechnet."
  KILL: Sonst — der Lotto bleibt eine Trajektorien-Eigenschaft,
     die sich nicht auf eigenwert-Niveau komprimieren laesst.
  UNENTSCHEIDEN: >30% ungueltige Punkte auf einer Seite.
  SEKUNDAER (Bericht): mittlerer Absolutfehler alt vs neu je
    Zelle; Streuung der Vorhersage ueber Seeds vs gemessene
    Streuung (Lotto-Amplitude getroffen?).
'''
import math
import os
import sys
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f34_exploration import IS
from cgfn.field import drive_input

FIXED = 9999
LAMS = [0.6, 1.0, 1.4]
DT = 0.1
SUBSTEPS = 5


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


def make_dyn(H, W, Q, T, Uw, arm):
    def sech2(y):
        return 1.0 - torch.tanh(y).pow(2)

    if arm == 'N':
        def dyn(z, dz, x_t):
            f = -z + torch.tanh(z) @ W.T + x_t @ Uw.T
            g = -dz + torch.einsum('ij,bjk->bik', W,
                                   sech2(z).unsqueeze(-1) * dz)
            return f, g
        return dyn
    if arm == 'NC':
        def dyn(z, dz, x_t):
            y = z @ Q
            t = torch.tanh(y)
            f = -z + (t @ T.T) @ Q.T + x_t @ Uw.T
            inner = sech2(y).unsqueeze(-1) * (dz @ Q)
            w = torch.einsum('ij,bjk->bik', T, inner)
            g = -dz + w @ Q.T
            return f, g
        return dyn
    raise ValueError(arm)


def benettin_custom(dyn, x_seq, H):
    B, Tn, _ = x_seq.shape
    z = torch.zeros(B, H)
    dz = torch.eye(H).unsqueeze(0).expand(B, H, H).contiguous()
    log_sum = torch.zeros(H)
    t_total = 0.0
    for t in range(Tn):
        x_t = x_seq[:, t]
        for _ in range(SUBSTEPS):
            f1, g1 = dyn(z, dz, x_t)
            z2 = z + 0.5 * DT * f1
            dz2 = dz + 0.5 * DT * g1
            f2, g2 = dyn(z2, dz2, x_t)
            z = z + DT * f2
            dz = dz + DT * g2
            t_total += DT
            Q_, R_ = torch.linalg.qr(dz)
            diag_r = torch.diagonal(R_, dim1=-2, dim2=-1).abs()
            log_sum += torch.log(diag_r.clamp_min(1e-30)).mean(dim=0)
            dz = Q_
    spectrum = (log_sum / max(t_total, 1e-9)).sort(descending=True).values
    return float(spectrum[0]), z


def rollout_states(H, dyn, x_seq):
    B, Tn, _ = x_seq.shape
    z = torch.zeros(B, H)
    eye = torch.eye(H).unsqueeze(0).expand(B, H, H)
    states = []
    for t in range(Tn):
        x_t = x_seq[:, t]
        for _ in range(SUBSTEPS):
            f1, _ = dyn(z, eye, x_t)
            f2, _ = dyn(z + 0.5 * DT * f1, eye, x_t)
            z = z + DT * f2
        states.append(z)
    return torch.stack(states, dim=1)


def predict_ben1(H, W, Q, T, Uw, arm, x_seq):
    '''Feldgesetz-Vorhersage aus Einmal-Trajektorie.'''
    dyn = make_dyn(H, W, Q, T, Uw, arm)
    states = rollout_states(H, dyn, x_seq)      # (B, Tn, H)
    F = Q if arm == 'NC' else torch.eye(H)
    y = states @ F                              # Rahmen-Koordinaten
    d = (1.0 - torch.tanh(y).pow(2)).mean(dim=(0, 1))   # (H,)
    I = torch.eye(H)
    J_neu = -I + W @ (F * d.unsqueeze(0)) @ F.T
    J_alt = -I + W @ torch.diag(d)              # altes Gesetz: F=I immer
    ben_neu = torch.linalg.eigvals(J_neu).real.max().item()
    ben_alt = torch.linalg.eigvals(J_alt).real.max().item()
    return ben_neu, ben_alt


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(list(xs)), ranks(list(ys))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float('nan')


def main():
    out_path = 'results/t1_field_law_repair_results.txt'
    csv_path = 'results/t1_field_law_repair_table.csv'
    os.makedirs('results', exist_ok=True)
    seeds = list(range(10))
    x_seq = drive_input(4, 48, IS)

    L = []
    L.append('=' * 118)
    L.append('T1 — FELDGESETZ-REPARATUR (registriert: PASS nur wenn P1 NC-Fehler<=0.05 UND')
    L.append('P2 Spearman(N)>=0.8 in >=3/4 Zellen UND P3 alt>=2xneu auf NC; f64, 10 Seeds)')
    L.append('Vorhersage: J = -I + W F diag(<sech^2(y)>) F^T (neu) vs W diag(d) alt (F17/F18)')
    L.append('=' * 118)

    rows = []
    cell_stats = {}
    for H in (32, 128):
        for s_nn in (0.0, 0.8):
            L.append('')
            L.append('########## H=' + str(H) + ' s=' + str(s_nn) + ' ##########')
            block = {}
            for arm in ('N', 'NC'):
                gem, pred_neu, pred_alt = [], [], []
                for lam in LAMS:
                    for seed in seeds:
                        W, Q, T = factored_WQT(H, lam, s_nn, seed, FIXED, FIXED)
                        Uw = Q[:, 0:1].contiguous()
                        dyn = make_dyn(H, W, Q, T, Uw, arm)
                        ben, _ = benettin_custom(dyn, x_seq, H)
                        b_neu, b_alt = predict_ben1(H, W, Q, T, Uw, arm, x_seq)
                        gem.append(ben)
                        pred_neu.append(b_neu)
                        pred_alt.append(b_alt)
                        rows.append(dict(H=H, s=s_nn, arm=arm, lam=lam,
                                         seed=seed, ben=ben, neu=b_neu,
                                         alt=b_alt))
                ok = [i for i in range(len(gem))
                      if gem[i] == gem[i] and abs(gem[i]) < 1e6]
                err_neu = [abs(pred_neu[i] - gem[i]) for i in ok]
                err_alt = [abs(pred_alt[i] - gem[i]) for i in ok]
                med_neu = float(torch.tensor(err_neu).median()) if err_neu else float('nan')
                med_alt = float(torch.tensor(err_alt).median()) if err_alt else float('nan')
                sp = spearman([pred_neu[i] for i in ok], [gem[i] for i in ok]) if ok else float('nan')
                st_pred = float(torch.tensor([pred_neu[i] for i in ok]).std()) if len(ok) > 1 else float('nan')
                st_gem = float(torch.tensor([gem[i] for i in ok]).std()) if len(ok) > 1 else float('nan')
                block[arm] = dict(med_neu=med_neu, med_alt=med_alt, sp=sp,
                                  st_pred=st_pred, st_gem=st_gem, n=len(ok))
                L.append('Arm %s: n=%d | median|neu-gem|=%.4f median|alt-gem|=%.4f | '
                         'Spearman(neu,gem)=%.3f | std gem=%.4f std pred=%.4f'
                         % (arm, len(ok), med_neu, med_alt, sp, st_gem, st_pred))
            cell_stats[(H, s_nn)] = block
            Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('H,s,arm,lam,seed,ben_gemessen,ben_neu,ben_alt\n')
        for r in rows:
            fh.write('%d,%s,%s,%.1f,%d,%.6g,%.6g,%.6g\n'
                     % (r['H'], r['s'], r['arm'], r['lam'], r['seed'],
                        r['ben'], r['neu'], r['alt']))

    L.append('')
    L.append('--- T1-AUSWERTUNG (nur registrierte Groessen zaehlen) ---')
    nc_meds, nc_alt_meds = [], []
    sp_cells = 0
    for (H, s_nn), b in sorted(cell_stats.items()):
        nc_meds.append(b['NC']['med_neu'])
        nc_alt_meds.append(b['NC']['med_alt'])
        ok_sp = (b['N']['sp'] == b['N']['sp'] and b['N']['sp'] >= 0.8)
        sp_cells += 1 if ok_sp else 0
        L.append('Zelle H=%d s=%s: NC median|neu-gem|=%.4f (<=0.05? %s), '
                 'NC median|alt-gem|=%.4f, Spearman(N)=%.3f (>=0.8? %s)'
                 % (H, s_nn, b['NC']['med_neu'],
                    'JA' if b['NC']['med_neu'] <= 0.05 else 'NEIN',
                    b['NC']['med_alt'], b['N']['sp'],
                    'JA' if ok_sp else 'NEIN'))
    p1 = all(m <= 0.05 for m in nc_meds)
    p2 = sp_cells >= 3
    med_neu_all = float(torch.tensor(nc_meds).mean())
    med_alt_all = float(torch.tensor(nc_alt_meds).mean())
    p3 = med_alt_all >= 2 * max(med_neu_all, 1e-9)
    L.append('P1 (NC-Fehler): %s | P2 (Lotto-Rangfolge): %d/4 | P3 (alt %.4f >= 2*neu %.4f): %s'
             % ('JA' if p1 else 'NEIN', sp_cells, med_alt_all, med_neu_all,
                'JA' if p3 else 'NEIN'))

    gate_bad = any(r['ben'] != r['ben'] for r in rows)
    if gate_bad:
        verdict = 'UNENTSCHEIDEN — ungueltige Messungen.'
    elif p1 and p2 and p3:
        verdict = ('PASS — Das Seed-Lotto ist BERECHENBAR: Das gerahmte '
                   'Feldgesetz J = -I + W F diag(<sech^2(y)>) F^T trifft den '
                   'getriebenen Exponenten (und die Rangfolge des Seed-'
                   'Lottos) aus Eigenwerten plus einer Einmal-Gain-Messung. '
                   'G2s irreduzibles Schicksal ist ein berechenbares '
                   'Schicksal. Das Feld hat seit F17 am falschen Jacobianer '
                   'gerechnet — die Reparatur ist ein Einzeiler.')
    else:
        verdict = ('KILL — Der Lotto laesst sich nicht auf Eigenwert-Niveau '
                   'komprimieren: Er bleibt eine Trajektorien-Eigenschaft. '
                   'Das gerahmte Gesetz ist bestenfalls eine Mittelwert-'
                   'Naeherung.')
    L.append('')
    L.append('VERDICT: ' + verdict)
    L.append('=== FERTIG ===')
    out = '\n'.join(L) + '\n'
    Path(out_path).write_text(out, encoding='utf-8')
    print(out)
    print('[i] Ergebnisse nach ' + out_path + ' und ' + csv_path + ' geschrieben.')


if __name__ == '__main__':
    main()
