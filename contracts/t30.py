'''T30-Contract — FENSTER-READOUT v1 (registriert XVII Nachtrag 54,
2026-11-16, VOR dem Lauf; neuer Stein nach T29-R5, Linie dort
TOT, keine Reparatur — neue registrierte Frage). Konsequenz aus
T29-R5: NOW (gedaechtnisfrei) lernt 1.00 in BEIDEN Konjugationen
(40/40) — die Geometrie killt gedaechtnisfreies Lernen nicht.
MEM mit Ridge auf dem Final-State: 0.00 auch bei a=0 —
Instrument-Verdacht: der Final-State-Readout ist zu schwach
fuer delayed recall (der Gradienten-Kopf schaffte 0.50).
FRAGE (Landscape-Punkt 2, Prozess statt Ergebnis): Haelt der
Reservoir-Schalter, wenn der Readout die ganze Rechenspur
bekommt — ein Zustands-FENSTER statt nur des Final-State?
  PASS-BESTAETIGT: Kollaps haelt bei fairer Lesespur -> der
    Schalter ist echt an Gedaechtnis gebunden.
  KILL-ARTIFAKT: Kollaps verschwindet -> T23-T25 waren
    teilweise Readout-Artefakt (Lesart verengt sich).
Design: exakt die T20/T23-Maschine (H=48, s=2.2, wb=0.6),
Ridge-Readout auf KONKATENIERTEM Zustands-Fenster der letzten
W=6 Frames (Feature-Dim 288), lam = 1.0 fix, Train 512,
Test 1024, P_lernt = acc >= 0.95, beide Aufgaben gleich.
Aufgaben: MEM (delayed recall, Referenz) und NOW
(gedaechtnisfreier Zufalls-Lehrer auf dem letzten Frame,
Seed 50000+s, Kontrolle aus T29-R5).
Stufen a in {0, 1}: MEM 1190..1209 / 1210..1229,
NOW 1230..1249 / 1250..1269. 80 Runs, f64, ~15 min.
Sanity: NOW_A0 >= 0.6 UND MEM_A0 >= 0.5 (mit fairer
Lesespur MUSZ die Referenz klar lernen — strenger als
T29-R5, weil das Instrument repariert wurde).
PRIMAER (Doppelklausel): dm30 = P_MEM(0) - P_MEM(1) falls
(P_MEM(0) >= 0.5 UND P_NOW(0) >= 0.6 UND Differenz >= 3*SE
gemessen), sonst -1.
  PASS — SCHALTER-BESTAETIGT: dm30 >= 0.30 — der Kollaps
    haelt bei fairer Lesespur; Schalter = Gedaechtnis.
  KILL — READOUT-ARTIFAKT: dm30 < 0.30 (mit intakter
    Sanity) — der T23-T25-Kollaps war teilweise eine
    Frage der Lesespur, nicht der Geometrie. Grauzone
    (dm30 in 0.0..0.30) komplementaer zu KILL registriert.
  UNENTSCHEIDEN (Crash): Sanity verletzt, ndiv > 6 je
    Block. Keine R5 mehr auf dieser Linie — dann ist die
    Frage mit Ridge-Instrumenten geschlossen.
Band (R8): P_NOW(0) in [0.80, 1.00] (T29-R5: 1.00, n=20,
verengt registriert); P_MEM-Arme ERSTFAHRT ohne Band (das
Fenster-Instrument ist neu — keine nachgeklaubten
Intervalle, Lehre T26).
Recon (2026-11-16): Basin Geometry and Reliable Recall
(arXiv 2609.01914) — Becken-Geometrie vs Recall; kein
Doppel. Instrument: experiments/t30_fenster_readout.py;
Verdikt generiert vom W1-Kernel. Rollen: block.''';

H = 48
Q_SEED = 2020
BLOCKS = {('mem', 0): (1190, 1209), ('mem', 1): (1210, 1229),
          ('now', 0): (1230, 1249), ('now', 1): (1250, 1269)}
DM30_ZERO = 0.0


def _seeds(rng):
    return list(range(rng[0], rng[1] + 1))


def contract_t30():
    arms = []
    for (task, a), rng in sorted(BLOCKS.items()):
        arms.append(dict(name='%s_A%d' % (task.upper(), a), H=H,
                         kind='norm', role='block', s=2.2, wb=0.6,
                         conj_a=a, task=task,
                         seeds=_seeds(rng),
                         invariants=[dict(metric='ndiv', op='le',
                                          expect=6)]))
    return dict(
        measures=['pf', 'colnorm', 'ndiv', 'dm30'],
        params=dict(
            blocks={str(k): list(v) for k, v in BLOCKS.items()},
            q_seed=Q_SEED, fixed_p0=4242, test_seed=777,
            t_seq=24, tau=1.0, dt=0.1, substeps=5, wb=0.6,
            a_lo=4, a_hi=8, b_lo=16, b_hi=20, hold=3,
            teacher_seed0=50000, t_hidden=8,
            window=6, ridge_lam=1.0, p_crit=0.95,
            n_train=512, n_test=1024,
            dm30_zero=DM30_ZERO),
        arms=arms,
        sanity=[dict(id='sanity-now', text='NOW a=0 lernt '
                     '(Kontrolle)', predicate=[dict(
                         metric='NOW_A0.pf', op='ge', value=0.6)]),
                dict(id='sanity-mem', text='MEM a=0 lernt klar '
                     'mit fairer Lesespur', predicate=[dict(
                         metric='MEM_A0.pf', op='ge', value=0.5)])],
        verdict=[
            dict(id='PASS — SCHALTER-BESTAETIGT',
                 text=('dm30 >= 0.30 mit Doppelklausel: der '
                       'Kollaps haelt bei fairer Lesespur — der '
                       'Schalter ist echt an Gedaechtnis gebunden'),
                 predicate=[dict(metric='MEM_A0.dm30', op='ge',
                                 value=0.3)]),
            dict(id='KILL — READOUT-ARTIFAKT',
                 text=('dm30 < 0.30 mit intakter Sanity: der '
                       'T23-T25-Kollaps war teilweise eine Frage '
                       'der Lesespur, nicht der Geometrie; Grauzone '
                       '0.0..0.30 komplementaer registriert'),
                 predicate=[dict(metric='MEM_A0.dm30', op='lt',
                                 value=0.3)]),
        ])