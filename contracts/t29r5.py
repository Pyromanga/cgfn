'''T29-R5-Contract — SCHALTER-RIDGE v1 (registriert XVII Nachtrag 53-
Ergaenzung, 2026-11-16, VOR dem Lauf; die EINE erlaubte Reparatur
nach T29 UNENTSCHEIDEN). Zwei ehrliche Selbstkritik-Punkte sind
die Reparatur: (1) T29s RND-Task war SELBST eine Gedaechtnis-
Aufgabe (Label hing an der Gesamtsequenz) — Zufaelligkeit und
Gedaechtnis-Forderung waren verkoppelt. (2) Gradienten-Readout
starb als Instrument (RND_A0 = 0.00 auch bei a = 0). R5 killt
beide: Ridge-Readout (kein Gradienten-Training, lam fix
registriert) und NOW-Task (Zufalls-Lehrer nur auf dem
AKTUELLEN Eingabe-Frame, gedaechtnisfrei).
Design: exakt die T20/T23-Maschine (H=48, s=2.2, wb=0.6,
delayed-recall-Eingabeprotokoll), zwei Aufgaben auf DEMSELBEN
Eingabeprotokoll:
  MEM — delayed recall, Ridge-Readout (Referenz)
  NOW — Zufalls-Lehrer je Seed auf dem letzten Eingabe-Frame
    (A1: 8x2, a2: Einheitsvektor, Seed 50000+s), gedaechtnisfrei
Stufen a in {0, 1}, je Aufgabe 1 frischer 20er-Block:
  MEM a=0: 1110..1129 | MEM a=1: 1130..1149
  NOW a=0: 1150..1169 | NOW a=1: 1170..1189
80 Runs, f64, kein Gradienten-Training (billig, ~10 min).
Lernkriterium (registriert): P_lernt = acc >= 0.95 (Ridge,
beide Aufgaben gleich). lam = 1.0 fix, Train 512, Test 1024.
Sanity: NOW_A0 >= 0.6 (gedaechtnisfreie Aufgabe muss
ueberwiegend lernen) UND MEM_A0 >= 0.25 (T22-Band unten).
PRIMAER (Doppelklausel): dm29r5 = P_NOW(1) - P_MEM(1) falls
(P_NOW(1) >= 0.5 UND P_MEM(1) <= 0.15 UND P_NOW(0) >= 0.6
UND P_MEM(0) >= 0.25 UND Differenz >= 3 * SE gemessen),
sonst -1.
  PASS — SCHALTER-GEDAECHTNIS: dm29r5 >= 0.0 — der Kollaps
    ist an Gedaechtnis gebunden; eine gedaechtnisfreie
    Zufalls-Aufgabe kippt nicht (Schalter = Aufgaben-
    Reservoir-Schnittstelle, Gedaechtnis-Anteil).
  KILL — SCHALTER-ALLGEMEIN: dm29r5 < 0.0 — auch die
    gedaechtnisfreie Aufgabe kollabiert unter Konjugation
    (Geometrie kippt Lernen generell im Reservoir-Regime).
  Grauzone (P_NOW(1) in 0.15..0.50 mit intakter Sanity)
  resolves KOMPLEMENTAER zu KILL — registriert VOR dem Lauf.
  UNENTSCHEIDEN (Crash): Sanity verletzt, ndiv > 6 je Block.
  Danach ist die Linie TOT (R5 verbraucht), egal wie das
  Ergebnis aussieht — registriert.
Band (R8): P_MEM(0) in [0.25, 0.64], P_MEM(1) in [0.00, 0.22]
(unveraendert aus N53). NOW-Arme: ERSTFAHRT ohne Band.
Recon-Nachbar (2026-11-16): Basin Geometry and Reliable
Recall of Dynamical Memories in RC (arXiv 2609.01914) —
Becken-Geometrie vs verlaessliches Recall; kein Doppel.
Instrument: experiments/t29r5_schalter_ridge.py; Verdikt
generiert vom W1-Kernel. Rollen: block (Aufgaben-x-Stufen).''';

H = 48
Q_SEED = 2020
BLOCKS = {('mem', 0): (1110, 1129), ('mem', 1): (1130, 1149),
          ('now', 0): (1150, 1169), ('now', 1): (1170, 1189)}
DM29R5_ZERO = 0.0


def _seeds(rng):
    return list(range(rng[0], rng[1] + 1))


def contract_t29r5():
    arms = []
    for (task, a), rng in sorted(BLOCKS.items()):
        arms.append(dict(name='%s_A%d' % (task.upper(), a), H=H,
                         kind='norm', role='block', s=2.2, wb=0.6,
                         conj_a=a, task=task,
                         seeds=_seeds(rng),
                         invariants=[dict(metric='ndiv', op='le',
                                          expect=6)]))
    return dict(
        measures=['pf', 'colnorm', 'ndiv', 'dm29r5'],
        params=dict(
            blocks={str(k): list(v) for k, v in BLOCKS.items()},
            q_seed=Q_SEED, fixed_p0=4242, test_seed=777,
            t_seq=24, tau=1.0, dt=0.1, substeps=5, wb=0.6,
            a_lo=4, a_hi=8, b_lo=16, b_hi=20, hold=3,
            teacher_seed0=50000, t_hidden=8,
            ridge_lam=1.0, p_crit=0.95, n_train=512, n_test=1024,
            dm29r5_zero=DM29R5_ZERO),
        arms=arms,
        sanity=[dict(id='sanity-now', text='NOW a=0 lernt '
                     '(gedaechtnisfreie Kontrolle)', predicate=[dict(
                         metric='NOW_A0.pf', op='ge', value=0.6)]),
                dict(id='sanity-mem', text='MEM a=0 im T22-Band '
                     'unten (Referenz)', predicate=[dict(
                         metric='MEM_A0.pf', op='ge', value=0.25)])],
        verdict=[
            dict(id='PASS — SCHALTER-GEDAECHTNIS',
                 text=('dm29r5 >= 0.0 mit Doppelklausel: der '
                       'Kollaps ist an Gedaechtnis gebunden; die '
                       'gedaechtnisfreie Zufalls-Aufgabe kippt nicht'),
                 predicate=[dict(metric='MEM_A0.dm29r5', op='ge',
                                 value=0.0)]),
            dict(id='KILL — SCHALTER-ALLGEMEIN',
                 text=('dm29r5 < 0.0: auch die gedaechtnisfreie '
                       'Aufgabe kollabiert — Geometrie kippt Lernen '
                       'generell im Reservoir-Regime; Grauzone '
                       '0.15..0.50 komplementaer registriert'),
                 predicate=[dict(metric='MEM_A0.dm29r5', op='lt',
                                 value=0.0)]),
        ])