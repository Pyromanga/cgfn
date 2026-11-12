'''T27-Contract — SCHALTER-TRAIN v1 (registriert XVII Nachtrag 48,
Linie TRAIN, erster Gradienten-Stein). FRAGE: Kippt ein
GRADIENTEN-TRAINIERTES Netz denselben Schalter wie das
eingefrorene Reservoir? MLP (feedforward, bewusste
Vereinfachung): H=48, tanh, Input-Dim 48 (= T_SEQ x 2,
flattened), KEIN Bias (G5-Audit-Lektion: versteckter Bias-
Lotto), Output-Head 2x48. Init: W1 = Q_a W_init Q_a^T mit
W_init aus der T20/T25-Konstruktion (Q_SEED=2020) — exakt
dieselbe Konjugationstiefe. Training: Adam lr=5e-3, 600
Steps, batch=32, delayed recall (t_seq=24), Test 1024.
Stufen a in {0, 1, 24}, je 1 frischer 20er-Block (a=0:
910..929, a=1: 930..949, a=24: 950..969). Kein HIGH48-Anker:
R6-Sanity ist der a=0-Block — P(0) muss >= 0.25 liegen
(Band [0.25, 0.64], T22 10-Block-Statistik).
PRIMAER (Doppelklausel R1, Lehre T25): peff = P(0) - P(1)
falls (P(1) <= 0.15 UND P(24) <= 0.15 UND P(0) - P(1) >=
3 * SE(im Lauf gemessen) UND P(0) im Band), sonst -1.
Kodierung als Verdict-Mass peff an JEDEM Arm (R2).
  PASS — SCHALTER-TRAIN: peff >= 0.0
  KILL — TRAIN-NETZ-RESISTENT: peff < 0.0 (umfasst P > 0.30
    und Differenz < 3 * SE; die Grauzone 0.15..0.30 resolves
    KOMPLEMENTAER zu KILL — registriert VOR dem Lauf, keine
    Nachverhandlung)
  TEIL-VERDICT (Bericht, kein Kernel-Praedikat): P kollabiert,
    aber DM_final trennt die Stufen nicht
    (SCHALTER-OHNE-ORDNUNGSPARAMETER)
  UNENTSCHEIDEN (Crash): Sanity P(0) < 0.25, ndiv > 6 je
    Block, Trainings-Divergenz > 10 Prozent. EINE Reparatur
    (R5), dann Linie tot. R4 erfuellt.
Band (mechanisch, R8-Haertung, Nachtrag 48): P(0) in [0.25,
0.64] | P(1) in [0.00, 0.30] | P(24) in [0.00, 0.30] |
DM_init(0) in [8.80, 19.80] (T25 n=11, R8).
Instrument: experiments/t27_schalter_train.py; Verdikt
generiert vom W1-Kernel. Rollen: block (Stufen-Bloecke).'''

H = 48
Q_SEED = 2020
STAGES = [0, 1, 24]
BLOCKS = {0: (910, 929), 1: (930, 949), 24: (950, 969)}
PF_PASS = 0.15
PEFF_ZERO = 0.0


def _seeds(rng):
    return list(range(rng[0], rng[1] + 1))


def contract_t27():
    arms = []
    for a in STAGES:
        arms.append(dict(name='A%d' % a, H=H, kind='norm',
                         role='block', s=2.2, wb=0.6, conj_a=a,
                         seeds=_seeds(BLOCKS[a]),
                         invariants=[dict(metric='ndiv', op='le',
                                          expect=6)]))
    return dict(
        measures=['pf', 'colnorm', 'ndiv', 'dm_init', 'dm_final',
                  'peff'],
        params=dict(
            stages=list(STAGES),
            blocks={str(k): list(BLOCKS[k]) for k in BLOCKS},
            q_seed=Q_SEED, fixed_p0=4242, test_seed=777,
            t_seq=24, steps=600, batch=32, lr=5e-3, lam_max=0.6,
            wb=0.6, a_lo=4, a_hi=8, b_lo=16, b_hi=20, hold=3,
            pf_pass=PF_PASS, peff_zero=PEFF_ZERO),
        arms=arms,
        sanity=[dict(id='sanity', text='P(0) im Band (a=0-Block '
                     'ist Anker, R6): unten 0.25', predicate=[dict(
                         metric='A0.pf', op='ge', value=0.25)])],
        verdict=[
            dict(id='PASS — SCHALTER-TRAIN',
                 text=('peff >= 0.0: P(1) <= 0.15 UND P(24) <= 0.15 '
                       'UND P(0) - P(1) >= 3 * SE UND P(0) im Band — '
                       'Bruecke Reservoir -> trainiert BESTAETIGT; '
                       'naechster Stein: Ordnungsparameter im '
                       'trainierten Netz (DM_final-Analog)'),
                 predicate=[dict(metric='A0.peff', op='ge',
                                 value=0.0)]),
            dict(id='KILL — TRAIN-NETZ-RESISTENT',
                 text=('peff < 0.0 (P > 0.30 oder Differenz < 3 * SE; '
                       'Grauzone 0.15..0.30 komplementaer zu KILL): '
                       'Der Schalter ist eine Eigenschaft des '
                       'Eingefroren-Reservoirs — Grenze des Befunds '
                       'praecise gezeichnet'),
                 predicate=[dict(metric='A0.peff', op='lt',
                                 value=0.0)]),
        ])
