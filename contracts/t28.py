'''T28-Contract — ZUFALLS-LEHRER v1 (registriert XVII Nachtrag 51,
Linie TRAIN, zweiter Stein nach dem T27-Instrument-Tod). FRAGE:
Kippt die Konjugation Q_a W_init Q_a^T die Lernbarkeit eines
GRADIENTEN-TRAINIERTEN Netzes, wenn die AUFGABE selbst hoch-
entropisch ist? HIGH-ENTROPIE-DESIGN (Lehre aus T27: die feste
delayed-recall-Aufgabe war vom flachen MLP ungelernbar — P=0.00
in ALLEN Armen inkl. Kontrolle): je Seed ein FRISCHER zufaelliger
nichtlinearer Lehrer (2-Schichten-tanh, 8 verborgene Einheiten,
Gewichte aus dem Seed), die Entropie liegt in der Aufgabe, nicht
in der Kalibrierung. Lernkriterium kalibrierungsfrei: R2 >= 0.9
(finaler Test-MSE <= 0.10 * Varianz der Targets) — keine
absoluten Schwellen, keine Nachverhandlung.
MLP: H=48, tanh, KEIN Bias, W1 = Q_a W_init Q_a^T (exakt
T20/T25/T27, Q_SEED=2020), linearer Skalar-Head. Adam lr=5e-3,
800 Steps, batch=64, Test 4096 frische Samples.
Stufen a in {0, 1, 24}, je 1 frischer 20er-Block (a=0: 970..989,
a=1: 990..1009, a=24: 1010..1029). Kein Anker: R6-Sanity ist der
a=0-Block — P(0) >= 0.5 (Kontrolle muss UEBERWIEGEND lernen;
strenger als T27, weil das Kriterium kalibrierungsfrei ist).
PRIMAER (Doppelklausel R1, Lehre T25/T27): peff = P(0) - P(1)
falls (P(1) <= 0.15 UND P(24) <= 0.15 UND P(0) - P(1) >=
3 * SE(im Lauf gemessen) UND P(0) >= 0.5), sonst -1.
  PASS — SCHALTER-HIGH-ENTROPIE: peff >= 0.0 — der Schalter
    kippt die Lernbarkeit auch bei Aufgaben-Entropie; Bruecke
    Reservoir -> trainiert BESTAETIGT am lernbaren Instrument.
  KILL — ENTROPIE-RESISTENT: peff < 0.0 — bei zufaelligen
    Lehrern kippt nichts; der Schalter ist ein Phaenomen der
    AUFGABEN-Struktur (delayed recall), nicht der Geometrie.
  TEIL-VERDICT (Bericht): P kollabiert, aber DM_final trennt
    nicht — SCHALTER-OHNE-ORDNUNGSPARAMETER.
  UNENTSCHEIDEN (Crash): Sanity P(0) < 0.5, ndiv > 6 je Block,
    Divergenz > 10 Prozent. EINE Reparatur (R5), dann Linie tot.
Band (R8, ehrlich duenn): DM_init(0) in [8.80, 19.80] (T25 n=11).
ALLE aufgabenbezogenen Groessen sind ERSTFAHRT ohne Band —
registriert als solche, keine nachgeklaubten Intervalle.
Instrument: experiments/t28_zufalls_lehrer.py; Verdikt generiert
vom W1-Kernel. Rollen: block (Stufen-Bloecke).''';

H = 48
Q_SEED = 2020
STAGES = [0, 1, 24]
BLOCKS = {0: (970, 989), 1: (990, 1009), 24: (1010, 1029)}
PF_PASS = 0.15
PEFF_ZERO = 0.0


def _seeds(rng):
    return list(range(rng[0], rng[1] + 1))


def contract_t28():
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
            q_seed=Q_SEED, din=48, t_hidden=8, steps=800,
            batch=64, lr=5e-3, lam_max=0.6, wb=0.6, r2_pass=0.9,
            pf_pass=PF_PASS, peff_zero=PEFF_ZERO),
        arms=arms,
        sanity=[dict(id='sanity', text='P(0) >= 0.5 (a=0-Block '
                     'ist Anker, R6 — Kontrolle muss ueberwiegend '
                     'lernen)', predicate=[dict(
                         metric='A0.pf', op='ge', value=0.5)])],
        verdict=[
            dict(id='PASS — SCHALTER-HIGH-ENTROPIE',
                 text=('peff >= 0.0: P(1) <= 0.15 UND P(24) <= 0.15 '
                       'UND P(0) - P(1) >= 3 * SE UND P(0) >= 0.5 — '
                       'der Schalter kippt die Lernbarkeit auch bei '
                       'Aufgaben-Entropie; Bruecke Reservoir -> '
                       'trainiert BESTAETIGT am lernbaren Instrument'),
                 predicate=[dict(metric='A0.peff', op='ge',
                                 value=0.0)]),
            dict(id='KILL — ENTROPIE-RESISTENT',
                 text=('peff < 0.0: bei zufaelligen Lehrern kippt '
                       'nichts — der Schalter ist ein Phaenomen der '
                       'AUFGABEN-Struktur (delayed recall), nicht '
                       'der Geometrie; Grenze des Befunds praecise'),
                 predicate=[dict(metric='A0.peff', op='lt',
                                 value=0.0)]),
        ])