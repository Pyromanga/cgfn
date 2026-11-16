'''T29-Contract — SCHALTER-TASK v1 (registriert XVII Nachtrag 53,
2026-11-13, VOR dem Lauf; Konsequenz aus T26+T28). FRAGE: Ist der
Reservoir-Schalter (P kollabiert bei a >= 1, T23/T24/T25) eine
GEDAECHTNIS-Eigenschaft der Aufgabe oder eine allgemeine
Geometrie-Eigenschaft des konjugierten Reservoirs? Design: exakt
die T20/T23-Maschine (H=48, s=2.2, wb=0.6, delayed-recall-
Eingabeprotokoll, Gradienten-Readout, 600 Steps), aber ZWEI
Aufgaben-Familien auf DEMSELBEN Eingabeprotokoll:
  MEM — delayed recall (Referenz; Kollaps bekannt: P(1) ~ 0.05,
    T23/T24/T25)
  RND — Zufalls-Lehrer je Seed (Lehre T28): binaeres Label aus
    einer frischen zufaelligen tanh-Funktion der GESAMTEN
    Eingabesequenz (A1: 48x8, a2: Einheitsvektor, Seed 50000+s)
Stufen a in {0, 1}, je Aufgabe 1 frischer 20er-Block:
  MEM a=0: 1030..1049 | MEM a=1: 1050..1069
  RND a=0: 1070..1089 | RND a=1: 1090..1109
80 Runs, f64. Kein Anker: Sanity ist RND a=0 (P >= 0.5) UND
MEM a=0 (P >= 0.25, T22-Band).
PRIMAER (Doppelklausel R1): dm29 = P_RND(1) - P_MEM(1) falls
(P_RND(1) >= 0.5 UND P_MEM(1) <= 0.15 UND P_RND(0) >= 0.5 UND
P_MEM(0) >= 0.25 UND Differenz >= 3 * SE gemessen), sonst -1.
  PASS — SCHALTER-GEDAECHTNIS: dm29 >= 0.0 — der Kollaps ist
    an Gedaechtnis-Aufgaben gebunden; Geometrie allein kippt
    kein Lernen (konsistent mit T28). Konsequenz: der Schalter
    ist eine Eigenschaft der Aufgaben-Reservoir-Schnittstelle.
  KILL — SCHALTER-ALLGEMEIN: dm29 < 0.0 (umfasst RND kollabiert
    auch, P_RND(1) <= 0.15, UND fehlenden Referenz-Kollaps; die
    Grauzone P_RND(1) in 0.15..0.50 resolves KOMPLEMENTAER zu
    KILL — registriert VOR dem Lauf). Konsequenz: der Kollaps
    ist aufgabenunabhaengige Geometrie.
  UNENTSCHEIDEN (Crash): Sanity verletzt, ndiv > 6 je Block,
    Divergenz > 10 Prozent. EINE Reparatur (R5), dann Linie tot.
Band (R8): P_MEM(0) in [0.25, 0.64] (T22/T23/T24, R8-Verbreiterung
nach T26-Lehre) | P_MEM(1) in [0.00, 0.22] (T23/T24/T25).
RND-Arme: ERSTFAHRT ohne Band (registriert, Lehre T26: nicht
nachklauben). Recon-Nachbar (2026-11-13): Task-Matched Memory,
Not Criticality, Decides Which Frozen Reservoir Helps (MDPI
Entropy 28(11):1215) — waehlt Reservoir nach Aufgabe, fragt
NICHT nach Geometrie-Kollaps per Konjugation; kein Doppel.
Instrument: experiments/t29_schalter_task.py; Verdikt generiert
vom W1-Kernel. Rollen: block (Aufgaben-x-Stufen-Bloecke).''';

H = 48
Q_SEED = 2020
STAGES = [0, 1]
BLOCKS = {('mem', 0): (1030, 1049), ('mem', 1): (1050, 1069),
          ('rnd', 0): (1070, 1089), ('rnd', 1): (1090, 1109)}
DM29_ZERO = 0.0


def _seeds(rng):
    return list(range(rng[0], rng[1] + 1))


def contract_t29():
    arms = []
    for (task, a), rng in sorted(BLOCKS.items()):
        arms.append(dict(name='%s_A%d' % (task.upper(), a), H=H,
                         kind='norm', role='block', s=2.2, wb=0.6,
                         conj_a=a, task=task,
                         seeds=_seeds(rng),
                         invariants=[dict(metric='ndiv', op='le',
                                          expect=6)]))
    return dict(
        measures=['pf', 'colnorm', 'ndiv', 'dm29'],
        params=dict(
            stages=list(STAGES),
            blocks={str(k): list(v) for k, v in BLOCKS.items()},
            q_seed=Q_SEED, fixed_p0=4242, test_seed=777,
            t_seq=24, tau=1.0, dt=0.1, substeps=5, steps=600,
            batch=32, lr=5e-3, lam_max=0.6, wb=0.6,
            a_lo=4, a_hi=8, b_lo=16, b_hi=20, hold=3,
            teacher_seed0=50000, t_hidden=8,
            dm29_zero=DM29_ZERO),
        arms=arms,
        sanity=[dict(id='sanity-rnd', text='RND a=0 lernt '
                     '(Kontrolle der Zufalls-Aufgabe)', predicate=[dict(
                         metric='RND_A0.pf', op='ge', value=0.5)]),
                dict(id='sanity-mem', text='MEM a=0 im T22-Band '
                     'unten (Referenz-Kollaps-Traeger)', predicate=[dict(
                         metric='MEM_A0.pf', op='ge', value=0.25)])],
        verdict=[
            dict(id='PASS — SCHALTER-GEDAECHTNIS',
                 text=('dm29 >= 0.0: P_RND(1) >= 0.5 UND P_MEM(1) '
                       '<= 0.15 UND Differenz >= 3 * SE — der Kollaps '
                       'ist an Gedaechtnis-Aufgaben gebunden; '
                       'Geometrie allein kippt kein Lernen (T28-'
                       'konsistent); Schalter = Aufgaben-Reservoir-'
                       'Schnittstelle'),
                 predicate=[dict(metric='MEM_A0.dm29', op='ge',
                                 value=0.0)]),
            dict(id='KILL — SCHALTER-ALLGEMEIN',
                 text=('dm29 < 0.0: RND kollabiert auch (P_RND(1) '
                       '<= 0.15) oder Referenz-Kollaps fehlt — der '
                       'Kollaps ist aufgabenunabhaengige Geometrie; '
                       'Grauzone 0.15..0.50 komplementaer registriert'),
                 predicate=[dict(metric='MEM_A0.dm29', op='lt',
                                 value=0.0)]),
        ])