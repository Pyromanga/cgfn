'''T31-Contract — VOLL-AEQUIVARIANTE ROTATION v1 (registriert XVII
Nachtrag 55, 2026-11-17, VOR dem Lauf; Antwort auf den
Readout-Einwand). EINWAND: T23-T25 rotieren nur W (b und Uw
bleiben in der alten Basis, tanh sitzt in Absolutkoordinaten)
— der Kollaps koennte ein Artefakt der TEIL-Rotation sein.
FRAGE: Haelt der Schalter, wenn die Rotation VOLL
aequivariant ist — W → Q W Q^T, b → Q b, Uw → Q Uw, mit
trainiertem Readout (der als trainiert linear unter Rotation
kompensieren kann)?
  PASS — SCHALTER-IMMUN: Kollaps repliziert im PART-Arm
    (P_PART(1) <= 0.15, wie T23) UND haelt im FULL-Arm
    (P_FULL(1) <= 0.15) — der Befund ist gegen jeden
    Readout-Einwand immun: echte Dynamik-Zerstoerung.
  KILL — ROTATIONS-ARTEFAKT: P_PART(1) <= 0.15 UND
    P_FULL(1) >= 0.5 — der Kollaps war ein Artefakt der
    Teil-Rotation (nur W gedreht); T23-T25-Lesart verengt
    sich auf 'W-only-Rotation gegen festen Anker'.
  Sonst (Grauzone: P_FULL(1) in 0.15..0.50 oder PART
  repliziert nicht): UNENTSCHEIDEN — Lesart offen.
Design: exakt die T20/T23-Maschine (H=48, s=2.2, wb=0.6,
delayed-recall-Eingabeprotokoll, Gradienten-Readout,
600 Steps). Arme: rot=part (nur W, exakt T23) und rot=full
(W, b, Uw alle mit demselben Q gedreht), Stufen a {0,1}:
  PART a=0: 1270..1289 | PART a=1: 1290..1309
  FULL a=0: 1310..1329 | FULL a=1: 1330..1349
80 Runs, f64, ~30 min. Bei a=0 sind PART und FULL
identisch (Q=I) — frische Bloecke dienen als
Replikations-Kontrolle.
Sanity: PART_A0 >= 0.25 (T22-Band) UND FULL_A0 >= 0.25.
Band (R8): P_PART(1) in [0.00, 0.22] (T23/T24/T25);
P_FULL-Arme ERSTFAHRT ohne Band (registriert).
Recon (2026-11-17): Einwand extern (Agenten-Dialog);
naechster Nachbar bleibt MDPI Entropy 28(11):1215.
Instrument: experiments/t31_aequivariant_rotation.py;
Verdikt generiert vom W1-Kernel. Rollen: block.''';

H = 48
Q_SEED = 2020
BLOCKS = {('part', 0): (1270, 1289), ('part', 1): (1290, 1309),
          ('full', 0): (1310, 1329), ('full', 1): (1330, 1349)}


def _seeds(rng):
    return list(range(rng[0], rng[1] + 1))


def contract_t31():
    arms = []
    for (rot, a), rng in sorted(BLOCKS.items()):
        arms.append(dict(name='%s_A%d' % (rot.upper(), a), H=H,
                         kind='norm', role='block', s=2.2, wb=0.6,
                         conj_a=a, rot=rot,
                         seeds=_seeds(rng),
                         invariants=[dict(metric='ndiv', op='le',
                                          expect=6)]))
    return dict(
        measures=['pf', 'colnorm', 'ndiv', 'immun', 'artefakt'],
        params=dict(
            blocks={str(k): list(v) for k, v in BLOCKS.items()},
            q_seed=Q_SEED, fixed_p0=4242, test_seed=777,
            t_seq=24, tau=1.0, dt=0.1, substeps=5, steps=600,
            batch=32, lr=5e-3, lam_max=0.6, wb=0.6,
            a_lo=4, a_hi=8, b_lo=16, b_hi=20, hold=3),
        arms=arms,
        sanity=[dict(id='sanity-part', text='PART a=0 im '
                     'T22-Band (Referenz)', predicate=[dict(
                         metric='PART_A0.pf', op='ge', value=0.25)]),
                dict(id='sanity-full', text='FULL a=0 im '
                     'T22-Band (Kontrolle)', predicate=[dict(
                         metric='FULL_A0.pf', op='ge', value=0.25)])],
        verdict=[
            dict(id='PASS — SCHALTER-IMMUN',
                 text=('Kollaps repliziert (P_PART(1) <= 0.15) '
                       'UND haelt unter voller aequivanter Rotation '
                       '(P_FULL(1) <= 0.15) — echte Dynamik-Zer-'
                       'stoerung, gegen jeden Readout-Einwand immun'),
                 predicate=[dict(metric='PART_A0.immun', op='ge',
                                 value=1.0)]),
            dict(id='KILL — ROTATIONS-ARTEFAKT',
                 text=('P_PART(1) <= 0.15 UND P_FULL(1) >= 0.5: '
                       'der Kollaps war ein Artefakt der Teil-'
                       'Rotation; T23-T25-Lesart verengt sich'),
                 predicate=[dict(metric='PART_A0.artefakt', op='ge',
                                 value=1.0)]),
        ])