'''T24-Contract — SCHALTER-RICHTUNG v1 (registriert XVII
Nachtrag 43, maschinengeneriert vom P4-Generator v1.2,
L5-Durchlauf 3). EINE Zelle (H=48, s=2.2, wb=0.6),
Konjugationstiefe a=1 via W' = Q_1 W Q_1^T (Spektrum identisch,
Q aus QR(I + 1*G_q)), 5 Rotationsachsen q_seed in
{2020, 5150, 8080, 1111, 4242}, je 20 frische Seeds 610..709;
Control a=0 (Seeds 710..729); Anchor HIGH48 (s=5.5, Seeds
730..749). q_seed=2020 ist die Replikations-Achse (R7: T23 mass
bei a=1 den Wert 0.025). Primaer pqmax = max(P_Q ueber die
5 Achsen), registriert an JEDEM Arm (R2, T21/T23-Lektion).
  PASS (RICHTUNGS-AGNOSTISCH): pqmax <= 0.15 — jede getestete
    Achse kippt, Symmetriebruch-Mechanismus (Anschluss G6)
  KILL (RICHTUNGS-SPEZIFISCH): pqmax >= 0.30 — mindestens eine
    Achse laesst Lernen zu, Achsen-Stein (top-Singulaerrichtung)
  UNENTSCHEIDEN: 0.15 < pqmax < 0.30 oder Crash (R6-Sanity
    Anchor pf < 0.5, ndiv > 6 je Arm, colnorm-Verletzung).
    EINE Reparatur (R5), dann Linie tot. R4 erfuellt.
Band (mechanisch, R8-HAERTUNG, Nachtrag 43): P_Q je Achse in
[0.00, 0.22] (Quelle T23 a=1 frisch 0.025, n=1-Quelle,
+50% Weite) | pqmax in [0.05, 0.25] | P(0)-Control in
[0.25, 0.64] (T22 10-Block: 0.445 +- 2*0.096).
KEINE std-Klausel: pqmax hat keinen Nenner (Praezedenz
Nachtrag 42 — std-Klauseln nur an Nenner-fuehrenden Massen).
Instrument: experiments/t24_schalter_richtung.py; Verdikt
generiert vom W1-Kernel. Rollen: anchor (SANITY), block.'''

H = 48
S = 2.2
WB = 0.6
Q_SEEDS = [2020, 5150, 8080, 1111, 4242]
S_ANCHOR = 5.5
AXIS_RANGES = {2020: (610, 629), 5150: (630, 649), 8080: (650, 669),
               1111: (670, 689), 4242: (690, 709)}
CTRL_RANGE = (710, 729)
ANCHOR_RANGE = (730, 749)


def _inv_norm(s):
    return [dict(metric='colnorm', op='eq', expect=s, tol=1e-6),
            dict(metric='ndiv', op='le', expect=6)]


def _seeds(rng):
    return list(range(rng[0], rng[1] + 1))


def contract_t24():
    arms = [dict(name='HIGH48', H=H, kind='norm', role='anchor',
                 s=S_ANCHOR, wb=WB, conj_a=0, q_seed=2020,
                 seeds=_seeds(ANCHOR_RANGE),
                 invariants=_inv_norm(S_ANCHOR)),
            dict(name='A0_CTRL', H=H, kind='norm', role='block',
                 s=S, wb=WB, conj_a=0, q_seed=2020,
                 seeds=_seeds(CTRL_RANGE),
                 invariants=_inv_norm(S))]
    for qs in Q_SEEDS:
        arms.append(dict(name='Q%d' % qs, H=H, kind='norm',
                         role='block', s=S, wb=WB, conj_a=1,
                         q_seed=qs, seeds=_seeds(AXIS_RANGES[qs]),
                         invariants=_inv_norm(S)))
    return dict(
        measures=['pf', 'colnorm', 'ndiv', 'pq', 'pqmax'],
        params=dict(
            q_seeds=list(Q_SEEDS),
            axis_seeds={str(k): list(_seeds(AXIS_RANGES[k])) for k in Q_SEEDS},
            ctrl_seeds=_seeds(CTRL_RANGE),
            seeds_anchor=_seeds(ANCHOR_RANGE),
            s_anchor=S_ANCHOR, fixed_p0=4242, test_seed=777,
            t_seq=24, tau=1.0, dt=0.1, substeps=5, steps=600,
            batch=32, lr=5e-3, lam_max=0.6, wb=WB,
            a_lo=4, a_hi=8, b_lo=16, b_hi=20, hold=3,
            pq_pass=0.15, pq_kill=0.30),
        arms=arms,
        sanity=[dict(id='sanity', text='HIGH48 lernt (Anker weit '
                     'oberhalb der Rampe, R6)', predicate=[dict(
                         metric='HIGH48.pf', op='ge', value=0.5)])],
        verdict=[
            dict(id='PASS — RICHTUNGS-AGNOSTISCH',
                 text='pqmax <= 0.15: JEDE getestete 1-dim-'
                      'Konjugation kippt das Lernen — Symmetriebruch-'
                      'Mechanismus (Anschluss G6); naechster Stein: '
                      'Ordnungsparameter des Kipppunkts',
                 predicate=[dict(metric='Q2020.pqmax', op='le',
                                 value=0.15)]),
            dict(id='KILL — RICHTUNGS-SPEZIFISCH',
                 text='pqmax >= 0.30: Mindestens eine Achse laesst '
                      'Lernen zu — Achsen-Lotto; naechster Stein: '
                      'top-Singulaerrichtung vs Zufallsrichtung',
                 predicate=[dict(metric='Q2020.pqmax', op='ge',
                                 value=0.30)]),
        ])
