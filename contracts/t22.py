'''T22-Contract — BLOCK-ROBUSTHEIT ALT vs NEU (registriert XVII
Nachtrag 38, L5-Durchlauf 1: erster LLM-Vorschlag). EINE Zelle
(H=48, s=2.2, wb=0.6). 5 neue disjunkte 20er-Bloecke 310..409;
Anchor HIGH48 mit 20 frischen Seeds 410..429.
PRIMAER (VERDIKT): zalt = (0.75 - mean(P_neu5)) / std(P_neu5),
  registriert an jedem Arm (0.75 = P48(2.2) des ALTEN Blocks
  190..209 aus T20-R2).
  PASS — ALT-AUSREISSER: zalt >= 2.5
  KILL — KONTINUITAET:  zalt < 2.5
UNENTSCHEIDEN (Crash): std(P_neu5) < 0.02 (z-Mass sinnlos),
  Sanity gerissen, ndiv > 6 je Block, colnorm-Verletzung.
SEKUNDAER: Block-P je Block, 10-Block-Gesamtverteilung (5 aus
  Nachtrag-37-Reanalyse als Referenzkonstante, 5 neu gemessen).
Rollen: anchor (Sanity), block (neue 20er-Bloecke).'''

H_CELL = 48
S_CELL = 2.2
S_ANCHOR = 5.5
BLOCKS = [list(range(310 + 20 * i, 330 + 20 * i)) for i in range(5)]
ANCHOR_SEEDS = list(range(410, 430))
P_ALT_220 = 0.75


def _inv_norm(s):
    return [dict(metric='colnorm', op='eq', expect=s, tol=1e-6),
            dict(metric='ndiv', op='le', expect=6)]


def contract_t22():
    arms = [dict(name='HIGH48', H=H_CELL, kind='norm', role='anchor',
                 s=S_ANCHOR, wb=0.6, seeds=list(ANCHOR_SEEDS),
                 invariants=_inv_norm(S_ANCHOR))]
    for i, blk in enumerate(BLOCKS):
        arms.append(dict(name='B%d' % i, H=H_CELL, kind='norm',
                         role='block', s=S_CELL, wb=0.6, seeds=list(blk),
                         invariants=_inv_norm(S_CELL)))
    return dict(
        measures=['pf', 'colnorm', 'ndiv', 'zalt'],
        params=dict(
            h_cell=H_CELL, s_cell=S_CELL, s_anchor=S_ANCHOR,
            blocks=[list(b) for b in BLOCKS],
            seeds_anchor=list(ANCHOR_SEEDS), p_alt_220=P_ALT_220,
            blocks_reanalyse=[0.35, 0.40, 0.55, 0.50, 0.35],
            fixed_p0=4242, test_seed=777,
            t_seq=24, tau=1.0, dt=0.1, substeps=5, steps=600,
            batch=32, lr=5e-3, lam_max=0.6, a_lo=4, a_hi=8,
            b_lo=16, b_hi=20, hold=3),
        arms=arms,
        sanity=[dict(id='sanity', text='HIGH48 lernt (Anker weit oberhalb der Rampe)',
                     predicate=[dict(metric='HIGH48.pf', op='ge', value=0.5)])],
        verdict=[
            dict(id='PASS — ALT-AUSREISSER',
                 text='zalt >= 2.5: Der alte Block (0.75) ist Extremwert '
                      'der Block-Verteilung — T16-T19-Kette war ein '
                      'Block-Unfall',
                 predicate=[dict(metric='B0.zalt', op='ge', value=2.5)]),
            dict(id='KILL — KONTINUITAET',
                 text='zalt < 2.5: 0.75 liegt innerhalb der Block-'
                      'Verteilung — Einzelblock-Trends bedeutungslos',
                 predicate=[dict(metric='B0.zalt', op='lt', value=2.5)]),
        ])
