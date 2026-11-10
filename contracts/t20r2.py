'''T20-R2-Contract — DM-TREPPE (registriert XVII Nachtrag 31).
Die EINE Reparatur von T20 (danach Linie tot). Konjugation war ein
Schalter; R2 nimmt den glatten Regler: WB steuert die Diagonallast
der Eigenbasis direkt. H=48 fix, Spektrum fix, Seeds 190..209.
PRIMAER (VERDIKT): pfspread220 = max-min von P_lernt(s=2.2)
  ueber die WB-Stufen, registriert an jedem Arm.
  PASS — DM-TREPPE-KAUSAL: pfspread220 >= 0.4
  KILL — DM-NEBENSPUR:     pfspread220 < 0.4
SEKUNDAER (Bericht, kein Crash): s50rep je Stufe — Wert oder
  -1.0 = ZENSIERT (Zensierung ist Messergebnis, kein Fehler).
UNENTSCHEIDEN (Crash): DM-Bereich < 10 Prozent relativ, Sanity
  gerissen, ndiv > 6 von 20, colnorm-Verletzung. Danach Linie tot.
Rollen: anchor (Sanity), sgrid (WB-Stufen x s-Raster).'''

H = 48
S_ANCHOR = 5.5
S_GRID = [1.83, 2.2, 2.9]
WB_STAGES = [0.6, 0.45, 0.3, 0.15]


def _inv_norm(s):
    return [dict(metric='colnorm', op='eq', expect=s, tol=1e-6),
            dict(metric='ndiv', op='le', expect=6)]


def contract_t20r2():
    arms = [dict(name='HIGH48', H=H, kind='norm', role='anchor', s=S_ANCHOR,
                 wb=0.6, invariants=_inv_norm(S_ANCHOR))]
    for wb in WB_STAGES:
        for s in S_GRID:
            arms.append(dict(name='WB%s_s%s' % (wb, s), H=H, kind='norm',
                             role='sgrid', s=s, wb=wb,
                             invariants=_inv_norm(s)))
    return dict(
        measures=['pf', 'colnorm', 'ndiv', 's50rep', 'pfspread220'],
        params=dict(
            seeds=list(range(190, 210)), H_list=[H], s_grid=list(S_GRID),
            wb_stages=list(WB_STAGES), s_anchor=S_ANCHOR,
            fixed_p0=4242, test_seed=777,
            t_seq=24, tau=1.0, dt=0.1, substeps=5, steps=600,
            batch=32, lr=5e-3, lam_max=0.6, a_lo=4, a_hi=8,
            b_lo=16, b_hi=20, hold=3),
        arms=arms,
        sanity=[dict(id='sanity', text='HIGH48 lernt (Anker weit oberhalb der Rampe)',
                     predicate=[dict(metric='HIGH48.pf', op='ge', value=0.5)])],
        verdict=[
            dict(id='PASS — DM-TREPPE-KAUSAL',
                 text='pfspread220 >= 0.4: P_lernt(2.2) hoert auf die '
                      'WB-Stufe — Diagonallast ist kausal',
                 predicate=[dict(metric='WB0.6_s1.83.pfspread220', op='ge', value=0.4)]),
            dict(id='KILL — DM-NEBENSPUR',
                 text='pfspread220 < 0.4: P_lernt(2.2) bleibt gleich ueber '
                      'alle WB-Stufen — DM ist Kovariate ohne Kausalitaet',
                 predicate=[dict(metric='WB0.6_s1.83.pfspread220', op='lt', value=0.4)]),
        ])
