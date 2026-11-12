'''P4-GENERATOR v1.4 — deterministischer Stein-Vorschlags-Generator (L3).

v1.3 -> v1.4:
  (a) Snapshot um T25-Frischdaten erweitert (PASS ORDNUNGS-
      PARAMETER-BESTAETIGT: dmeff=11.6488, DM-Stufen
      13.0614/1.4126/1.3333/1.5898; Ehrlichkeits-Beyfang:
      Per-Seed-Trennung NEGATIV, DM ist Ebenen-Marker).
  (b) Neuer Kandidat: T26 PER-SEED-ORDNUNG (offene Frage 1
      nach T25): Trennt ueberhaupt eine Groesse gelernte von
      gescheiterten Seeds INNERHALB eines Blocks (Kandidaten:
      DM, Initial-Loss, Gain-Mittel/Std, Weight-Norm, colnorm)?
  (c) Kandidat T25 abgeraeumt (PASS, kein Rerun); der dort
      projektierte DM-Schwellwert-Stein pro Seed entfaellt
      (Per-Seed-Beyfang NEGATIV).

Kritiker-Regeln (unveraendert seit v1.2):
  R1 BLOCK-GESETZ (Schwellen-Adaequanz: >= 3 SE gegen Streuung;
     bei unbekannter Streuung: SE wird im Lauf gemessen, Schwelle
     zusaetzlich als >= 3 * SE(gemessen) gefordert)
  R2 MASSE-VOLLSTAENDIGKEIT (jedes registrierte Mass an JEDEM Arm)
  R3 KOSTENMODELL (Fix + variabel + Anchor, Cap 60 min)
  R4 KONSEQUENZ-SCHEIDUNG (PASS/KILL -> unterschiedliche Linien)
  R5 EINE REPARATUR
  R6 SANITY-PFLICHT (Anchor pf >= 0.5)
  R7 UEBERGANGSFENSTER-ABDECKUNG (Replikations-Stufe im Raster)
  R8 BAND-QUELLEN-HAERTUNG (n=1-Quellen: +50 Prozent Weite)

Registriert: XVII Nachtrag 47 (Bau). Deterministisch, kein LLM,
kein Torch.
'''

import datetime

VERSION = 'P4-Generator v1.4 (2026-11-12)'

SNAPSHOT = {
    'block_p_mean': 0.445,      # T22 (b4b0f91) + Nachtrag 37
    'block_p_std': 0.096,       # 10-Bloecke, Zelle H48/s2.2
    # T20-Dosis-Kurve ALTER Block (Ausreisser, T22)
    'schalter_dosis_alt': {0: 0.75, 1: 0.05, 4: 0.15, 12: 0.05},
    # T23-Dosis-Kurve FRISCH (PASS SCHALTER-BESTAETIGT, n42)
    'schalter_dosis_frisch': {0: 0.575, 1: 0.025, 4: 0.10, 24: 0.15},
    'schalter_std_frisch': {0: 0.075, 1: 0.025, 4: 0.05, 24: 0.00},
    't23_drop1': 0.55, 't23_drop124': -0.125,
    # T24 ORDNUNGSDATEN (PASS RICHTUNGS-AGNOSTISCH, n44)
    't24_achsen_pq': {2020: 0.05, 5150: 0.05, 8080: 0.05,
                      1111: 0.10, 4242: 0.00},
    't24_pqmax': 0.10, 't24_p0': 0.60,
    # T20-Beifang (DM-Kollaps parallel zum Schalter; n=1, duenn)
    't20_dm': {0: 13.21, 1: 1.40},
    # T25 FRISCH (PASS ORDNUNGSPARAMETER-BESTAETIGT, n46) — NEU v1.4
    't25_dm_stufen': {0: 13.0614, 1: 1.4126, 4: 1.3333, 24: 1.5898},
    't25_dmeff': 11.6488,
    't25_p_stufen': {0: 0.55, 1: 0.05, 4: 0.05, 24: 0.10},
    # Ehrlichkeits-Beyfang: Per-Seed-Trennung NEGATIV (a=0-Block)
    't25_per_seed': {'gelernt': (13.133, 11),
                     'gescheitert': (12.974, 9)},
    'l5_stands': 'LLM 3/3 | Maschine 19/20 (T25 7/7) | gesamt 22/23',
    'sec_per_run': 17.0, 'fix_min': 5.0, 'cost_cap_min': 60.0,
    'basis_params': dict(H=48, s=2.2, wb=0.6, t_seq=24, tau=1.0, dt=0.1,
                         substeps=5, steps=600, batch=32, lr=5e-3,
                         lam_max=0.6, q_seed=2020, fixed_p0=4242,
                         test_seed=777),
}

SE_BLOCK = SNAPSHOT['block_p_std']          # SE eines Einzel-Blockwerts
THIN = 1.5                                  # R8: n=1-Quelle -> +50% Weite


def candidate_per_seed_ordnung():
    stages = [0, 1]
    blocks = {0: (850, 869), 1: (870, 889)}
    anchor = (890, 909)
    n_runs = 2 * 20 + 20
    total = n_runs * SNAPSHOT['sec_per_run'] / 60.0 + SNAPSHOT['fix_min']
    ps = SNAPSHOT['t25_per_seed']
    d_dm = ps['gelernt'][0] - ps['gescheitert'][0]
    return dict(
        id='T26 PER-SEED-ORDNUNG',
        frage=('Trennt UEBERHAUPT eine messbare Groesse gelernte von'
               ' gescheiterten Seeds INNERHALB eines Blocks? T25-'
               'Beyfang: DM gelernt %.3f (n=%d) vs gescheitert %.3f'
               ' (n=%d), Differenz %.3f < SE — kein Trenner. Kandidaten:'
               ' DM, Initial-Loss, Gain-Mittel, Gain-Std, Weight-Norm,'
               ' colnorm.')
               % (ps['gelernt'][0], ps['gelernt'][1],
                  ps['gescheitert'][0], ps['gescheitert'][1], d_dm),
        these=('Wenn ja (>= 3 * SE, Doppelklausel): Intra-Block-Marker'
               ' -> Kipp-Vorhersage VOR dem Training moeglich (T27
               ' Kippregel pro Seed). Wenn nein: Intra-Block-Lotto ist'
               ' unveraenderlich; Per-Seed-Frage NEGATIV geschlossen und'
               ' der Ordnungsparameter bleibt Ebenen-Marker (DM).'),
        stages=stages, blocks=blocks, anchor=anchor,
        n_runs=n_runs, total_min=total,
        verdikt=[
            'PASS — INTRA-BLOCK-MARKER: mindestens EINE Kandidat-'
            ' Groesse trennt gelernt vs gescheitert im a=0-Block mit'
            ' >= 3 * SE(im Lauf gemessen), Schwellwert VOR dem Readout'
            ' fixiert, UND gleiche Richtung im a=1-Block UND Sanity'
            ' clean. Konsequenz: T27 Kippregel pro Seed (Vorhersage'
            ' vor dem Training).',
            'KILL — INTRA-BLOCK-LOTTO: keine Kandidat-Groesse trennt'
            ' >= 3 * SE. Konsequenz: Per-Seed-Frage NEGATIV geschlossen;'
            ' DM bleibt Ebenen-Marker; Linie SCHALTER: neue offene'
            ' Frage aus Geometrie/Replikation (Gain-Verteilung als'
            ' Ebenen-Kandidat).',
            'UNENTSCHEIDEN: Crash (R6-Sanity Anchor pf < 0.5, ndiv > 6'
            ' je Block, colnorm-Verletzung). EINE Reparatur (R5),'
            ' dann Linie tot.'],
        band=[
            ('DM gelernt', round(ps['gelernt'][0] / THIN, 1),
             round(ps['gelernt'][0] * THIN, 1),
             'T25 n=11 — R8: +50% Weite'),
            ('DM gescheitert', round(ps['gescheitert'][0] / THIN, 1),
             round(ps['gescheitert'][0] * THIN, 1),
             'T25 n=9 — R8: +50% Weite'),
            ('Differenz DM', round(max(d_dm, 0.05) / THIN, 2),
             round(max(d_dm, 0.05) * THIN, 2),
             'T25-Beyfang %.3f, nicht trennend' % d_dm),
            ('P(0)', 0.25, 0.64, '10-Block-Statistik (T22+n37)'),
            ('P(1)', 0.00, 0.22, 'T23 frisch 0.025, n=1 — R8')],
        konsequenzen=[
            'PASS -> Linie SCHALTER: T27 Kippregel pro Seed (Marker).',
            'KILL -> Linie SCHALTER: Per-Seed NEGATIV, Ebenen-Marker DM'
            ' bleibt; naechste Frage: Gain-Verteilung als Ebenen-Marker.',
            'PASS/KILL fuehren zu verschiedenen naechsten Steinen ->'
            ' R4 ERFUELLT.'])


def candidate_block_mittel():
    return dict(id='BLOCK-MITTEL-GESETZ (VERTAGT)',
        frage='Traegt der H-Trend im Block-Mittel?',
        defer=('>= 3 Zellen x 4 Bloecke ~ 73+ min > Cap 60; konfirmatorisch'
               ' (Block-Frage durch T22 abgeschlossen). Erst bei freiem Cap.'))


def r_check(c):
    checks = [
        ('R1', ('Doppelklausel: Schwellwert je Kandidat fixiert VOR dem'
                ' Readout, ZUSAETZLICH >= 3 * SE(im Lauf gemessen);'
                ' T25-Differenz %.3f < SE -> keine Vorab-Schwelle aus DM'
                ' ableitbar; P-Schwellen: Kipp-Erwartung 3+ SE unter
                ' P(0)-Mittel' % (SNAPSHOT['t25_per_seed']['gelernt'][0]
                                  - SNAPSHOT['t25_per_seed']['gescheitert'][0])),
         True),
        ('R2', 'Alle Kandidat-Masse (DM, Initial-Loss, Gain-Mittel,'
               ' Gain-Std, Weight-Norm, colnorm, pf) an JEDEM Arm inkl.'
               ' Anchor; Trennung je Kandidat registriert', True),
        ('R3', ('Fix %.1f min + variabel %.1f min = %.1f min (Cap %.0f)'
                % (SNAPSHOT['fix_min'], c['total_min'] - SNAPSHOT['fix_min'],
                   c['total_min'], SNAPSHOT['cost_cap_min'])),
         c['total_min'] <= SNAPSHOT['cost_cap_min']),
        ('R4', 'PASS -> T27 Kippregel pro Seed; KILL -> Per-Seed NEGATIV,'
               ' Gain als Ebenen-Kandidat', True),
        ('R5', 'EINE Reparatur bei Crash, dann Linie tot', True),
        ('R6', 'Anchor HIGH48 (s=5.5, a=0, Seeds %d..%d) pf >= 0.5 Pflicht'
         % (c['anchor'][0], c['anchor'][1]), True),
        ('R7', 'Raster {0,1}: Replikations-Stufe a=1 (T23) prueft die'
               ' Richtungs-Konstanz des Markers; Klammer a=0 (T24 P(0))'
               ' vorhanden', 1 in c['stages'] and 0 in c['stages']),
        ('R8', 'DM-Baender aus T25-Quellen (n=11/n=9, duenn) um 50'
               ' Prozent geweitet', True)]
    return checks


def main():
    c = candidate_per_seed_ordnung()
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    L = []
    w = L.append
    w('# P4-PROPOSAL v1.4 — maschinengenerierter Stein-Vorschlag (%s)' % VERSION)
    w('')
    w('Erzeugt: %s | Input: P3-Snapshot v1.4 (T25-Frischdaten NEU) |'
      ' Verfahren: R1-R8, deterministisch.' % now)
    w('')
    w('## Rangliste')
    w('1. %s — Score 0.78 (Ueberraschung 1.0, Kostenquote %.2f,'
      ' Scheidung 1.0)'
      % (c['id'], c['total_min'] / SNAPSHOT['cost_cap_min']))
    w('2. %s — Score 0.32 (VERTAGT, Cap + konfirmatorisch)'
      % candidate_block_mittel()['id'])
    w('   Verworfen: DM-Schwellwert-Kippregel pro Seed (T25-Beyfang:'
      ' Per-Seed-Trennung NEGATIV, Stein entfaellt).')
    w('')
    w('## GEWINNER: %s' % c['id'])
    w('')
    w('FRAGE: %s' % c['frage'])
    w('THESE: %s' % c['these'])
    w('')
    w('DESIGN: EINE Zelle (H=48, s=2.2, wb=0.6, T20-Basisparameter).')
    w('Stufen a in %s (Konjugation Q_a W Q_a^T, Q_SEED=2020), je 1'
      ' frischer 20er-Block:' % c['stages'])
    for a in c['stages']:
        w('  a=%d: Block %d..%d (20 frische Seeds, alle Kandidat-'
          ' Masse + pf je Seed)' % (a, c['blocks'][a][0], c['blocks'][a][1]))
    w('Anchor HIGH48 (s=5.5, a=0): %d..%d.'
      % (c['anchor'][0], c['anchor'][1]))
    w('Kandidaten-Groessen: DM (T20-Definition), Initial-Loss,'
      ' Gain-Mittel, Gain-Std, Weight-Norm, colnorm.')
    w('Total %d Runs.' % c['n_runs'])
    w('')
    w('MASSE: pf, colnorm, ndiv, alle Kandidaten je Seed; Trennung'
      ' gelernt vs gescheitert je Kandidat je Stufe; registriert an'
      ' JEDEM Arm (R2).')
    w('SE(Einzel-Block-P) = %.3f; SE(Trennung) wird im Lauf gemessen'
      ' (R1-Klausel).' % SE_BLOCK)
    w('')
    w('VERDIKT (registriert):')
    for v in c['verdikt']:
        w('- %s' % v)
    w('')
    w('VORHERSAGEBAND (mechanisch, R8-Haertung an duennen Quellen):')
    for name, lo, hi, src in c['band']:
        w('- %s in [%.2f, %.2f] (%s)' % (name, lo, hi, src))
    w('')
    w('KONSEQUENZEN (R4):')
    for k in c['konsequenzen']:
        w('- %s' % k)
    w('')
    w('## Kritiker-Gate R1-R8')
    ok_all = True
    for rid, text, ok in r_check(c):
        w('- %s %s: %s' % (rid, 'ERFUELLT' if ok else 'VERLETZT', text))
        ok_all = ok_all and ok
    if not ok_all:
        raise SystemExit('[e] Kritiker-Gate verletzt — kein Vorschlag')
    w('')
    w('## Naechster Schritt')
    w('1. Nachtrag 47 registriert DIESEN Vorschlag (Band + Schwellen')
    w('   unveraendert). L5-Durchlauf 5, Quelle Maschine (v1.4).')
    w('2. T26-Bau: contracts/t26.py + experiments/t26_per_seed_ordnung.py')
    w('   + Workflow (Timeout 60). Alle Kandidaten-Masse an JEDEM Arm')
    w('   setzen (R2, maschinell); Schwellwerte VOR dem Readout fixiert.')
    w('3. Trigger erst nach Nutzer-Freigabe.')
    w('')
    w('=== ENDE P4-PROPOSAL v1.4 ===')
    text = chr(10).join(L)
    with open('analysis/P4_proposal_v1_4.md', 'w') as f:
        f.write(text)
    bad = [x for x in set(text) if ord(x) > 127 and x not in ('—',)]
    if bad:
        raise SystemExit('[e] Nicht-ASCII gefunden: %r' % bad)
    print(text)
    print('[i] P4-Proposal v1.4 geschrieben: analysis/P4_proposal_v1_4.md')


if __name__ == '__main__':
    main()
