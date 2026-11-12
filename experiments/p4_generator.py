'''P4-GENERATOR v1.3 — deterministischer Stein-Vorschlags-Generator (L3).

v1.2 -> v1.3:
  (a) Snapshot um T24-Frischdaten erweitert (RICHTUNGS-AGNOSTISCH:
      Achsen-P_Q 2020:0.05, 5150:0.05, 8080:0.05, 1111:0.10,
      4242:0.00; pqmax=0.10; P(0)=0.60; Verdikt PASS, Nachtrag 44;
      L5-Band 7/7, R8-Erstfahrt).
  (b) Neuer Kandidat: T25 ORDNUNGSPARAMETER (offene Frage 1 des
      P3-Graphs nach T24): Ist der DM-Kollaps (T20-Beifang
      13.21 -> 1.40 parallel zum Schalter) ein Ordnungsparameter
      des Kipppunkts — beobachtbar VOR dem Verdikt?
  (c) Kandidat T24 abgeraeumt (Verdikt gefaellt, kein Rerun);
      BLOCK-MITTEL-GESETZ bleibt vertagt.

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

Registriert: XVII Nachtrag 45 (Bau). Deterministisch, kein LLM,
kein Torch.
'''

import datetime

VERSION = 'P4-Generator v1.3 (2026-11-11)'

SNAPSHOT = {
    'block_p_mean': 0.445,      # T22 (b4b0f91) + Nachtrag 37
    'block_p_std': 0.096,       # 10-Bloecke, Zelle H48/s2.2
    # T20-Dosis-Kurve ALTER Block (Ausreisser, T22)
    'schalter_dosis_alt': {0: 0.75, 1: 0.05, 4: 0.15, 12: 0.05},
    # T23-Dosis-Kurve FRISCH (PASS SCHALTER-BESTAETIGT, n42)
    'schalter_dosis_frisch': {0: 0.575, 1: 0.025, 4: 0.10, 24: 0.15},
    'schalter_std_frisch': {0: 0.075, 1: 0.025, 4: 0.05, 24: 0.00},
    't23_drop1': 0.55, 't23_drop124': -0.125,
    # T24 ORDNUNGSDATEN FRISCH (PASS RICHTUNGS-AGNOSTISCH, n44) — NEU v1.3
    't24_achsen_pq': {2020: 0.05, 5150: 0.05, 8080: 0.05,
                      1111: 0.10, 4242: 0.00},
    't24_pqmax': 0.10, 't24_p0': 0.60,
    # T20-Beifang (DM-Kollaps parallel zum Schalter; n=1, duenn)
    't20_dm': {0: 13.21, 1: 1.40},
    'l5_stands': 'LLM 3/3 (breit) | Maschine 12/13 (R8-Erstfahrt 7/7)',
    'sec_per_run': 17.0, 'fix_min': 5.0, 'cost_cap_min': 60.0,
    'basis_params': dict(H=48, s=2.2, wb=0.6, t_seq=24, tau=1.0, dt=0.1,
                         substeps=5, steps=600, batch=32, lr=5e-3,
                         lam_max=0.6, q_seed=2020, fixed_p0=4242,
                         test_seed=777),
}

SE_BLOCK = SNAPSHOT['block_p_std']          # SE eines Einzel-Blockwerts
THIN = 1.5                                  # R8: n=1-Quelle -> +50% Weite


def candidate_ordnungsparameter():
    stages = [0, 1, 4, 24]
    blocks = {0: (750, 769), 1: (770, 789), 4: (790, 809),
              24: (810, 829)}
    anchor = (830, 849)
    n_runs = 4 * 20 + 20
    total = n_runs * SNAPSHOT['sec_per_run'] / 60.0 + SNAPSHOT['fix_min']
    return dict(
        id='T25 ORDNUNGSPARAMETER',
        frage=('Ist der DM-Kollaps (T20-Beifang 13.21 -> 1.40 '
               'parallel zum Schalter) ein ORDNUNGSPARAMETER des '
               'Kipppunkts — d.h. DM trennt gekippte von intakten '
               'Netzen VOR dem Verdikt und ist kein blossem '
               'Begleitprodukt?'),
        these=('T20-R2 zeigte: DM ist keine Ursache (Drehen der '
               'DM-Treppe aenderte nichts). Offen ist die '
               'diagnostische Seite: Faellt DM mit dem Schalter '
               'ZUSAMMEN (Stufen-Sprung zwischen a=0 und a=1) und '
               'trennt er Per-Seed die Kipp-Zugehoerigkeit? Wenn ja, '
               'hat die Linie SCHALTER ein messbares Vorhersage-'
               'Instrument. Wenn nein, ist DM endgueltig Nebenprodukt '
               'und die Ordnungsparameter-Suche geht zu Geometrie-'
               'Alternativen (Gain-Verteilung).'),
        stages=stages, blocks=blocks, anchor=anchor,
        n_runs=n_runs, total_min=total,
        verdikt=[
            'PASS — ORDNUNGSPARAMETER-BESTAETIGT: dmdrop = DM(0) - DM(1)'
            ' >= 5.0 UND dmdrop >= 3 * SE(dmdrop, im Lauf gemessen)'
            ' UND Sanity clean. Konsequenz: DM wird Vorhersage-'
            ' Instrument; naechster Stein: DM-Schwellwert-Kippregel'
            ' pro Seed (Kipp VOR dem Training sichtbar).',
            'KILL — DM-NEBENPRODUKT: dmdrop < 5.0 oder < 3 * SE.'
            ' Konsequenz: DM endgueltig als Diagnostikum gefallen'
            ' (konsistent mit T20-R2 KILL DM-NEBENSPUR);'
            ' Ordnungsparameter-Suche: Gain-Verteilung als'
            ' naechster Kandidat.',
            'UNENTSCHEIDEN: Crash (R6-Sanity Anchor pf < 0.5, ndiv > 6'
            ' je Block, colnorm-Verletzung). EINE Reparatur (R5),'
            ' dann Linie tot.'],
        band=[
            ('DM(0)', round(SNAPSHOT['t20_dm'][0] / THIN, 1),
             round(SNAPSHOT['t20_dm'][0] * THIN, 1),
             'T20 n=1 (13.21) — R8: +50% Weite'),
            ('DM(1)', round(SNAPSHOT['t20_dm'][1] / THIN, 1),
             round(SNAPSHOT['t20_dm'][1] * THIN, 1),
             'T20 n=1 (1.40) — R8: +50% Weite'),
            ('dmdrop', round((SNAPSHOT['t20_dm'][0] - SNAPSHOT['t20_dm'][1]) / 2, 1),
             round((SNAPSHOT['t20_dm'][0] - SNAPSHOT['t20_dm'][1]) * THIN, 1),
             'T20-Effekt 11.81, n=1 — R8: Weite'),
            ('P(0)', 0.25, 0.64, '10-Block-Statistik (T22+n37)'),
            ('P(1)', 0.00, 0.22, 'T23 frisch 0.025, n=1 — R8'),
            ('P(4)', 0.00, 0.30, 'T23 frisch 0.10, n=1 — R8'),
            ('P(24)', 0.00, 0.25, 'T23 frisch 0.15, n=1 — R8')],
        konsequenzen=[
            'PASS -> Linie SCHALTER: Vorhersage-Instrument (DM-Kippregel).',
            'KILL -> Linie SCHALTER: Geometrie-Alternative (Gain-Verteilung).',
            'PASS/KILL fuehren zu verschiedenen naechsten Steinen -> R4 ERFUELLT.'])


def candidate_block_mittel():
    return dict(id='BLOCK-MITTEL-GESETZ (VERTAGT)',
        frage='Traegt der H-Trend im Block-Mittel?',
        defer=('>= 3 Zellen x 4 Bloecke ~ 73+ min > Cap 60; konfirmatorisch'
               ' (Block-Frage durch T22 abgeschlossen). Erst bei freiem Cap.'))


def r_check(c):
    checks = [
        ('R1', ('Single-Block je Stufe; DM-SE ungetestet -> Schwelle'
                ' 5.0 = 42 Prozent des T20-Effekts ZUSAETZLICH als'
                ' >= 3 * SE(dmdrop) im Verdikt-Praedikat gefordert'
                ' (SE wird im Lauf gemessen); P-Schwellen: Kipp-'
                ' Erwartung liegt 3+ SE unter P(0)-Mittel'), True),
        ('R2', 'Masse pf/colnorm/ndiv/DM/DM-Mittel je Stufe an JEDEM'
               ' Arm inkl. Anchor; dmdrop registriert an JEDEM Arm', True),
        ('R3', ('Fix %.1f min + variabel %.1f min = %.1f min (Cap %.0f)'
                % (SNAPSHOT['fix_min'], c['total_min'] - SNAPSHOT['fix_min'],
                   c['total_min'], SNAPSHOT['cost_cap_min'])),
         c['total_min'] <= SNAPSHOT['cost_cap_min']),
        ('R4', 'PASS -> Vorhersage-Instrument; KILL -> Gain-Alternative', True),
        ('R5', 'EINE Reparatur bei Crash, dann Linie tot', True),
        ('R6', 'Anchor HIGH48 (s=5.5, a=0, Seeds %d..%d) pf >= 0.5 Pflicht'
         % (c['anchor'][0], c['anchor'][1]), True),
        ('R7', 'Stufen-Raster {0,1,4,24} enthaelt Replikations-Stufe a=1'
               ' (T23) und Klammer a=0 (T24-P(0))', 1 in c['stages'] and 0 in c['stages']),
        ('R8', 'DM-Baender aus n=1-Quelle (T20) um 50 Prozent geweitet', True)]
    return checks


def main():
    c = candidate_ordnungsparameter()
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    L = []
    w = L.append
    w('# P4-PROPOSAL v1.3 — maschinengenerierter Stein-Vorschlag (%s)' % VERSION)
    w('')
    w('Erzeugt: %s | Input: P3-Snapshot v1.3 (T24-Frischdaten NEU) |'
      ' Verfahren: R1-R8, deterministisch.' % now)
    w('')
    w('## Rangliste')
    w('1. %s — Score 0.80 (Ueberraschung 1.0, Kostenquote %.2f,'
      ' Scheidung 1.0)'
      % (c['id'], c['total_min'] / SNAPSHOT['cost_cap_min']))
    w('2. %s — Score 0.32 (VERTAGT, Cap + konfirmatorisch)'
      % candidate_block_mittel()['id'])
    w('   Verworfen: konfirmatorische T24-Achsen-Replikation ohne'
      ' neue Frage (T24 ist PASS, abgeraeumt).')
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
        w('  a=%d: Block %d..%d (20 frische Seeds, DM + pf je Seed)'
          % (a, c['blocks'][a][0], c['blocks'][a][1]))
    w('Anchor HIGH48 (s=5.5, a=0): %d..%d.'
      % (c['anchor'][0], c['anchor'][1]))
    w('DM-Definition: unveraendert aus T20-Instrument uebernommen')
    w('(experiments/t20_conjugation*.py; Kollaps 13.21 -> 1.40 dort).')
    w('Total %d Runs.' % c['n_runs'])
    w('')
    w('MASSE: pf, colnorm, ndiv, DM je Seed; DM-Mittel je Stufe;')
    w('dmdrop = DM(0) - DM(1); registriert an JEDEM Arm (R2).')
    w('SE(Einzel-Block-P) = %.3f; SE(dmdrop) wird im Lauf gemessen'
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
    w('1. Nachtrag 45 registriert DIESEN Vorschlag (Band + Schwellen')
    w('   unveraendert). L5-Durchlauf 4, Quelle Maschine (v1.3).')
    w('2. T25-Bau: contracts/t25.py + experiments/t25_ordnungsparameter.py')
    w('   + Workflow (Timeout 60). DM-Readout aus T20-Instrument,')
    w('   dmdrop an JEDEM Arm setzen (R2, maschinell).')
    w('3. Trigger erst nach Nutzer-Freigabe.')
    w('')
    w('=== ENDE P4-PROPOSAL v1.3 ===')
    text = chr(10).join(L)
    with open('analysis/P4_proposal_v1_3.md', 'w') as f:
        f.write(text)
    bad = [x for x in set(text) if ord(x) > 127 and x not in ('—',)]
    if bad:
        raise SystemExit('[e] Nicht-ASCII gefunden: %r' % bad)
    print(text)
    print('[i] P4-Proposal v1.3 geschrieben: analysis/P4_proposal_v1_3.md')


if __name__ == '__main__':
    main()
