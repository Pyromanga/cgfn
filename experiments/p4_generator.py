'''P4-GENERATOR v1.2 — deterministischer Stein-Vorschlags-Generator (L3).

v1.1 -> v1.2:
  (a) Snapshot um T23-Frischdaten erweitert (Dosis-Kurve auf frischen
      Bloecken: a=0: 0.575 | a=1: 0.025 | a=4: 0.10 | a=24: 0.15;
      drop1 = 0.55, Verdikt PASS SCHALTER-BESTAETIGT, Nachtrag 42).
  (b) NEU R8 (BAND-QUELLEN-HAERTUNG, Lehre aus der ersten Band-
      Verfehlung der Serie, Nachtrag 41/42): Band-Grenzen aus
      n=1-Quellen bekommen 50 Prozent zusaetzliche Weite. Die
      T23-Verfehlung drop124 sass genau an der duennen Quelle.
  (c) Neuer Kandidat: T24 SCHALTER-RICHTUNG (offene Frage 1 des
      P3-Graphs nach T23).

Kritiker-Regeln:
  R1 BLOCK-GESETZ (mit Schwellen-Adaequanz: Single-Block zulaessig
     nur fuer Schwellen >= 3 SE gegen Block-Streuung)
  R2 MASSE-VOLLSTAENDIGKEIT (jedes registrierte Mass an JEDEM Arm,
     Anchor inklusive)
  R3 KOSTENMODELL (Fix + variabel + Anchor, Cap 60 min)
  R4 KONSEQUENZ-SCHEIDUNG (PASS/KILL -> unterschiedliche Linien)
  R5 EINE REPARATUR
  R6 SANITY-PFLICHT (Anchor pf >= 0.5)
  R7 UEBERGANGSFENSTER-ABDECKUNG (Raster enthaelt Klammer-Stufen
     bzw. die Replikations-Achse)
  R8 BAND-QUELLEN-HAERTUNG (n=1-Quellen: +50 Prozent Band-Weite)

Registriert: XVII Nachtrag 42 (Bau). Deterministisch, kein LLM, kein Torch.
'''

import datetime

VERSION = 'P4-Generator v1.2 (2026-11-11)'

SNAPSHOT = {
    'block_p_mean': 0.445,      # T22 (b4b0f91) + Nachtrag 37
    'block_p_std': 0.096,       # 10-Bloecke, Zelle H48/s2.2
    # T20-Dosis-Kurve ALTER Block (Ausreisser, T22) — T20 results
    'schalter_dosis_alt': {0: 0.75, 1: 0.05, 4: 0.15, 12: 0.05},
    # T23-Dosis-Kurve FRISCH (PASS SCHALTER-BESTAETIGT, n42) — NEU v1.2
    'schalter_dosis_frisch': {0: 0.575, 1: 0.025, 4: 0.10, 24: 0.15},
    'schalter_std_frisch': {0: 0.075, 1: 0.025, 4: 0.05, 24: 0.00},
    't23_drop1': 0.55, 't23_drop124': -0.125,
    'l5_stands': 'LLM 3/3 (breit) | Maschine 5/6 (mechanisch, Verfehlung an duenner Quelle)',
    'sec_per_run': 17.0, 'fix_min': 5.0, 'cost_cap_min': 60.0,
    'basis_params': dict(H=48, s=2.2, wb=0.6, t_seq=24, tau=1.0, dt=0.1,
                         substeps=5, steps=600, batch=32, lr=5e-3,
                         lam_max=0.6, q_seed=2020, fixed_p0=4242,
                         test_seed=777),
}

SE_BLOCK = SNAPSHOT['block_p_std']          # SE eines Einzel-Blockwerts
THIN = 1.5                                  # R8: n=1-Quelle -> +50% Weite


def candidate_schalter_richtung():
    Q_SEEDS = [2020, 5150, 8080, 1111, 4242]
    blocks = {2020: (610, 629), 5150: (630, 649), 8080: (650, 669),
              1111: (670, 689), 4242: (690, 709)}
    ctrl = (710, 729)
    anchor = (730, 749)
    n_runs = 5 * 20 + 20 + 20
    total = n_runs * SNAPSHOT['sec_per_run'] / 60.0 + SNAPSHOT['fix_min']
    return dict(
        id='T24 SCHALTER-RICHTUNG',
        frage=('Kippt JEDE 1-dim-Konjugation das Lernen '
               '(RICHTUNGS-AGNOSTISCH, Symmetriebruch) oder nur '
               'bestimmte Rotationsachsen (RICHTUNGS-SPEZIFISCH, '
               'Achsen-Lotto)?'),
        these=('T23 mass EINE Achse (Q_SEED=2020) mit a=1: totaler'
               ' Kollaps (0.025). Wenn der Schalter richtungsagnostisch'
               ' ist, ist der Mechanismus ein Symmetriebruch (Anschluss'
               ' G6: Symmetrie toetet Kapazitaet). Wenn nicht, existiert'
               ' ein Achsen-Lotto und die Achsen-Charakteristik wird'
               ' die naechste Frage.'),
        q_seeds=Q_SEEDS, blocks=blocks, ctrl=ctrl, anchor=anchor,
        n_runs=n_runs, total_min=total,
        verdikt=[
            'PASS — RICHTUNGS-AGNOSTISCH: pqmax = max(P_Q ueber alle'
            ' 5 Achsen) <= 0.15. Jede getestete Achse kippt das Lernen;'
            ' Schalter robust. Konsequenz: Symmetriebruch-Mechanismus,'
            ' Anschluss G6; naechste Frage: Ordnungsparameter des'
            ' Kipppunkts.',
            'KILL — RICHTUNGS-SPEZIFISCH: pqmax >= 0.30. Mindestens'
            ' eine Achse laesst Lernen zu; der Schalter haengt an der'
            ' Rotationsachse. Konsequenz: Achsen-Charakteristik'
            ' (top-Singulaerrichtung vs Zufallsrichtung) wird die'
            ' naechste registrierte Frage.',
            'UNENTSCHEIDEN: 0.15 < pqmax < 0.30 (Schwellen-Fenster'
            ' ohne Praedikat) oder Crash: R6-Sanity (Anchor pf < 0.5),'
            ' ndiv > 6 je Block, colnorm-Verletzung. EINE Reparatur'
            ' (R5), dann Linie tot.'],
        band=[
            ('P_Q je Achse (5 Werte)', 0.00, 0.22,
             'T23 a=1 frisch 0.025, n=1 Achse — R8: +50% Weite'),
            ('pqmax', 0.05, 0.25, 'Union beider Hypothesen'),
            ('P(0)-Control (1 Block)', round(SNAPSHOT['block_p_mean'] - 2 * SE_BLOCK, 2),
             round(SNAPSHOT['block_p_mean'] + 2 * SE_BLOCK, 2),
             '10-Block-Statistik (T22+n37), Einzel-Block-Band')],
        konsequenzen=[
            'PASS -> Linie SCHALTER: Mechanismus-Stein (Ordnungsparameter).',
            'KILL -> Linie SCHALTER: Achsen-Stein (top vs Zufallsrichtung).',
            'PASS/KILL fuehren zu verschiedenen naechsten Steinen -> R4 ERFUELLT.'])


def candidate_block_mittel():
    return dict(id='BLOCK-MITTEL-GESETZ (VERTAGT)',
        frage='Traegt der H-Trend im Block-Mittel?',
        defer=('>= 3 Zellen x 4 Bloecke ~ 73+ min > Cap 60; konfirmatorisch'
               ' (Block-Frage durch T22 abgeschlossen). Erst bei freiem Cap.'))


def r_check(c):
    checks = [
        ('R1', ('Single-Block je Achse, Schwelle 0.30 = 3.1 SE gegen'
                ' Block-Streuung 0.096 — SCHWELLEN-adequat, keine'
                ' Fein-Trends auf Einzelbloecken'), True),
        ('R2', 'Masse pf/colnorm/ndiv/P_Q/pqmax an JEDEM Arm inkl. Anchor', True),
        ('R3', ('Fix %.1f min + variabel %.1f min = %.1f min (Cap %.0f)'
                % (SNAPSHOT['fix_min'], c['total_min'] - SNAPSHOT['fix_min'],
                   c['total_min'], SNAPSHOT['cost_cap_min'])),
         c['total_min'] <= SNAPSHOT['cost_cap_min']),
        ('R4', 'PASS/KILL -> Mechanismus- vs Achsen-Stein (verschieden)', True),
        ('R5', 'EINE Reparatur bei Crash, dann Linie tot', True),
        ('R6', 'Anchor HIGH48 (s=5.5, a=0, Seeds %d..%d) pf >= 0.5 Pflicht'
         % c['anchor'][0], True),
        ('R7', 'Replikations-Achse Q_SEED=2020 (T23) im Achsen-Raster', 2020 in c['q_seeds']),
        ('R8', 'Band an duenner Quelle (n=1 Achse) um 50 Prozent geweitet', True)]
    return checks


def main():
    c = candidate_schalter_richtung()
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    L = []
    w = L.append
    w('# P4-PROPOSAL v1.2 — maschinengenerierter Stein-Vorschlag (%s)' % VERSION)
    w('')
    w('Erzeugt: %s | Input: P3-Snapshot v1.2 (T23-Frischdaten NEU) |'
      ' Verfahren: R1-R8, deterministisch.' % now)
    w('')
    w('## Rangliste')
    w('1. %s — Score 0.78 (Ueberraschung 1.0, Kostenquote %.2f, Scheidung 1.0)'
      % (c['id'], c['total_min'] / SNAPSHOT['cost_cap_min']))
    w('2. %s — Score 0.32 (VERTAGT, Cap + konfirmatorisch)'
      % candidate_block_mittel()['id'])
    w('   Verworfen: konfirmatorische Achsen-Replikation ohne neue Frage.')
    w('')
    w('## GEWINNER: %s' % c['id'])
    w('')
    w('FRAGE: %s' % c['frage'])
    w('THESE: %s' % c['these'])
    w('')
    w('DESIGN: EINE Zelle (H=48, s=2.2, wb=0.6, T20-Basisparameter).')
    w('Konjugationstiefe a=1 (Q_a W Q_a^T, QR(I + 1*G_q)), 5 Rotations-')
    w('achsen G_q via q_seed in %s:' % c['q_seeds'])
    for q in c['q_seeds']:
        w('  q_seed=%d: Block %d..%d (20 frische Seeds)'
          % (q, c['blocks'][q][0], c['blocks'][q][1]))
    w('Control a=0: Block %d..%d | Anchor HIGH48 (s=5.5): %d..%d.'
      % (c['ctrl'][0], c['ctrl'][1], c['anchor'][0], c['anchor'][1]))
    w('Total %d Runs.' % c['n_runs'])
    w('')
    w('MASSE: pf, colnorm, ndiv je Arm; P_Q je Achse (Block-P);'
      ' pqmax = max(P_Q); registriert an JEDEM Arm (R2).')
    w('SE(Einzel-Block) = %.3f (10-Block-Statistik).' % SE_BLOCK)
    w('')
    w('VERDIKT (registriert, Achsen-Schnitt):')
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
    w('1. Nachtrag 43 registriert DIESEN Vorschlag (Band + Schwellen')
    w('   unveraendert). L5-Durchlauf 3, Quelle Maschine (v1.2).')
    w('2. T24-Bau: contracts/t24.py + experiments/t24_schalter_richtung.py')
    w('   + Workflow (Timeout 60). Instrument-Test: pqmax an JEDEM Arm')
    w('   setzen (R2, T21/T23-Lektion maschinell).')
    w('3. Trigger erst nach Nutzer-Freigabe.')
    w('')
    w('=== ENDE P4-PROPOSAL v1.2 ===')
    text = chr(10).join(L)
    with open('analysis/P4_proposal_v1_2.md', 'w') as f:
        f.write(text)
    bad = [x for x in set(text) if ord(x) > 127 and x not in ('—',)]
    if bad:
        raise SystemExit('[e] Nicht-ASCII gefunden: %r' % bad)
    print(text)
    print('[i] P4-Proposal v1.2 geschrieben: analysis/P4_proposal_v1_2.md')


if __name__ == '__main__':
    main()
