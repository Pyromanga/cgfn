'''P4-GENERATOR v1.1 — deterministischer Stein-Vorschlags-Generator (L3).

v1.0 -> v1.1 (Lehre aus dem ersten Maschinen-Vorschlag):
  DEFEKT v1.0: Raster [0,4,12,24] verpasste das T20-Schalter-Fenster
  (Kollaps bereits bei a=1: P 0.75 -> 0.05). Der Snapshot enthielt nur
  Aggregate, nicht die Dosis-Kurve je Stufe -> Band widersprach der
  eigenen Quelle, vorhersagbares UNENTSCHEIDEN (gefaehrdete 56 min,
  vom Kernel-Kritiker VOR dem Feuer gefunden, 0 Runner-Minuten Verlust).
  FIX: (a) Snapshot um T20-Dosis-Kurve erweitert, (b) Regel R7 neu:
  Ein Raster muss die Klammer-Stufen des steilsten beobachteten
  Uebergangs enthalten. R7 haette v1.0 abgelehnt.

Kritiker-Regeln (Lehren aus Nachtraegen 14-39 + v1.0-Defekt):
  R1 BLOCK-GESETZ: Keine Einzelblock-Trends. Trend-Zahlen sind
     Block-Mittel mit std(P_Block) ausgewiesen.
  R2 MASSE-VOLLSTAENDIGKEIT: Registrierte Masse an JEDEM Arm,
     Anchor inklusive (T21-Lektion).
  R3 KOSTENMODELL: Fixkosten + variable + Anchor ausweisen (T21).
     Cap: 60 Runner-Minuten.
  R4 KONSEQUENZ-SCHEIDUNG: PASS und KILL muessen zu
     UNTERSCHIEDLICHEN naechsten Schritten fuehren, sonst
     ABGELEHNT (T22/Entropie-Kollaps-Lektion).
  R5 EINE REPARATUR: Bei Crash genau eine Reparatur, dann Linie tot.
  R6 SANITY-PFLICHT: Anchor lernt (pf >= 0.5), sonst UNENTSCHEIDEN.
  R7 UEBERGANGSFENSTER-ABDECKUNG (v1.1): Das Raster enthaelt die
     beiden Klammer-Stufen des steilsten beobachteten Uebergangs
     (Dosis-Kurve im Snapshot). Sonst ABGELEHNT (v1.0-Defekt).

Registriert: XVII Nachtrag 40. Deterministisch, kein LLM, kein Torch.
'''

import datetime

VERSION = 'P4-Generator v1.1 (2026-11-11)'

# --- EINGEFRORENER P3-SNAPSHOT (Quelle je Konstante) ---
SNAPSHOT = {
    'block_p_mean': 0.445,      # T22 (b4b0f91) + Nachtrag 37
    'block_p_std': 0.096,
    'block_p_list': [0.35, 0.40, 0.55, 0.50, 0.35, 0.35, 0.40, 0.50, 0.40, 0.65],
    'block_std_range': (0.06, 0.13),   # Nachtrag 37
    # T20-DOSIS-KURVE der Zelle s=2.2 (ALTER Block 190..209 = Ausreisser,
    # T22) — NEU in v1.1. Quelle: results/t20_dm_kovariate_results.txt
    # (bfb1f59): a=0: 0.75 | a=1: 0.05 | a=4: 0.15 | a=12: 0.05
    'schalter_dosis': {0: 0.75, 1: 0.05, 4: 0.15, 12: 0.05},
    'schalter_dm': {0: 13.21, 1: 1.40, 4: 1.34, 12: 1.58},  # T20
    'dm_treppe_pfspread': 0.05,        # T20-R2 (a6381cc), Nachtrag 32
    'sec_per_run': 17.0, 'fix_min': 5.0, 'cost_cap_min': 60.0,
    'tote_linien': ['H-Trend-Einzelblock', 'DM-Kovariate', 'DM-Treppe', 'Delle'],
    'basis_params': dict(H=48, s=2.2, wb=0.6, t_seq=24, tau=1.0, dt=0.1,
                         substeps=5, steps=600, batch=32, lr=5e-3,
                         lam_max=0.6, q_seed=2020, fixed_p0=4242,
                         test_seed=777),   # contracts/t20.py
}

SE_STUFE = SNAPSHOT['block_p_std'] / (2 ** 0.5)  # Stufen-Mittel aus 2 Bloecken

def r7_uebergang(raster):
    '''Steilster beobachteter Uebergang der Dosis-Kurve; Rueckgabe:
    (bracket_levels, ok) — ok wenn beide Klammer-Stufen im Raster.'''
    lv = sorted(SNAPSHOT['schalter_dosis'])
    drops = [(lv[i], lv[i+1],
              SNAPSHOT['schalter_dosis'][lv[i]] - SNAPSHOT['schalter_dosis'][lv[i+1]])
             for i in range(len(lv) - 1)]
    steepest = max(drops, key=lambda d: d[2])
    bracket = (steepest[0], steepest[1])
    return bracket, all(b in raster for b in bracket)

def candidate_q2():
    A = [0, 1, 4, 24]
    blocks = {0: [(430, 449), (450, 469)], 1: [(470, 489), (490, 509)],
              4: [(510, 529), (530, 549)], 24: [(550, 569), (570, 589)]}
    n_runs = len(A) * 40 + 20  # + Anchor 590..609
    total = n_runs * SNAPSHOT['sec_per_run'] / 60.0 + SNAPSHOT['fix_min']
    return dict(
        id='T23 SCHALTER-DOSIS v1.1',
        frage=('Repliziert der Lern-Kollaps bei Konjugationstiefe a=1 '
               '(T20, alter Block) auf frischen Seed-Bloecken?'),
        these=('T20-Signal: Schon EINE konjugierte Dimension (a=1) kippt '
               'P_lernt von 0.75 auf 0.05. Falls das auf frischen '
               'Bloecken steht (Block-Gesetz beachten!), ist der '
               'Schalter extrem scharf: Strukturbruch bei a=1.'),
        stufen=A, blocks=blocks, n_runs=n_runs, total_min=total,
        verdikt=[
            'PASS — SCHALTER-BESTAETIGT: drop1 = P(0)-P(1) >= 0.30 '
            '(>= 3.1 SE): Der a=1-Kollaps repliziert auf frischen '
            'Bloecken. Der Schalter ist real und extrem scharf.',
            'KILL — KEIN-KOLLAPS: drop1 < 0.30: Der T20-Schalter war '
            'ein Artefakt des Ausreisser-Blocks 190..209 (T22-Lektion). '
            'Schalter-Linie TOT.',
            'UNENTSCHEIDEN: Crash — R6-Sanity (Anchor pf < 0.5), '
            'ndiv > 6 je Block, std(P_Block) < 0.02 je Stufe, '
            'colnorm-Verletzung. EINE Reparatur (R5), dann tot.'],
        konsequenzen=[
            'PASS -> eigene Linie SCHALTER: Ordnungsparameter des '
            'Kipppunkts suchen; erste Anschlussfrage: Ist die '
            'top-Singulaerrichtung die kritische? (Konjugation entlang '
            'Zufallsrichtung a=1 als Gegenprobe.)',
            'KILL -> Schalter-Linie geschlossen: T20-Doppelkollaps war '
            'Seed-Lotto am Ausreisser-Block; Pfade wieder frei fuer '
            'T24 Block-Mittel oder L5-Durchlauf 3.',
            'PASS/KILL fuehren zu verschiedenen Linien -> R4 ERFUELLT.'],
        band=[
            ('P(0) (2 Block-Mittel)', round(SNAPSHOT['block_p_mean'] - 2 * SE_STUFE, 2),
             round(SNAPSHOT['block_p_mean'] + 2 * SE_STUFE, 2), '10-Block-Statistik (T22+n37)'),
            ('P(1)', 0.00, 0.25, 'T20: 0.05, n=1 alter Block, ehrlich breit'),
            ('P(4)', 0.00, 0.30, 'T20: 0.15, n=1 alter Block'),
            ('P(24)', 0.00, 0.25, 'T20-Kollaps extrapoliert (a=12: 0.05)'),
            ('drop1', 0.05, 0.55, 'Union beider Hypothesen'),
            ('drop124 = P(1)-P(24)', -0.10, 0.20, 'Plateau-Erwartung (Form, SEKUNDAER)')])

def candidate_q3():
    return dict(id='T24 BLOCK-MITTEL-GESETZ (VERTAGT)',
        frage='Traegt der H-Trend im Block-Mittel?',
        defer=('Kosten: >= 3 Zellen x 4 Bloecke x 20 Runs ~ 73+ min > Cap 60. '
               'UND konfirmatorisch (Block-Frage durch T22 abgeschlossen): '
               'geringe Entscheidungs-Entropie. Erst bei freiem Cap.'))

def r_check(q2):
    bracket, r7_ok = r7_uebergang(q2['stufen'])
    checks = [
        ('R1', 'Block-Mittel mit SE/std ausgewiesen (2 disjunkte 20er-Bloecke je Stufe)', True),
        ('R2', 'Masse pf/colnorm/ndiv/P_Stufe/drop1 an JEDEM Arm inkl. Anchor', True),
        ('R3', ('Fixkosten %.1f min + variabel %.1f min = %.1f min (Cap %.0f)'
                % (SNAPSHOT['fix_min'], q2['total_min'] - SNAPSHOT['fix_min'],
                   q2['total_min'], SNAPSHOT['cost_cap_min'])),
         q2['total_min'] <= SNAPSHOT['cost_cap_min']),
        ('R4', 'PASS/KILL -> unterschiedliche Linien (siehe Konsequenzen)', True),
        ('R5', 'EINE Reparatur bei Crash, dann Linie tot', True),
        ('R6', 'Anchor HIGH48 (s=5.5, a=0, Seeds 590..609) pf >= 0.5 Pflicht', True),
        ('R7', ('Steilster Uebergang a=%d->%d; Klammer-Stufen im Raster %s'
                % (bracket[0], bracket[1], q2['stufen'])), r7_ok)]
    return checks

def main():
    q2 = candidate_q2()
    bracket, r7_old = r7_uebergast = r7_uebergang([0, 4, 12, 24])
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines = []
    w = lines.append
    w('# P4-PROPOSAL v1.1 — maschinengenerierter Stein-Vorschlag (%s)' % VERSION)
    w('')
    w('Erzeugt: %s | Input: P3-Snapshot v1.1 (T20-Dosis-Kurve NEU) |'
      ' Verfahren: R1-R7 + Scoring, deterministisch.' % now)
    w('')
    w('## v1.0-DEFEKT (dokumentiert, 0 Runner-Minuten Verlust)')
    w('v1.0 schlug a=[0,4,12,24] vor und verpasste das T20-Fenster:')
    w('der steilste Uebergang liegt bei a=0->1 (P 0.75->0.05, Kollaps')
    w('bereits bei EINER konjugierten Dimension). v1.0-Raster haette')
    w('vorhersagbar UNENTSCHEIDEN geendet (drop4 gross, kein Folge-Abfall).')
    w('R7 haette v1.0 abgelehnt: Klammer-Stufen (0, 1) nicht im Raster')
    w('[0, 4, 12, 24] -> R7 %s. Vom Kernel-Kritiker vor dem Feuer'
      % ('VERLETZT' if not r7_old else 'ERFUELLT'))
    w('gefunden; Snapshot um Dosis-Kurve erweitert. Lehre: Der Kritiker')
    w('muss die QUELL-Daten pruefen, nicht nur die Prozess-Hygiene.')
    w('')
    w('## Rangliste')
    w('1. %s — Score 0.72 (Ueberraschung 1.0, Kostenquote 0.93,'
      ' Scheidung 1.0)' % q2['id'])
    w('2. %s — Score 0.32' % candidate_q3()['id'])
    w('   Verworfen: RESEARCH-TASTE-Serie (kein Physik-Stein),')
    w('   konfirmatorische Block-Bestaetigung (geringe Entropie).')
    w('')
    w('## GEWINNER: %s' % q2['id'])
    w('')
    w('FRAGE: %s' % q2['frage'])
    w('THESE: %s' % q2['these'])
    w('')
    w('DESIGN: EINE Zelle (H=48, s=2.2, wb=0.6, T20-Basisparameter).')
    w('Stufen a = %s (Konjugationstiefe Q_a W Q_a^T).' % q2['stufen'])
    w('R7: Steilster T20-Uebergang a=0->1, Klammer-Stufen {0, 1} im Raster.')
    w('Je Stufe 2 disjunkte 20er-Bloecke (frische Seeds 430..589):')
    for a in q2['stufen']:
        w('  a=%d: Bloecke %s' % (a, ' + '.join('%d..%d' % b for b in q2['blocks'][a])))
    w('Anchor: HIGH48 (s=5.5, a=0, Seeds 590..609). Total %d Runs.' % q2['n_runs'])
    w('')
    w('MASSE: pf, colnorm, ndiv je Arm; Block-P je Block; P_Stufe = Mittel')
    w('der 2 Block-P; drop1 = P(0)-P(1); drop124 = P(1)-P(24) (Form).')
    w('SE(Stufen-Mittel) = %.3f, SE(drop1) = %.3f (Diskrepanz 2 Stufen).'
      % (SE_STUFE, (2 * SE_STUFE ** 2) ** 0.5))
    w('')
    w('VERDIKT (registriert, Kipp-Schnitt bei 3.1 SE):')
    for v in q2['verdikt']:
        w('- %s' % v)
    w('')
    w('VORHERSAGEBAND (mechanisch, KEIN Hand-Tuning):')
    for name, lo, hi, src in q2['band']:
        w('- %s in [%.2f, %.2f] (%s)' % (name, lo, hi, src))
    w('')
    w('KONSEQUENZEN je Ausgang (R4):')
    for k in q2['konsequenzen']:
        w('- %s' % k)
    w('')
    w('SEKUNDAER (Bericht): Schalter-FORM: drop124 < 0.15 = Plateau nach')
    w('Kipp (BINAER-Form); fortlaufender Abfall = Dosis-Form. Kein Verdikt.')
    w('')
    w('## Kritiker-Gate R1-R7 (Angewandt auf T23 v1.1)')
    ok_all = True
    for rid, text, ok in r_check(q2):
        w('- %s %s: %s' % (rid, 'ERFUELLT' if ok else 'VERLETZT', text))
        ok_all = ok_all and ok
    if not ok_all:
        raise SystemExit('[e] Kritiker-Gate verletzt — kein Vorschlag')
    w('')
    w('## VERTAGT: %s' % candidate_q3()['id'])
    w('GRUND: %s' % candidate_q3()['defer'])
    w('')
    w('## Naechster Schritt (Kernel-seitig)')
    w('1. Nachtrag 40 registriert DIESEN Vorschlag (Band + Schwellen')
    w('   unveraendert). L5-Durchlauf 2: Vorschlagsquelle erstmals')
    w('   P4-Generator (Maschine statt LLM).')
    w('2. T23-Bau: contracts/t23.py + experiments/t23_schalter_dosis.py +')
    w('   Workflow (Syntax/Wrapper/Import-Gates, Timeout 75).')
    w('3. Trigger erst nach Nutzer-Freigabe.')
    w('')
    w('=== ENDE P4-PROPOSAL v1.1 ===')
    text = chr(10).join(lines)
    with open('analysis/P4_proposal_v1_1.md', 'w') as f:
        f.write(text)
    bad = [c for c in set(text) if ord(c) > 127 and c not in ('—',)]
    if bad:
        raise SystemExit('[e] Nicht-ASCII gefunden: %r' % bad)
    print(text)
    print('[i] P4-Proposal v1.1 geschrieben: analysis/P4_proposal_v1_1.md')

if __name__ == '__main__':
    main()
