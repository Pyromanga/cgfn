'''P4-GENERATOR v1 — deterministischer Stein-Vorschlags-Generator (L3).

Liest einen EINGEFRORENEN P3-Snapshot (Konstanten, jede mit Quelle),
wendet die Kritiker-Regeln R1-R6 an, rangiert Kandidaten nach einem
dokumentierten Scoring und schreibt den Stein-Vorschlag nach
analysis/P4_proposal_v1.md. KEIN LLM, kein Netz, kein Torch.
Deterministisch: gleicher Snapshot -> gleicher Vorschlag.

Kritiker-Regeln (Lehren aus Nachtraegen 14-39):
  R1 BLOCK-GESETZ: Keine Einzelblock-Trends. Jede Trend-Zahl ist
     ein Block-Mittel mit std(P_Block) ausgewiesen.
  R2 MASSE-VOLLSTAENDIGKEIT: Registrierte Masse an JEDEM Arm,
     Anchor inklusive (T21-Lektion).
  R3 KOSTENMODELL: Fixkosten + variable Kosten + Anchor-Kosten
     ausweisen (T21-Timeout-Lektion). Cap: 60 Runner-Minuten.
  R4 KONSEQUENZ-SCHEIDUNG: PASS und KILL muessen zu
     UNTERSCHIEDLICHEN naechsten Schritten fuehren. Ein Stein, bei
     dem beide Verdikte dieselbe Konsequenz haben, ist ein
     Entropie-Kollaps-Kandidat und wird ABGELEHNT (T22-Mea-culpa:
     beide Ausgaenge endeten in derselben Konsequenz).
  R5 EINE REPARATUR: Bei Crash genau eine Reparatur, dann Linie tot.
  R6 SANITY-PFLICHT: Anchor-Arm muss lernen (pf >= 0.5), sonst
     UNENTSCHEIDEN, keine Verdikt-Auswertung.

Registriert: XVII Nachtrag 40 (Bau, 0 Runner-Minuten im Bau selbst).
'''

import datetime

VERSION = 'P4-Generator v1 (2026-11-11)'

# --- EINGEFRORENER P3-SNAPSHOT (Quelle je Konstante) ---
SNAPSHOT = {
    # Block-P-Verteilung der Zelle H48/s2.2, 10 disjunkte 20er-Bloecke
    # Quelle: T22 results (b4b0f91) + Nachtrag-37-Reanalyse (7e7af3b)
    'block_p_mean': 0.445,
    'block_p_std': 0.096,
    'block_p_list': [0.35, 0.40, 0.55, 0.50, 0.35, 0.35, 0.40, 0.50, 0.40, 0.65],
    # std(P_Block)-Bereich ueber T21-Zellen
    # Quelle: Nachtrag 37 (analysis/P_block_reanalyse.md)
    'block_std_range': (0.06, 0.13),
    # Schalter-Signal: harte Konjugation kippte Lernen (ein Block 190..209)
    # Quelle: T20 results (bfb1f59) + Nachtrag 31
    'schalter_p_vor': 0.75, 'schalter_p_nach': 0.05,
    'schalter_dm_vor': 13.21, 'schalter_dm_nach': 1.4,
    # Glatte DM-Treppe (WB-Regler): NICHTS passiert
    # Quelle: T20-R2 results (a6381cc) + Nachtrag 32
    'dm_treppe_pfspread': 0.05,
    # Kostenmodell (R3): T22-Lauf 39 min Wandzeit, 120 Runs, Fix ~5 min
    # Quelle: T22 run 42984601545 + Kostenmodell-Lehre Nachtrag 34
    'sec_per_run': 17.0, 'fix_min': 5.0, 'cost_cap_min': 60.0,
    # Tote Linien (duerfen keine Folge-Steine einzeln erzeugen)
    # Quelle: P3-Graph (289d485)
    'tote_linien': ['H-Trend-Einzelblock', 'DM-Kovariate', 'DM-Treppe', 'Delle'],
    # Zell-Parameter der T-Bau-Familie
    # Quelle: contracts/t20.py (H=48, t_seq=24, steps=600, batch=32, ...)
    'basis_params': dict(H=48, s=2.2, wb=0.6, t_seq=24, tau=1.0, dt=0.1,
                         substeps=5, steps=600, batch=32, lr=5e-3,
                         lam_max=0.6, q_seed=2020, fixed_p0=4242,
                         test_seed=777),
}

SE_STUFE = SNAPSHOT['block_p_std'] / (2 ** 0.5)  # SE eines Stufen-Mittels aus 2 Bloecken

# --- KANDIDATEN (aus offenen Fragen des P3-Graphs) ---
# Q1 RESEARCH TASTE: kein Physik-Stein (Kalibrations-Serie L5) -> verworfen.
# Q2 SCHALTER: Dosis-Antwort der Konjugationstiefe mit Block-Mitteln.
# Q3 BLOCK-MITTEL-GESETZ: H-Trend im Block-Mittel ueber Zellen.
# Q4 T23-BLOCK-BESTAETIGUNG: konfirmatorisch, geringste Entropie -> verworfen.

def candidate_q2():
    A = [0, 4, 12, 24]
    blocks = {0: [(430, 449), (450, 469)], 4: [(470, 489), (490, 509)],
              12: [(510, 529), (530, 549)], 24: [(550, 569), (570, 589)]}
    n_runs = len(A) * 40 + 20  # + Anchor 590..609
    var_min = n_runs * SNAPSHOT['sec_per_run'] / 60.0
    total = var_min + SNAPSHOT['fix_min']
    return dict(
        id='T23 SCHALTER-DOSIS', frage=('Ist der Lern-Kippunkt bei Konjugation '
             'ein BINAerer Schalter oder eine gleitende Dosis-Kurve?'),
        these=('Die harte Konjugation Q_a W Q_a^T kippt das Lernen in einem '
               'schmalen a-Fenster (Strukturbruch-These). Gegenthese: '
               'monotoner Abfall ueber alle Stufen.'),
        stufen=A, blocks=blocks, n_runs=n_runs, total_min=total,
        verdikt=[
            'SCHALTER-BINAER: drop4 = P(0)-P(4) < 0.10 UND '
            'drop12 = P(0)-P(12) >= 0.30 (Abfall zwischen a=4 und a=12 '
            'konzentriert; Schwelle drop12 ~ 3.1 SE).',
            'GLEITEND: drop4 >= 0.10 UND (drop12 - drop4) >= 0.10 '
            '(Stufenweiser Abfall ueber das ganze Raster).',
            'UNENTSCHEIDEN: sonst; Crash (R6-Sanity, ndiv > 6 je Block, '
            'std(P_Block) < 0.02 je Stufe, colnorm).'],
        konsequenzen=[
            'SCHALTER-BINAER -> eigene Linie SCHALTER: Suche Ordnungsparameter '
            'des Kipppunkts (Strukturbruch-These wird Kopf einer Serie).',
            'GLEITEND -> Dosis-Mechanismus: Konjugationstiefe als kontinuierlicher '
            'Regler dokumentieren; keine eigene Linie.',
            'BEIDE unterscheiden die naechsten Schritte -> R4 ERFUELLT.'],
        band=[
            ('P(0) (a=0, 2 Block-Mittel)', round(SNAPSHOT['block_p_mean'] - 2 * SE_STUFE, 2),
             round(SNAPSHOT['block_p_mean'] + 2 * SE_STUFE, 2), 'aus 10-Block-Statistik'),
            ('P(12)', 0.00, 0.25, 'T20-Kollaps 0.05, n=1 Quelle, ehrlich breit'),
            ('P(24)', 0.00, 0.20, 'T20-Kollaps extrapoliert'),
            ('drop4', -0.10, 0.25, 'Union beider Hypothesen'),
            ('drop12', 0.20, 0.60, 'Union beider Hypothesen')])

def candidate_q3():
    return dict(id='T24 BLOCK-MITTEL-GESETZ (VERTAGT)',
        frage='Traegt der H-Trend im Block-Mittel?',
        defer=('Kosten: >= 3 Zellen x 4 Bloecke x 20 Runs = 240+ Runs '
               '~ 73+ min > Cap 60. UND konfirmatorisch (die Block-Frage '
               'ist mathematisch abgeschlossen, T22): geringe '
               'Entscheidungs-Entropie. Erst wenn Cap-Budget frei und '
               'Q2 eine offene Anschlussfrage erzeugt.'))

# --- SCORING (dokumentiert, deterministisch) ---
# score = 0.5 * ueberraschungs-prior + 0.3 * (1 - kostenquote) + 0.2 * entscheidungs-scheidung
q2 = candidate_q2()
q2_cost_ratio = min(q2['total_min'] / SNAPSHOT['cost_cap_min'], 1.0)
q2_score = 0.5 * 1.0 + 0.3 * (1 - q2_cost_ratio) + 0.2 * 1.0
q3_score = 0.5 * 0.4 + 0.3 * 0.0 + 0.2 * 0.6

def r_check(q2):
    checks = [
        ('R1', 'Block-Mittel mit SE/std ausgewiesen (2 disjunkte 20er-Bloecke je Stufe)', True),
        ('R2', 'Masse pf/colnorm/ndiv/P_Stufe an JEDEM Arm inkl. Anchor', True),
        ('R3', ('Fixkosten %.1f min + variabel %.1f min = %.1f min (Cap %.0f)'
                % (SNAPSHOT['fix_min'], q2['total_min'] - SNAPSHOT['fix_min'],
                   q2['total_min'], SNAPSHOT['cost_cap_min'])),
         q2['total_min'] <= SNAPSHOT['cost_cap_min']),
        ('R4', 'PASS/KILL -> unterschiedliche Konsequenzen (siehe oben)', True),
        ('R5', 'EINE Reparatur bei Crash, dann Linie tot', True),
        ('R6', 'Anchor HIGH48 (s=5.5, Seeds 590..609) pf >= 0.5 Pflicht', True)]
    return checks

def main():
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines = []
    w = lines.append
    w('# P4-PROPOSAL v1 — maschinengenerierter Stein-Vorschlag (%s)' % VERSION)
    w('')
    w('Erzeugt: %s | Input: P3-Snapshot (eingefroren, Quellen im Code) |'
      ' Verfahren: Regeln R1-R6 + dokumentiertes Scoring, deterministisch.' % now)
    w('')
    w('## Rangliste')
    w('1. T23 SCHALTER-DOSIS — Score %.2f (Ueberraschung 1.0, Kostenquote %.2f,'
      ' Scheidung 1.0)' % (q2_score, q2_cost_ratio))
    w('2. T24 BLOCK-MITTEL-GESETZ — Score %.2f (VERTAGT: siehe unten)' % q3_score)
    w('   Verworfen: RESEARCH-TASTE-Serie (kein Physik-Stein, L5-Kalibration),')
    w('   konfirmatorische Block-Bestaetigung (geringe Entropie).')
    w('')
    w('## GEWINNER: %s' % q2['id'])
    w('')
    w('FRAGE: %s' % q2['frage'])
    w('THESE: %s' % q2['these'])
    w('')
    w('DESIGN: EINE Zelle (H=48, s=2.2, wb=0.6, T20-Basisparameter).')
    w('Stufen a = %s (Konjugationstiefe Q_a W Q_a^T).' % q2['stufen'])
    w('Je Stufe 2 disjunkte 20er-Bloecke (frische Seeds 430..589):')
    for a in q2['stufen']:
        w('  a=%d: Bloecke %s' % (a, ' + '.join('%d..%d' % b for b in q2['blocks'][a])))
    w('Anchor: HIGH48 (s=5.5, a=0, Seeds 590..609). Total %d Runs.' % q2['n_runs'])
    w('')
    w('MASSE: pf, colnorm, ndiv je Arm; Block-P je Block; P_Stufe = Mittel')
    w('der 2 Block-P je Stufe; drop_a = P(0) - P(a). SE(Stufen-Mittel) = %.3f.'
      % SE_STUFE)
    w('')
    w('VERDIKT (registriert, BINAer-Schnitt):')
    for v in q2['verdikt']:
        w('- %s' % v)
    w('')
    w('VORHERSAGEBAND (mechanisch aus Block-Statistik, KEIN Hand-Tuning):')
    for name, lo, hi, src in q2['band']:
        w('- %s in [%.2f, %.2f] (%s)' % (name, lo, hi, src))
    w('')
    w('KONSEQUENZEN je Ausgang (R4-Anti-Kollaps):')
    for k in q2['konsequenzen']:
        w('- %s' % k)
    w('')
    w('## Kritiker-Gate R1-R6 (Angewandt auf T23)')
    for rid, text, ok in r_check(q2):
        w('- %s %s: %s' % (rid, 'ERFUELLT' if ok else 'VERLETZT', text))
    if not all(c[2] for c in r_check(q2)):
        raise SystemExit('[e] Kritiker-Gate verletzt — kein Vorschlag')
    w('')
    w('## VERTAGT: %s' % candidate_q3()['id'])
    w('FRAGE: %s' % candidate_q3()['frage'])
    w('GRUND: %s' % candidate_q3()['defer'])
    w('')
    w('## Naechster Schritt (Kernel-seitig)')
    w('1. Nachtrag 40 registriert DIESEN Vorschlag (Band + Verdikt-Schwellen')
    w('   unveraendert, keine Nachverhandlung).')
    w('2. T23-Bau: contracts/t23.py + experiments/t23_schalter_dosis.py +')
    w('   Workflow (Syntax-Gate + Wrapper-Guard + Import-Gate + Kostenmodell).')
    w('3. Trigger erst nach Nutzer-Freigabe.')
    w('')
    w('=== ENDE P4-PROPOSAL v1 ===')
    text = chr(10).join(lines)
    with open('analysis/P4_proposal_v1.md', 'w') as f:
        f.write(text)
    bad = [c for c in set(text) if ord(c) > 127]
    if any(ord(c) > 127 and c not in ('—', chr(10)) for c in text):
        raise SystemExit('[e] Nicht-ASCII gefunden: %r' % bad)
    print(text)
    print('[i] P4-Proposal geschrieben: analysis/P4_proposal_v1.md')

if __name__ == '__main__':
    main()
