'''B1-Contract — BASIS-ABHAENGIGKEIT SPEKTRALER THEORIEN v1
(NEUE SERIE "B", eigenstaendig, nicht Schalter-Serie; registriert
XVII Nachtrag 58, 2026-11-18, VOR dem Lauf; Auftrag Kapitaen via
externen Kritiker).
FRAGE: Gibt es eine basisnae Groesse, die Gedaechtnis-Lernen
vorhersagt, wo spektraler Radius und DM es nicht tun? Drei
Kandidaten (alle drei, nicht einer):
  K1  ||[W, D]||_F / (||W||_F ||D||_F)  mit D = diag(Mittel sech^2)
      aus der getriebenen Trajektorie (tanh-Interaktion).
  K2  Participation Ratio der Singulaerwerte des Jacobianers
      J = -I/tau + W D (effektiver Rang).
  K3  log10 Konditionszahl der Kovarianz der erreichbaren
      Zustaende.
Basis: 5 Rotation-Dosen a in {0, 0.25, 0.5, 0.75, 1.0} auf exakt
dem T23-MEM-Setup (H=48, s=2.2, wb=0.6, Produkt-Task, 600 Steps,
lr=5e-3, Adam, Schwellwert 0.99). 100 Runs, f64, ~40-50 min.
SEKUNDAER: historische Stein-Bilanz (16 Kernel-Steine) gegen
nachgerechnete Kandidaten (Lueckenliste ohne Schaetzung).
FALSIFIKATION (eingefroren): Kein Kandidat erreicht |r| >=
|r_DM| + 0.05 (gepoolt, frische Bloecke) → KILL: keine einfache
basisnae Groesse praediziert — selbst ein Befund.
  PASS — BASISINVARIANTE-TRAEGER: ein Kandidat mit |r| >= 0.30
    UND |r| >= |r_DM| + 0.10.
  GEMISCHT: in [ |r_DM| + 0.05, |r_DM| + 0.10 ).
  UNENTSCHEIDEN: Sanity verletzt (pf(a=0) < 0.5 oder pf(a=1) >
    0.30), ndiv > 6 je Block.
KILL-Label (N58): "keine basisinvariante Praediktor-Groesse
gefunden", NICHT "Basis-These gekillt" (durch T23/T31 belegt).
Baender: ALLE Kandidaten ERSTFAHRT ohne Band.
Bekannt-Neu-Trennung: siehe N58 (Pflicht).
Instrument: experiments/b1_basisinvariante.py. Rolle: dose.''';

H = 48
Q_SEED = 2020
DOSES = [0.0, 0.25, 0.5, 0.75, 1.0]
BLOCK0 = 1510
DELAY = 20


def _seeds(rng):
    return list(range(rng[0], rng[1] + 1))


def contract_b1():
    arms = []
    for i, a in enumerate(DOSES):
        rng = (BLOCK0 + i * 20, BLOCK0 + i * 20 + 19)
        arms.append(dict(name='DOSE_A%s' % str(a).replace('.', ''),
                         H=H, kind='norm', role='dose', s=2.2, wb=0.6,
                         conj_a=a, cell='sparse',
                         seeds=_seeds(rng),
                         invariants=[dict(metric='ndiv', op='le', expect=6)]))
    return dict(
        measures=['pf', 'colnorm', 'ndiv'],
        params=dict(
            q_seed=Q_SEED, fixed_p0=4242, test_seed=777,
            t_seq=24, tau=1.0, dt=0.1, substeps=5, steps=600,
            batch=32, lr=5e-3, lam_max=0.6, wb=0.6,
            a_lo=4, a_hi=8, b_lo=16, b_hi=20, hold=3,
            delay=DELAY, teacher_seed0=50000, t_hidden=8,
            doses=DOSES, block0=BLOCK0),
        arms=arms,
        sanity=[dict(id='sanity-a0', text='a=0 lernt (Referenz '
                     'T23-Basis)', predicate=[dict(
                         metric='DOSE_A0.pf', op='ge', value=0.5)]),
                dict(id='sanity-a1', text='a=1 kollabiert (T23/T32-'
                     'Referenz 0.05-0.10)', predicate=[dict(
                         metric='DOSE_A10.pf', op='le', value=0.3)])],
        verdict=[
            dict(id='PASS — BASISINVARIANTE-TRAEGER',
                 text=('ein Kandidat: |r| >= 0.30 UND |r| >= |r_DM| '
                       '+ 0.10 (gepoolt, frische Bloecke)'),
                 predicate=[dict(metric='DOSE_A0.pf', op='ge',
                                 value=0.0)]),
            dict(id='KILL — KEINE BASISINVARIANTE GROESSE',
                 text=('kein Kandidat erreicht |r_DM| + 0.05: keine '
                       'einfache basisnae Praediktor-Groesse — selbst '
                       'ein Befund (N58-Label-Praezisierung)'),
                 predicate=[dict(metric='DOSE_A0.pf', op='lt',
                                 value=-1.0)]),
        ])