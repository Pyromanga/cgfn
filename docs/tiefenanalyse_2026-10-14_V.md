# Tiefenanalyse Runde V (2026-10-14) — Messmethodik & Struktur-Fragen

Runde V liefert (a) die exakte Messvorschrift fuer F25 (Beggs-Plenz +
Clauset-Statistik + Quasi-Kritikalitaets-Lese), (b) die erste
Struktur-These fuer F26 (strukturierte statt unstrukturter
Nicht-Normalitaet).

## 1. Beggs & Plenz 2003: die Original-Messvorschrift
- Branching-Parameter sigma = Verhaeltnis Aktivitaet in Bin t+1 zu
  Bin t innerhalb einer Lawine; kritisch bei sigma = 1.
- Lawine = Kaskade aufeinanderfolgender Bins mit >= 1 aktivem Kanal;
  size = Summe der Events, duration = Bin-Zahl.
- Kortex-Konsens (Review, Runde III): m = 0.94-0.998, NICHT 1 —
  das System ist 'reverberating', subkritisch. Exponent size ~ -3/2
  stammt aus Eurich et al. 2002 (gekoppelte Schwellenelemente) und
  dem kritischen Verzweigungsprozess (Zapperi 1995, nu = 3/2).
- F25 setzt das 1:1 um: per-Unit-Schwelle (99%-Quantile, MEA-Analogon),
  Lawinen = Bin-Binde mit >0 Events, m via Autokovarianz (C1/C0,
  Priesemann-Standard) UND Beggs-Plenz-Bin-Ratio — zwei Schaetzer,
  ein Konsens-Check.

## 2. Clauset/Shalizi/Newman 2009: Power-Law-Statistik ohne Suennde
- Exponenten NIEMALS per least-squares-Log-Log-Fit, sondern per
  diskretem MLE: alpha = 1 + n / sum(ln(x/(xmin-0.5))).
- xmin per KS-Statistik-Minimierung; Plausibilitaet gegen Exponential/
  Lognormal per Likelihood-Ratio (PLOS ONE 2011 bestatigt das fuer
  Neuronen-Lawinen: ML ~ -1.5, KS-Test akzeptiert Power Law, LR-Test
  verwirft Exponential).
- F25 implementiert MLE+KS (LR-Gegenmodelle kommen in F25b, wenn
  die Exponenten im Spiel sind).

## 3. Quasi-Kritikalitaet: getriebene Systeme liegen auf einer LINIE
Nanowire-Perspektive (IOP): stim-getriebene Netzwerke liegen nicht
auf einem kritischen Punkt, sondern entlang einer (tauS, tauT)-Linie,
die die crackling-Skalierungsrelation (tauT-1)/(tauS-1) ~ gamma nur
NAHE erfuellt. F25 misst gamma doppelt: (i) als Regressionssteigung
log<S>(T) vs log T, (ii) als (tauT-1)/(tauS-1) — Uebereinstimmung =
Quasi-Kritikalitaets-Signatur (kein exakter Punkt, aber Skalierung).

## 4. Die grosse Warnung (eLife: 'Neural criticality from effective
latent variables'): Lawinen-Kritikalitaet entsteht auch als BEIPRODUKT
- Populationen, getrieben von wenigen quasi-stationaeren latenten
  Variablen, zeigen approximative Power-Laws OHNE Feintuning (Priesemann
  & Shriki 2018: getunte inhomogene Poisson-Summen).
- Heisst: m im Kortex-Fenster ist NOTWENDIG, nicht HINREICHEND als
  'Ressourcen-Beleg'.
- F25-Konsequenz (schon implementiert): m wird ueber einen lam-Sweep
  gemessen. Nur MONOTONE m(lam)-Wanderung plus MC-Optimum im/rechts
  vom Fenster trennt 'echte Regime-Koordinate' von 'Beiprodukt der
  Rausch-Anteilskorrelation'. Das ist der Unterschied zwischen
  'wir haben eine Zahl repliziert' und 'wir haben ein Gesetz'.

## 5. F26-These: STRUKTURIERTE Nicht-Normalitaet zahlt mehr
- Orhan/Pitkow 2019 (Volltext via Runde IV): trainierte orthogonale
  RNNs entwickeln versteckte KETTEN-artige feedforward Strukturen;
  Ganguli-Linie: maximale Fisher-Memory braucht eine versteckte
  feedforward-Architektur, die superlinear amplifiziert, mit input-
  Konnektivitaet, die zur Architektur PASST.
- Baggio et al. 2020 (Science Advances): Nicht-Normalitaet = effektive
  feedforward-Pfade; identifizierbar ueber die Schur-Zerlegung — die
  streng-obere Dreiecksmatrix Delta enthaelt die 'hidden chains'.
- Non-reciprocal Networks (arXiv:2507.19127): lokale nicht-normale
  Dynamik erlaubt Rausch-Exkursionen, STRUKTURELLE Asymmetrie
  entscheidet, ob sie synchron (Feedback) oder sequenziell
  (Feedforward) propagieren — zwei verschiedene Ressourcen.
- F26-DESIGN (noch nicht implementiert, braucht F24-Freigabe):
  Vergleiche bei festem kappa und festem lambda: unstrukturierte
  Schur-Stoerungen vs. BAND-STruktur (Einzel-Kette: Delta obere
  Diagonale, Ordnung festgelegt) vs. mehrere kurze Ketten.
  Vorhersage: bei gleichem kappa gewinnt die strukturierte Variante,
  weil die Kette das Amplifikationsprofil in die Zeitrichtung legt
  (sequentielle Abtastung der Vergangenheit — Ganguli-SNR-Argument).
  Das waere der Uebergang von 'Nicht-Normalitaet ist gut' zu
  'SO BAUT MAN SIE'.

## 6. Kleiner Anschluss: Pinning/Sequenz-Generation (Rajavarman?)
'Neural sequences emerge from minimally structured circuitry'
(Cell/Neuron): Sequenzen koennen aus grob unstrukturierter
Rekurrenz + Input-Kooperation entstehen statt aus synfire chains.
- Vorsicht fuer F26: strukturierte Kette muss gegen 'unstrukturiert
  + Training' antreten, nicht nur gegen 'unstrukturiert fix'. Sonst
  messen wir Architektur gegen Lernen, nicht Struktur gegen Rauschen.

## Netto-Erkenntnis Runde V
F25 ist jetzt die schaerfte Version seiner selbst: zwei m-Schaetzer,
MLE/KS-Exponenten, doppelt gemessenes gamma, lam-Sweep gegen die
Beiprodukt-Falle, hidden-Sweep fuer die Systemgrosse-Vorhersage,
MC an jedem Gitterpunkt. Und dahinter steht mit F26 erstmals eine
BAU-Frage: nicht OB Nicht-Normalitaet zahlt, sondern WELCHE FORM.

## Quellen (Runde V)
- Beggs & Plenz 2003 (J Neurosci 23:11167): sigma-Schaetzer, -3/2
- Eurich, Hermann, Ernst 2002: alpha=1.5 Vorhersage
- Clauset, Shalizi, Newman 2009 (SIAM Review): MLE/KS/LR-Framework
- PLOS ONE 2011 (e0019779): statistische Unterstuetzung fuer
  Power-Laws in Neuronen-Lawinen (ML ~ -1.5)
- IOP Nanoscale-Netzwerke (2632-072X/ac3ad3): Quasi-Kritikalitaet,
  (tauS,tauT)-Linie, Skalierungsrelation
- eLife reviewed-preprint 89337: Latent-Variable-Kritikalitaet ohne
  Feintuning (Priesemann & Shriki 2018)
- Orhan & Pitkow 2019: versteckte Ketten in trainierten RNNs
- Baggio, Rutten, Hennequin, Zampieri 2020: Schur-Delta als
  hidden-chain-Detektor
- arXiv:2507.19127: nicht-reziproke Netzwerke — zwei Formen der
  Nicht-Normalitaet (lokal vs strukturell)
