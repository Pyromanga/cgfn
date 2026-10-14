# Tiefenanalyse Runde IX (2026-10-14) — ANGRIFF AUF DIE EIGENEN GRUNDLAGEN

Der Auftrag: 'Wenn die Grundlagen richtig waeren, haetten wir AGI.'
Richtig. Und die erste Anwendung gilt uns selbst. Diese Runde fuehrt
den schaerfen Angriff, den das Repository kennt — gegen die eigenen
Saeulen. Format: Säule -> Angriff -> Ueberlebenstest.

## SAEULE 1: 'MC ist die Ressource' — ANGRIFF: MC misst vielleicht
### nichts, was Task-Performance vorhoraussagt.
- 2607.17858 (Mode-Task-Analyse): nominell hoehere Kapazitaet kann
  in der Praxis SCHLECHTER sein, wenn die task-relevanten Features
  in schwachen Moden liegen. MC ist Moden-agnostisch.
- 2405.06561 (RC-Benchmark-Kritik): NARMA 'should be approached with
  caution'; lineare Readout-Metriken sind fragil.
- Der Jaeger-Vermutung-Befund (Sci Rep 2017): LINEARE Netzwerke sind
  optimal fuer Memory — jede Nichtlinearitaet degradiert MC. Unsere
  kappa-Vergleiche koennen Nichtlinearitaets-Grade vermischen.
- Folgerung: Wenn MC nicht mit echter Task-Adaption korreliert, ist
  die 'Ressource' Ornament. => F30 (implementiert): Grid aus
  (lam, s), pro Konfiguration MC UND (a) nichtlineare Gedraechtnis-
  Readout-Performance, (b) echte Task-Adaptionszeit. KILL: schwache
  Korrelation => Ressourcen-Begriff des Papiers faellt auf 'F12
  misst einen Readout-Artefakt' zurueck.

## SAEULE 2: 'm im Kortex-Fenster' — ANGRIFF: Der Schätzer selbst
### ist unter Subsampling verzerrt — und wir subsamplen.
- Wilting/Priesemann 2018: der konventionelle lineare Regressions-
  Schaetzer (aka C1/C0-Autokovarianz!) ist 'strongly biased under
  subsampling'. DER MR-Schaetzer wurde entwickelt, weil genau unser
  Verfahren verzerrt.
- IOP 2022 'Brain criticality beyond avalanches': drei konkurrierende
  Erklaerungen fuer dieselben Signaturen (superkritisch + Subsampling
  vs subkritisch + Antrieb vs Oszillations-Uebergang) — die Daten
  trennen sie NICHT. Subsampling kann Power-Laws in Lognormals
  verwandeln und Exponenten verstecken.
- UNSERE Schwachstelle konkret: F25 misst an ALLEN Units (kein
  neuronales Subsampling), ABER die 99%-Quantil-Schwellung ist ein
  aehnlicher Filter — m haengt von q ab, und niemand hat gesagt, wie.
  => F25 muss um einen q-Robustheits-Sweep erweitert werden
  (q in {0.95, 0.98, 0.99, 0.995}); m nur als Regime-Koordinate
  akzeptiert, wenn es q-stabil bleibt. Sonst: F25-Ergebnis nur als
  'schwellen-abhaengige Stationsaussage' lesen.

## SAEULE 3: 'Kante/Fenster ist der Rahmen' — ANGRIFF: Vielleicht ist
### intrinsische Dynamik fuer Berechnung schlicht IRRELEVANT.
- 2110.07238: RNNs lernen chaotische Dynamik schwer, aber die
  Loesungsidee (fixed-point/Spektrum-Zwangsbedingungen) 'disables
  multistability and chaos' — die Autoren halten das fuer BEITRAG
  zur Expressivitaet. Umgekehrt: AL-RNN-Linie (2610.01369, Brenner/
  Durstewitz): erstaunlich wenige nichtlineare Units reichen, um
  chaotische Systeme zu rekonstruieren; das meiste ist stueckweise
  linear. Vielleicht leistet 'Dynamik-Regime' viel weniger als
  'Aufgabenstruktur + wenige nichtlineare Knoten'.
- Unser Gegenmittel ist ehrlich: F12/F24 messen MC als FUNKTION des
  Regimes am SELBEN Netz — das ist der einzige Weg, Regime-Relevanz
  gegen Task-Argumente zu verteidigen. Aber: wenn F30 zeigt, dass
  Regime-Effekte auf MC nicht auf Tasks durchschlagen, sind wir
  genau hier tot. Deshalb ist F30 vor F24 zu feuern (25 Min).

## SAEULE 4: 'Kortex-Analogie' — ANGRIFF: Anthropozentrismus.
- Unser F25 sucht m im Fenster 0.94-0.998, weil der Kortex da ist.
  Aber vielleicht ist der Kortex dort aus Gruenden, die mit
  Berechnung nichts zu tun haben (Energie, Gefässversorgung,
  Entwicklungsbiologie). Die eLife-Latent-Variablen-Warnung (Runde V)
  sagt aehnliches: Signaturen ohne Funktion.
- Ueberlebenstest: nur wenn m (oder lambda) im Fenster mit einer
  MESSBAREN Ressource (MC, Adaptionszeit) verkoppelt ist — F25
  misst MC am selben Gitter, gut — darf die Analogie als Motivation
  bleiben, nicht als Beleg.

## SAEULE 5: '32 Neuronen auf CPU' — ANGRIFF: Alles ist klein.
- Reproduzierbarkeits-Lage (Nature-Umfrage: >70% scheitern an
  Reproduktion fremder Experimente): unser Setup ist deterministisch
  (Seeds), das ist unser Fels. ABER Skalierung ist [THESE]: nichts
  in unseren Daten sagt, dass kappa-Effekte bei hidden=512 mit
  trainierten Netzen ueberleben. F30/hidden-Sweep ist die erste
  Sonde; ernsthafte Antwort erst mit F29+ (SSM-Massstab).
- Und: 4-8 Seeds bei Effektgroessen, die oft seed-abhaengig sind —
  die F20b-Lehre (H4 fiel nach 8-Seed-Haertung, p=0.732) ist das
  Warnbeispiel. ALLE neuen Batterie-Ergebnisse muessen per
  Permutationstest gehaertet werden, bevor sie Papier-Saetze werden.

## SAEULE 6 (die Meta-Saeule): 'Wenn die Grundlagen richtig waeren,
### haetten wir AGI' — ANGRIFF: Die Grundlagen der Industrie SIND
### vielleicht richtig — und AGI ist Ingenieursfrage.
- Harte Wahrheit: Backprop + Skalierung + Attention haben Systeme
  gebaut, die die meisten AGI-Definitionen streifen. Vielleicht ist
  'Physik des Betriebszustands' ein schoenes Mosaiksteinchen ohne
  Notwendigkeit. Unser countered claim: Transformer haben KEINE
  intrinsische rekurrente Dynamik — sie sind die perfekte Vermeid-
  ungsstrategie des ganzen Problems, das wir studieren. Wenn
  Weltmodell-Agenten (die ZURUECK zur Rekurrenz muessen, weil
  Interaktion online ist) der naechste Schritt sind, wird unsere
  Physik genau dann relevant, wenn die Industrie auf unser Terrain
  zurueckkehren MUSS. Das ist eine Timing-These, keine Notwendig-
  keits-These. Ehrlich so nennen.

## Das Ergebnis des Angriffs (nicht demuetigieren — schaerfen):
1. F30 VOR F24 feuern (Null-Test der Metrik, 25 Min, killt am
   schnellsten).
2. F25 um q-Robustheits-Sweep erweitern (Schwellen-Artefakt).
3. Papiersprache aendern: 'Ressource' heisst kuenftig 'task-validierte
   Ressource' — erst nach F30.
4. Kortex-Analogie als Motivation deklassifizieren, nicht Beleg.
5. Timing-These (Rekurrenz-Rueckkehr) als THESE markieren.
6. Permutations-Haertung als Pflicht fuer jeden Papier-Satz.

## Quellen (Runde IX — die Gegenfront)
- 2607.17858: mode-task Kapazitaet, 'higher capacity may perform
  worse' (Metrik-Angriff)
- 2405.06561: RC-Benchmark-Kritik, NARMA-Vorsicht
- Sci Rep 2017 (s41598-017-10257-6): Jaeger-Vermutung, Linearitaet
- Wilting/Priesemann 2018 + Frontiers 911245 + IOP ac2071:
  Subsampling-Bias, MR-Schaetzer, 3 konkurrierende Erklaerungen
- 2110.07238: fixed-point-Zwang tötet Multistabilitaet/Chaos
- 2610.01369 / Brenner/Durstewitz: wenige nichtlineare Units genuegen
- Nature-Reproduzierbarkeits-Umfrage (>70% scheitern): unser Setup
  ist seed-deterministisch — der eine Vorteil, den wir verteidigen
  koennen und muessen.
