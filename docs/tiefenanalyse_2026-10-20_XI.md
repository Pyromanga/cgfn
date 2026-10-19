# Tiefenanalyse XI (2026-10-20): Die Ratte im Schwanz — vier Zirkularitaeten und ihr mathematischer Schnitt

**Auftrag des Kapitaens:** systemtheoretisch/mathematisch herangehen,
finden, wo sich die aktuellen Methoden die Ratte in den Schwanz beissen,
und das mathematisch/programmatisch korrekt aufloesen. Test/Validierung
sobald Kontingent da ist (F31, implementiert, NICHT getriggert).

Die vier Ratten — ehrlich, an unserem eigenen Stack:

---

## R1: Die offene Schleife (Linearisierung ohne Kovarianz-Schluss)

**Objekte.** CT-RNN: dz/dt = -z/tau + tanh(z) W^T + U u(t).
Tangente: J(z) = -I/tau + W diag(sech^2(z)). Das Feld-Modul
(predict_driven_ftle_field) setzt d = <sech^2(z)> aus EINER generierten
Trajektorie ein und liest lambda aus dem maximalen Eigenwert von
J(d) = -I/tau + W diag(d).

**Die Ratte.** Die Trajektorie, aus der d stammt, wird von derselben
Dynamik erzeugt, die die Linearisierung beschreiben soll — die Schleife
wird nie geschlossen: d ist Funktion der Zustandsverteilung, die
Zustandsverteilung ist Funktion der (linearisierten) Dynamik, die von d
abhaengt. Der Homeostat (F21/F22) reguliert dann lambda_field — eine
Koordinate, deren Abweichung von der wahren Stabilitaet (Kokkyklus-
Benettin-FTLE) von genau der Nicht-Normalitaet erzeugt wird, die die
Ressource sein soll. Regelkreis, der das falsche Signal zurueckkoppelt.

**Nebenbefund (Symptom, nicht Bug):** sech2_field_iteration iteriert tot —
der Rollout steht ausserhalb der Schleife; jede Iteration rechnet dasselbe
d_new. Die Schleife war nie geschlossen, nicht einmal syntaktisch.

**Der Schnitt: DMFT-Fixpunkt.** Schliesse
    d = m(Sigma_11..Sigma_HH),   d_i = E_{z~N(0,Sigma_ii)}[sech^2(z)]  (Quadratur)
    J(d) Sigma + Sigma J(d)^T = -q_eff B B^T,   q_eff = sigma_u^2 * Delta
Iteriere bis Konvergenz. Kontraktion solange das System im linearen
Regime bleibt (dm/ds^2 beschraenkt, sech^2 ist Lipschitz-1 nach unten
geklammert); am supra-linearen Rand existiert KEIN stabiler Fixpunkt —
das ist selbst eine Aussage: dort gibt es keinen stationaeren
Betriebspunkt, den der Homeostat halten koennte. Das ist die
Sompolinsky-Closure (Crisanti/Sompolinsky 1988-Stil) fuer das getriebene
subkritische Regime. Implementiert: resource.dmft_fixed_point.

**F31-P1 (Schluss-d):** d_dmft vs d_emp (rollout-gemessen). Im linearen
Regime sollten sie zusammenfallen; die relative Abweichung wachst mit
Antriebsstaerke — sie ist unser Nichtlinearitaets-Meter.
**F31-P2 (Schluss-lambda):** |lambda_dmft - lambda_benettin| vs
|lambda_open - lambda_benettin| uebers Gitter. VORAB genagelt: DMFT
gewinnt im Mittel. FALLS NEIN: die Selbstkonsistenz ist nicht der
Engpass — d_emp enthaelt Information aus jenseits-Mean-Field-
Fluktuationen, und die DMFT-Geschichte ist falsch herum zu lesen. Beide
Ausgaenge sind Erkenntnis.

## R2: Der kappa-Zoo — falsch etikettierte Invarianten

field.py nennt sv_max/sv_min - 1 eine 'Kreiss constant'. Das ist eine
Konditionszahl. Die vier ehrlichen Invarianten (resource.kappa_zoo):
    kappa_cond = sv_max/sv_min          (Kondition)
    kappa_T    = cond(V)               (Eigenvektor-Kondition: DIE
                                        Nicht-Normalitaet diagonalisierbarer
                                        Matrizen)
    delta_H    = sqrt(||J||_F^2 - sum|lam_i|^2)   (Henrici)
    K_Kreiss   = sup_s Re(s)|| (sI-J)^{-1} ||    (Resolvente; wir messen
                 die reelle-Achsen-Untergrenze)
Satz (Trefethen-Embree): K_Kreiss <= g <= e*n*K_Kreiss, wobei
g = sup_t ||exp(J t)|| das direkt messbare transiente Wachstum ist.

**Zweiter Fund:** J = -I/tau + W diag(d) ist AUCH bei symmetrischem W
nicht-normal (W diag(d) ist nicht symmetrisch fuer nicht-konstantes d).
Die Ressource ist ein (W, d)-Eigentum — der Antrieb ko-konstruiert die
Nicht-Normalitaet. Fenster-Koordinate kappa ist NICHT kappa(W).
F31-P4 misst, welches der vier Invarianten log E und log g am besten
vorhersagt.

## R3: Die Vermischung — (lambda, kappa) sind Projektionen eines Skalars

**Kernsatz.** Fuer die getriebene Linearisierung
    dz/dt = J z + B u,  u weiss (PSD q_eff),
ist der Zustand stationaer mit Kovarianz Sigma, geloest aus
    J Sigma + Sigma J^T = -q_eff B B^T.
Tr(Sigma) ist EIN observables Skalar, das BEIDE Fenster-Achsen exakt
traegt: fuer normale Systeme (diag(lambda_i)) reduziert
    Tr Sigma_norm = sum_i q_i / (-2 Re lambda_i)
auf die lambda-Achse; der Exzess
    E := Tr Sigma / Tr Sigma_norm
IST die kappa-Achse. Messbar: Sigma_emp aus rollierender Kovarianz
(O(H^2) pro Schritt, kein QR, keine Tangenten-Integration) —
Homeostat-v3-Zielgroesse ist Tr(Sigma), nicht ein geschaetztes lambda.

**Ehrlichkeits-Falle (selbst gefunden, self-kill):** die in der
Reservoir-Literatur behauptete Identitaet 'MC = Tr(Sigma)/sigma^2' ist
FALSCH. Skalar-Gegenbeispiel: z_{t+1} = a z_t + b u_t gibt
MC = sum_k corr^2 = (1-a^2) sum a^{2k-2} = 1 fuers ALLE b, a —
waehrend Tr Sigma/sigma^2 = b^2/(1-a^2). MC ist MAHALANOBIS-gewichtet
(optimaler Readout pro Lag: corr2_max(k) = sigma^2 b^T A^{(k-1)T}
Sigma^{-1} A^{k-1} b), Tr Sigma/sigma^2 ist die H2-Rauschverstaerkung.
Zwei verschiedene exakte Funktionale derselben (J, B). Beide exakt
implementiert (resource.mc_linear_prediction). Konsequenz:
F31-P3 (corr(MC_theorie, MC_gemessen) >= 0.8) ist ein KONSISTENZ-Test
des linearen Regimes, KEINE Evidenz fuer die Ressourcen-These. Die
These lebt und stirbt mit F30 (Transfer auf Trainings-Tasks).

**SSM-Schaerfung:** fuer Diagonalsysteme (Mamba/S4-Familie) ist
E identisch 1 — der kappa-Anteil der Ressource ist dort STRUKTURELL
null. F26-These, jetzt als Exzess-Identitaet messbar scharf.

## R4: Der Optimierer als geschlossener Kreis (EoS)

Die Edge-of-Stability-Literatur behandelt Sharpness (lambda_max H) als
Eigentum der Verlustlandschaft. Ratte: lambda_max(H) organisiert sich
unter dem Optimierer SELBST zur Stabilitaetskante (2/eta-Regel) — es ist
ein Fixpunkt des Update-Operators, nicht der Landschaft. Der Heavy-Ball-
Kompagnon
    M_i = [[1+beta-eta h_i, -beta], [1, 0]]   (pro Modus)
ist nicht-normal, wird defekt bei (1+beta-eta h_i)^2 = 4 beta — und
genau an der Kantengeometrie (|lambda| -> 1) ist der pro-Modus-Defekt
maximal. Der messbare Tatbestand ist der Update-Kokkyklus:
    g_opt = sup_t ||Delta_t|| / ||Delta_0||   (Zwillings-Proben)
resource.update_cocycle_growth powert den Kompagnon analytisch (billig);
F31-P5 misst g_opt(beta) — VORHERSAGE: monoton in beta. E3b-Schaerfung:
der Antagonist ist nicht das Hessian, sondern der Kompagnon-Kokkyklus.
F27/F28 (kappa(W)-Drift, targeted vs random) laufen unveraendert — F31-P5
legt die Messlatte dafuer, was 'richtig' waere (g_opt statt kappa(W)).

---

## F31 — die Test-Batterie (implementiert, Marker NICHT gesetzt)

experiments/f31_dmft_closure.py + src/cgfn/resource.py. Vorab genagelt:
  P1: d_dmft ≈ d_emp im linearen Regime; d_gap wachst mit Antrieb/kappa.
  P2: |lam_dmft - lam_ben| < |lam_open - lam_ben| im Mittel. Kill bei
      Gegenteil = DMFT-Geschichte falsch herum (s.o.).
  P3: corr(MC_opt_linear, MC_gemessen) >= 0.8; Tr(Sigma)-Schluss:
      corr(h2_emp, h2_dmft) hoch.
  P4: E wachst mit s; log E korreliert am besten mit dem kappa-Mass,
      das die Transient-Theorie nennt (Erwartung: kappa_T/Kreiss vor
      kappa_cond).
  P5: g_opt monoton in beta (Heavy-Ball-Kompagnon).
Laufzeit ~30-40 Min (Gitter 4 lam x 3 s x 2 Seeds, H=32, Rollout
T=2500 + Benettin + DMFT-Iterationen, alles CPU-torch).

## Was NICHT geloest ist (Pflicht-Selbstkritik)

- DMFT ist Gaussian-Mean-Field: Korrelationen vernachlaessigt — der
  Fehler geht als d_gap in P1 ins Ergebnis, ist aber nicht separierbar
  von RK2-Diskretisierungsrauschen und dem U-Bias (fixe Verschiebung des
  Arbeitspunkts, in der Linearisierung ignoriert).
- Der Kokkyklus-Benettin misst die wahre getriebene Stabilitaet entlang
  EINER Trajektorie; die DMFT macht eine Aussage ueber die ENSEMBLE-
  Verteilung. Beide koennen recht haben und trotzdem differieren — P2
  misst die Differenz, erklaert sie nicht.
- R4 bleibt ein Demo am quadratischen Verlust; echte EoS-Selbst-
  organisation braucht nichtlineare Netze. F31-P5 ist die Messlatte,
  nicht der Beweis.
- Die(news-Lage)-Synthese aus Runde X bleibt duenn belegt; hier hat sich
  nichts geaendert.

## Feuerplan-Update

F31 rutsckt DIREKT hinter F30 (Plan aktualisiert): erst der Null-Test,
dann das Messgeraet, mit dem F24/F25/F26/F27/F28 zu lesen sind.

**Kernsatz der Runde:** Die Ressource ist kein 2D-Fenster, sondern ein
observables Skalar — Tr(Sigma) — mit einem geschlossenen Modellsatz
(DMFT) und einer normalen Referenz (E = 1). Wer Tr(Sigma) halten kann,
haelt lambda und kappa gleichzeitig. Das ist der Homeostat-Vertrag 2.0.
