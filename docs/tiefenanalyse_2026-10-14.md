# Tiefenanalyse II — Theorie-Ursprünge & Querverbindungen (2026-10-14, abends)

Vertiefung auf vier Wurzeln: Mean-Field-Theorie, Echo-State/Reservoir, biologische Homöostase, moderne SSMs. Ergebnis: DREI direkte Treffer zwischen Literatur und unseren Messdaten, und eine neue testbare Vorhersage (F24-Kandidat).

---

## 1. Der Ursprung: Sompolinsky-Crisanti-Sommers 1988
"Chaos in Random Neural Networks" ([PRL 61, 259](https://link.aps.org/doi/10.1103/PhysRevLett.61.259)):
Mean-Field-Theorie, exakt im N->inf — Übergang stationär->chaotisch bei kritischem Gain g, maximaler Lyapunov-Exponent analytisch.
**Querverbindung:** Unsere designed CT-RNNs sind buchstäblich dieses Modell (rate neurons, random asymmetric W, tanh). Unser lam-Sweep in F12/F20 IST ein g-Sweep. Alles, was wir messen, hat hier seinen theoretischen Anker. Zitationsfundament des Papers.

Neu 2026: "Predictable Mean-Field Chaos" ([arXiv 2606.08805](https://arxiv.org/html/2606.08805)): Krylov-Wachstumsrate alpha >= lambda_L — eine ZWEITE Rate neben dem Lyapunov-Exponent. Interessant für unsere F18-Kette: Wir testeten omega (widerlegt) — Krylov-alpha ist der nächste Kandidat nach dem gleichen Muster (fit-frei testbar).

## 2. DER GROSSE TREFFER: Getriebene Netzwerke & "lokal expandierend, asymptotisch stabil"
"Transition to chaos and signal response in driven random neural networks" (Wainrib & Galtier, [JuSER](https://juser.fz-juelich.de/record/828621)):
- Zeitvariabler Input SUPPRIMIERT die langsamen internen Dynamiken am Chaos-Übergang
- **Memory-Kapazität PEAKT in einem Regime, in dem die Dynamik lokal expandierend, aber asymptotisch stabil ist**

**Querverbindung zu UNSEREN Daten — der Durchbruch der Lektüre:**
Das ist EXAKT unser Regime B aus F18c: kappa>=6 -> transientes Wachstum (lokal expandierend, T0-Defizit, Kreiss-Amplifikation) bei negativem getriebenem Benettin (asymptotisch stabil). Die Literatur sagt: **dieses Regime ist keine Pathologie, sondern die optimale Speicher-Resource.**

KONSEQUENZ (neue testbare Vorhersage, F24-KANDIDAT):
Wenn lokal-expandierend-asymptotisch-stabil optimal für Memory ist, dann sollte bei FESTEM lambda (z.B. -0.3) die MC mit kappa ZUNEHMEN (transientes Wachstum als Ressource), nicht abnehmen. Unser kappa-Cap (F16/F21-Arm B) könnte Memory VERHINDERN statt schützen! F21 beantwortet das implizit (Arm C hat keinen Cap, Arm B schon) — aber F24 macht es direkt: MC gegen kappa bei festem lambda, designed Gitter.

## 3. Echo-State & Lyapunov: unsere Mess-Philosophie hat Namen
- "A local Echo State Property through the largest Lyapunov exponent" (Galtier & Wainrib, [Neural Networks 2016](https://www.sciencedirect.com/science/article/abs/pii/S0893608015002828)): Mean-Field-Theorie GETRIEBENER Netzwerke, LLE berechenbar, lokale ESP. Das ist genau unser Messgerät — methodisch etablierter Boden; unsere F11-Artefakt-Kritik bleibt der neue Teil.
- Reservoir-Literatur ([ESP-Übersicht](https://www.emergentmind.com/topics/echo-state-property-esp), [IOP-Constraints](https://iopscience.iop.org/article/10.1088/2632-2153/aca1f6)): LLE MIT Input ist die richtige Größe (Input wirkt kontraktiv auf tanh-Reservoirs) — bestätigt F14: GETRIEBEN ist die korrekte Messung.
- Spin-Eis-Reservoirs ([Sci Rep 2025](https://www.nature.com/articles/s41598-025-93189-w)): MC-Änderung drastisch nahe der Grenze negativ/Null = "edge of ESP". SPANNUNG zu unserem F12 (MC-Optimum bei -0.33, nicht 0): beide nahe, aber unterhalb Null; unser Optimum ist klar subkritisch mit Abfall zur Kante hin — genau die Nuance, die Edge-of-Chaos-Rhetorik übersieht. Ehrlich im Paper diskutieren.

## 4. Biologische Homöostase: unser F21 hat 30 Jahre Neuro-Prezedenz
- Synaptic Scaling (Turrigiano; [Übersicht](https://en.wikipedia.org/wiki/Synaptic_scaling)): multiplicative globale Skalierung, hält relative Gewichte, reguliert Firing-Rate. Das biologische Vorbild unseres Regelkreises: GLOBALE Lageregelung ohne lokale Information zu zerstören.
- Homeostatic Scaling of Excitability (HSE, [PLOS CompBio 2012](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1002494)): verhindert Quieszenz UND Sättigung; **funktioniert solange Feedback-Erregung nicht stärker als Feedback-Hemmung** — KONKRETE Struktur-Vorhersage für unseren Homeostat: bei stark rekurrenter Erregung (unsere lam=1.2/1.4-Designs!) kann ein globaler Regler scheitern. F21-Auswertung: Versagen dort ist HSE-erwartet, kein Bug.
- PNAS 2020 ([Link](https://www.pnas.org/doi/10.1073/pnas.1918368117)): synaptic scaling stellt Korrelationen/Struktur wieder her, intrinsic plasticity stellt Raten wieder her — ZWEI getrennte homöostatische Kanäle. Analogie: unser lambda-Regler (Rate) + möglicher kappa-Regler (Struktur) — F22/F23-Kandidat für einen ZWEI-KANAL-Homeostaten.

## 5. Mamba/SSM: die moderne Architektur ist ein impliziter Lyapunov-Homeostat
- "Mamba SSMs Are Lyapunov-Stable Learners" ([arXiv 2406.00209](https://arxiv.org/pdf/2406.00209)): Beweis: Mamba-Block-Rekurrenz ist Lyapunov-stabil => robustes Fine-Tuning vs. Transformer (die divergieren). Die Industrie baut bereits implizit Lyapunov-stabile Homeostaten — ohne es so zu nennen. Unser Projekt liefert die Mess-Sprache und Trainings-Dynamik-Ansicht dazu.
- MemMamba ([arXiv 2510.03279](https://arxiv.org/html/2510.03279v1)): Stabilität != Memory! Mambas Zustands-Update ist stabil, aber Frühinformation verfällt exponentiell — Langzeit-Memory bleibt das Problem. Echo unserer F12-Resource-Kurve: Stabilität ist nötig, aber zu viel Stabilität tötet die Resource. Der Tradeoff, den wir in 32 Units vermessen, begrenzt SSM-Skalierung.
- NeurIPS 2024 ([Feature Learning in SSMs](https://proceedings.neurips.cc/paper_files/paper/2024/file/9c7eeda2dc98e61baa9a5884afd231bc-Paper-Conference.pdf)): Stabilität bei Init via spektraler Skalierung; Mamba-GitHub warnt selbst vor fp16-Instabilität der Rekurrenz ([state-spaces/mamba](https://github.com/state-spaces/mamba)) — die rekurrente Dynamik ist das empfindliche Organ. Genau dort misst unser Benettin-Sensor.

---

## SYN THESE — Die Karte der Querverbindungen

  Sompolinsky 1988 (Theorie-Anker)
        |
  g-Sweep = unser lam-Sweep (F12/F20)
        |
  Wainrib-Galtier: getrieben, lokal expandierend + asymptotisch stabil
        |          == unser Regime B (F18c, kappa>=6)
        |          == SPEICHER-RESOURCE, nicht Pathologie
        |
  F24-KANDIDAT: MC vs kappa bei festem lambda
  (Wenn MC mit kappa steigt: transientes Wachstum ist Memory-Ressource;
   unser kappa-Cap braucht ein oberes Zusatzfenster, kein hartes Cap)
        |
  Homeostat (F21/F22)  <---> Turrigiano/HSE/PNAS:
        globale Lageregelung, 2 Kanäle (rate + struktur),
        Vorhersage: Regler versagt bei starker Rekurrenz-Erregung
        |
  Mamba = industrieller impliziter Homeostat (Lyapunov-stabil),
  aber Stabilität != Memory (MemMamba) == unsere Resource-Kurve
        |
  Krylov-alpha (neu 2026) = nächster fit-freier Prädiktions-Kandidat
  nach omega-Widerlegung (F18e)

## Neue Experiment-Konsequenzen (Reset-Day-Plan ergänzt)
1. **F24 (NEU): Transientes Wachstum als Memory-Ressource** — MC gegen kappa bei festem lambda=-0.3, designed Gitter (s-Werte sweepen, lam fest). Vorhersage aus Wainrib-Galtier: MC steigt mit kappa. Wenn JA: kappa-Cap-Design ändern (Fenster statt Deckel), F12-Optimum tiefer erklärbar (Optimum = Tradeoff Expansion vs. Kontraktion, nicht einfach subkritisch).
2. F21-Auswertung um zwei Blickwinkel erweitern: (a) C vs B prüft indirekt F24 (Regler ohne Cap gegen Cap), (b) Homeostat-Fehlschlag bei lam=1.2/1.4-Zellen ist HSE-erwartet, kein Bug.
3. F22/F23: Zwei-Kanal-Homeostat (lambda + kappa) als biologisch motivierte Erweiterung vormerken.
4. Krylov-alpha als F18f-Kandidat parken (fit-frei, wie omega, aber mit Prädiktions-Bedeutung statt Wachstums-Obergrenze).
