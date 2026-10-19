# Tiefenanalyse X (2026-10-19): KI, die KI erzeugt — und der Durchbruch-Pfad

**Modus:** Runde X der Research-Trajektorie. Frage des Kapitaens:
"Wie wuerde KI KI von Grund auf generieren — und was haben wir,
was andere nicht haben?" Ehrliche Antwort vorweg: Wir haben keine
bessere Gradientenregel und kein groesseres Budget. Wir haben ein
MESSGERAET und einen REGELKREIS, die im Markt niemand benutzt:
das getriebene Benettin-Fenster (lambda x kappa, F12/F18c) und den
Homeostat (F21/F22). Diese Analyse verdrahtet beides mit der
Selbstzeugungs-Literatur zu einem Durchbruch-Pfad.

---

## 1. Die Selbstzeugungs-Literatur — Bestandsaufnahme (kritisch)

### 1.1 Goedel-Maschine (Schmidhuber 2003ff.)
Idee: ein Programm, das jeden Rewrite, fuer den es einen FORMAL
BEWEIS der Nutzensteigerung gibt, sofort anwendet.
- Staerke: optimalitaets-a-la-Kolmogorov, keine Spruenge ins Blindis.
- SCHWACHE (praktisch toedlich): Beweispflicht. In stochastischem ML
  gibt es keine beweisbaren Nutzen-Garantien — Verlustlandschaften,
  Datenverteilungen, Hardware-Rauschen sind alle unregelmaessig.
  Kein Goedel-Rewrite war je auf einem echten Lernproblem gelaufen
  (Stand: nur Toy-Instanzen).
- UNSERE ERSETZUNG: statistische Goedel-Maschine. Der formale Beweis
  wird ersetzt durch einen *Ressourcen-Vertrag*: ein Rewrite/architecture-
  Candidate wird nur zugelassen, wenn der getriebene Betriebspunkt
  (lambda_aufgetrieben, kappa) im Fenster bleibt — gemessen ONLINE
  vom Homeostat. Statt "bewiesen besser" heisst es "veraetzt im
  Ressourcen-Fenster". Das ist schwaecher als ein Beweis, aber
  MESSBAR und SKALIERBAR — und es ist exakt die Garantie-Art, die
  stochastische Systeme ueberhaupt hergeben koennen.

### 1.2 AI-GA (Clune 2019): 3 Saeulen
(a) Meta-Lernen der Architektur, (b) Meta-Lernen des Lernalgorithmus,
(c) Generierung von Lernaufgaben + Umgebungen. Clunes Diagnose: (c)
ist der Engpass — der Suchraum der Probleme ist unterversorgt.
- UNSER ANSATZPUNKT: Wir haben ein Fitness-Signal, das KEIN Task-
  Return braucht: die Ressourcen-Koordinaten (ben, kappa) selbst.
  Open-Endedness-Suche mit Ressourcen-Vertrag als Invariante =
  Saeule (c) wird intern erzeugt: Umgebungen werden so gesampelt,
  dass das System am Rand des Fensters bleibt (maximale
  Transient-Ressource ohne Divergenz).

### 1.3 POET / Open-Endedness (Wang, Lehman, Stanley; osborne-Papers)
Umgebungen und Agent co-evolvieren, minimale Kriterien statt Ziel.
Bekannte Fehlerte: Reward-Hacking der Environment-Generierung,
Schwimmbad-Pluralitaet (viele Umgebungen, keine Tiefe).
- KRITIK AN UNS SELBST: Unser Ressourcen-Fenster kann genauso
  gehackt werden (ein Netz, das nur in der Ecke des Fensters sitzt
  und nichts lernt, waere "im Vertrag"). Deshalb F30: Null-Test,
  der zuerst prueft, ob Ressourcen-Koordinaten ueberhaupt Task-
  Speed vorhersagen (corr >= +0.4 vorab festgenagelt). Ohne F30
  ist der ganze Selbstzeugungspfad eine schoene Geschichte.

### 1.4 Intrinsische Motivation (Curiosity/RND/Empowerment)
Alle belohnen Neuheit/Informationsgewinn im Zustandsraum.
- Verbindung: Neugier ohne Stabilitaets-Anker erzeugt Edge-of-Chaos-
  Drift (unsere E3b-These: Optimierer und Neugier erodieren die
  Ressource). Homeostat = die fehlende Gegenkraft: Neugier-Pull
  (Exploration) + Ressourcen-Vertrag (Anker) = Zoom: F21-Regelkreis
  ist ein Zucker-Reward auf einem stabilen Sockel.

---

## 2. Durchbruch-Synthese: Warum JETZT (Nachrichtenlage Okt 2026)

- Die Frontier-Lobby-Liste Okt 2026 (Gemini 4 Argon, GPT-6 Astra,
  Claude 5.5, Muse Wonder, Mistral Large 4) hat ein gemeinsames
  neues Motiv: STATE ZURUECK (persistent memory: Hy Memory 2.0,
  personal agents bei Meta/Mistral). Der Markt rennt in unsere
  Richtung — alle bauen aber GEDAECHTNIS ALS ARCHITEKTUR (Module,
  Retrieval, Hybrid), keiner als BETRIEBSZUSTAND (getriebene
  Dynamik im Ressourcen-Fenster).
- "Frontier models fail at long-horizon autonomy" + rogue-Modelle:
  die Industrie hat ein Kontroll-Problem und ein Persistenz-Problem.
  Unsere These: beides ist EIN Problem — der Betriebszustand driftet
  aus dem Ressourcen-Fenster (E3b-Antagonist, vierfacher Tod),
  deshalb vergisst und entgleist das System.
- Copy-Liste (funktionierendes klauen): Mamba-Selektivitaet,
  DeepSeek-Effizienz-Rezepte, Personal-Agent-Loop.
- Einzigartiges Asset (nicht kopierbar ohne unser Messgeraet):
  Online-Messung von getriebenem lambda und kappa + Homeostat.
  Mamba/S4 sind diagonal (kappa~1) — die Selektivitaet, die der
  Markt liebt, ist bei ihnen UNMOEGLICH mit Ressourcen-Regime B.
  Das ist F26/F29-Gelaende.

## 3. Der Pfad (5 Stufen, wieder abwaertskompatibel zu Runde VIII)

1. F30 Null-Test: Ressourcen-Koordinaten muessen Task-Speed
   vorhersagen, sonst STOP (Kern-Kill-Kriterium).
2. F24: kappa zahlt bei festem lambda (MC steigt mit kappa).
3. F27/F28: Optimierer als Antagonist (kappa-Drift, targeted vs
   random bei identischer Norm).
4. F21/F22: Homeostat haelt das Fenster unter Training.
5. DARUEBER: statistische Goedel-Maschine — Architektur-Suche +
   Environment-Sampling mit Ressourcen-Vertrag als Zulassungs-
   criterion (AI-GA-Saeule c, POET-Anker). Das waere F31.

## 4. Was NOCH KEINER sieht (Kapitaens-Frage, Teil 2)

Wenn F24 UND F30 positiv sind, gilt: Speicher ist keine Struktur,
sondern eine SCHICHTUNG von Nicht-Normalitaet ueber einem sub-
kritischen getriebenen Betriebspunkt. Dann ist "Gedaechtnis
hinzufuegen" (was Meta/Tencent gerade modulweise bauen) die falsche
Achse — richtig waere: den Betriebspunkt messen und an der
Ressource festklemmen (Homeostat). Kein Frontier-Labor hat ein
 lambda-Messgeraet im Trainingsloop; die Lyapunov-Learning-Schiene
(ICML 2025) will lambda->0 ZIELEN — nach F12 genau falsch herum.
Das ist unser Fenster zur AGI-Luecke: nicht groessere Modelle,
sonsten der 2D-Ressourcen-Vertrag in jeder Trainingsstunde.

## 5. Selbstkritik (Pflichtprogramm)

- Der ganze Pfad haengt an F30. Wenn corr(MC, Task-Speed) < 0.4
  faellt, sind die Ressourcen-Koordinaten nur noch Reservoir- Folklore.
- Die Goedel-Maschinen-Ersetzung ist eine Analogie, keine Theorie.
  Es fehlt ein Satz der Form "Rewrite unter Ressourcen-Vertrag
  konvergiert mindestens so gut wie ..." — bis dahin: Hypothese.
- m-Schaetzer in F25 bleibt subsampling-empfindlich (Wilting/
  Priesemann) — q-Robustheits-Sweep wird eingebaut, bevor F25
  Feuerung ist.
- News-Beweis ist Thin-Slicing: "State kommt zurueck" kann auch
  nur Product-Marketing sein.

**Naechste Schritte:** (1) F25 q-Sweep einbauen (dieser Commit-
Serie), (2) Reset-Day-Feuerreihenfolge auf F30-first umstellen,
(3) F31-Skizze (statistische Goedel-Maschine) nur NACH F30.
