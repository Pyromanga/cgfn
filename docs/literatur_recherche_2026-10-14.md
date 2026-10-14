# Literatur-Recherche — 2026-10-14 (Kontingent-Pause, Recherche-Tag)

## Der wichtigste Fund: Lyapunov Learning (ICML 2025)
**"Lyapunov Learning at the Onset of Chaos"** ([arXiv 2506.12810](https://arxiv.org/html/2506.12810v1), [OpenReview](https://openreview.net/forum?id=DM3q9t8I0o))
- Regularisiert den max. Lyapunov-Exponenten waehrend des Trainings: L = L_MSE + alpha*|lambda| — Ziel: lambda -> 0 (Edge of Chaos).
- Anwendung: Regime-Shifts in nicht-stationaeren Zeitreihen (Lorenz-Parameterwechsel!), Continual Learning.
- Sie argumentieren: SSMs/Mamba, orthogonale RNN-Init, Jacobian-Norm-Penalties sind alle implizite "Lyapunov-zentrierte" Mechanismen.
**Unsere Position dagegen (aus eigener Evidenz, F12/F20b/F20):**
- Wir messen BENETTIN GETRIEBEN (artefaktfrei) — ihr lambda wird aus Sequenz-Spektren geschaetzt, und die Edge-of-Chaos-Literatur misst teils den SVD-Artefakt (F11/F13).
- Unsere gehaertete Ressourcen-Kurve (F12: MC-Optimum bei ben ~ -0.33) sagt: **subkritisch, nicht an der Kante**. F20b haertet: kein Gradienten-Pull zur Kante, Start-abhaengige Traegheit.
- => TESTBARER WETTKAMPF: Homeostat-Target -0.3 (unsere These) vs Target 0 (ihre These) vs frei. F21 hat jetzt einen C0-Arm.
- Zusaetzlich: "weakly chaotic regime kann gut performen" ([PMC11118502](https://pmc.ncbi.nlm.nih.gov/articles/PMC11118502/)) — die Kante ist umstritten; wir liefern die artefaktfreie Messseite.

## Methodik-Fundament (bestaetigt unser Vorgehen)
- **Engelken et al., "Lyapunov spectra of chaotic RNNs"** ([PRR 2023](https://www.columbia.edu/cu/neurotheory/Larry/EngelkenPRR23.pdf)): orthonormale Basis entlang der Trajektorie mit analytischen Jacobis = genau unsere Benettin-QR-Implementierung; trainierte Netze akkumulieren nahe-null Exponenten.
- **Frontiers 2022, "On Lyapunov Exponents for RNNs"** ([Frontiers](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2022.818799/full)): LE aus getriebenen versteckten Zustaenden = unser getriebenes Messgeraet.
- RNNs sind nicht-autonom — "Chaos"/Kante fuer getriebene Systeme heikel ([PMC8389338](https://pmc.ncbi.nlm.nih.gov/articles/PMC8389338/) fuer ESNs). Unsere getriebene Benettin-Messung ist die saubere Antwort.

## Kritik an Edge-of-Chaos (unterstuetzt F20b)
- **Avalanche- und Edge-of-Chaos-Kritikalitaet koexistieren nicht noetig** ([AIP Chaos 2017](https://pubs.aip.org/aip/cha/article/27/4/047408/322534)): lambda1 in allen drei Regimen positiv — "Kritikalitaet" ist messgeraet-abhaengig.
- **Transition to chaos = rich/lazy learning shift** ([PMC11118502](https://pmc.ncbi.nlm.nih.gov/articles/PMC11118502/)): Kante markiert Lern-Regime-Wechsel, nicht Optimalitaet — konsistent mit subkritischem Ressourcen-Optimum.

## Nicht-Normalitaet (unser kappa-Faden hat direkte Anschlussliteratur!)
- **"Non-normal spectral signatures of instability in NN training dynamics"** ([arXiv 2605.23476](https://arxiv.org/html/2605.23476v1)): kappa(V) als Fruehwarn-Indikator fuer transiente Amplifikation BEVOR der Spektralradius 1 ueberschreitet — die Optimierungs-Seite unseres F16-kappa-Drifts.
- **Science Advances 2019, "Non-normal networks"** ([Science](https://www.science.org/doi/10.1126/sciadv.aau9403)): reale Netzwerke nicht-normal, numerische Abszisse >> spektrale — sie nuetzen omega als Reaktivitaetsmass (wir widerlegten omega als FTLE-Praediktor in F18e; die Spannung ist diskussionswert).
- **nnRNN (NeurIPS)** ([Paper](http://papers.neurips.cc/paper/9513-non-normal-recurrent-neural-network-nnrnn-learning-long-time-dependencies-while-improving-expressivity-with-transient-dynamics.pdf)): nicht-normale Konnektivitaet als FEATURE (transiente Dynamik loest PTB) — Regime B ist zugleich Ausdrucksmacht; kappa-Cap darf nicht zu hart sein.
- **Coding with transient trajectories** ([PLOS CompBio](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1007655)): transiente Amplifikation als Berechnungs-Ressource — Trefethen-Rahmen fuer unser F18-Zwei-Regime-Gesetz.

## Continual Learning Landschaft (F23 einordnen)
- Uebersicht: van de Ven et al. ([arXiv 2403.05175](https://arxiv.org/abs/2403.05175)) — 3 Familien: Regularisierung (EWC), Replay, dynamische Architekturen. UNSER Homeostat ist KEINE davon: rein dynamisches Regularisierungsprinzip ohne Gewichtsdistanz, ohne Replay, ohne Wachstum. Das ist die Luecke.
- Titans (Google) & CH-HNN ([Nature Comms](https://www.nature.com/articles/s41467-025-56405-9)) gehen architekturell ran — wir gehen dynamisch ran: keine zusaetzliche Hardware, nur ein inneres Stabilitaetsziel.

## Konsequenzen (direkt eingebaut)
1. **F21 erhaelt C0-Arm** (Target 0 = ICML-These) gegen C (Target -0.3 = unsere These).
2. **F23-Metric**: Lorenz-Regime-Shift ist ihr Hauptbenchmark — wir messen beides.
3. **Paper-Positionierung**: artefaktfrei liegt das Optimum subkritisch; Kanten-Homeostase (ICML) vs subkritische Homeostase (wir) — wer auf MC/Transfer gewinnt, gewinnt die Designfrage. F21 entscheidet.
4. kappa(V)-Fruehwarn-Paper verbindet F16-Drift mit Training-Instabilitaet — Zitationsziel.
