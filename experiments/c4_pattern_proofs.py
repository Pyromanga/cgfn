#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C4: Muster-Beweis-Abdeckung -- Jonnys Mustergruppen-These, testbar gemacht.

These: Schach zerfaellt in eine Handvoll endlicher Mustergruppen, auf denen
der Spielwert konstant ist. Wenn Muster + Beweis 100% Abdeckung mit 100%
Praezision erreichen, ist die Klasse OHNE Enumerations-Bruteforce geloest.

Methode: klassische Muster-Theoreme (Key Squares, Opposition/Paritaet,
Turmbauer-Zone, Dominanz) als reine Praedikate gegen EXAKTE Syzygy-WDL-Werte
ueber ALLE legalen Positionen jeder 3-Mann-Klasse. Kein Lernen, kein Fitten.

Vereinfachung (dokumentiert): 50-Zuege-Regel ignoriert; Syzygy ±1 (cursed)
zaehlt als Sieg/Niederlage, wie in der klassischen Theorie.

Fix 2026-11-03 (Run 1-3 waren fake-gruen in 14-21s):
- Mirror-Downloads wie C3 (sesse.net zuerst, lichess 3-4-5-wdl als Fallback,
  Nicht-Leer-Gate pro Datei; KBvK.rtbw ist WIRKLICH nur ~80 Bytes gross).
- Sanity-Gate: jede Klasse muss probe-bar sein, sonst LAUT rot enden statt
  gruenem Fake-Commit ohne Zahlen.
"""
import os
import sys
import datetime
import chess
import chess.syzygy

RESULTS = os.environ.get("C4_RESULTS", "results/c4_pattern_proof_results.txt")
TB_DIR = os.environ.get("C4_TB_DIR", "tablebases")
WIN, DRAW, LOSS = "WIN", "DRAW", "LOSS"
LINES = []


def log(msg):
    print(msg)
    LINES.append(msg)


PTMAP = {
    "KQvK": chess.QUEEN,
    "KRvK": chess.ROOK,
    "KNvK": chess.KNIGHT,
    "KBvK": chess.BISHOP,
    "KPvK": chess.PAWN,
}

MIRRORS = [
    "http://tablebase.sesse.net/syzygy/3-4-5/{name}.rtbw",
    "https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/{name}.rtbw",
]


def write_results():
    with open(RESULTS, "w") as f:
        f.write("\n".join(LINES) + "\n")


def download_tables():
    os.makedirs(TB_DIR, exist_ok=True)
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    names = list(PTMAP.keys())

    def dl(name):
        p = os.path.join(TB_DIR, name + ".rtbw")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return True
        for mirror in MIRRORS:
            url = mirror.replace("{name}", name)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"})
                with urllib.request.urlopen(req, timeout=30) as r, open(p, "wb") as f:
                    while True:
                        chunk = r.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                if os.path.getsize(p) > 0:
                    log("table ok: " + name + " <- " + url + " (" + str(os.path.getsize(p)) + " bytes)")
                    return True
                os.remove(p)
                log("leerer Body von " + url + " — naechster Mirror")
            except Exception as e:
                log("table FAILED: " + name + " <- " + url + " " + str(e))
        return False

    with ThreadPoolExecutor(5) as ex:
        oks = list(ex.map(dl, names))
    return all(oks)


def sanity(tb):
    """Lautes Scheitern statt Fake-Gruen: jede Klasse muss probe-bar sein."""
    import random
    rng = random.Random(0)
    fails = []
    for cls, pt in PTMAP.items():
        ok = 0
        for _ in range(200):
            b = chess.Board(None)
            wk = rng.randrange(64)
            bk = rng.randrange(64)
            ps = rng.randrange(64)
            if wk == bk or chess.square_distance(wk, bk) < 2 or ps in (wk, bk):
                continue
            if pt == chess.PAWN and ps // 8 in (0, 7):
                continue
            b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
            b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
            b.set_piece_at(ps, chess.Piece(pt, chess.WHITE))
            b.turn = chess.WHITE
            try:
                tb.probe_wdl(b)
                ok += 1
            except Exception:
                pass
            if ok >= 5:
                break
        if ok < 1:
            fails.append(cls)
    if fails:
        log("SANITY FEHLGESCHLAGEN fuer " + ", ".join(fails) +
            " — Tablebase defekt/leer. LAUTER ABRRUCH, kein Fake-Commit.")
        return False
    log("Sanity ok: alle 5 Klassen probe-bar.")
    return True


def gen_positions(cls):
    pt = PTMAP[cls]
    for wk in range(64):
        for bk in range(64):
            if wk == bk or chess.square_distance(wk, bk) < 2:
                continue
            if pt == chess.PAWN:
                extra = [chess.square(f, r) for f in range(8) for r in range(1, 7)]
            else:
                extra = [s for s in range(64) if s not in (wk, bk)]
            for ps in extra:
                if ps in (wk, bk):
                    continue
                for turn in (chess.WHITE, chess.BLACK):
                    b = chess.Board(None)
                    b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
                    b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
                    b.set_piece_at(ps, chess.Piece(pt, chess.WHITE))
                    b.turn = turn
                    if b.is_valid():
                        yield b


def preds_kpvk(b):
    pawn = None
    for sq, pc in b.piece_map().items():
        if pc.piece_type == chess.PAWN:
            pawn = sq
    f = chess.square_file(pawn)
    r = chess.square_rank(pawn)
    wk = b.king(chess.WHITE)
    bk = b.king(chess.BLACK)
    out = []
    # P1: Key Squares (File b..g, Bauer Rang 2..5): Koenig 2 Ruecke davor = Gewinn
    if 1 <= f <= 6 and r <= 4:
        keys = [chess.square(f + d, r + 2) for d in (-1, 0, 1)]
        if wk in keys:
            out.append(("key_squares_win", WIN))
    # P2: weisser Koenig direkt vor dem Bauern und Schwarz am Zug = Gewinn
    front = chess.square(f, r + 1)
    if wk == front and b.turn == chess.BLACK:
        out.append(("wk_front_black_to_move_win", WIN))
    # P3: schwarzer Koenig direkt vor dem Bauern, vertikale Opposition = Remis
    if bk == front and r >= 1 and wk == chess.square(f, r - 1):
        out.append(("opposition_draw", DRAW))
    # P4: Turmbauer, schwarzer Koenig in der Ziehzone (Ecke +/- 1) = Remis
    if f in (0, 7):
        corner = chess.square(f, 7)
        if chess.square_distance(bk, corner) <= 1:
            out.append(("rook_pawn_zone_draw", DRAW))
    return out


def preds_dominance_win(b):
    return [("dominance_win", WIN)]


def preds_insufficient_draw(b):
    return [("insufficient_material_draw", DRAW)]


PRED = {"KPvK": preds_kpvk, "KQvK": preds_dominance_win, "KRvK": preds_dominance_win,
        "KNvK": preds_insufficient_draw, "KBvK": preds_insufficient_draw}


def truth_of(b, tb):
    wdl = tb.probe_wdl(b)  # Sicht: Spieler am Zug
    if wdl == 0:
        return DRAW
    val = WIN if wdl > 0 else LOSS
    if b.turn == chess.BLACK:
        val = LOSS if val == WIN else WIN
    return val


def analyze(cls, tb):
    t0 = datetime.datetime.now()
    n = 0
    dist = {WIN: 0, DRAW: 0, LOSS: 0}
    covered = 0
    correct = 0
    conflicts = 0
    per_pred = {}
    uncovered = 0
    uncovered_ex = []
    for b in gen_positions(cls):
        n += 1
        truth = truth_of(b, tb)
        dist[truth] += 1
        preds = PRED[cls](b)
        if not preds:
            uncovered += 1
            if len(uncovered_ex) < 8:
                uncovered_ex.append((b.fen(), truth))
            continue
        covered += 1
        vals = set(v for _, v in preds)
        if len(vals) > 1:
            conflicts += 1
        else:
            if vals.pop() == truth:
                correct += 1
        for name, v in preds:
            d = per_pred.setdefault(name, [0, 0])
            d[0] += 1
            if v == truth:
                d[1] += 1
    dt = (datetime.datetime.now() - t0).total_seconds()
    log("")
    log("=== Klasse " + cls + " ===")
    log("Positionen (legal, beide Zugrechte): " + str(n) + "  [" + str(round(dt, 1)) + "s]")
    log("Wahre Werte: WIN " + str(dist[WIN]) + " / DRAW " + str(dist[DRAW]) + " / LOSS " + str(dist[LOSS]))
    cov = 100.0 * covered / max(n, 1)
    acc_cov = 100.0 * correct / max(covered, 1)
    solved = 100.0 * correct / max(n, 1)
    log("Abdeckung durch Muster: " + str(round(cov, 2)) + "%")
    log("Praezision auf abgedeckten: " + str(round(acc_cov, 2)) + "%")
    log("Konflikte (Muster widersprechen sich): " + str(conflicts))
    log("BEWEISBAR GELÖST (korrekt gedeckt / alle): " + str(round(solved, 2)) + "%")
    log("Unabgedeckt: " + str(uncovered) + " (" + str(round(100.0 * uncovered / max(n, 1), 2)) + "%)")
    for name in per_pred:
        fired, corr = per_pred[name]
        log("  Praedikat " + name + ": gefeuert " + str(fired) + ", Praezision " + str(round(100.0 * corr / max(fired, 1), 2)) + "%")
    for fen, truth in uncovered_ex:
        log("  Beispiel unabgedeckt: " + fen + " -> " + truth)
    return {"cls": cls, "n": n, "coverage": cov, "precision_cov": acc_cov, "solved": solved}


def main():
    log("C4 Muster-Beweis-Abdeckung -- gestartet " + datetime.datetime.utcnow().isoformat() + "Z")
    log("These: Schach = Handvoll endlicher Mustergruppen mit konstantem Wert.")
    log("Test: a-priori Muster-Praedikate vs. exakte Syzygy-Werte, ALLE legalen Positionen.")
    if not download_tables():
        log("ABBRUCH: Tablebase-Download fehlgeschlagen.")
        write_results()
        raise SystemExit(1)
    tb = chess.syzygy.open_tablebase(TB_DIR)
    if not sanity(tb):
        write_results()
        raise SystemExit(1)
    results = []
    for cls in list(PTMAP.keys()):
        try:
            results.append(analyze(cls, tb))
        except Exception as e:
            log("Klasse " + cls + " FEHLER: " + str(e))
            import traceback
            log(traceback.format_exc())
    tb.close()
    if not results:
        log("KEINE Klasse analysiert — LAUT ROT statt Fake-Gruen.")
        write_results()
        raise SystemExit(1)
    log("")
    log("=== Skalierungstabelle (Jonny-These: bleibt die Regelzahl klein?) ===")
    log("Klasse   | n        | Coverage | Praezision | beweisbar geloest")
    for r in results:
        log(r["cls"].ljust(8) + " | " + str(r["n"]).rjust(8) + " | " +
            (str(round(r["coverage"], 1)) + "%").rjust(8) + " | " +
            (str(round(r["precision_cov"], 1)) + "%").rjust(10) + " | " +
            (str(round(r["solved"], 1)) + "%").rjust(6))
    log("")
    log("Leseart: 100% beweisbar geloest = Klasse ohne Bruteforce GELÖST.")
    log("Luecken = dort, wo die klassische Musterlehre unvollstaendig ist;")
    log("sie sind der naive Naechste-Schritt der Muster-Expansion.")
    write_results()
    log("Ergebnis geschrieben: " + RESULTS)


if __name__ == "__main__":
    main()
