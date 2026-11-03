#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C4: Muster-Beweis-Abdeckung -- Jonnys Mustergruppen-These, testbar gemacht.

These: Schach zerfaellt in eine Handvoll endlicher Mustergruppen, auf denen
der Spielwert konstant ist. Wenn Muster + Beweis 100% Abdeckung mit 100%
Praezision erreichen, ist die Klasse OHNE Enumerations-Bruteforce geloest.

Methode: klassische Muster-Theoreme (Key Squares, Opposition/Paritaet,
Turmbauer-Zone, Dominanz) werden als reine Praedikate codiert und gegen
EXAKTE Syzygy-WDL-Werte ueber ALLE legalen Positionen jeder 3-Mann-Klasse
getestet. Kein Lernen, kein Fitten: a-priori-Regeln.

Vereinfachung (dokumentiert): 50-Zuege-Regel ignoriert; Syzygy ±1 (cursed)
zaehlt als Sieg/Niederlage, wie in der klassischen Theorie.
"""
import os
import datetime
import chess
import chess.syzygy

RESULTS = "results/c4_pattern_proof_results.txt"
TB_DIR = "results/tb345"
WIN, DRAW, LOSS = "WIN", "DRAW", "LOSS"
LINES = []


def log(msg):
    print(msg)
    LINES.append(msg)


def download_tables():
    os.makedirs(TB_DIR, exist_ok=True)
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    names = ["KQvK", "KRvK", "KPvK", "KNvK", "KBvK"]

    def dl(name):
        p = os.path.join(TB_DIR, name + ".rtbw")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return True
        url = "https://tablebase.lichess.ovh/tables/standard/3-4-5/" + name + ".rtbw"
        try:
            with urllib.request.urlopen(url, timeout=30) as r, open(p, "wb") as f:
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
            log("table ok: " + name + " (" + str(os.path.getsize(p)) + " bytes)")
            return True
        except Exception as e:
            log("table FAILED: " + name + " " + str(e))
            return False

    with ThreadPoolExecutor(5) as ex:
        oks = list(ex.map(dl, names))
    return all(oks)


PTMAP = {"KQvK": chess.QUEEN, "KRvK": chess.ROOK, "KNvK": chess.KNIGHT,
         "KBvK": chess.BISHOP, "KPvK": chess.PAWN}


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
        log("  Praedikat " + name + ": ge feuert " + str(fired) + ", Praezision " + str(round(100.0 * corr / max(fired, 1), 2)) + "%")
    for fen, truth in uncovered_ex:
        log("  Beispiel unabgedeckt: " + fen + " -> " + truth)
    return {"cls": cls, "n": n, "coverage": cov, "precision_cov": acc_cov, "solved": solved}


def main():
    log("C4 Muster-Beweis-Abdeckung -- gestartet " + datetime.datetime.utcnow().isoformat() + "Z")
    log("These: Schach = Handvoll endlicher Mustergruppen mit konstantem Wert.")
    log("Test: a-priori Muster-Praedikate vs. exakte Syzygy-Werte, ALLE legalen Positionen.")
    if not download_tables():
        log("ABBRUCH: Tablebase-Download fehlgeschlagen.")
        raise SystemExit(1)
    tb = chess.syzygy.open_tablebase(TB_DIR)
    results = []
    for cls in ["KQvK", "KRvK", "KPvK", "KNvK", "KBvK"]:
        try:
            results.append(analyze(cls, tb))
        except Exception as e:
            log("Klasse " + cls + " FEHLER: " + str(e))
    tb.close()
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
    log("sie sind der naive Naechste-Baueme-Schritt der Muster-Expansion.")
    with open(RESULTS, "w") as f:
        f.write("\n".join(LINES) + "\n")
    log("Ergebnis geschrieben: " + RESULTS)


if __name__ == "__main__":
    main()
