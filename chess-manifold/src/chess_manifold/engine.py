"""Stockfish-Wrapper: Evaluation in Centipawns -> Gewinnwahrscheinlichkeit."""
import glob
import math
import os
import shutil
import tarfile
import tempfile
import urllib.request

import chess
import chess.engine

WINPROB_LAMBDA = 0.00368208  # Standard-Sigmoid-Kalibrierung (lichess-artig)
MATE_SCORE = 100000


def cp_to_winprob(cp: float) -> float:
    """Centipawns (aus Weiss-Sicht) -> Gewinnwahrscheinlichkeit Weiss in [0,1]."""
    cp = max(-2000.0, min(2000.0, float(cp)))
    return 0.5 + 0.5 * (2.0 / (1.0 + math.exp(-WINPROB_LAMBDA * cp)) - 1.0)


def find_stockfish():
    """Binary-Suche: env -> PATH -> Download von offiziellen Releases. None = Fallback-Modus."""
    p = os.environ.get("STOCKFISH_PATH")
    if p and os.path.isfile(p):
        return p
    w = shutil.which("stockfish")
    if w:
        return w
    urls = [
        "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64.tar",
        "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar",
    ]
    for url in urls:
        try:
            tmp = tempfile.mkdtemp(prefix="sf_")
            tgz = os.path.join(tmp, "sf.tar")
            urllib.request.urlretrieve(url, tgz)
            with tarfile.open(tgz) as tf:
                tf.extractall(tmp)
            cands = [f for f in glob.glob(os.path.join(tmp, "**", "stockfish*"), recursive=True)
                     if os.path.isfile(f) and os.access(f, os.X_OK)]
            if cands:
                return cands[0]
        except Exception:
            continue
    return None


def heuristic_eval_cp(board: chess.Board) -> float:
    """Material-Fallback, falls keine Engine verfuegbar (wird im Report markiert)."""
    vals = {chess.PAWN: 100, chess.KNIGHT: 305, chess.BISHOP: 325,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
    diff = sum(len(board.pieces(pt, chess.WHITE)) * v for pt, v in vals.items()) - \
           sum(len(board.pieces(pt, chess.BLACK)) * v for pt, v in vals.items())
    return float(diff)


class Engine:
    def __init__(self, path=None, depth=12):
        self.path = path or find_stockfish()
        self.depth = depth
        self._engine = None
        if self.path:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
            self._engine.configure({"Threads": 1, "Hash": 64})

    @property
    def available(self) -> bool:
        return self._engine is not None

    def eval_cp(self, board: chess.Board, depth: int = None) -> float:
        """Weiss-sichtige Bewertung in cp (Matt = +-MATE_SCORE)."""
        depth = depth or self.depth
        if self._engine is None:
            return heuristic_eval_cp(board)
        try:
            info = self._engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].white().score(mate_score=MATE_SCORE)
        except chess.engine.EngineTerminatedError:
            self._engine = None
            return heuristic_eval_cp(board)
        return float(score)

    def close(self):
        if self._engine:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None
