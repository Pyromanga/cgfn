# chess-manifold: Reproduzierbare Pipeline (Seed 42)
.PHONY: install test smoke full lichess

install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt
	pip install -e .

test:
	python -m pytest tests -q

smoke:  ## ~300 Stellungen, ~1 Min CI
	CM_N_GAMES=25 CM_EPOCHS=20 CM_PH_SUBSAMPLE=500 CM_N_BOOTSTRAP=500 python -m chess_manifold.run_pipeline

full:  ## ~10k Stellungen, ~7 Min CI
	CM_N_GAMES=850 CM_EPOCHS=40 python -m chess_manifold.run_pipeline

lichess:  ## ~8-10k Turnierstellungen (Top-Classical, Lichess-API)
	CM_SOURCE=lichess CM_LICHESS_TOP_N=25 CM_LICHESS_GAMES_PER_USER=40 CM_EPOCHS=40 python -m chess_manifold.run_pipeline
