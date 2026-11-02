.PHONY: install test smoke full clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	python -m pytest tests -q

# Smoke: ~300 Stellungen, wenige Epochen (~3-5 min Runner-Zeit)
smoke:
	CM_N_GAMES=25 CM_EPOCHS=20 CM_PH_SUBSAMPLE=500 CM_N_BOOTSTRAP=500 python -m chess_manifold.run_pipeline

# Full: ~10.000 Stellungen (Empfehlung aus dem Aufgabenpapier)
full:
	CM_N_GAMES=850 CM_EPOCHS=40 CM_PH_SUBSAMPLE=1200 CM_N_BOOTSTRAP=2000 python -m chess_manifold.run_pipeline

clean:
	rm -rf results/* data/*.csv
