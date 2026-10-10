.PHONY: install dev test lint serve index search chat eval-set eval docker-up docker-down demo demo-data clean

PY ?= python

install:            ## full install (models download on first use)
	$(PY) -m pip install -r requirements.txt

dev:                ## lightweight install for tests / CI
	$(PY) -m pip install -r requirements-ci.txt

test:
	$(PY) -m pytest -q

lint:
	ruff check voicevault scripts finetune tests

serve:              ## web app on http://localhost:8000
	uvicorn voicevault.server.app:app --host 0.0.0.0 --port 8000 --reload

index:              ## ingest + index everything in data/raw
	$(PY) scripts/index.py data/raw -v

search:
	$(PY) scripts/search.py

chat:
	$(PY) scripts/chat.py

eval-set:           ## draft an eval set from your index (review it afterwards!)
	$(PY) scripts/make_eval_set.py -n 40 --out eval/eval_set.jsonl

eval:               ## retrieval ablation + LLM-judge answer quality + latency + feedback
	$(PY) scripts/evaluate.py all --eval-set eval/eval_set.jsonl

OFFLINE = EMBED_BACKEND=hash LLM_PROVIDER=extractive STT_PROVIDER=fake TTS_PROVIDER=fake \
          VISION_PROVIDER=none INDEX_DIR=data/demo_index

demo-data:          ## build the sample library into the offline demo index
	$(OFFLINE) $(PY) scripts/make_demo_data.py --index --no-vision

demo:               ## offline demo web app: no API keys, no model downloads
	$(OFFLINE) uvicorn voicevault.server.app:app --port 8000

docker-up:
	docker compose up --build -d && echo "→ http://localhost:8000"

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache data/processed/* data/index data/demo_index
	find . -name __pycache__ -prune -exec rm -rf {} +
