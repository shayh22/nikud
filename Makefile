PY ?= python3
BOOK ?= book.docx

.PHONY: help test fetch fetch-safe eval-set corpus lexicon quotes baseline report book clean-derived

help:
	@echo "שלב 1: make eval-set && make baseline"
	@echo "שלב 2: make fetch && make corpus          (fetch-safe = בלי CC-BY-NC)"
	@echo "שלב 3: make lexicon && make quotes && make report"
	@echo "שלב 4: $(PY) src/train.py --smoke   ואז ריצה מלאה"
	@echo "שלב 5: make book BOOK=path/to/book.docx"
	@echo "שלב 6: $(PY) src/review.py queue --docx \$$BOOK && $(PY) src/review.py run"

test:
	$(PY) tests/test_engine.py

fetch:
	$(PY) src/fetch_sefaria.py

fetch-safe:
	$(PY) src/fetch_sefaria.py --commercial-safe

eval-set:
	$(PY) src/build_eval.py

corpus:
	$(PY) src/build_corpus.py

lexicon:
	$(PY) src/build_lexicon.py

quotes:
	$(PY) src/quotes.py

# שלב 1 — הבסיס שכל שלב אחריו נמדד מולו.
baseline:
	$(PY) src/evaluate.py --engines null lexicon lexicon+quotes dictabert full \
		--title "שלב 1 — בסיס" --out reports/baseline.md

report:
	$(PY) src/evaluate.py --engines lexicon+quotes full \
		--title "הערכה" --out reports/evaluation.md

book:
	$(PY) pipeline/docx_pipeline.py $(BOOK) out/$(notdir $(BOOK)) \
		--engine full --report reports/book.json

clean-derived:
	rm -rf data/corpus/*.jsonl lexicon/forms.json lexicon/bigrams.json \
	       lexicon/quotes.json lexicon/names.json lexicon/review_queue.json
