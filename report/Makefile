# make                    figures, tables and relatorio.pdf
# make RESULTS="../results/<job>/results.csv ..."
#
# By default, every results.csv under ../results with the columns of the
# current ada-bench.sh (header starting with "experiment") is used.

PYTHON  ?= ../.venv/bin/python
RESULTS ?= $(shell grep -l '^experiment,' ../results/*/results.csv 2>/dev/null)

relatorio.pdf: relatorio.md include.lua build/.stamp
	pandoc $< --columns=1000 --lua-filter include.lua --pdf-engine=tectonic -o $@

build/.stamp: ../scripts/report.py $(RESULTS)
	$(PYTHON) ../scripts/report.py $(RESULTS) -o build
	touch $@

clean:
	rm -rf build relatorio.pdf

.PHONY: clean
