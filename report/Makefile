# make                    figures, tables and relatorio.pdf
# make RESULTS="../results/<job>/results.csv ..."
#
# By default, job 1034793 (Release build; earlier jobs ran -O0 builds).

PYTHON  ?= ../.venv/bin/python
RESULTS ?= ../results/1034793.ada/results.csv

relatorio.pdf: relatorio.md include.lua build/.stamp
	pandoc $< --columns=1000 --lua-filter include.lua --pdf-engine=tectonic -o $@

build/.stamp: ../scripts/report.py $(RESULTS)
	$(PYTHON) ../scripts/report.py $(RESULTS) -o build
	touch $@

clean:
	rm -rf build relatorio.pdf

.PHONY: clean
