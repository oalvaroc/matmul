# make                    figures, tables and relatorio.pdf
# make JOB=../results/<job>
#
# Job 1035570: Release build (-O3), 5 repetitions of every case, first one checked.

PYTHON ?= $(abspath ../.venv/bin/python)
JOB    ?= ../results/1035570.ada

relatorio.pdf: relatorio.md include.lua build/.analysis build/fig_esquema.pdf
	pandoc $< --columns=1000 --lua-filter include.lua --pdf-engine=tectonic -o $@

build/.analysis: analysis.py $(JOB)/results.csv
	@mkdir -p build
	$(PYTHON) analysis.py $(JOB) -o build > build/analysis.log
	touch $@

build/fig_esquema.pdf: diagrama.py
	@mkdir -p build
	$(PYTHON) diagrama.py $@

clean:
	rm -rf build relatorio.pdf

.PHONY: clean
