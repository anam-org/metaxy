.PHONY: poster assets clean distclean

INKSCAPE ?= inkscape
TECTONIC ?= tectonic

SVGS := $(wildcard assets/*.svg)
PDFS := $(SVGS:.svg=.svg.pdf)

poster: poster.pdf

poster.pdf: poster.tex assets/example.py $(PDFS)
	$(TECTONIC) poster.tex

assets/%.svg.pdf: assets/%.svg
	$(INKSCAPE) "$<" --export-type=pdf --export-filename="$@"

assets: $(PDFS)

clean:
	rm -f poster.pdf

distclean: clean
	rm -f assets/*.svg.pdf
