# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
EN_SOURCEDIR  = en
EN_BUILDDIR   = _build/en

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

html-en:
	@$(SPHINXBUILD) -b html "$(EN_SOURCEDIR)" "$(EN_BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "English documentation built in $(EN_BUILDDIR)/"

html-all: html html-en
	@echo "Both Chinese and English documentation built."

clean-en:
	@rm -rf "$(EN_BUILDDIR)"

clean-all: clean clean-en
	@echo "Both Chinese and English build directories cleaned."

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
