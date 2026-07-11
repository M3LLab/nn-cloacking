#!/bin/sh
# build.sh - rebuild the Wave Motion paper with a correct bibliography.
#
# Citations are compiled by BibTeX, not pdflatex. Running pdflatex alone reuses
# a stale (or emptied) .bbl, so every [n] renders as [?]. This script runs the
# full chain via latexmk - the project's original builder (see .fdb_latexmk) -
# doing a CLEAN build so a broken .bbl can never be reused. Missing figures are
# made non-fatal so the bibliography always resolves.

set -u
cd "$(dirname "$0")"
JOB=elsarticle-template-num

if command -v latexmk >/dev/null 2>&1; then
    latexmk -C "$JOB" >/dev/null 2>&1 || true              # nuke stale artifacts (incl. empty .bbl)
    latexmk -pdf -bibtex -interaction=nonstopmode -f "$JOB".tex || true
else
    echo "latexmk not found; using manual pdflatex/bibtex chain."
    rm -f "$JOB".aux "$JOB".bbl "$JOB".blg                 # never reuse a broken .bbl
    pdflatex -interaction=nonstopmode "$JOB" || true       # writes .aux (\bibdata, \citation)
    bibtex   "$JOB"                          || true       # .aux -> .bbl
    pdflatex -interaction=nonstopmode "$JOB" || true       # pull in .bbl
    pdflatex -interaction=nonstopmode "$JOB" || true       # settle [n] numbering
fi

undef=$(grep -c "Citation.*undefined" "$JOB".log 2>/dev/null)
undef=${undef:-0}
echo "-----------------------------------------------"
if [ "$undef" -eq 0 ]; then
    echo "OK: all citations resolved -> $JOB.pdf"
else
    echo "WARNING: $undef undefined citation(s) remain - see $JOB.log"
fi
