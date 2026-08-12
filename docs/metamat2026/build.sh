#!/bin/sh
# build.sh - rebuild the metamat2026 Beamer presentation.
#
# Beamer needs two pdflatex passes: the first writes the .aux/.toc/.nav files
# that drive navigation and the table of contents; the second pulls them back
# in so cross-references and the outline resolve. nonstopmode keeps a missing
# logo or figure from halting the build.

set -u
cd "$(dirname "$0")"

# compile twice for nav/toc; nonstopmode so missing logos don't halt
pdflatex -interaction=nonstopmode -halt-on-error=0 presentation.tex > /tmp/pl1.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error=0 presentation.tex > /tmp/pl2.log 2>&1

echo "-----------------------------------------------"
if [ -f presentation.pdf ]; then
    echo "OK: built presentation.pdf"
else
    echo "ERROR: presentation.pdf not produced - see /tmp/pl2.log"
fi
