#!/bin/sh

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

gnuplot -p -e "datafile='$1'" $SCRIPT_DIR/lossplot.p
