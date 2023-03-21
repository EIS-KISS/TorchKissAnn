#!/bin/sh

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

gnuplot -p -e "dir='$1'" $SCRIPT_DIR/singleaccumplot.p
