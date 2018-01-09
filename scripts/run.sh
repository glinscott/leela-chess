#!/bin/bash

# Debugging
#echo '/tmp/core.%e.%p.%t' | sudo tee /proc/sys/kernel/core_pattern
#ulimit -c unlimited

# Add -debug to see engine output
# Missing -repeat and -openings (-t2 gives some randomness...)
./cutechess-cli -debug -rounds 100 -tournament gauntlet -concurrency 1 \
 -pgnout results.pgn \
 -engine name=lc_new cmd=/home/gary/tmp/leela-chess/src/lczero arg="-t2" arg="--weights=/home/gary/tmp/leela-chess/src/newweights.txt" \
 -engine name=lc_base cmd=/home/gary/tmp/leela-chess/src/lczero arg="-t2" arg="--weights=/home/gary/tmp/leela-chess/src/weights.txt" \
 -each proto=uci tc=inf
