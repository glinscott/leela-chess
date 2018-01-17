#!/bin/bash

# Debugging
#echo '/tmp/core.%e.%p.%t' | sudo tee /proc/sys/kernel/core_pattern
#ulimit -c unlimited

# Add -debug to see engine output
# Missing -repeat and -openings (-t2 gives some randomness...)
#./cutechess-cli -rounds 100 -tournament gauntlet -concurrency 1 \
./cutechess-cli \
 -engine name=supervised cmd=/home/fhuizing/Workspace/leela-chess/build/lczero arg="-t2" arg="--weights=/home/fhuizing/Workspace/leela-chess/scripts/newweights.txt" \
 -engine name=random cmd=/home/fhuizing/Workspace/leela-chess/build/lczero arg="-t2" arg="--weights=/home/fhuizing/Workspace/leela-chess/scripts/weights.txt" \
 -each proto=uci tc=inf \
 -rounds 100 \
 -tournament gauntlet \
 -concurrency 2 \
 -pgnout results.pgn
