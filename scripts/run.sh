#!/bin/bash

# Debugging
#echo '/tmp/core.%e.%p.%t' | sudo tee /proc/sys/kernel/core_pattern
#ulimit -c unlimited

# Add -debug to see engine output
# Missing -repeat and -openings (-n gives some randomness)
DIR=$PWD/../src
./cutechess-cli -rounds 100 -tournament gauntlet -concurrency 1 \
 -pgnout results.pgn \
 -engine name=lc_new cmd=$DIR/lczero arg="-n" arg="-t1" arg="--weights=$DIR/newweights.txt" \
 -engine name=lc_base cmd=$DIR/lczero arg="-n" arg="-t1" arg="--weights=$DIR/weights.txt" \
 -each proto=uci tc=inf
