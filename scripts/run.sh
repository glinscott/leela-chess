#!/bin/bash

# Add -debug to see engine output
WDR=$HOME/Workspace/chess/lczero-weights
DIR=$PWD/../build
NXT=lc_gen4
CUR=lc_gen3

cutechess-cli -rounds 100 -tournament gauntlet -concurrency 4 \
 -pgnout results.pgn \
 -engine name=$NXT cmd=$DIR/lczero arg="--threads=1" arg="--noise" arg="--weights=$WDR/gen4-64x6.txt" arg="--playouts=800" arg="--noponder" arg="--gpu=0" \
 -engine name=$CUR cmd=$DIR/lczero arg="--threads=1" arg="--noise" arg="--weights=$WDR/gen3-64x6.txt" arg="--playouts=800" arg="--noponder" arg="--gpu=1" \
 -each proto=uci tc=inf

mv -v results.pgn "$NXT-vs-$CUR.pgn"
