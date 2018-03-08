#!/bin/bash

# Add -debug to see engine output
WDR=$HOME/Workspace/chess/lczero-weights
DIR=$PWD/../build
NXT=lc_gen2
CUR=lc_gen1

cutechess-cli -rounds 50 -tournament gauntlet -concurrency 4 \
 -pgnout results.pgn \
 -engine name=$NXT cmd=$DIR/lczero arg="--threads=1" arg="--noise" arg="--weights=$WDR/gen2-64x6.txt" arg="--playouts=800" arg="--noponder" arg="--gpu=0" tc=inf \
 -engine name=$CUR cmd=$DIR/lczero arg="--threads=1" arg="--noise" arg="--weights=$WDR/seed-64x6.txt" arg="--playouts=800" arg="--noponder" arg="--gpu=1" tc=inf \
 -each proto=uci

mv -v results.pgn "$NXT-vs-$CUR.pgn"
