#!/bin/bash

# Add -debug to see engine output
WDR=$HOME/Workspace/chess/lczero-weights
DIR=$PWD/../build
NXT=ID55
CUR=ID45

cutechess-cli -rounds 100 -tournament gauntlet -concurrency 4 \
 -pgnout results.pgn \
 -engine name=$NXT cmd=$DIR/lczero-new arg="--threads=1" arg="--noise" arg="--weights=$WDR/ID55" arg="--playouts=800" arg="--noponder" arg="--gpu=0" \
 -engine name=$CUR cmd=$DIR/lczero arg="--threads=1" arg="--noise" arg="--weights=$WDR/ID45" arg="--playouts=800" arg="--noponder" arg="--gpu=1" \
 -each proto=uci tc=inf

mv -v results.pgn "$NXT-vs-$CUR.pgn"
