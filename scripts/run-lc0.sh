#!/usr/bin/env bash

# Add -debug to see engine output
WDR=/tmp
DIR=$PWD/../build
NXT=ID1
CUR=ID0

cutechess-cli -rounds 100 -tournament gauntlet -concurrency 4 \
 -pgnout results.pgn \
 -engine name=$NXT cmd=lc0 arg="--threads=1" arg="--noise" arg="--weights=$WDR/weights.txt" arg="--backend=cudnn" arg="--temperature=1" arg="--tempdecay-moves=20" \
 -engine name=$CUR cmd=lc0 arg="--threads=1" arg="--noise" arg="--weights=$WDR/weights.txt" arg="--backend=random" arg="--temperature=1" arg="--tempdecay-moves=20" \
 -each nodes=800 proto=uci tc=inf

mv -v results.pgn "$NXT-vs-$CUR.pgn"
