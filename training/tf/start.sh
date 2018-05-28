#!/usr/bin/env bash

set -e

CONFIG=$1
GAMES=$2

NET="/tmp/weights.txt"
NETDIR="/home2/networks/upload"
GAMEFILE="$HOME/.lc0.dat"

if [ ! -f "$GAMEFILE" ]
then
  echo "File $GAMEFILE must contain a single number, exiting now!"
  exit 1
fi

game_num=$(cat $GAMEFILE)
game_num=$((game_num + GAMES))
echo "Starting with training.$game_num.gz as last game in window"

while true
do
  if [ -f "/home/folkert/data/run1/training.${game_num}.gz" ]
  then
    echo ""
    unbuffer ./train.py --cfg=$CONFIG --output=$NET 2>&1 | tee /home2/logs/$(date +%Y%m%d-%H%M%S).log
    TSTAMP=$(date +"%Y_%m%d_%H%M_%S_%3N")
    FILE="$NETDIR/$TSTAMP.lc0"
    mv -v $NET $FILE
    echo $game_num > $GAMEFILE
    game_num=$((game_num + GAMES))
    echo "waiting for training.$game_num.gz"
  else
    echo -n "."
    sleep 60
  fi
done
