#!/usr/bin/env bash

set -e

CONFIG=$1
ADDRESS=$2
NET="/tmp/weights.txt"

while true
do
  ./train.py --cfg=$CONFIG --output=$NET | tee $HOME/logs/$(date +%Y%m%d-%H%M%S).log

  # prepare network for uploading
  CHECKSUM=$(sha256sum $NET | awk '{print $1}')
  mv $NET $CHECKSUM
  gzip -9 $CHECKSUM
  FILE="$CHECKSUM.gz"

  # upload in the background and continue next training session
  echo "Uploading '$FILE' to $ADDRESS"
  curl -s -F "file=@${FILE}" -F "training_id=1" -F "layers=6" -F "filters=64" $ADDRESS &
done
