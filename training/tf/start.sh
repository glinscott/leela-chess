#!/usr/bin/env bash

set -e

CONFIG=$1
ADDRESS=$2
NET="/tmp/weights.txt"
NETDIR="/home2/networks/upload"

while true
do
  unbuffer ./train.py --cfg=$CONFIG --output=$NET 2>&1 | tee /home2/logs/$(date +%Y%m%d-%H%M%S).log

  # prepare network for uploading
  CHECKSUM=$(sha256sum $NET | awk '{print $1}')
  mv -v $NET $NETDIR/$CHECKSUM
  gzip -9 $NETDIR/$CHECKSUM
  FILE="$NETDIR/$CHECKSUM.gz"

  # upload in the background and continue next training session
  echo "Uploading '$FILE' to $ADDRESS"
  curl -s -F "file=@${FILE}" -F "training_id=1" -F "layers=10" -F "filters=128" $ADDRESS &
done
