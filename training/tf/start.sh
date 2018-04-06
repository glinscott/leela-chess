#!/usr/bin/env bash

set -e

NET="/tmp/weights.txt"
CONFIG="$(pwd)/configs/test.yaml"
ADDRESS="http://162.217.248.187/upload_network"

while true
do
  ./train.py --cfg=$CONFIG --output=$NET

  # prepare network for uploading
  CHECKSUM=$(sha256sum $NET | awk '{print $1}')
  mv $NET $CHECKSUM
  gzip -9 $CHECKSUM
  FILE="$CHECKSUM.gz"

  # upload in the background and continue next training session
  echo "Uploading '$FILE'"
  curl -s -F "file=@${FILE}" -F "training_id=9" -F "layers=6" -F "filters=64" $ADDRESS &
done
