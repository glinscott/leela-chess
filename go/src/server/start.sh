#!/bin/bash

go build main.go
pkill -f main
nohup ./prod.sh & >server.out
