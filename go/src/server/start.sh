#!/bin/bash

pkill -f main
nohup ./prod.sh & >server.out
