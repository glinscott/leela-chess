#!/bin/bash

mkdir data
./lczero --weights=weights.txt -t2 -n --start=train >training.out
