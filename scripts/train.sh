#!/bin/bash

mkdir data
./lczero --weights=weights.txt --randomize -n --start=train >training.out
