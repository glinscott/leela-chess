#!/bin/bash

intexit() {
    # Kill all subprocesses
    kill -HUP -$$
}

trap intexit INT

mkdir data
./lczero --weights=weights.txt --randomize -n -t1 --start="train 1" >training.out &
./lczero --weights=weights.txt --randomize -n -t1 --start="train 2" >training2.out &

wait
