#!/bin/bash

rm -fr build
mkdir build
pushd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
popd
