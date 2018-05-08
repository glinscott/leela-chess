#!/bin/bash

rm -fr build
mkdir build
pushd build
cmake .. -DCMAKE_BUILD_TYPE=Release
#cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
#cmake .. -DCMAKE_BUILD_TYPE=Debug
make
popd
