#!/usr/bin/env bash

set -e

BUILDTYPE=$1

if [ -z ${BUILDTYPE} ]
then
  BUILDTYPE=release
fi

rm -fr build
meson build --buildtype ${BUILDTYPE} --prefix ${INSTALL_PREFIX:-/usr/local}

pushd build

if [ -n "${INSTALL_PREFIX}" ]
then
  ninja install
else
  ninja
fi

popd
