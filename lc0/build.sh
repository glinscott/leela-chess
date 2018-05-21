#!/usr/bin/env bash

set -e

export CC=gcc CXX=g++

BUILDTYPE=$1

if [ -z "${BUILDTYPE}" ]
then
  BUILDTYPE=debug
fi

BUILDDIR=build/${BUILDTYPE}

rm -fr ${BUILDDIR}
meson ${BUILDDIR} --buildtype ${BUILDTYPE} --prefix ${INSTALL_PREFIX:-/usr/local}

pushd ${BUILDDIR}

if [ -n "${INSTALL_PREFIX}" ]
then
  ninja install
else
  ninja
fi

popd
