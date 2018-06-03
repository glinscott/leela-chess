#!/bin/bash
set -e
cwd="`pwd`"

# parse command line arguments

shared=false
cuda=false

for key in "$@"; do
    case $key in
        --shared)
        shared=true
        ;;
        --cuda)
        cuda=true
        ;;
    esac
done

# add repository with recent versions of compilers
apt-get -y update
apt-get -y install software-properties-common
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get -y clean

# install requirements
apt-get -y update
apt-get -y install \
  build-essential \
  curl \
  clang \
  git \
  cmake \
  unzip \
  autoconf \
  autogen \
  libtool \
  mlocate \
  zlib1g-dev \
  g++-6 \
  python \
  python3-numpy \
  python3-dev \
  python3-pip \
  python3-wheel \
  unzip \
  wget

if $shared; then
    apt-get -y update
    apt-get -y install openjdk-8-jdk
    # Running bazel inside a `docker build` command causes trouble, cf:
    #   https://github.com/bazelbuild/bazel/issues/134
    # The easiest solution is to set up a bazelrc file forcing --batch.
    echo "startup --batch" >>/etc/bazel.bazelrc
    # Similarly, we need to workaround sandboxing issues:
    #   https://github.com/bazelbuild/bazel/issues/418
    echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
	>>/etc/bazel.bazelrc
    # Install the most recent bazel release.
    export BAZEL_VERSION=0.11.0
    mkdir bazel
    cd bazel
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE
    chmod +x bazel-*.sh
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
    cd "$cwd"
    rm -f bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
    export PATH="$PATH:/usr/local/bin"
fi
if $cuda; then
    # install libcupti
    apt-get -y install cuda-command-line-tools-9-1
fi

apt-get -y clean

# when building TF with Intel MKL support, `locate` database needs to exist
updatedb

### build and install tensorflow_cc ###

cd "$cwd"
mkdir tensorflow_cc/tensorflow_cc/build
cd tensorflow_cc/tensorflow_cc/build
# configure only shared or only static library
if $shared; then
    cmake -DTENSORFLOW_STATIC=OFF -DTENSORFLOW_SHARED=ON ..;
else
    cmake ..;
fi
make && make install
cd "$cwd"
rm -rf tensorflow_cc/tensorflow_cc/build

### cleanup ###

cd "$cwd"
rm -rf ~/.cache
