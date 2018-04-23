FROM ubuntu:16.04

# Install
RUN apt-get -qq update
RUN apt-get install -y cmake g++
RUN apt-get install -y libboost-all-dev libopenblas-dev opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev zlib1g-dev
RUN apt-get install -y clinfo python3-pip python3-dev
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade tensorflow
RUN clinfo
RUN python3 -V
RUN pip3 -V
COPY . /src/

# GPU build
WORKDIR /src/gpu/
RUN CXX=g++ CC=gcc cmake ..
RUN cmake --build . --target lczero --config Release -- -j2

# CPU build
WORKDIR /src/cpu/
RUN CXX=g++ CC=gcc cmake -DFEATURE_USE_CPU_ONLY=1 ..
RUN cmake --build . --config Release -- -j2
RUN ./tests

# PYTHON tests
WORKDIR /src/training/tf/
RUN python3 chunkparser.py
RUN python3 shufflebuffer.py
