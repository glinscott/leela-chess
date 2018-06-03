FROM nvidia/cuda:9.0-cudnn7-devel

WORKDIR /lczero
ADD . /lczero


RUN apt-get update && apt-get install -y git

# clone tensorflow_cc repository
RUN git clone https://github.com/FloopCZ/tensorflow_cc.git

# build and install
RUN ./install.sh --shared --cuda

RUN pip3 --no-cache-dir install meson ninja

# build lc0
RUN ./build.sh
