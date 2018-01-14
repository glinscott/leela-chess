# Introduction

This is an adaptation of [GCP](https://github.com/gcp)'s [Leela Zero](https://github.com/gcp/leela-zero/)
repository to chess, using Stockfish's position representation and move generation. (No heuristics or prior
knowledge are carried over from Stockfish.)

The goal is to build a strong UCT chess AI following the same type of techniques as AlphaZero, as
described in [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815).

We will need to do this with a distributed project, as it requires a huge amount of compute.

# Contributing

The project is not quite ready to launch the distributed training component.

## Weights

The weights are located at https://github.com/glinscott/lczero-weights.  The current weights are
just randomly initialized.

# Training

After compiling lczero (see below), try the following:
```
cd src
cp ../scripts/train.sh .
./train.sh
```

This should launch lczero in training mode.  It will begin self-play games,
using the weights from weights.txt (initial weights can be downloaded from the
repo above).  The training data will be written into the data subdirectory.

Once you have enough games, you can simply kill the process.

To run the training process, you need to have CUDA and Tensorflow installed.
See the instructions on the Tensorflow page (I used the pip installation method
into a virtual environment).  NOTE: You need a GPU accelerated version of
Tensorflow to train, the CPU version doesn't support the input data format that
is used.
```
cd training/tf
./parse.py ../../src/data/training
```

That will bring up Tensorflow and start running training.  You should see output
like this:
```
2018-01-12 09:57:00.089784: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX TITAN X, pci bus id: 0000:02:00.0, compute capability: 5.2)
2018-01-12 09:57:13.126277: I tensorflow/core/kernels/shuffle_dataset_op.cc:110] Filling up shuffle buffer (this may take a while): 43496 of 65536
2018-01-12 09:57:18.175088: I tensorflow/core/kernels/shuffle_dataset_op.cc:121] Shuffle buffer filled.
step 100, policy loss=7.25049 mse=0.0988732 reg=0.254439 (0 pos/s)
step 200, policy loss=6.80895 mse=0.0904644 reg=0.255358 (3676.48 pos/s)
step 300, policy loss=6.33088 mse=0.0823623 reg=0.256656 (3652.74 pos/s)
step 400, policy loss=5.86768 mse=0.0748837 reg=0.258076 (3525.1 pos/s)
step 500, policy loss=5.42553 mse=0.0680195 reg=0.259414 (3537.3 pos/s)
step 600, policy loss=5.0178 mse=0.0618027 reg=0.260582 (3600.92 pos/s)
...
step 2000, training accuracy=96.9141%, mse=0.00218292
Model saved in file: /home/gary/tmp/leela-chess/training/tf/leelaz-model-2000
```

It saves out the new model every 2000 steps.  To evaluate the model, I do the
following:
```
cd src
cp ../training/tf/leelaz-model-2000.txt ./newweights.txt
cd ../scripts
./run.sh
```

This runs an evaluation match using cutechess-cli.

# Compiling

## Requirements

* GCC, Clang or MSVC, any C++14 compiler
* boost 1.58.x or later (libboost-all-dev on Debian/Ubuntu)
* BLAS Library: OpenBLAS (libopenblas-dev) or (optionally) Intel MKL
* zlib library (zlib1g & zlib1g-dev on Debian/Ubuntu)
* Standard OpenCL C headers (opencl-headers on Debian/Ubuntu, or at
  https://github.com/KhronosGroup/OpenCL-Headers/tree/master/opencl22/)
* OpenCL ICD loader (ocl-icd-libopencl1 on Debian/Ubuntu, or reference implementation at https://github.com/KhronosGroup/OpenCL-ICD-Loader)
* An OpenCL capable device, preferably a very, very fast GPU, with recent
  drivers is strongly recommended (OpenCL 1.2 support should be enough, even
  OpenCL 1.1 might work). If you do not have a GPU, modify config.h in the
  source and remove the line that says `#define USE_OPENCL`.
* Tensorflow 1.4 or higher (for training)
* The program has been tested on Linux.

## Example of compiling - Ubuntu 16.04

    # Install dependencies
    sudo apt install libboost-all-dev libopenblas-dev opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev zlib1g-dev

    # Test for OpenCL support & compatibility
    sudo apt install clinfo && clinfo

    # Clone github repo
    git clone git@github.com:glinscott/leela-chess.git
    cd leela-chess
    mkdir build && cd build
    cmake ..
    make

# Other projects

* [mokemokechicken/reversi-alpha-zero](https://github.com/mokemokechicken/reversi-alpha-zero)
* [Zeta36/chess-alpha-zero](https://github.com/Zeta36/chess-alpha-zero)
* [benediamond/chess-alpha-zero](https://github.com/benediamond/chess-alpha-zero/)

