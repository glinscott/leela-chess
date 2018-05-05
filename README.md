[![Linux Build Status](https://travis-ci.org/glinscott/leela-chess.svg?branch=master)](https://travis-ci.org/glinscott/leela-chess)
[![Windows Build status](https://ci.appveyor.com/api/projects/status/w2nymx3wpd0d1da1/branch/master?svg=true)](https://ci.appveyor.com/project/glinscott/leela-chess/branch/master)

# Introduction

This is an adaptation of [GCP](https://github.com/gcp)'s [Leela Zero](https://github.com/gcp/leela-zero/) repository to chess, using Stockfish's position representation and move generation. (No heuristics or prior knowledge are carried over from Stockfish.)

The goal is to build a strong UCT chess AI following the same type of techniques as AlphaZero, as described in [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815).

We will need to do this with a distributed project, as it requires a huge amount of computations.

Please visit the LCZero forum to discuss: https://groups.google.com/forum/#!forum/lczero, or the github issues.

# Contributing

For precompiled binaries, see:
* [wiki](https://github.com/glinscott/leela-chess/wiki)
* [wiki/Getting-Started](https://github.com/glinscott/leela-chess/wiki/Getting-Started)

For live status: http://lczero.org

The rest of this page is for users who want to compile the code themselves.
Of course, we also appreciate code reviews, pull requests and Windows testers!

# Compiling

## Requirements

* GCC, Clang or MSVC, any C++14 compiler
* boost 1.54.x or later (libboost-all-dev on Debian/Ubuntu)
* BLAS Library: OpenBLAS (libopenblas-dev) or (optionally) Intel MKL
* zlib library (zlib1g & zlib1g-dev on Debian/Ubuntu)
* Standard OpenCL C headers (opencl-headers on Debian/Ubuntu, or at
  https://github.com/KhronosGroup/OpenCL-Headers/tree/master/opencl22/)
* OpenCL ICD loader (ocl-icd-libopencl1 on Debian/Ubuntu, or reference implementation at https://github.com/KhronosGroup/OpenCL-ICD-Loader)
* An OpenCL capable device, preferably a very, very fast GPU, with recent
  drivers is strongly recommended but not required. (OpenCL 1.2 support should be enough, even
  OpenCL 1.1 might work).
* Tensorflow 1.4 or higher (for training)
* The program has been tested on Linux.


## Example of compiling - Ubuntu 16.04

    # Install dependencies
    sudo apt install cmake g++ git libboost-all-dev libopenblas-dev opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev zlib1g-dev

    # Test for OpenCL support & compatibility
    sudo apt install clinfo && clinfo

    # Clone git repo
    git clone https://github.com/glinscott/leela-chess.git
    cd leela-chess
    git submodule update --init --recursive
    mkdir build && cd build
    
    # Configure
    cmake ..

    # Or configure without GPU support
    cmake -DFEATURE_USE_CPU_ONLY=1 ..

    # Build and run tests
    make
    ./tests

# Compiling Client

See https://github.com/glinscott/leela-chess/tree/master/go/src/client/README.md.
This client will produce self-play games and upload them to http://lczero.org. 
A central server uses these self-play game data as inputs for the training process.

## Weights

The weights from the distributed training are downloadable from http://lczero.org/networks. The best one is the top network that has some games played on it.

Weights that we trained to prove the engine was solid are here https://github.com/glinscott/lczero-weights. The best weights obtained through supervised learning on a human dataset were with elo ratings > 2000.

# Training

The training pipeline resides in `training/tf`, this requires tensorflow running on linux (Ubuntu 16.04 in this case). 

## Data preparation

In order to start a training session you first need to download trainingdata from http://lczero.org/training_data. This data is packed in tar.gz balls each containing 10'000 games or chunks as we call them. Preparing data requires the following steps:

```
tar -xzf games11160000.tar.gz
ls training.* | parallel gzip {}
```

This repacks each chunk into a gzipped file ready to be parsed by the training pipeline. Note that the `parallel` command uses all your cores and can be installed with `apt-get install parallel`.

## Training pipeline

Now that the data is in the right format one can configure a training pipeline. This configuration is achieved through a yaml file, see `training/tf/configs/example.yaml`:

```yaml
%YAML 1.2
---
name: 'kb1-64x6'                       # ideally no spaces
gpu: 0                                 # gpu id to process on

dataset: 
  num_chunks: 100000                   # newest nof chunks to parse
  train_ratio: 0.90                    # trainingset ratio
  input: '/path/to/chunks/*/draw/'     # supports glob

training:
    batch_size: 2048                   # training batch
    total_steps: 140000                # terminate after these steps
    shuffle_size: 524288               # size of the shuffle buffer
    lr_values:                         # list of learning rates
        - 0.02
        - 0.002
        - 0.0005
    lr_boundaries:                     # list of boundaries
        - 100000
        - 130000
    policy_loss_weight: 1.0            # weight of policy loss
    value_loss_weight: 1.0             # weight of value loss
    path: '/path/to/store/networks'    # network storage dir

model:
  filters: 64
  residual_blocks: 6
...
```

The configuration is pretty self explanatory, if you're new to training I suggest looking at the [machine learning glossary](https://developers.google.com/machine-learning/glossary/) by google. Now you can invoke training with the following command:

```bash
./train.py --cfg configs/example.yaml --output /tmp/mymodel.txt
```

This will initialize the pipeline and start training a new neural network. You can view progress by invoking tensorboard:

```bash
tensorboard --logdir leelalogs
```

If you now point your browser at localhost:6006 you'll see the trainingprogress as the trainingsteps pass by. Have fun!

## Restoring models

The training pipeline will automatically restore from a previous model if it exists in your `training:path` as configured by your yaml config. For initializing from a raw `weights.txt` file you can use `training/tf/net_to_model.py`, this will create a checkpoint for you.

## Supervised training

Generating trainingdata from pgn files is currently broken and has low priority, feel free to create a PR.

# Other projects

* [mokemokechicken/reversi-alpha-zero](https://github.com/mokemokechicken/reversi-alpha-zero)
* [Zeta36/chess-alpha-zero](https://github.com/Zeta36/chess-alpha-zero)
* [benediamond/chess-alpha-zero](https://github.com/benediamond/chess-alpha-zero/)

# License

The code is released under the GPLv3 or later, except for ThreadPool.h, cl2.hpp and the clblast_level3 subdir, which have specific licenses (compatible with GPLv3) mentioned in those files.
