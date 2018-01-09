# Introduction

This is an adaptation of [GCP](https://github.com/gcp)'s [Leela Zero](https://github.com/gcp/leela-zero/)
repository to chess, using Stockfish's position representation and move generation. (No heuristics or prior
knowledge are carried over from Stockfish.)

The goal is to build a strong UCT chess AI following the same type of techniques as AlphaZero, as
described in [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1713.01815).

We will need to do this with a distributed project, as it requires a huge amount of compute.

# Contributing

The project is not quite ready to launch the distributed training component.

## Weights

The weights are located at https://github.com/glinscott/lczero-weights.  The current weights are
just randomly initialized.

# Training

TODO (it is running in the `training/tf` directory).

# Compiling

## Requirements

TODO (see https://github.com/gcp/leela-zero/)

# Other projects
[mokemokechicken/reversi-alpha-zero](https://github.com/mokemokechicken/reversi-alpha-zero)
[Zeta36/chess-alpha-zero](https://github.com/Zeta36/chess-alpha-zero)
[benediamond/chess-alpha-zero](https://github.com/benediamond/chess-alpha-zero/)

