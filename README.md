# Introduction

This is a _work-in-progress_ adaptation of [GCP](https://github.com/gcp)'s [Leela Zero](https://github.com/gcp/leela-zero/) repository to chess, using Stockfish's position representation and move generation. (No heuristics or prior knowledge are carried over from Stockfish.) When complete, it should ultimately be a faithful replication of AlphaZero, as described in [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815). The goal is to have a very fast C++ MCTS utility for chess. (See [mokemokechicken/reversi-alpha-zero](https://github.com/mokemokechicken/reversi-alpha-zero), [Zeta36/chess-alpha-zero](https://github.com/Zeta36/chess-alpha-zero), and [benediamond/chess-alpha-zero](https://github.com/benediamond/chess-alpha-zero/) for existing Python analogues.)

The code is not functional yet, but it shouldn't be too far off. Please see the [Issues](https://github.com/benediamond/leela-chess/issues) section for remaining tasks. Collaboration is welcome!

# Requirements

Like Leela Zero, Leela Chess requires boost and BLAS, among others. Please see Leela Zero's [README](https://github.com/gcp/leela-zero/blob/master/README.md) for details.
