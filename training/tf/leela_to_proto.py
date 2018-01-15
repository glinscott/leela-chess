#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

import sys
import os
import glob
import multiprocessing as mp
import parse
import argparse


def get_chunks(data_prefix):
    return glob.glob(data_prefix + "*.gz")


def get_configuration():
    """Returns a populated cli configuration"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-samples', default=100000,
        help='Number of training examples to convert to tensorflow format',
        type=int)
    parser.add_argument('-r', '--train-ratio', default=0.75,
        help='Ratio for training / testing', type=float)
    parser.add_argument('-o', '--output', default=".",
        help='Output directory', type=str)
    parser.add_argument('directories', default="", nargs='+',
        help='Directories with leela-chess chunks in gzip format')
    return parser.parse_args()


def main(args):
    cfg = get_configuration()

    chunks = []
    for d in cfg.directories:
        chunks += get_chunks(d)

    print("Found {0} chunks".format(len(chunks)))

    if not chunks:
        return

    parser = parse.ChunkParser(chunks)
    gen = parser.parse_chunk()

    num_train = int(cfg.num_samples*cfg.train_ratio)
    num_test = cfg.num_samples - num_train
    print("Generating {0} training-, {1} testing-samples".format(num_train, num_test))
    if cfg.output != ".":
        os.makedirs(cfg.output)

    with open('{}/train.bin'.format(cfg.output), 'wb') as f:
        for _ in range(num_train):
            data = next(gen)
            f.write(data)

    with open('{}/test.bin'.format(cfg.output), 'wb') as f:
        for _ in range(num_test):
            data = next(gen)
            f.write(data)

    print("Written data to {}*.bin".format(cfg.output))

if __name__ == "__main__":
    main(sys.argv[1:])
    mp.freeze_support()
