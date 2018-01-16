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
import random

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


def generate_dataset(chunks, num_samples, filename):
    parser = parse.ChunkParser(chunks, 16)
    gen = parser.parse_chunk()

    with open(filename, 'wb') as f:
        for _ in range(num_samples):
            f.write(next(gen))
        print("Written dataset to {}".format(filename))


def main(args):
    cfg = get_configuration()

    chunks = []
    for d in cfg.directories:
        chunks += parse.get_chunks(d)

    print("Found {0} chunks".format(len(chunks)))

    if len(chunks) < 10:
        print("Not enough chunks")
        return 1

    random.shuffle(chunks)

    num_train = int(len(chunks)*cfg.train_ratio)
    num_train_samples = int(cfg.num_samples*cfg.train_ratio)
    num_test_samples = cfg.num_samples - num_train_samples
    print("Generating {0} training-, {1} testing-samples".format(num_train_samples, num_test_samples))

    if cfg.output != ".":
        os.makedirs(cfg.output)

    generate_dataset(chunks[:num_train], num_train_samples, "{}/train.bin".format(cfg.output))
    generate_dataset(chunks[num_train:], num_test_samples, "{}/test.bin".format(cfg.output))


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main(sys.argv[1:]))
