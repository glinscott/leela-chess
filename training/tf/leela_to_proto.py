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
import yaml
import os
import glob
import multiprocessing as mp
import parse
import argparse
import random


def generate_dataset(chunks, num_samples, filename, skip, start):
    parser = parse.ChunkParser(chunks, skip, start)
    gen = parser.parse_chunk()

    with open(filename, 'ba') as f:
        for _ in range(num_samples):
            f.write(next(gen))
        print("Written dataset to {} pass {}/{}".format(filename, start//2, (skip-2)//2))


def main(args):
    if len(sys.argv) != 2:
        print("Usage: {} config.yaml".format(sys.argv[0]))
        return 1

    cfg = yaml.safe_load(open(sys.argv[1], 'r').read())
    print(yaml.dump(cfg, default_flow_style=False))

    chunks = []
    for d in glob.glob(cfg['dataset']['input']):
        chunks += parse.get_chunks(d)

    print("Found {0} chunks".format(len(chunks)))

    if len(chunks) < 10:
        print("Not enough chunks")
        return 1

    random.shuffle(chunks)
    skip = cfg['dataset']['skip']
    num_samples = 200*len(chunks) // skip # a chunk contains 200 samples
    num_train = int(len(chunks)*cfg['dataset']['train_ratio'])
    num_train_samples = int(num_samples*cfg['dataset']['train_ratio'])
    num_test_samples = num_samples - num_train_samples
    print("Generating {} training-, {} testing-samples".format(num_train_samples*skip//2, num_test_samples*skip//2))

    if not os.path.exists(cfg['dataset']['path']):
        os.makedirs(cfg['dataset']['path'])

    for start in range(0, skip, 2):
        filename = os.path.join(cfg['dataset']['path'], 'train.bin')
        generate_dataset(chunks[:num_train], num_train_samples, filename, skip, start)
        filename = os.path.join(cfg['dataset']['path'], 'test.bin')
        generate_dataset(chunks[num_train:], num_test_samples, filename, skip, start)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
    mp.freeze_support()
