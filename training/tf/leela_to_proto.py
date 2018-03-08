#!/usr/bin/env python3
#
#    This file is part of Leela Chess.
#    Copyright (C) 2018 Folkert Huizinga
#
#    Leela Chess is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Chess is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

import sys
import yaml
import os
import glob
import multiprocessing as mp
import parse
import argparse
import random

PASSES = 2

def generate_dataset(chunks, num_samples, filename, i):
    parser = parse.ChunkParser(chunks)
    gen = parser.parse_chunk()

    with open(filename, 'ba') as f:
        for _ in range(num_samples):
            f.write(next(gen))
        print("Written dataset to {} pass {}/{}".format(filename, i+1, PASSES))


def main(args):
    if len(sys.argv) != 2:
        print("Usage: {} config.yaml".format(sys.argv[0]))
        return 1

    cfg = yaml.safe_load(open(sys.argv[1], 'r').read())
    print(yaml.dump(cfg, default_flow_style=False))

    chunks = []
    for d in glob.glob(cfg['dataset']['input']):
        chunks += parse.get_chunks(d)

    num_chunks = cfg['dataset']['num_chunks']
    if len(chunks) < num_chunks:
        print("Not enough chunks")
        return 1

    print("sorting {} chunks...".format(len(chunks)), end='')
    chunks.sort(key=os.path.getmtime, reverse=True)
    print("[done]")

    chunks = chunks[:num_chunks]
    random.shuffle(chunks)
    assert(len(chunks) == num_chunks)

    # a chunk contains 200 samples and we're making 2 passes through a chunk when reading it in parse.py
    num_samples = parse.CHUNK_PASSES*200*num_chunks // parse.SKIP 
    num_train = int(num_chunks*cfg['dataset']['train_ratio'])
    num_train_samples = int(num_samples*cfg['dataset']['train_ratio'])
    num_test_samples = num_samples - num_train_samples
    print("Generating {} training-, {} testing-samples".format(num_train_samples*PASSES, num_test_samples*PASSES))

    if not os.path.exists(cfg['dataset']['path']):
        os.makedirs(cfg['dataset']['path'])

    for i in range(0, PASSES):
        filename = os.path.join(cfg['dataset']['path'], 'train.bin')
        generate_dataset(chunks[:num_train], num_train_samples, filename, i)
        filename = os.path.join(cfg['dataset']['path'], 'test.bin')
        generate_dataset(chunks[num_train:], num_test_samples, filename, i)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
    mp.freeze_support()
