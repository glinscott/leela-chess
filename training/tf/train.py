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

import binascii
import os
import yaml
import sys
import glob
import gzip
import random
import math
import multiprocessing as mp
import numpy as np
import os
import time
import tensorflow as tf
from tfprocess import TFProcess
from chunkparser import ChunkParser

SKIP = 16

def get_checkpoint(root_dir):
    checkpoint = os.path.join(root_dir, 'checkpoint')
    with open(checkpoint, 'r') as f:
        cp = f.readline().split()[1][1:-1]
    return cp


def get_chunks(data_prefix):
    return glob.glob(data_prefix + "*.gz")


def get_latest_chunks(path, num_chunks):
    chunks = []
    for d in glob.glob(path):
        chunks += get_chunks(d)

    if len(chunks) < num_chunks:
        print("Not enough chunks")
        sys.exit(1)

    print("sorting {} chunks...".format(len(chunks)), end='')
    chunks.sort(key=os.path.getmtime, reverse=True)
    print("[done]")
    chunks = chunks[:num_chunks]
    print("{} - {}".format(os.path.basename(chunks[-1]), os.path.basename(chunks[0])))
    random.shuffle(chunks)
    return chunks


class FileDataSrc:
    """
        data source yielding chunkdata from chunk files.
    """
    def __init__(self, chunks):
        self.chunks = []
        self.done = chunks
    def next(self):
        if not self.chunks:
            self.chunks, self.done = self.done, self.chunks
            random.shuffle(self.chunks)
        if not self.chunks:
            return None
        while len(self.chunks):
            filename = self.chunks.pop()
            try:
                with gzip.open(filename, 'rb') as chunk_file:
                    self.done.append(filename)
                    return chunk_file.read()
            except:
                print("failed to parse {}".format(filename))


def benchmark(parser):
    """
        Benchmark for parser
    """
    gen = parser.parse()
    batch=100
    while True:
        start = time.time()
        for _ in range(batch):
            next(gen)
        end = time.time()
        print("{} pos/sec {} secs".format( ChunkParser.BATCH_SIZE * batch / (end - start), (end - start)))


def benchmark1(t):
    """
        Benchmark for full input pipeline, including tensorflow conversion
    """
    batch=100
    while True:
        start = time.time()
        for _ in range(batch):
            t.session.run([t.next_batch],
                    feed_dict={t.training: True, t.learning_rate: 0.01, t.handle: t.train_handle})

        end = time.time()
        print("{} pos/sec {} secs".format( ChunkParser.BATCH_SIZE * batch / (end - start), (end - start)))


def main():
    if len(sys.argv) != 2:
        print("Usage: {} config.yaml".format(sys.argv[0]))
        return 1

    cfg = yaml.safe_load(open(sys.argv[1], 'r').read())
    print(yaml.dump(cfg, default_flow_style=False))

    num_chunks = cfg['dataset']['num_chunks']
    chunks = get_latest_chunks(cfg['dataset']['input'], num_chunks)

    num_train = int(num_chunks*cfg['dataset']['train_ratio'])
    shuffle_size = cfg['training']['shuffle_size']
    ChunkParser.BATCH_SIZE = cfg['training']['batch_size']

    root_dir = os.path.join(cfg['training']['path'], cfg['name'])
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    #bench_parser = ChunkParser(FileDataSrc(chunks[:1000]), shuffle_size=1<<14, sample=SKIP, batch_size=ChunkParser.BATCH_SIZE)
    #benchmark(bench_parser)

    train_parser = ChunkParser(FileDataSrc(chunks[:num_train]),
            shuffle_size=shuffle_size, sample=SKIP, batch_size=ChunkParser.BATCH_SIZE)
    #benchmark(train_parser)
    dataset = tf.data.Dataset.from_generator(
        train_parser.parse, output_types=(tf.string, tf.string, tf.string))
    dataset = dataset.map(ChunkParser.parse_function)
    dataset = dataset.prefetch(4)
    train_iterator = dataset.make_one_shot_iterator()

    test_parser = ChunkParser(FileDataSrc(chunks[num_train:]), batch_size=ChunkParser.BATCH_SIZE)
    dataset = tf.data.Dataset.from_generator(
        test_parser.parse, output_types=(tf.string, tf.string, tf.string))
    dataset = dataset.map(ChunkParser.parse_function)
    dataset = dataset.prefetch(4)
    test_iterator = dataset.make_one_shot_iterator()

    tfprocess = TFProcess(cfg)
    tfprocess.init(dataset, train_iterator, test_iterator)

    if os.path.exists(os.path.join(root_dir, 'checkpoint')):
        cp = get_checkpoint(root_dir)
        tfprocess.restore(cp)

    # Sweeps through all test chunks statistically
    num_evals = int(round(((num_chunks-num_train) * (200 / SKIP)) / ChunkParser.BATCH_SIZE))
    print("Using {} evaluation batches".format(num_evals))

    # while True:
    for _ in range(cfg['training']['total_steps']):
        tfprocess.process(ChunkParser.BATCH_SIZE, num_evals)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
    mp.freeze_support()
