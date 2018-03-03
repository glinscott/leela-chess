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

import yaml
import sys
import os
import numpy as np
import time
import tensorflow as tf
import parse
from tfprocess import TFProcess

class BinaryParser:
    def __init__(self, filename, batchsize):
        self.batch_size = batchsize
        self.example_len = 15438 # size of a tf.train.Example in binary
        self.f = open(filename, 'rb')
        self.len = os.stat(filename).st_size
        assert(self.len % self.example_len == 0)
        print("{} examples, {} bytes per sample".format(self.num_samples(), self.example_len))

    def num_samples(self):
        return self.len // self.example_len

    def parse_chunk(self):
        chunk_size = self.example_len*self.batch_size
        while True:
            # cache a batch
            size = min(chunk_size, self.len-self.f.tell())
            data = self.f.read(size)
            if size < chunk_size:
                self.f.seek(0, os.SEEK_SET)
                data += self.f.read(chunk_size-size)

            for i in range(self.batch_size):
                yield data[i*self.example_len:i*self.example_len+self.example_len]


def dataset_iterator(filename, batch_size):
    parser = BinaryParser(filename, batch_size)
    ds = tf.data.Dataset.from_generator(
        parser.parse_chunk, output_types=(tf.string))
    ds = ds.shuffle(1 << 18)
    ds = ds.map(parse._parse_function)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(8)
    iterator = ds.make_one_shot_iterator()
    return iterator.get_next(), parser


def main():
    if len(sys.argv) != 2:
        print("Usage: {} config.yaml".format(sys.argv[0]))
        return 1

    cfg = yaml.safe_load(open(sys.argv[1], 'r').read())
    print(yaml.dump(cfg, default_flow_style=False))

    batch_size = cfg['training']['batch_size']

    filename = os.path.join(cfg['dataset']['path'], 'train.bin')
    train_next_batch, parser = dataset_iterator(filename, batch_size)
    print("Creating trainingset from {}".format(filename))
    num_eval = parser.num_samples() // batch_size
    print("Train epoch in {} steps".format(num_eval))

    filename = os.path.join(cfg['dataset']['path'], 'test.bin')
    test_next_batch, parser = dataset_iterator(filename, batch_size)
    print("Creating testset from {}".format(filename))
    num_eval = parser.num_samples() // batch_size
    print("Test epoch in {} steps".format(num_eval))
    
    tfprocess = TFProcess(cfg, train_next_batch, test_next_batch, num_eval)

    root_dir = os.path.join(cfg['training']['path'], cfg['name'])
    if os.path.exists(os.path.join(root_dir, 'checkpoint')):
        checkpoint = parse.get_checkpoint(root_dir)
        tfprocess.restore(checkpoint)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        print("Created output directory: {}".format(root_dir))

    while True:
        tfprocess.process(batch_size)


if __name__ == "__main__":
    sys.exit(main())
