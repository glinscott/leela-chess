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
import numpy as np
import time
import tensorflow as tf
import parse
from tfprocess import TFProcess

class Parser:
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
    parser = Parser(filename, batch_size)
    ds = tf.data.Dataset.from_generator(
        parser.parse_chunk, output_types=(tf.string))
    ds = ds.shuffle(65536)
    ds = ds.map(parse._parse_function)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(16)
    iterator = ds.make_one_shot_iterator()
    return iterator.get_next(), parser

def main(args):
    train_next_batch, parser = dataset_iterator(sys.argv[1], parse.BATCH_SIZE)
    print("Creating trainingset from {}".format(sys.argv[1]))
    num_eval = parser.num_samples() // parse.BATCH_SIZE
    print("Train epoch in {} steps".format(num_eval))

    test_next_batch, parser = dataset_iterator(sys.argv[2], parse.BATCH_SIZE)
    print("Creating testset from {}".format(sys.argv[2]))
    num_eval = parser.num_samples() // parse.BATCH_SIZE
    print("Test epoch in {} steps".format(num_eval))
    
    tfprocess = TFProcess(train_next_batch, test_next_batch, num_eval)
    if args and len(sys.argv) == 4:
        print("Restoring neural net from {}".format(sys.argv[3]))
        tfprocess.restore(sys.argv[3])
    while True:
        tfprocess.process(parse.BATCH_SIZE)

if __name__ == "__main__":
    main(sys.argv[1:])
