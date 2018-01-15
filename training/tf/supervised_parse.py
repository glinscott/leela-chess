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
import sys
import glob
import gzip
import random
import math
import multiprocessing as mp
import numpy as np
import time
import tensorflow as tf
import parse
from tfprocess import TFProcess

class Parser:
    def __init__(self, filename):
        self.data = None
        self.example_len = 15438
        with open(filename, 'rb') as f:
            self.data = f.read()
            print("{} examples, {} bytes".format(len(self.data)/self.example_len, self.example_len))

    def num_samples(self):
        return len(self.data) // self.example_len

    def parse_chunk(self):
        n = self.num_samples()
        i = 0
        while True:
            yield self.data[i*self.example_len:i*self.example_len+self.example_len]
            i += 1
            i %= n


def main(args):

    print("Creating trainingset from {}".format(sys.argv[1]))
    train_parser = Parser(sys.argv[1])
    train_ds = tf.data.Dataset.from_generator(
        train_parser.parse_chunk, output_types=(tf.string))
    train_ds = train_ds.shuffle(65536)
    train_ds = train_ds.map(parse._parse_function)
    train_ds = train_ds.batch(parse.BATCH_SIZE)
    train_ds = train_ds.prefetch(16)
    iterator = train_ds.make_one_shot_iterator()
    train_next_batch = iterator.get_next()

    print("Creating testset from {}".format(sys.argv[2]))
    test_parser = Parser(sys.argv[2])
    test_ds = tf.data.Dataset.from_generator(
        train_parser.parse_chunk, output_types=(tf.string))
    test_ds = test_ds.map(parse._parse_function)
    batch_size = 2**int(np.log2(test_parser.num_samples()/15))
    print("Test batch size {}".format(batch_size))
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(16)
    iterator = test_ds.make_one_shot_iterator()
    test_next_batch = iterator.get_next()

    tfprocess = TFProcess(train_next_batch, test_next_batch)
    if args and len(sys.argv) == 4:
        tfprocess.restore(sys.argv[3])
    while True:
        tfprocess.process(parse.BATCH_SIZE)

if __name__ == "__main__":
    main(sys.argv[1:])
    mp.freeze_support()
