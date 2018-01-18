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
from tfprocess import TFProcess

DATA_ITEM_LINES = 121

BATCH_SIZE = 512

class ChunkParser:
    def __init__(self, chunks):
        self.flat_planes = []
        for r in range(0, 255):
            self.flat_planes.append(bytes([r]*64))

        # Start worker processes, leave 1 for TensorFlow
        workers = max(1, mp.cpu_count() - 1)
        print("Using {} worker processes.".format(workers))
        self.readers = []

        for _ in range(workers):
            read, write = mp.Pipe(False)
            mp.Process(target=self.task,
                       args=(chunks, write)).start()
            self.readers.append(read)

    def convert_train_data(self, text_item):
        """
            Convert textual training data to a tf.train.Example

            Converts a set of 121 lines of text into a pythonic dataformat.
            [[plane_1],[plane_2],...],...
            [probabilities],...
            winner,...
        """
        # We start by building a list of 112 planes, each being a 8*8=64 element array
        # of type np.uint8
        planes = []
        for plane in range(0, 112):
            hex_string = text_item[plane]
            array = np.unpackbits(np.frombuffer(bytearray.fromhex(hex_string), dtype=np.uint8))
            planes.append(array)

        # We flatten to a single array of len 112*8*8, type=np.uint8
        planes = np.concatenate(planes)
        # Convert the array to a byte string
        planes = [ planes.tobytes() ]

        # Now we add the final planes
        for plane in range(112, 119):
            stm = min(int(text_item[plane]), 254)
            planes.append(self.flat_planes[stm])
        planes.append(self.flat_planes[0])

        # Flatten all planes to a single byte string
        planes = b''.join(planes)
        assert len(planes) == (120 * 8 * 8)

        # Load the probabilities.
        probabilities = np.array(text_item[119].split()).astype(float)
        if np.any(np.isnan(probabilities)):
            # Work around a bug in leela-zero v0.3, skipping any
            # positions that have a NaN in the probabilities list.
            return False, None
        assert len(probabilities) == 1924

        # Load the game winner color.
        winner = float(text_item[120])
        assert winner == 1.0 or winner == -1.0 or winner == 0.0

        # Construct the Example protobuf
        example = tf.train.Example(features=tf.train.Features(feature={
            'planes' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[planes])),
            'probs' : tf.train.Feature(float_list=tf.train.FloatList(value=probabilities)),
            'winner' : tf.train.Feature(float_list=tf.train.FloatList(value=[winner]))}))
        return True, example.SerializeToString()

    def task(self, chunks, writer):
        while True:
            random.shuffle(chunks)
            for chunk in chunks:
                with gzip.open(chunk, 'r') as chunk_file:
                    file_content = chunk_file.readlines()
                    item_count = len(file_content) // DATA_ITEM_LINES
                    for item_idx in range(item_count):
                        pick_offset = item_idx * DATA_ITEM_LINES
                        item = file_content[pick_offset:pick_offset + DATA_ITEM_LINES]
                        str_items = [str(line, 'ascii').strip() for line in item]
                        success, data = self.convert_train_data(str_items)
                        if success:
                            # Send it down the pipe.
                            writer.send_bytes(data)

    def parse_chunk(self):
        while True:
            for r in self.readers:
                yield r.recv_bytes()

def get_chunks(data_prefix):
    return glob.glob(data_prefix + "*.gz")


#
# Tests to check that records can round-trip successfully
def generate_fake_pos():
    """
        Generate a random game position.
        Result is ([[64] * 112], [1]*5, [1924], [1])
    """
    # 1. 112 binary planes of length 64
    planes = [[float(f) for f in np.random.randint(2, size=64).tolist()] for plane in range(112)]
    for i in range(5):
        planes.append([np.random.randint(2)] * 64)
    planes.append([np.random.randint(50)] * 64)
    planes.append([np.random.randint(200)] * 64)
    planes.append([0.0] * 64)
    # 2. 1924 probs
    probs = np.random.random(size=1924).tolist()
    # 3. And a winner: 1, 0, -1
    winner = [ float(np.random.randint(3)) - 1 ]
    return (planes, probs, winner)

def run_test(parser):
    """
        Test game position decoding.
    """

    # First, build a random game position.
    planes, probs, winner = generate_fake_pos()

    # Convert that to a text record in the same format
    # generated by dump_supervised
    items = []
    for p in range(112):
        h = np.packbits([int(x) for x in planes[p]]).tobytes()
        h = binascii.hexlify(h).decode('ascii')
        items.append(h)
    for p in range(112, 119):
        items.append(str(int(planes[p][0])))
    # then probs
    items.append(' '.join([str(x) for x in probs]))
    # and finally a winner
    items.append(str(int(winner[0])))

    # Have an input string. Running it through parsing to see
    # if it gives the same result we started with.
    # We need a tf.Session() as we're going to use the tensorflow
    # decoding framework for part of the parsing.
    with tf.Session() as sess:
        result = parser.convert_train_data(items)
        assert result[0] == True
        # We got back a serialized tf.train.Example, which we need to decode.
        graph = _parse_function(result[1])
        data = sess.run(graph)
        data = (data[0].tolist(), data[1].tolist(), data[2].tolist())

        # Check that what we got out matches what we put in.
        assert data[0] == planes
        assert len(data[1]) == len(probs)
        for i in range(len(probs)):
            if abs(data[1][i] - probs[i]) > 1e-6:
                assert False
        assert data[2] == winner
    print("Test parse passes")


# Convert a tf.train.Example protobuf into a tuple of tensors
# NB: This conversion is done in the tensorflow graph, NOT in python.
def _parse_function(example_proto):
    features = {"planes": tf.FixedLenFeature((1), tf.string),
                "probs": tf.FixedLenFeature((1924), tf.float32),
                "winner": tf.FixedLenFeature((1), tf.float32)}
    parsed_features = tf.parse_single_example(example_proto, features)
    # We receives the planes as a byte array, but we really want
    # floats of shape (120, 8*8), so decode, cast, and reshape.
    planes = tf.decode_raw(parsed_features["planes"], tf.uint8)
    planes = tf.to_float(planes)
    planes = tf.reshape(planes, (120, 8*8))
    # the other features are already in the correct shape as return as-is.
    return planes, parsed_features["probs"], parsed_features["winner"]

def benchmark(parser):
    gen = parser.parse_chunk()
    while True:
        start = time.time()
        for _ in range(10000):
            next(gen)
        end = time.time()
        print("{} pos/sec {} secs".format( 10000. / (end - start), (end - start)))

def main(args):

    train_data_prefix = args.pop(0)

    chunks = get_chunks(train_data_prefix)
    print("Found {0} chunks".format(len(chunks)))

    if not chunks:
        return

    parser = ChunkParser(chunks)

    run_test(parser)
    #benchmark(parser)

    dataset = tf.data.Dataset.from_generator(
        parser.parse_chunk, output_types=(tf.string))
    dataset = dataset.shuffle(65536)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(16)
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    tfprocess = TFProcess(next_batch)
    if args:
        restore_file = args.pop(0)
        tfprocess.restore(restore_file)
    while True:
        tfprocess.process(BATCH_SIZE)

if __name__ == "__main__":
    main(sys.argv[1:])
    mp.freeze_support()
