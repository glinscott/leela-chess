#!/usr/bin/env python3
#
#    This file is part of Leela Chess.
#    Copyright (C) 2018 Folkert Huizinga
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
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

import binascii
import glob
import gzip
import itertools
import math
import multiprocessing as mp
import numpy as np
import random
import shufflebuffer as sb
import struct
import sys
import threading
import time
import tensorflow as tf
import unittest

# binary version
VERSION = struct.pack('i', 2)

# 14*8 planes, 4 castling, 1 color, 1 50rule, 1 movecount, 1 unused
DATA_ITEM_LINES = 121

# Interface for a chunk data source.
class ChunkDataSrc:
    def __init__(self, items):
        self.items = items
    def next(self):
        if not self.items:
            return None
        return self.items.pop()

class ChunkParser:
    # static batch size
    BATCH_SIZE = 8
    def __init__(self, chunkdatasrc, shuffle_size=1, sample=1, buffer_size=1, batch_size=256, workers=None):
        """
            Read data and yield batches of raw tensors.

            'chunkdatasrc' is an object yeilding chunkdata
            'shuffle_size' is the size of the shuffle buffer.
            'sample' is the rate to down-sample.
            'workers' is the number of child workers to use.

            The data is represented in a number of formats through this dataflow
            pipeline. In order, they are:

            chunk: The name of a file containing chunkdata

            chunkdata: type Bytes. Either mutiple records of v1 format, or multiple records
            of v2 format.

            v1: The original text format describing a move. 121 lines long. VERY slow
            to decode.

            v2: Packed binary representation of v1. Fixed length, no record seperator.
            The most compact format. Data in the shuffle buffer is held in this
            format as it allows the largest possible shuffle buffer. Very fast to
            decode. Preferred format to use on disk.

            raw: A byte string holding raw tensors contenated together. This is used
            to pass data from the workers to the parent. Exists because TensorFlow doesn't
            have a fast way to unpack bit vectors. 7950 bytes long.
        """

        # Build a series of flat planes with values 0..255
        self.flat_planes = []
        for i in range(256):
            self.flat_planes.append(bytes([i]*64))

        # set the down-sampling rate
        self.sample = sample
        # set the mini-batch size
        self.batch_size = batch_size
        # set number of elements in the shuffle buffer.
        self.shuffle_size = shuffle_size
        # Start worker processes, leave 2 for TensorFlow
        if workers is None:
            workers = max(1, mp.cpu_count() - 2)
        print("Using {} worker processes.".format(workers))

        # Start the child workers running
        self.readers = []
        for _ in range(workers):
            read, write = mp.Pipe(duplex=False)
            mp.Process(target=self.task,
                    args=(chunkdatasrc, write)).start()
            self.readers.append(read)
            write.close()
        self.init_structs()

    def init_structs(self):
        # struct.Struct doesn't pickle, so it needs to be separately
        # constructed in workers.

        # V2 Format (8604 bytes total)
        # int32 version (4 bytes)
        # 1924 float32 probabilities (7696 bytes)
        # 112 packed bit planes (896 bytes)
        # uint8 castling us_ooo (1 byte)
        # uint8 castling us_oo (1 byte)
        # uint8 castling them_ooo (1 byte)
        # uint8 castling them_oo (1 byte)
        # uint8 side_to_move (1 byte)
        # uint8 rule50_count (1 byte)
        # uint8 move_count (1 byte)
        # int8 result (1 byte)
        self.v2_struct = struct.Struct('4s7696s896sBBBBBBBb')

    @staticmethod
    def parse_function(planes, probs, winner):
        """
        Convert v2 to tensors
        """
        planes = tf.decode_raw(planes, tf.uint8)
        probs = tf.decode_raw(probs, tf.float32)
        winner = tf.decode_raw(winner, tf.float32)

        planes = tf.to_float(planes)

        planes = tf.reshape(planes, (ChunkParser.BATCH_SIZE, 120, 8*8))
        probs = tf.reshape(probs, (ChunkParser.BATCH_SIZE, 1924))
        winner = tf.reshape(winner, (ChunkParser.BATCH_SIZE, 1))

        return (planes, probs, winner)

    def convert_v1_to_v2(self, text_item):
        """
            Convert v1 text format to v2 packed binary format

            Converts a chess chunk of ascii text into a byte string
            [[plane_1],[plane_2],...],...
            [probabilities],...
            winner,...
        """
        # We start by building a list of 112 planes, each being a 8*8=64
        # element array of type np.uint8
        planes = []
        for plane in range(0, 112):
            hex_string = text_item[plane]
            array = np.unpackbits(np.frombuffer(bytearray.fromhex(hex_string), dtype=np.uint8))
            planes.append(array)

        # We flatten to a single array of len 112*8*8, type=np.uint8
        planes = np.concatenate(planes)
        planes = np.packbits(planes).tobytes()

        # Now we extract the non plane information
        us_ooo = int(text_item[112])
        us_oo = int(text_item[113])
        them_ooo = int(text_item[114])
        them_oo = int(text_item[115])
        stm = int(text_item[116])
        rule50_count = min(int(text_item[117]), 255)
        move_count = min(int(text_item[118]), 255)

        # Load the probabilities.
        probabilities = np.array(text_item[119].split()).astype(np.float32)
        assert(len(probabilities) == 1924)
        probs = probabilities.tobytes()
        assert(len(probs) == 1924 * 4)

        # Load the game winner color.
        winner = int(text_item[120])
        assert winner == 1 or winner == -1 or winner == 0

        return True, self.v2_struct.pack(VERSION, probs, planes, us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count, move_count, winner)


    def convert_v2_to_tuple(self, content):
        """
            Convert v2 binary training data to packed tensors 

            v2 struct format is
                int32 version (4 bytes)
                1924 float32 probabilities (7696 bytes)
                112 packed bit planes (896 bytes)
                uint8 castling us_ooo (1 byte)
                uint8 castling us_oo (1 byte)
                uint8 castling them_ooo (1 byte)
                uint8 castling them_oo (1 byte)
                uint8 side_to_move (1 byte)
                uint8 rule50_count (1 byte)
                uint8 move_count (1 byte)
                int8 result (1 byte)

            packed tensor formats are
                float32 winner (4 bytes)
                float32 probs (1924 * 4 bytes)
                uint8 planes (120 * 8 * 8 bytes)
        """
        (ver, probs, planes, us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count, move_count, winner) = self.v2_struct.unpack(content)
        # Unpack planes.
        planes = np.unpackbits(np.frombuffer(planes, dtype=np.uint8))
        planes = planes.tobytes() + self.flat_planes[us_ooo] + self.flat_planes[us_oo] + self.flat_planes[them_ooo] + self.flat_planes[them_oo] + self.flat_planes[stm] + self.flat_planes[rule50_count] + self.flat_planes[move_count] + self.flat_planes[0]
        assert len(planes) == (120 * 8 * 8), len(planes)
        winner = float(winner)
        assert winner == 1.0 or winner == -1.0 or winner == 0.0, winner
        winner = struct.pack('f', winner)

        return (planes, probs, winner)

    def convert_chunkdata_to_v2(self, chunkdata):
        """
            Take chunk of unknown format, and return it as a list of
            v2 format records.
        """
        if chunkdata[0:4] == b'\1\0\0\0':
            # invalidated by bug see issue #119
            return
        elif chunkdata[0:4] == VERSION:
            #print("V2 chunkdata")
            for i in range(0, len(chunkdata), self.v2_struct.size):
                if self.sample > 1:
                    # Downsample, using only 1/Nth of the items.
                    if random.randint(0, self.sample-1) != 0:
                        continue  # Skip this record.
                yield chunkdata[i:i+self.v2_struct.size]
        else:
            #print("V1 chunkdata")
            file_chunkdata = chunkdata.splitlines()

            result = []
            for i in range(0, len(file_chunkdata), DATA_ITEM_LINES):
                if self.sample > 1:
                    # Downsample, using only 1/Nth of the items.
                    if random.randint(0, self.sample-1) != 0:
                        continue  # Skip this record.
                item = file_chunkdata[i:i+DATA_ITEM_LINES]
                str_items = [str(line, 'ascii') for line in item]
                success, data = self.convert_v1_to_v2(str_items)
                if success:
                    yield data

    def task(self, chunkdatasrc, writer):
        """
            Run in fork'ed process, read data from chunkdatasrc, parsing, shuffling and
            sending v2 data through pipe back to main process.
        """
        self.init_structs()
        while True:
            chunkdata = chunkdatasrc.next()
            if chunkdata is None:
                break
            for item in self.convert_chunkdata_to_v2(chunkdata):
                # NOTE: This requires some more thinking, we can't just apply a
                # reflection along the horizontal or vertical axes as we would
                # also have to apply the reflection to the move probabilities
                # which is non trivial for chess.

                # symmetry = random.randrange(8)
                # item = self.v2_apply_symmetry(symmetry, item)
                writer.send_bytes(item)

    def v2_gen(self):
        """
            Read v2 records from child workers, shuffle, and yield
            records.
        """
        sbuff = sb.ShuffleBuffer(self.v2_struct.size, self.shuffle_size)
        while len(self.readers):
            #for r in mp.connection.wait(self.readers):
            for r in self.readers:
                try:
                    s = r.recv_bytes()
                    s = sbuff.insert_or_replace(s)
                    if s is None:
                        continue  # shuffle buffer not yet full
                    yield s
                except EOFError:
                    print("Reader EOF")
                    self.readers.remove(r)
        # drain the shuffle buffer.
        while True:
            s = sbuff.extract()
            if s is None:
                return
            yield s

    def tuple_gen(self, gen):
        """
            Take a generator producing v2 records and convert them to tuples.
            applying a random symmetry on the way.
        """
        for r in gen: 
            yield self.convert_v2_to_tuple(r)

    def batch_gen(self, gen):
        """
            Pack multiple records into a single batch
        """
        # Get N records. We flatten the returned generator to
        # a list because we need to reuse it.
        while True:
            s = list(itertools.islice(gen, self.batch_size))
            if not len(s):
                return
            yield ( b''.join([x[0] for x in s]),
                    b''.join([x[1] for x in s]),
                    b''.join([x[2] for x in s]) )

    def parse(self):
        """
            Read data from child workers and yield batches
            of raw tensors
        """
        gen = self.v2_gen()        # read from workers
        gen = self.tuple_gen(gen)  # convert v2->tuple
        gen = self.batch_gen(gen)  # assemble into batches
        for b in gen:
            yield b



#
# Tests to check that records can round-trip successfully
class ChunkParserTest(unittest.TestCase):
    def generate_fake_pos(self):
        """
            Generate a random game position.
            Result is ([[64] * 112], [1]*5, [1924], [1])
        """
        # 0. 112 binary planes of length 64
        planes = [np.random.randint(2, size=64).tolist() for plane in range(112)]

        # 1. generate the other integer data
        integer = []
        for i in range(5):
            integer.append(np.random.randint(2))
        integer.append(np.random.randint(100))
        integer.append(np.random.randint(255))

        # 2. 1924 probs
        probs = np.random.randint(9, size=1924).tolist()

        # 3. And a winner: 1, 0, -1
        winner = [ float(np.random.randint(3) - 1) ]
        return (planes, integer, probs, winner)


    def test_parsing(self):
        """
            Test game position decoding pipeline.

            We generate a V1 record, and feed it all the way
            through the parsing pipeline to final tensors,
            checking that what we get out is what we put in.
        """
        batch_size=256
        # First, build a random game position.
        planes, integer, probs, winner, chunkdata = self.v1_gen()

        # feed batch_size copies into parser
        chunkdatasrc = ChunkDataSrc([chunkdata for _ in range(batch_size*2)])
        parser = ChunkParser(chunkdatasrc,
                shuffle_size=1, workers=1,batch_size=batch_size)

        # Get one batch from the parser.
        batchgen = parser.parse()
        data = next(batchgen)

        # Convert batch to python lists.
        batch = ( np.reshape(np.frombuffer(data[0], dtype=np.uint8), (batch_size, 120, 64)).tolist(),
                  np.reshape(np.frombuffer(data[1], dtype=np.float32), (batch_size, 1924)).tolist(),
                  np.reshape(np.frombuffer(data[2], dtype=np.float32), (batch_size, 1)).tolist() )

        # Check that every record in the batch is a some valid symmetry
        # of the original data.
        for i in range(batch_size):
            data = (batch[0][i][:112], [batch[0][i][j][0] for j in range(112,119)], batch[1][i], batch[2][i])
            assert data == (planes, integer, probs, winner)

        print("Test parse passes")

        # drain parser
        for _ in batchgen:
            pass


    def v1_gen(self):
        """
            Generate a batch of random v1 fake positions
        """
        # First, build a random game position.
        planes, integer, probs, winner = self.generate_fake_pos()

        # Convert that to a v1 text record.
        items = []
        for p in range(112):
            h = str(np.packbits([int(x) for x in planes[p]]).tobytes().hex())
            items.append(h + "\n")
        # then integer info
        for i in integer:
            items.append(str(i) + "\n")
        # then probabilities
        items.append(' '.join([str(x) for x in probs]) + "\n")
        # and finally if the side to move is a winner
        items.append(str(int(winner[0])) + "\n")

        # Convert to a chunkdata byte string and return original data
        return planes, integer, probs, winner, ''.join(items).encode('ascii')


    def test_tensorflow_parsing(self):
        """
            Test game position decoding pipeline including tensorflow.
        """
        batch_size = ChunkParser.BATCH_SIZE
        chunks = []
        original = []
        for i in range(batch_size):
            planes, integer, probs, winner, chunk = self.v1_gen()
            chunks.append(chunk)
            original.append((planes, integer, probs, winner))

        # feed batch_size copies into parser
        chunkdatasrc = ChunkDataSrc(chunks)
        parser = ChunkParser(chunkdatasrc,
                shuffle_size=1, workers=1,batch_size=batch_size)

        # Get one batch from the parser
        batchgen = parser.parse()
        data = next(batchgen)
        probs = np.frombuffer(data[1], dtype=np.float32, count=1924*batch_size)
        probs = probs.reshape(batch_size, 1924)
        planes = np.frombuffer(data[0], dtype=np.uint8, count=120*8*8*batch_size)
        planes = planes.reshape(batch_size, 120, 8*8)
        planes = planes.astype(np.float32)
        winner = np.frombuffer(data[2], dtype=np.float32, count=1*batch_size)

        # Pass it through tensorflow
        with tf.Session() as sess:
            graph = ChunkParser.parse_function(data[0], data[1], data[2])
            tf_planes, tf_probs, tf_winner = sess.run(graph)

            for i in range(batch_size):
                assert (probs[i] == tf_probs[i]).all()
                assert (planes[i] == tf_planes[i]).all()
                assert (winner[i] == tf_winner[i]).all()

        print("Test parse passes")

        # drain parser
        for _ in batchgen:
            pass



if __name__ == '__main__':
    unittest.main()
