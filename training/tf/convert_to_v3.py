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

# TODO:
# - The chunkparser convert_v3_to_tuple is untested.
#     I just took a guess at it.
# - Modify lczero code to detect version in weights and switch
#     to V3 automatically
#
# Done:
# - output binary for V3, not text
# - display/flip policy output
# - remove rep2 plane
# - normalize rule50_count to be in [0,1].
# - refactor to use chunckparser.py where appropriate.
#     Still more can be done here but it's closer now.
#
# Plan not to change:
# - remove move_count?
# - I vote to keep side_to_move, but it may not be needed either.
# - LZGo actually added black_to_move white_to_move planes to help
#     it recognize the edge of the board more easily.

import array
import binascii
import chunkparser
import glob
import gzip
import itertools
import math
import numpy as np
import random
import re
import shutil
import struct
import sys
import threading
import time
import unittest
import argparse
from collections import defaultdict

# VERSION of the training data file format
#     1 - Text, oldflip
#     2 - Binary, oldflip
#     3 - Binary, newflip
#     b'\1\0\0\0' - Invalid, see issue #119
#
# Note: VERSION1 does not include a version in the header, it starts with
# text hex characters. This means any VERSION that is also a valid ASCII
# hex string could potentially be a training file. Most of the time it
# will be "00ff", but maybe games could get split and then there are more
# "00??" possibilities.
#
# Also note "0002" is actually b'\0x30\0x30\0x30\0x32' (or maybe reversed?)
# so it doesn't collide with VERSION2.
#
VERSION1 = chunkparser.VERSION1
VERSION2 = chunkparser.VERSION2
VERSION3 = chunkparser.VERSION3

V2_BYTES = 8604
V3_BYTES = 8280

# Us   -- uppercase
# Them -- lowercase
PIECES = "PNBRQKpnbrqk"

class Board:
    def __init__(self):
        self.clear_board()

    def clear_board(self):
        self.board = []
        for rank in range(8):
            self.board.append(list("."*8))
        self.reps = 0

    def describe(self):
        s = []
        for rank in range(8):
            s.append("".join(self.board[rank]))
        s.append("reps {}  ".format(self.reps))
        return s

class TrainingStep:
    def __init__(self, version):
        self.version = version
        # Construct a fake parser just to get access to it's variables
        self.parser = chunkparser.ChunkParser(chunkparser.ChunkDataSrc([]), workers=1)
        self.NUM_HIST = 8
        self.NUM_PIECE_TYPES = 6
        self.V2_NUM_PLANES = self.NUM_PIECE_TYPES*2+2  # = 14 (6*2 us/them pieces, rep1, rep2)
        self.V3_NUM_PLANES = self.NUM_PIECE_TYPES*2+1  # = 13 (6*2 us/them pieces, rep1 (no rep2))
        if self.version <= 2:
            self.NUM_PLANES = self.V2_NUM_PLANES
        else:
            self.NUM_PLANES = self.V3_NUM_PLANES
        self.NUM_REALS = 7  # 4 castling, 1 color, 1 50rule, 1 movecount
        self.NUM_OUTPUTS = 2 # policy, value
        self.NUM_PLANES_BYTES = self.NUM_PLANES*4
        self.NUM_PLANES_BYTES = self.NUM_PLANES*4
        self.NUM_PLANES_BYTES = self.NUM_PLANES*4

        self.V2_NUM_POLICY_MOVES = 1924 # (7696 bytes)
        self.V3_NUM_POLICY_MOVES = 1858 # (7432 bytes)
        if self.version <= 2:
            self.NUM_POLICY_MOVES = self.V2_NUM_POLICY_MOVES
        else:
            self.NUM_POLICY_MOVES = self.V3_NUM_POLICY_MOVES


        # This one is used for V1 text
        self.DATA_ITEM_LINES = self.NUM_HIST*self.NUM_PLANES+self.NUM_REALS+self.NUM_OUTPUTS # 121
        # Note: The C++ code adds 1 unused REAL to make the number even.

        self.init_structs()
        self.init_move_map()
        self.history = []
        self.probs = []
        for history in range(self.NUM_HIST):
            self.history.append(Board())
        self.us_ooo = 0
        self.us_oo  = 0
        self.them_ooo = 0
        self.them_oo  = 0
        self.us_black = 0
        self.rule50_count = 0


    def init_structs(self):
        self.v2_struct = self.parser.v2_struct
        self.v3_struct = self.parser.v3_struct

        if self.version <= 2:
            self.this_struct = self.v2_struct
        else:
            self.this_struct = self.v3_struct

    def init_move_map(self):
        self.old_move_map = defaultdict(lambda:-1)
        self.new_white_move_map = defaultdict(lambda:-1)
        self.new_black_move_map = defaultdict(lambda:-1)
        self.old_rev_move_map = {}
        self.new_rev_white_move_map = {}
        self.new_rev_black_move_map = {}

        fin = open("new_movelist.txt")
        moves = [x.strip() for x in fin.readlines()]
        for idx, m in enumerate(moves):
            self.new_white_move_map[m] = idx
            self.new_rev_white_move_map[idx] = m
            m_black = m.translate(str.maketrans("12345678", "87654321"))
            self.new_black_move_map[m_black] = idx
            self.new_rev_black_move_map[idx] = m_black

        fin = open("old_movelist.txt")
        moves = [x.strip() for x in fin.readlines()]
        for idx, m in enumerate(moves):
            self.old_rev_move_map[idx] = m
            self.old_move_map[m] = idx

        #for m in moves:
        #    print("debug", m, self.old_move_map[m], self.new_white_move_map[m], self.new_black_move_map[m])

    def clear_hist(self):
        for hist in range(self.NUM_HIST):
            self.history.clear_board()

    def update_board(self, hist, piece, bit_board):
        """
            Update the ASCII board representation
        """
        for r in range(8):
            for f in range(8):
                # Note: Using 8-1-f because both the text and binary have the
                # column bits reversed fhom what this code expects
                if bit_board & (1<<(r*8+(8-1-f))):
                    assert(self.history[hist].board[r][f] == ".")
                    self.history[hist].board[r][f] = piece

    def describe(self):
        s = ""
        if self.us_black:
            s += "us = Black\n"
            s += "(Note the black pieces are CAPS, black moves up, but A1 is in lower left)\n"
        else:
            s += "us = White\n"
        s += "rule50_count {} b_ooo b_oo, w_ooo, w_oo {} {} {} {}\n".format(
            self.rule50_count, self.us_ooo, self.us_oo, self.them_ooo, self.them_oo)
        s += "  abcdefgh\n"
        rank_strings = [[]]
        for rank in reversed(range(8)):
            rank_strings[0].append("{}".format(rank+1))
        for hist in range(self.NUM_HIST):
            rank_strings.append(self.history[hist].describe())
        for rank in range(8):
            for hist in range(self.NUM_HIST):
                s += rank_strings[hist][rank] + " "
            s += "\n"
        sum = 0.0
        top_moves = {}
        for idx, prob in enumerate(self.probs):
            if prob > 0.01:
                top_moves[idx] = prob
            sum += prob
        for idx, prob in sorted(top_moves.items(), key=lambda x:-x[1]):
            if self.version <= 2:
                s += "{} {:4.1f}%\n".format(self.old_rev_move_map[idx], prob*100)
            else:
                s += "{} {:4.1f}%\n".format(self.new_rev_white_move_map[idx], prob*100)
        #print("debug prob sum", sum, "cnt", len(self.probs))
        return s

    def update_reals(self, text_item):
        self.us_ooo = int(text_item[self.NUM_HIST*self.NUM_PLANES+0])
        self.us_oo = int(text_item[self.NUM_HIST*self.NUM_PLANES+1])
        self.them_ooo = int(text_item[self.NUM_HIST*self.NUM_PLANES+2])
        self.them_oo = int(text_item[self.NUM_HIST*self.NUM_PLANES+3])
        self.us_black = int(text_item[self.NUM_HIST*self.NUM_PLANES+4])
        self.rule50_count = min(int(text_item[self.NUM_HIST*self.NUM_PLANES+5]), 255)
        # should be around 99-102ish
        assert self.rule50_count < 105

    def display_v1(self, ply, text_item):
        for hist in range(self.NUM_HIST):
            for idx, piece in enumerate(PIECES):
                self.update_board(hist, piece, int(text_item[hist*self.NUM_PLANES+idx], 16))
            if (text_item[hist*self.NUM_PLANES+12] != "0000000000000000"):
                self.history[hist].reps = 1
                assert (text_item[hist*self.NUM_PLANES+12] == "ffffffffffffffff")
            # It's impossible to have a position in training with reps=2
            # Because that is already a draw.
            assert (text_item[hist*self.NUM_PLANES+13] == "0000000000000000")

        # Now we extract the non plane information
        self.update_reals(text_item)
        print("ply {} move {} (Not actually part of training data)".format(
            ply+1, (ply+2)//2))
        print(self.describe())

    def flip_single_v1_plane(self, plane):
        # Split hexstring into bytes (2 ascii chars), reverse, rejoin
        # This causes a vertical flip
        return "".join([plane[x:x+2] for x in reversed(range(0, len(plane), 2))])

    def display_v2_or_v3(self, ply, content):
        if self.version <= 2:
            (ver, probs, planes, us_ooo, us_oo, them_ooo, them_oo, us_black, rule50_count, move_count, winner) = self.this_struct.unpack(content)
        else:
            (ver, probs, planes, rule50_count, us_ooo, us_oo, them_ooo, them_oo, us_black, move_count, winner, unused) = self.this_struct.unpack(content)
        assert self.version == int.from_bytes(ver, byteorder="little")
        # Enforce move_count to 0
        move_count = 0
        # Unpack planes.
        for hist in range(self.NUM_HIST):
            for idx, piece in enumerate(PIECES):
                start = hist*self.NUM_PLANES*8+idx*8
                end = start + 8
                self.update_board(hist, piece, int.from_bytes(planes[start:end], byteorder="big"))
            if planes[self.NUM_PLANES*8+12*8:self.NUM_PLANES*8+12*8+8] != struct.pack('II', 0, 0):
                self.history[hist].reps = 1
                assert planes[self.NUM_PLANES*8+12*8:self.NUM_PLANES*8+12*8+8] == struct.pack('II', 0xffffffff, 0xffffffff)
            if self.version <= 2:
                # It's impossible to have a position in training with reps=2
                # Because that is already a draw.
                # Note version 3 this plane doesn't even exist
                assert planes[hist*self.NUM_PLANES*8+13*8:hist*self.NUM_PLANES*8+13*8+8] == struct.pack('II', 0, 0)
        self.us_ooo = us_ooo
        self.us_oo = us_oo
        self.them_ooo = them_ooo
        self.them_oo = them_oo
        self.us_black = us_black
        if self.version == 2:
            self.rule50_count = rule50_count
        else:
            self.rule50_count = int(round(rule50_count*100))
        for idx in range(0, len(probs), 4):
            self.probs.append(struct.unpack("f", probs[idx:idx+4])[0])
        print("ply {} move {} (Not actually part of training data)".format(
            ply+1, (ply+2)//2))
        print(self.describe())

    def convert_v2_to_v3(self, content, fout):
        # TODO Half the code uses these decode fields
        # and the other half directly uses content...
        (ver, probs, planes, us_ooo, us_oo, them_ooo, them_oo, us_black, rule50_count, move_count, winner) = self.v2_struct.unpack(content)

        idx = 0
        new_content = VERSION3 # int32 version (4 bytes)
        idx += 4

        # Loop over all new_move_idx in the new map, and pick
        # the probs from the old map
        for new_move_idx in range(len(self.new_rev_white_move_map)):
            # Get the algebraic move
            if us_black:
                move = self.new_rev_black_move_map[new_move_idx]
            else:
                move = self.new_rev_white_move_map[new_move_idx]
            # Find the old_move_idx
            old_move_idx = self.old_move_map[move]
            # Grab the prob from the old_move_idx, and append in order
            # as we loop over new_move_idx
            new_content += content[idx+old_move_idx*4:idx+old_move_idx*4+4]
        idx += self.NUM_POLICY_MOVES*4

        # TODO: Loop over unused old map promotion moves and assert they are zero

        for hist in range(self.NUM_HIST):
            flip = hist % 2 == 1
            if not flip:
                us_offset = 0
                them_offset = self.NUM_PIECE_TYPES
            else:
                # V1 and V2 had us/them wrong for odd history planes
                us_offset = self.NUM_PIECE_TYPES
                them_offset = 0
            for offset in [us_offset, them_offset]:
                for i in range(self.NUM_PIECE_TYPES):
                    start = idx+hist*self.NUM_PLANES*8+offset*8+i*8
                    if flip:
                        new_content += bytes(reversed(content[start:start+8]))
                    else:
                        new_content += content[start:start+8]
            new_content += content[idx+hist*self.NUM_PLANES*8+12*8:idx+hist*self.NUM_PLANES*8+12*8+8] # reps=1
            # Omit reps=2 plane
            # new_content += content[idx+hist*self.NUM_PLANES*8+13*8:idx+hist*self.NUM_PLANES*8+13*8+8] # reps=2
        idx += self.NUM_HIST*self.NUM_PLANES*8

        new_content += struct.pack("f", rule50_count/100)
        new_content += content[idx:idx+5]   # us_ooo, us_oo, them_ooo, them_oo, us_black
        new_content += content[idx+6:idx+8] # move_count, winner
        new_content += struct.pack("B", 0)  # unused
        fout.write(new_content)

def main(args):
    for filename in args.files:
        foutname = "v3bin_" + re.sub("\.gz", "", filename)
        fout = open(foutname, "wb")
        #print("Parsing {}".format(filename))
        with gzip.open(filename, 'rb') as f:
            chunkdata = f.read()
            if chunkdata[0:4] == b'\1\0\0\0':
                print("Invalid version")
            elif chunkdata[0:4] == VERSION2:
                #print("debug Version2")
                for i in range(0, len(chunkdata), V2_BYTES):
                    ts = TrainingStep(2)
                    if args.display:
                        ts.display_v2_or_v3(i//V2_BYTES, chunkdata[i:i+V2_BYTES])
                    if args.convert:
                        ts.convert_v2_to_v3(chunkdata[i:i+V2_BYTES], fout)
            elif chunkdata[0:4] == VERSION3:
                #print("debug Version3")
                for i in range(0, len(chunkdata), V3_BYTES):
                    ts = TrainingStep(3)
                    if args.display:
                        ts.display_v2_or_v3(i//V3_BYTES, chunkdata[i:i+V3_BYTES])
                    if args.convert:
                        raise Exception("Convert from V3 to V3?")
            else:
                parser = chunkparser.ChunkParser(chunkparser.ChunkDataSrc([chunkdata]), workers=1)
                gen1 = parser.convert_chunkdata_to_v2(chunkdata)
                ply = 1
                for t1 in gen1:
                    ts = TrainingStep(2)
                    if args.display:
                        ts.display_v2_or_v3(ply, t1)
                    if args.convert:
                        ts.convert_v2_to_v3(t1, fout)
                    ply += 1
                    # TODO maybe detect new games and reset ply count
                    # It's informational only
                for _ in parser.parse():
                    # TODO: What is happening here?
                    #print("debug drain", len(_))
                    pass
        if args.convert:
            fout.close()
            with open(foutname, "rb") as fin:
                with gzip.open(foutname+".gz", "wb") as fout:
                    shutil.copyfileobj(fin, fout)

if __name__ == '__main__':
    usage_str = """
This script can parse training files and display them,
or convert them to another format."""

    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=usage_str)
    parser.add_argument(
            "--display", action="store_true",
            help="Display a visualization of the training data")
    parser.add_argument(
            "--convert", action="store_true",
            help="Convert training data to V3")
    parser.add_argument(
            "files", type=str, nargs="+",
            help="Debug data files (training*.gz)")
    args = parser.parse_args()

    if not args.display and not args.convert:
        args.display = True

    main(args)
