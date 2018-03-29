#!/usr/bin/env python3

import sys
import chunkparser
import numpy as np
import gzip

with gzip.open(sys.argv[1], 'rb') as f:
    v1 = f.read()

with gzip.open(sys.argv[2], 'rb') as f:
    v2 = f.read()

parser = chunkparser.ChunkParser(chunkparser.ChunkDataSrc([v1, v2]), workers=1)
gen1 = parser.convert_chunkdata_to_v2(v1)
gen2 = parser.convert_chunkdata_to_v2(v2)

for t1 in gen1:
    t2 = next(gen2)
    t1 = parser.convert_v2_to_tuple(t1)
    t2 = parser.convert_v2_to_tuple(t2)
    p1 = np.frombuffer(t1[1], dtype=np.float32, count=1924)
    p2 = np.frombuffer(t2[1], dtype=np.float32, count=1924)
    pl1 = np.frombuffer(t1[0], dtype=np.uint8, count=120*8*8)
    pl2 = np.frombuffer(t2[0], dtype=np.uint8, count=120*8*8)
    assert((pl1[0] == pl2[0]).all())
    assert(t1[2] == t2[2])
    a = np.argsort(p1[p1>0])
    b = np.argsort(p2[p2>0])
    assert((a == b).all())

# drain the parser
for _ in parser.parse():
    pass
