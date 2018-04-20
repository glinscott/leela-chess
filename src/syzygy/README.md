TODO List:
* Get cmake to work, for now MSVC or `cd src; make` works.
 
Test procedures:
 
```
position fen 8/8/3k4/8/8/8/3P4/3K4 w - - 0 1
d

 +---+---+---+---+---+---+---+---+
 |   |   |   |   |   |   |   |   |
 +---+---+---+---+---+---+---+---+
 |   |   |   |   |   |   |   |   |
 +---+---+---+---+---+---+---+---+
 |   |   |   | k |   |   |   |   |
 +---+---+---+---+---+---+---+---+
 |   |   |   |   |   |   |   |   |
 +---+---+---+---+---+---+---+---+
 |   |   |   |   |   |   |   |   |
 +---+---+---+---+---+---+---+---+
 |   |   |   |   |   |   |   |   |
 +---+---+---+---+---+---+---+---+
 |   |   |   | P |   |   |   |   |
 +---+---+---+---+---+---+---+---+
 |   |   |   | K |   |   |   |   |
 +---+---+---+---+---+---+---+---+

go
RNG seed: 0xda290309d7b8e800 (thread: 466331404530731046)
info depth 4 nodes 2 nps 200 score cp -319 time 4 pv d1e2 d6e5
info depth 5 nodes 3 nps 333 score cp -223 time 5 pv d1c2 d6c5
info depth 6 nodes 6 nps 625 score cp -97 time 7 pv d1e2 d6e5 e2e3 e5d5
0 info depth 8 nodes 17 nps 640 score cp 3 time 24 pv d1e2 d6e5 e2e3 e5d5 e3d3 d5e5
0 info depth 9 nodes 29 nps 683 score cp 31 time 40 pv d1e2 d6e5 e2e3 e5d5 e3d3 d5e5
info depth 10 nodes 54 nps 898 score cp 52 time 58 pv d1e2 d6e5 e2e3 e5d5 e3d3 d5e5 d3e3
0 0 0 info depth 11 nodes 100 nps 1100 score cp 67 time 89 pv d1e2 d6e5 e2e3 e5d5 e3d3 d5e5 d3e3 e5d5
0 0 0 0 0 0 0 info depth 12 nodes 188 nps 1272 score cp 76 time 146 pv d1e2 d6e5 e2e3 e5d5 d2d4 d5d6 e3e4 d6e6
0 0 -1936485568 0 0 0 0 0 0 0 0 0 0 0 0 0 0 info depth 13 nodes 336 nps 1516 score cp 88 time 220 pv d1e2 d6e5 e2e3 e5d5 d2d4 d5d6 e3e4 d6e6 d4d5
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 info depth 14 nodes 602 nps 1861 score cp 94 time 322 pv d1e2 d6e5 e2e3 e5d5 d2d4 d5d6 e3e4 d6e6 d4d5 e6d6
0 0 0 0 0 0 0 0 738384240 0 0 0 0 0 0 0 0 0 0 0 0 
info string    d4 ->      12   (V: 51.51%) (N:  7.49%) PV: d4 Kd5 Ke2 Kxd4
info string   Kc1 ->      23   (V: 59.32%) (N:  5.28%) PV: Kc1 Kd5 Kc2 Kc4 d3+ Kd4
info string   Ke1 ->      31   (V: 58.62%) (N:  8.21%) PV: Ke1 Ke5 Ke2 Kd4 d3 Kd5 Ke3 Ke5
info string    d3 ->      78   (V: 60.97%) (N: 10.67%) PV: d3 Kd5 Ke2 Kd4 Kd2 Kd5 Ke3 Ke5 d4+
info string   Kc2 ->     229   (V: 60.46%) (N: 28.16%) PV: Kc2 Kc5 Kd3 Kd5 Ke3 Ke5 d4+ Kd5 Kd3 Kd6 Ke4 Ke6
info string   Ke2 ->     329   (V: 61.04%) (N: 40.19%) PV: Ke2 Ke5 Ke3 Kd5 d4 Kd6 Ke4 Ke6 d5+ Kd6 Kd4
```
