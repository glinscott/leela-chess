#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017 Andy Olsen
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
import re
import argparse

def parseGames(filenames, resignrate):
    #print("Begin Analyze resign rate {:0.0f}%".format(resignrate))
    games = len(filenames)
    correct_resigns = 0
    incorrect_resigns = 0
    total_plys = 0
    resign_plys = 0
    for filename in filenames:
        #print("debug filename: {}".format(filename))
        plynum = 0
        who_resigned = None
        resign_plynum = None
        for line in open(filename).readlines():
            m = re.match("^Score: (\S+)", line)
            if m:
                (score,) = m.groups()
                score = float(score)
                #print("debug Score", score)
            m = re.match("^info string stm\s+(\S+) winrate\s+(\S+)%", line)
            if m:
                plynum += 1
                total_plys += 1
                if who_resigned:
                    # We already found who resigned.
                    # continue after just counting plys
                    continue
                resign_plys += 1
                (stm, winrate) = m.groups()
                winrate = float(winrate)
                if winrate < resignrate:
                    who_resigned = stm
                    resign_plynum = plynum
                    #print("debug stm, winrate, plynum", stm, winrate, plynum)
        #print("debug who_resigned {} resign_playnum {} total_plynum {}".format(who_resigned, resign_plynum, plynum))
        if ((score == -1 and who_resigned == "Black") or
            (score ==  1 and who_resigned == "White")):
            #print("debug incorrect resign")
            incorrect_resigns += 1
        elif who_resigned != None:
            #print("debug correct resign")
            correct_resigns += 1
    print("Analyze resign rate {:0.0f}%".format(resignrate))
    print("incorrect {:0.0f}%\ncorrect {:0.0f}%\nplies saved {:0.0f}%".format(
        incorrect_resigns/games*100, correct_resigns/games*100, (total_plys-resign_plys)/total_plys*100))
    print()

if __name__ == "__main__":
    usage_str = """
This script analyzes the debug output from `client -debug`
to determine the impact of various resign thresholds.

Process flow:
  Run `client -debug`

  Analyze results with this script:
    ./resign_analysis.py logs-###/*.log
"""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=usage_str)
    parser.add_argument("data", metavar="files", type=str, nargs="+", help="client debug log files (logs-###/*.log)")
    args = parser.parse_args()
    for rr in (50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.0):
        parseGames(args.data, rr)
