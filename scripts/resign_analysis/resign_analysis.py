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
    very_incorrect_resigns = 0
    total_plys = 0
    resign_plys = 0
    draws = 0
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
        if ((score !=  1 and who_resigned == "Black") or
            (score != -1 and who_resigned == "White")):
            incorrect_resigns += 1
            #print("debug incorrect resign filename {} who_resigned {} resign_playnum {} total_plynum {}".format(
                #filename, who_resigned, resign_plynum, plynum))
        elif who_resigned != None:
            #print("debug correct resign")
            correct_resigns += 1
        if ((score == -1 and who_resigned == "Black") or
            (score ==  1 and who_resigned == "White")):
            very_incorrect_resigns += 1
            #print("debug very incorrect resign filename {} who_resigned {} resign_playnum {} total_plynum {}".format(
                #filename, who_resigned, resign_plynum, plynum))
        if score == 0:
            draws += 1
    assert incorrect_resigns >= very_incorrect_resigns
    assert incorrect_resigns + correct_resigns <= games
    resigned_games = incorrect_resigns + correct_resigns
    if resigned_games == 0:
        print("RR {:2.0f}% NR 100%".format(resignrate))
    else:
        print("RR {:2.0f}% Draw {:4.1f}% NR {:4.1f}% I {:4.1f}%/{:4.1f}% VI {:4.1f}%/{:4.1f}% C {:4.1f}%/{:4.1f}% PS {:4.1f}%".format(
            resignrate,
            draws/games*100,
            (games - resigned_games)/games*100,
            incorrect_resigns/games*100,
            incorrect_resigns/resigned_games*100,
            very_incorrect_resigns/games*100,
            very_incorrect_resigns/resigned_games*100,
            correct_resigns/games*100,
            correct_resigns/resigned_games*100,
            (total_plys-resign_plys)/total_plys*100))

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
    print("Analyzing {} games".format(len(args.data)))
    print("RR   - Resign Rate")
    print("Draw - number of drawn games")
    print("NR   - No Resign")
    print("I    - Incorect Resign (would have resigned, but drew or won)")
    print("VI   - Very Incorect Resign (would have resigned, but won)")
    print("C    - Correct Resign (would have resigned, and actually lost")
    print("PS   - Plies Saved")
    print("I, VI, C display two values. Including / excluding the NR games")
    for rr in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0):
        parseGames(args.data, rr)
