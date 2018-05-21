#!/usr/bin/env python

# This file is part of Leela Chess.
# Copyright (C) 2018 Folkert Huizinga
#
# Leela Chess is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Leela Chess is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Leela Chess. If not, see <http://www.gnu.org/licenses/>.

import psycopg2
import argparse
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import chess
import os

def get_configuration():
    """
    Returns a populated configuration
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='gorm',
                        help="database name")
    parser.add_argument('--username', type=str, default='',
                        help="database username", required=True)
    parser.add_argument('--password', type=str, default='',
                        help="database password", required=True)
    parser.add_argument('--limit', type=int, default=1,
                        help="plot last LIMIT network statistics")
    parser.add_argument('--output', type=str, default='/tmp',
                        help="output directory to store plots")

    return parser.parse_args()


def plot_stats(stats, name, cfg):
    """
    Plots various chess statistics
    """
    types  = ['o', '.', '^', 'v', '+', 'x']
    colors = ['w', 'k', 'b', 'g', 'm', 'r']
    edges  = ['k', 'k', 'b', 'g', 'm', 'r']
    w = np.sum(stats['white'])
    b = np.sum(stats['black'])
    d = np.sum(stats['draw'])
    t = w + b + d
    w = int(np.round(w / float(t) * 100))
    b = int(np.round(b / float(t) * 100))
    d = int(np.round(d / float(t) * 100))
    max_plies = stats['white'].size
    fig=plt.figure(figsize=(12, 5), dpi=80)
    plt.xlim(0, max_plies)
    plt.xlabel('ply (half-move)')
    plt.ylabel('games')
    games = int(np.sum(stats['plycount']))
    cutoff = (stats['plycount'][max_plies-1][0] / float(games)) * 100
    plt.title('{} games, (w {}%, b {}%, d {}%) - {:.2f}% cutoff [{}]'.format(games, w, b, d, cutoff, name))

    for i, k in enumerate(['white', 'black', 'nomaterial', 'stalemate', '3-fold', '50-move']):
        stats[k][stats[k] == 0] = np.nan
        plt.plot(range(1, max_plies), stats[k][:max_plies-1], "{}".format(types[i]+colors[i]), label=k, markeredgecolor=edges[i])

    plt.legend()
    filename = os.path.join(cfg.output, '{}.png'.format(name))
    fig.savefig(filename, bbox_inches='tight')
    print("Saved as `{}'".format(filename))


def main(cfg):
    conn = psycopg2.connect(dbname=cfg.database, user=cfg.username, password=cfg.password)
    cur = conn.cursor()
    cur.execute("SELECT * FROM networks ORDER BY id DESC LIMIT {};".format(cfg.limit))
    cur2 = conn.cursor()

    for i in range(cur.rowcount):
        netrow = cur.fetchone()
        cur2.execute("SELECT * FROM training_games WHERE network_id = {};".format(netrow[0]))
        MAX_PLIES = 450
        names = ['plycount', 'checkmate', 'stalemate', 'gameover', 'nomaterial', 'white', 'black', 'draw', '3-fold', '50-move']
        stats = {}
        failed = 0

        for name in names:
            stats[name] = np.zeros((MAX_PLIES, 1), dtype=np.float32)

        for j in range(cur2.rowcount):
            row = cur2.fetchone()
            board = chess.Board()
            filename = '/home/web/leela-chess/go/src/server/pgns/run1/' + str(row[0]) + '.pgn'
            try:
              with open(filename) as f:
                moves = f.read().split('.')
            except:
              print('Unable to find', filename)
              continue

            fail = False
            for move in moves[1:]:
                san = move.split()
                try:
                    board.push_san(san[0])
                except:
                    fail = True
                    break

                if len(san) > 1:
                    try:
                        board.push_san(san[1])
                    except:
                        fail = True
                        break

            plies = len(board.move_stack) - 1
            if not fail and plies < MAX_PLIES:
                stats['plycount'][plies] += 1
                if board.is_checkmate():
                    stats['checkmate'][plies] += 1
                    if plies % 2 == 0:
                        stats['white'][plies] += 1
                    else:
                        stats['black'][plies] += 1
                else:
                    stats['draw'][plies] += 1

                if board.is_stalemate():
                    stats['stalemate'][plies] += 1
                elif board.is_insufficient_material():
                    stats['nomaterial'][plies] += 1
                elif board.can_claim_threefold_repetition():
                    stats['3-fold'][plies] += 1
                elif board.can_claim_fifty_moves():
                    stats['50-move'][plies] += 1

                if board.is_game_over():
                    stats['gameover'][plies] += 1

            elif fail:
                failed += 1

        plot_stats(stats, netrow[3][:8], cfg)



if __name__ == "__main__":
    cfg = get_configuration()
    main(cfg)
