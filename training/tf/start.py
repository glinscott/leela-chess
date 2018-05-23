# This file is part of Leela Chess Zero.
# Copyright (C) 2018 The LCZero Authors
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
# along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import multiprocessing as mp
import yaml
import inotify.adapters
import os
import tensorflow as tf

from chunkparser import ChunkParser
from tfprocess import TFProcess
from train import get_latest_chunks
from train import get_checkpoint
from train import FileDataSrc
from train import SKIP


def run_cycle(cmd, cfg, chunks, tfprocess):
    root_dir = os.path.join(cfg['training']['path'], cfg['name'])
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    train_ratio = cfg['dataset']['train_ratio']
    num_train = int(len(chunks)*train_ratio)
    shuffle_size = cfg['training']['shuffle_size']
    train_parser = ChunkParser(FileDataSrc(chunks[:num_train]),
            shuffle_size=shuffle_size, sample=SKIP, batch_size=ChunkParser.BATCH_SIZE)
    dataset = tf.data.Dataset.from_generator(
        train_parser.parse, output_types=(tf.string, tf.string, tf.string))
    dataset = dataset.map(ChunkParser.parse_function)
    dataset = dataset.prefetch(4)
    train_iterator = dataset.make_one_shot_iterator()

    shuffle_size = int(shuffle_size*(1.0-train_ratio))
    test_parser = ChunkParser(FileDataSrc(chunks[num_train:]), 
            shuffle_size=shuffle_size, sample=SKIP, batch_size=ChunkParser.BATCH_SIZE)
    dataset = tf.data.Dataset.from_generator(
        test_parser.parse, output_types=(tf.string, tf.string, tf.string))
    dataset = dataset.map(ChunkParser.parse_function)
    dataset = dataset.prefetch(4)
    test_iterator = dataset.make_one_shot_iterator()

    tfprocess.init(dataset, train_iterator, test_iterator)

    if os.path.exists(os.path.join(root_dir, 'checkpoint')):
        cp = get_checkpoint(root_dir)
        tfprocess.restore(cp)

    # Sweeps through all test chunks statistically
    num_evals = (len(chunks)-num_train)*10 // ChunkParser.BATCH_SIZE
    print("Using {} evaluation batches".format(num_evals))

    for _ in range(cfg['training']['total_steps']):
        tfprocess.process(ChunkParser.BATCH_SIZE, num_evals)

    tfprocess.save_leelaz_weights(cmd.output)

    train_parser.shutdown()
    test_parser.shutdown()


def main(cmd):
    with open(cmd.cfg, 'r') as f:
        cfg = yaml.safe_load(f.read())

    print(yaml.dump(cfg, default_flow_style=False))

    # load latest chunks from disk
    num_chunks = cfg['dataset']['num_chunks']
    chunks = get_latest_chunks(cfg['dataset']['input'], num_chunks)

    num_new_games = num_chunks * (cmd.new_games / 100)
    ChunkParser.BATCH_SIZE = cfg['training']['batch_size']

    # install inotifier that watches on input dir
    train_i = inotify.adapters.Inotify()
    train_i.add_watch(cfg['dataset']['input'])

    tfprocess = TFProcess(cfg)

    # run a cycle
    run_cycle(cmd, cfg, chunks, tfprocess)
    
    i = 0
    for event in train_i.event_gen(yield_nones=False):
        (_, type_names, path, filename) = event

        if 'IN_CLOSE_WRITE' in type_names:
            chunks.pop()
            chunks.insert(0, os.path.join(path, filename))
            i += 1
            print("added {}".format(filename))

        if i >= num_new_games:
            i = 0
            run_cycle(cmd, cfg, chunks, tfprocess)
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
    'Tensorflow pipeline for training Leela Chess.')
    argparser.add_argument('--cfg', type=str, 
        help='yaml configuration with training parameters')
    argparser.add_argument('--output', type=str, 
        help='file to store weights in')
    argparser.add_argument('--new-games', type=int, 
        help='percentage of new games in window before next cycle')

    mp.set_start_method('spawn')
    main(argparser.parse_args())
