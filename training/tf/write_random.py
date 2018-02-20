#!/usr/bin/env python3

import sys
import yaml
from tfprocess import TFProcess
import tensorflow as tf

def main():
    if len(sys.argv) != 2:
        print("Usage: {} config.yaml".format(sys.argv[0]))
        return 1

    cfg = yaml.safe_load(open(sys.argv[1], 'r').read())
    print(yaml.dump(cfg, default_flow_style=False))

    batch = [
            tf.placeholder(tf.float32, [None, 120, 8 * 8]),
            tf.placeholder(tf.float32, [None, 1924]),
            tf.placeholder(tf.float32, [None, 1]),
    ]
    tfprocess = TFProcess(cfg, batch)
    tfprocess.save_leelaz_weights('weights.txt')

if __name__ == '__main__':
    main()
