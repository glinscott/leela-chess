#!/usr/bin/env python3

from tfprocess import TFProcess
import tensorflow as tf

def main():
    batch = [
            tf.placeholder(tf.float32, [None, 120, 8 * 8]),
            tf.placeholder(tf.float32, [None, 1924]),
            tf.placeholder(tf.float32, [None, 1]),
    ]
    tfprocess = TFProcess(batch)
    tfprocess.save_leelaz_weights('weights.txt')

if __name__ == '__main__':
    main()
