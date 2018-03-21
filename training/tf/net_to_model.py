#!/usr/bin/env python3
import tensorflow as tf
import os
import sys
import yaml
import textwrap
from tfprocess import TFProcess


YAMLCFG = """
%YAML 1.2
---
name: 'online-64x6'
gpu: 0

dataset:
    num_chunks: 200000
    train_ratio: 0.90

training:
    batch_size: 2048
    total_steps: 60000
    shuffle_size: 1048576
    lr_values:
        - 0.04
        - 0.002
    lr_boundaries:
        - 35000
    policy_loss_weight: 1.0
    value_loss_weight: 1.0
    path: /dev/null

model:
    filters: 64
    residual_blocks: 6
...
"""
YAMLCFG = textwrap.dedent(YAMLCFG).strip()
cfg = yaml.safe_load(YAMLCFG)

with open(sys.argv[1], 'r') as f:
    weights = []
    for e, line in enumerate(f):
        if e == 0:
            #Version
            print("Version", line.strip())
            if line != '1\n':
                raise ValueError("Unknown version {}".format(line.strip()))
        else:
            weights.append(list(map(float, line.split(' '))))
        if e == 2:
            filters = len(line.split(' '))
            print("Channels", filters)
    blocks = e - (4 + 14)
    if blocks % 8 != 0:
        raise ValueError("Inconsistent number of weights in the file")
    blocks //= 8
    print("Blocks", blocks)

cfg['model']['filters'] = filters
cfg['model']['residual_blocks'] = blocks
print(yaml.dump(cfg, default_flow_style=False))

x = [
    tf.placeholder(tf.float32, [None, 120, 8*8]),
    tf.placeholder(tf.float32, [None, 1924]),
    tf.placeholder(tf.float32, [None, 1])
    ]

tfprocess = TFProcess(cfg)
tfprocess.init_net(x)
tfprocess.replace_weights(weights)
path = os.path.join(os.getcwd(), cfg['name'])
save_path = tfprocess.saver.save(tfprocess.session, path, global_step=0)
print("Writted model to {}".format(path))
