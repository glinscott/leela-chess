#!/usr/bin/env python

# This file is part of Leela Chess.
# Copyright (C) 2018 Brian Konzman
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


# This script only works for 15x196 networks

from weights_file import read_weights_file
from scipy.stats import skew
from scipy.stats import kurtosis
import numpy as np
import os

import matplotlib.pyplot as plt


directory = os.getcwd()

data_points = list()
ids = list()

files = os.listdir(directory)
for file in files:
    id = int(file.split('_')[1].split('.')[0])
    ids.append(id)

ids.sort()

data_points = list()
policy_skews = list()
value_skews = list()
ids_used = list()
first_conv_inputs_skews = list()
second_conv_inputs_skews = list()
middle_conv_inputs_skews = list()
last_conv_inputs_skews = list()

FIRST_15x196_ID = 227
POLICY_FCL_INDEX = 128

for id_number in ids:
    filename = 'weights_' + str(id_number) + '.txt.gz'
    if filename.endswith(".gz") and id_number >= FIRST_15x196_ID:
        file = os.path.join(directory, filename)
        filters, blocks, weights = read_weights_file(file)

        if type(weights) == list:
            policy_skews.append(skew(weights[POLICY_FCL_INDEX]))

            ids_used.append(id_number)
            print('ID' + str(id_number) + ' complete')
        else:
            print(type(weights))
            print(str(id_number))

        continue
    else:
        continue

plt.plot(ids_used, policy_skews)

plt.show()
