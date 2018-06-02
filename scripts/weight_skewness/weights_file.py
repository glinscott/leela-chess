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

# Only supports weights version 2

import gzip

LEELA_WEIGHTS_VERSION = '2'


def read_weights_file(filename):
    if '.gz' in filename:
        opener = gzip.open
    else:
        opener = open
    with opener(filename, 'r') as f:
        version = f.readline().decode('ascii')
        if version != '{}\n'.format(LEELA_WEIGHTS_VERSION):
            raise ValueError("Invalid version {}".format(version.strip()))
        weights = []
        for e, line in enumerate(f):
            line = line.decode('ascii')
            weight = list(map(float, line.split(' ')))
            weights.append(weight)
            if e == 1:
                filters = len(line.split(' '))
        blocks = e - (3 + 14)
        if blocks % 8 != 0:
            raise ValueError("Inconsistent number of weights in the file")
        blocks //= 8
    return (filters, blocks, weights)