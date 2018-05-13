# This file is part of Leela Chess.
# Copyright (C) 2018 github username so-much-meta
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

# Downloads all weights files from lczero.org, does not try to download files already present in destination folder

from bs4 import BeautifulSoup
import os
import requests
import shutil

page = requests.get("http://lczero.org/networks")
soup = BeautifulSoup(page.content, 'html.parser')
network_table = soup.find_all('tr')

download_links = dict()

for i in range(1, len(network_table)):
    row = list(network_table[i])
    net_number = str(row[1]).split('>')[1].split('<')[0]
    download_links[net_number] = 'http://lczero.org' + str(row[3]).split('href="')[1].split('"')[0]

most_recent_net_number = str(list(network_table[1])[1]).split('>')[1].split('<')[0]

directory = os.getcwd()
all_weights = list()
id_numbers = list()

# remove links that we have already downloaded
for filename in os.listdir(directory):
    if filename.endswith(".gz"):
        net_id = filename.split('_')[1].split('.')[0]
        if net_id in download_links.keys():
            download_links.pop(net_id)


keys = download_links.keys()
for k in keys:
    link = download_links.get(k)
    r = requests.get(link, stream=True)
    if r.status_code == 200:
        with open(directory + 'weights_' + k + '.txt.gz', 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)