#!/bin/python

# TorchKissAnn - A collection of tools to train various types of Machine learning
# algorithms on various types of EIS data
# Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
#
# This file is part of TorchKissAnn.
#
# TorchKissAnn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TorchKissAnn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.


import matplotlib
import matplotlib.pyplot as plt
import csv
import os
import sys

outDir = ""

def plotHistogram(buckets, lables, name):
    fig, ax = plt.subplots()
    ax.bar(lables, buckets, width = 1)
    ax.set(xlim=(-0.5, len(buckets)-0.5), xticks=lables, ylim=(0, max(buckets)+max(buckets)*0.1))
    fig.set_size_inches(14, 8)
    plt.savefig(outDir + '/' + name + ".png")
    plt.close()

def plotHistrogramFromCsv(filename):
    with open(filename) as file:
        reader = csv.reader(file, delimiter = ',', quotechar='\"')
        line = reader.__next__()
        if line[0] != "class_predictions":
            print(f"{filename} is not a valid histrogram file")
            return
        line = reader.__next__()
        buckets = []
        for cell in line:
            buckets.append(float(cell))
        plotHistogram(buckets, range(0, len(buckets)), os.path.splitext(os.path.basename(filename))[0])


def plotCsvDir(directory):
    entrylist = os.listdir(directory)
    for dirent in entrylist:
        path = directory + '/' + dirent
        if os.path.isfile(path):
            print(f"plotting {path}")
            plotHistrogramFromCsv(path)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} [TorchKissAnn log DIR] [OUTPUT DIR]")
        exit()
    outDir = sys.argv[2]
    os.makedirs(outDir, exist_ok=True)
    entrylist = os.listdir(sys.argv[1])
    dirlist = []

    font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 14}
    matplotlib.rc('font', **font)
    for dirent in entrylist:
        path = sys.argv[1] + '/' + dirent
        isDir = os.path.isdir(path)
        if isDir:
            print(f"Entering {path}")
            plotCsvDir(path)
