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
    ax.bar(lables, buckets, width=1, edgecolor="white")
    plt.semilogy()
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.xticks(rotation=90)
    ax.set(xlim=(-0.5, len(buckets)-0.5), xticks=lables, ylim=(0.9, max(buckets)+max(buckets)*0.1))
    fig.set_size_inches(14, 8)
    plt.savefig(outDir + '/' + name + ".png")
    plt.close()

def plotHistrogramFromReport(filename):
    with open(filename) as file:
        reader = csv.reader(file, delimiter = ',', quotechar='\"')
        line = reader.__next__()
        if line[0] != "Name":
            print(f"{filename} is not a valid histrogram file")
            return
        line = reader.__next__()
        line = reader.__next__()
        lables = reader.__next__()
        lables[:] = [cell.strip() for cell in lables]
        line = reader.__next__()
        buckets = []
        for cell in line:
            buckets.append(float(cell))
        plotHistogram(buckets, lables, os.path.splitext(os.path.basename(filename))[0])

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} [DATASETREPORT] [OUTPUT DIR]")
        exit()

    font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 22}
    matplotlib.rc('font', **font)
    outDir = sys.argv[2]
    os.makedirs(outDir, exist_ok=True)
    plotHistrogramFromReport(sys.argv[1])
