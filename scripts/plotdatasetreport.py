#!/bin/python

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
