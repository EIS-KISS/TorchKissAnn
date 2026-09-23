#!/usr/bin/env python3
# TorchKissAnn - A collection of tools to train various types of Machine learning
# algorithms on various types of EIS data
# Copyright (C) 2026 Carl Klemm <carl@uvos.xyz>
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

"""Plot a confusion matrix as a heatmap in the style of scripts/shootoutplot.py.

Reads a confusion matrix csv and writes a grayscale PDF heatmap plus, with --csv, a row
normalized copy. The input format is the one the test tool dumps: the first line holds the
class names separated by commas (a trailing comma is tolerated), every following line is one
true class and holds the counts per predicted class, so the diagonal is the correct
predictions.

Like the shootout heatmaps the colormap is inverted gray, so the dark cells carry the most
weight, and every cell is labelled with black or white text whichever reads better on it. The
matrix is printed as raw counts by default; --normalize divides every row by its sum, turning
the diagonal into the per class recall and the cells into probabilities, which is the form that
compares classes of different sizes. The overall accuracy and the per class recall are printed
to stdout either way.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Match the shootout plots: an inverted gray colormap so the dark cells are the heavy ones,
# and TrueType outlines so the labels stay scalable once the figure is included in the thesis.
plt.rcParams["image.cmap"] = "gray_r"
plt.rcParams["pdf.fonttype"] = 42


def textShade(value: float, low: float, high: float) -> str:
	"""Black or white label text, whichever reads better on the gray cell behind it."""
	return "white" if high - low > 0 and value > (low + high) / 2 else "black"


def readConfusion(path: Path):
	"""Pull the class names and the count matrix out of a confusion csv."""
	lines = [line for line in path.read_text().splitlines() if line.strip()]
	if len(lines) < 2:
		raise SystemExit(f"{path} holds no matrix, expected a name line and one row per class")

	def split(line: str) -> list:
		fields = [field.strip() for field in line.split(",")]
		while fields and fields[-1] == "":
			fields.pop()
		return fields

	names = split(lines[0])
	rows = []
	for line in lines[1:]:
		fields = split(line)
		rows.append([float(field) if field else 0.0 for field in fields])
	matrix = np.array(rows, dtype=float)

	# the names and the matrix have to describe the same grid; pad the shorter of the two with
	# numeric labels rather than dropping data, so a malformed header still yields a plot
	size = max(len(names), matrix.shape[1])
	names += [str(index) for index in range(len(names), size)]
	if matrix.shape[1] < size:
		matrix = np.pad(matrix, ((0, 0), (0, size - matrix.shape[1])))
	if matrix.shape[0] < size:
		matrix = np.pad(matrix, ((0, size - matrix.shape[0]), (0, 0)))
	return names, matrix[:size, :size]


def plotConfusion(names: list, matrix: np.ndarray, outpath: Path,
				  normalize: bool, skipZero: bool, title: str) -> None:
	"""Draw the heatmap: grayscale cells, per cell label, class names on both axes."""
	data = matrix.copy()
	if normalize:
		totals = data.sum(axis=1, keepdims=True)
		totals[totals == 0] = 1.0
		data = data / totals
	size = len(names)

	fig, ax = plt.subplots(figsize=(max(8, 0.62 * size + 4), max(7, 0.52 * size + 3)))
	image = ax.imshow(data, aspect="auto", vmin=0, vmax=np.nanmax(data))
	ax.set_xticks(range(size), names, rotation=90, ha="center", fontsize=15)
	ax.set_yticks(range(size), names, fontsize=15)
	low, high = np.nanmin(data), np.nanmax(data)
	formatValue = (lambda value: f"{value:.2f}") if normalize else (lambda value: f"{value:.0f}")
	for i in range(size):
		for j in range(size):
			value = data[i][j]
			if np.isnan(value) or (skipZero and value == 0):
				continue
			ax.text(j, i, formatValue(value), ha="center", va="center", fontsize=14,
					color=textShade(value, low, high))
	ax.set_xlabel("Predicted", fontsize=20)
	ax.set_ylabel("True", fontsize=20)
	ax.set_title(title if title is not None else ("Confusion matrix" + (", row normalized" if normalize else "")), fontsize=20)
	fig.colorbar(image, ax=ax, fraction=0.046)
	fig.tight_layout()
	fig.savefig(outpath, dpi=140)
	plt.close(fig)


def summarize(names: list, matrix: np.ndarray) -> None:
	"""Print the overall accuracy and the per class recall to stdout."""
	totals = matrix.sum(axis=1)
	correct = np.diag(matrix)
	grand = matrix.sum()
	recall = np.divide(correct, totals, out=np.full_like(correct, np.nan), where=totals > 0)
	order = np.argsort(-np.nan_to_num(recall))
	print(f"overall accuracy: {correct.sum() / grand:.4f} "
		  f"({int(correct.sum())} of {int(grand)} spectra correct)")
	print("per class recall (true positives / true count):")
	for index in order:
		count = int(totals[index])
		if count == 0:
			continue
		print(f"  {names[index]:<16} {recall[index]:.4f}  ({int(correct[index])}/{count})")


def main():
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("confusion", type=Path, help="the confusion matrix csv")
	parser.add_argument("-o", "--output", type=Path, default=None,
						help="output pdf (default: the input stem with .pdf)")
	parser.add_argument("-n", "--normalize", action="store_true",
						help="divide each row by its sum, plotting probabilities instead of counts")
	parser.add_argument("--skip-zero", action="store_true",
						help="leave empty cells unlabelled instead of printing 0")
	parser.add_argument("--title", default=None, help="override the plot title")
	parser.add_argument("--csv", type=Path, default=None,
						help="also write the (optionally normalized) matrix as a csv with names")
	args = parser.parse_args()

	names, matrix = readConfusion(args.confusion)
	outpath = args.output or args.confusion.with_suffix(".pdf")
	plotConfusion(names, matrix, outpath, args.normalize, args.skip_zero, args.title)

	if args.csv:
		data = matrix.copy()
		if args.normalize:
			totals = data.sum(axis=1, keepdims=True)
			totals[totals == 0] = 1.0
			data = data / totals
		header = ",".join(names)
		body = "\n".join(name + "," + ",".join(f"{value:g}" for value in row)
						for name, row in zip(names, data))
		args.csv.write_text(header + "\n" + body + "\n")

	summarize(names, matrix)
	print(f"wrote {outpath}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
