#!/usr/bin/env python3
#
# TorchKissAnn -- machine learning for electrochemical impedance spectroscopy
# Copyright (C) 2026 Philipp Sohnius
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""
Aggregate the long classification shootout runs into convergence tables and plots.

	./scripts/shootoutlongplot.py [-d shootout/long] [-b shootout]

Reads <dir>/results/<job>/ (job names NNN_e<EPOCHS>_<architecture>) plus the parent
run's summary/trials.csv for the short-budget baseline and dataset_report_test.txt for
the class names. Writes into <dir>/summary/:

  convergence.csv        per architecture: epochs run, best epoch, best/final accuracy,
                         accuracy at the old budget, gain after it, stillImproving
  learning_curves.pdf    test accuracy vs epoch, one line per architecture
  gain_after_20.pdf      how much accuracy the epochs beyond the old budget bought
  ranking_shift.pdf      old budget vs long accuracy per architecture
  class_accuracy.pdf     per class accuracy at the best epoch, per architecture
  report.md              the tables above

Unlike the regression follow up there is no circuit dimension: the classification
shootout trains on a single dataset, so each architecture is one run and the tables
and plots are per architecture only.

The question this answers: most short-budget trials were still improving when they
stopped, so was that ranking an artifact of the truncated budget?
"""

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

# same grayscale style as shootoutplot.py
GRAY_SHADES = [f"{v:.2f}" for v in np.linspace(0.0, 0.72, 8)]
LINE_STYLES = ("-", "--", "-.", ":")
plt.rcParams["axes.prop_cycle"] = cycler(linestyle=LINE_STYLES) * cycler(color=GRAY_SHADES)
plt.rcParams["image.cmap"] = "gray_r"

BAR_HATCHES = ("", "//", "\\\\", "xx", "..", "||", "--", "++")
plt.rcParams["hatch.linewidth"] = 1.8

plt.rcParams["pdf.fonttype"] = 42

# the long jobs are NNN_e<EPOCHS>_<architecture>; architecture names have no underscores
JOB_RE = re.compile(r"^\d+_e\d+_(?P<arch>.+)$")


def styleBars(axes, title: str = None) -> None:
	"""Make the first bar series solid black, the rest white with black hatches.

	The legend has to be created after the patches are restyled, because matplotlib copies
	the patch appearance into the legend handles when the legend is built.
	"""
	for index, container in enumerate(axes.containers):
		face = "black" if index == 0 else "white"
		hatch = BAR_HATCHES[index % len(BAR_HATCHES)]
		for patch in container:
			patch.set_facecolor(face)
			patch.set_hatch(hatch)
			patch.set_edgecolor("black")
			patch.set_linewidth(0.7)
	handles, labels = axes.get_legend_handles_labels()
	if handles:
		axes.legend(handles, labels, title=title)


def textShade(value: float, low: float, high: float) -> str:
	"""Black or white label text, whichever reads better on the gray cell behind it."""
	return "white" if high - low > 0 and value > (low + high) / 2 else "black"


def readClassNames(path):
	"""The class names from a torchkissann_datasetinfo report, in index order."""
	if not os.path.isfile(path):
		return []
	lines = [line for line in open(path).read().splitlines() if line.strip()]
	for index, line in enumerate(lines):
		if [field.strip() for field in line.split(",")][:3] == ["0", "1", "2"]:
			return [field.strip() for field in lines[index + 1].split(",")]
	return []


def splitJob(job):
	"""NNN_e<EPOCHS>_<architecture> -> the architecture, or None if it does not match."""
	m = JOB_RE.match(job)
	return m.group("arch") if m else None


def readCurves(resultsdir):
	"""-> {arch: {epoch: testAccuracy}}. Each long job trains one network, one trial."""
	curves = {}
	for job in sorted(os.listdir(resultsdir)):
		arch = splitJob(job)
		if arch is None:
			continue
		path = os.path.join(resultsdir, job, "curves.csv")
		if not os.path.isfile(path):
			continue
		series = {}
		with open(path, newline="") as fh:
			for row in csv.DictReader(fh):
				try:
					series[int(row["epoch"])] = float(row["testAccuracy"])
				except (ValueError, KeyError):
					continue
		if series:
			curves[arch] = series
	return curves


def readClassAccuracy(resultsdir):
	"""-> {arch: {class: accuracy}} at the epoch with the best mean per class accuracy."""
	out = {}
	for job in sorted(os.listdir(resultsdir)):
		arch = splitJob(job)
		if arch is None:
			continue
		path = os.path.join(resultsdir, job, "classaccuracy.csv")
		if not os.path.isfile(path):
			continue
		byEpoch = defaultdict(dict)
		with open(path, newline="") as fh:
			for row in csv.DictReader(fh):
				try:
					byEpoch[int(row["epoch"])][int(row["class"])] = float(row["accuracy"])
				except (ValueError, KeyError):
					continue
		if not byEpoch:
			continue
		best = max(byEpoch, key=lambda e: sum(byEpoch[e].values()) / len(byEpoch[e]))
		out[arch] = byEpoch[best]
	return out


def readBaseline(base):
	"""the short shootout: best test accuracy per architecture family."""
	path = os.path.join(base, "summary", "trials.csv")
	if not os.path.isfile(path):
		return {}
	best = {}
	with open(path, newline="") as fh:
		for row in csv.DictReader(fh):
			if row.get("status") != "ok" or row.get("family", "") == "":
				continue
			try:
				acc = float(row["bestAccuracy"])
			except (ValueError, KeyError):
				continue
			family = row["family"]
			if family not in best or acc > best[family]:
				best[family] = acc
	return best


def mean(values):
	values = [v for v in values if v is not None and math.isfinite(v)]
	return sum(values) / len(values) if values else None


def fmt(v, nd=4):
	if v is None or (isinstance(v, float) and not math.isfinite(v)):
		return ""
	return f"{v:.{nd}f}"


def markdownTable(headers, rows):
	lines = ["| " + " | ".join(headers) + " |",
	         "|" + "|".join(["---"] * len(headers)) + "|"]
	for row in rows:
		lines.append("| " + " | ".join(str(c) for c in row) + " |")
	return "\n".join(lines)


def main():
	parser = argparse.ArgumentParser(description="plot the long classification shootout")
	parser.add_argument("-d", "--dir", default="shootout/long",
	                    help="the long run directory (default: shootout/long)")
	parser.add_argument("-b", "--base", default=None,
	                    help="the short shootout run (default: parent of --dir)")
	parser.add_argument("--baseline-epoch", type=int, default=20,
	                    help="the old epoch budget (default: 20)")
	args = parser.parse_args()

	resultsdir = os.path.join(args.dir, "results")
	if not os.path.isdir(resultsdir):
		sys.exit(f"no results in {resultsdir} -- run scripts/shootout_long.sh first")
	base = args.base or os.path.dirname(os.path.abspath(args.dir))

	classNames = readClassNames(os.path.join(base, "dataset_report_test.txt"))
	curves = readCurves(resultsdir)
	classAcc = readClassAccuracy(resultsdir)
	baseline = readBaseline(base)

	archs = sorted(curves)
	if not archs:
		sys.exit("no parsable long runs found")

	summarydir = os.path.join(args.dir, "summary")
	os.makedirs(summarydir, exist_ok=True)
	baseKey = f"accAt{args.baseline_epoch}"
	maxEpochs = max(max(series) for series in curves.values())

	# ---- convergence, per architecture ----
	conv = []
	for arch in archs:
		series = curves[arch]
		epochs = sorted(series)
		bestEpoch = max(epochs, key=lambda e: series[e])
		atBase = series.get(args.baseline_epoch)
		conv.append({
			"architecture": arch,
			"epochs": epochs[-1],
			"bestEpoch": bestEpoch,
			"bestAcc": series[bestEpoch],
			"finalAcc": series[epochs[-1]],
			baseKey: atBase,
			"gainAfterBase": None if atBase is None else series[bestEpoch] - atBase,
			"stillImproving": bestEpoch >= epochs[-1] - 5,
			"baselineAcc": baseline.get(arch),
		})
	conv.sort(key=lambda r: -r["bestAcc"])
	with open(os.path.join(summarydir, "convergence.csv"), "w", newline="") as fh:
		writer = csv.DictWriter(fh, fieldnames=list(conv[0].keys()))
		writer.writeheader()
		writer.writerows(conv)

	# ---- learning curves, one line per architecture ----
	fig, ax = plt.subplots(figsize=(8, 5))
	# no gray shades in the convergence plot: the lines differ in style, not in shade
	ax.set_prop_cycle(cycler(linestyle=LINE_STYLES) * cycler(color=["black"]))
	for arch in archs:
		series = curves[arch]
		epochs = sorted(series)
		ax.plot(epochs, [series[e] for e in epochs], label=arch, lw=1.4)
	ax.axvline(args.baseline_epoch, color="0.4", ls=":", lw=1.2)
	ax.set_xlabel("epoch")
	ax.set_ylabel("test accuracy")
	ax.set_title(f"Test accuracy over up to {maxEpochs} epochs, one line per architecture\n"
	             f"(dotted: the {args.baseline_epoch} epoch budget)")
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize=8, loc="lower right")
	fig.tight_layout()
	fig.savefig(os.path.join(summarydir, "learning_curves.pdf"), dpi=140)
	plt.close(fig)

	# ---- gain beyond the old budget ----
	fig, ax = plt.subplots(figsize=(8, 4.2))
	gains = [next(r["gainAfterBase"] for r in conv if r["architecture"] == a) for a in archs]
	ax.bar(range(len(archs)), [g if g is not None else 0 for g in gains],
	       color="0.15", edgecolor="black", linewidth=0.7)
	ax.set_xticks(range(len(archs)))
	ax.set_xticklabels(archs, fontsize=8)
	ax.axhline(0, color="black", lw=0.8)
	ax.set_ylabel(f"accuracy gained after epoch {args.baseline_epoch}")
	ax.set_title("What the epochs beyond the old budget bought")
	ax.grid(True, axis="y", alpha=0.3)
	fig.tight_layout()
	fig.savefig(os.path.join(summarydir, f"gain_after_{args.baseline_epoch}.pdf"), dpi=140)
	plt.close(fig)

	# ---- ranking shift ----
	fig, ax = plt.subplots(figsize=(8, 4.2))
	width = 0.38
	old = [baseline.get(a) for a in archs]
	new = [next(r["bestAcc"] for r in conv if r["architecture"] == a) for a in archs]
	ax.bar([j - width / 2 for j in range(len(archs))], [o if o is not None else 0 for o in old],
	       width, label=f"{args.baseline_epoch} epochs (short shootout)")
	ax.bar([j + width / 2 for j in range(len(archs))], [n if n is not None else 0 for n in new],
	       width, label="long run")
	ax.set_xticks(range(len(archs)))
	ax.set_xticklabels(archs, fontsize=8)
	ax.set_ylabel("best test accuracy")
	ax.set_title("Short budget against the long run, per architecture")
	ax.grid(True, axis="y", alpha=0.3)
	styleBars(ax)
	fig.tight_layout()
	fig.savefig(os.path.join(summarydir, "ranking_shift.pdf"), dpi=140)
	plt.close(fig)

	# ---- per class accuracy at the best epoch ----
	if classAcc:
		# rows are the classes, ordered by their mean accuracy across the architectures so
		# the hardest classes sit at the bottom
		classIndex = sorted({c for per in classAcc.values() for c in per},
		                    key=lambda c: -mean([per.get(c) for per in classAcc.values()]))
		labels = [classNames[c] if c < len(classNames) else str(c) for c in classIndex]
		pivot = np.array([[classAcc.get(a, {}).get(c, np.nan) for c in classIndex] for a in archs],
		                 dtype=float).T
		fig, ax = plt.subplots(figsize=(2.2 * len(archs) + 3.5, 0.34 * len(classIndex) + 3))
		image = ax.imshow(pivot, vmin=0, vmax=1, aspect="auto")
		ax.set_xticks(range(len(archs)), archs, fontsize=8)
		ax.set_yticks(range(len(classIndex)), labels, fontsize=7)
		for i in range(pivot.shape[0]):
			for j in range(pivot.shape[1]):
				value = pivot[i][j]
				if not np.isnan(value):
					ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
							color=textShade(value, 0, 1))
		ax.set_title("Per class test accuracy at each architecture's best epoch")
		fig.colorbar(image, ax=ax, fraction=0.046)
		fig.tight_layout()
		fig.savefig(os.path.join(summarydir, "class_accuracy.pdf"), dpi=140)
		plt.close(fig)

	# ---- report ----
	def cell(v):
		if v is None:
			return ""
		if isinstance(v, bool):
			return "yes" if v else "no"
		if isinstance(v, float):
			return fmt(v)
		return v

	lines = [f"# Long classification shootout: {len(archs)} architectures", ""]
	lines.append(markdownTable(
		["architecture", "epochs", "best ep", "best acc", "final acc",
		 f"acc @ {args.baseline_epoch}", "gain", "improving", f"short ({args.baseline_epoch} ep)"],
		[[cell(r[k]) for k in ("architecture", "epochs", "bestEpoch", "bestAcc", "finalAcc",
		                      baseKey, "gainAfterBase", "stillImproving", "baselineAcc")]
		 for r in conv]))
	lines += ["", f"Plots: `learning_curves.pdf`, `gain_after_{args.baseline_epoch}.pdf`, "
	          f"`ranking_shift.pdf`, `class_accuracy.pdf`.", ""]
	with open(os.path.join(summarydir, "report.md"), "w") as fh:
		fh.write("\n".join(lines))

	print(f"wrote {summarydir}/report.md")
	print(f"{len(conv)} runs, architectures: {', '.join(archs)}")


if __name__ == "__main__":
	main()
