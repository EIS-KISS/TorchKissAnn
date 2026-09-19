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

"""Aggregate the classification shootout of scripts/shootout.sh into tables and plots.

Reads <outdir>/results/*/results.csv, curves.csv and classaccuracy.csv plus the
torchkissann_datasetinfo reports and writes <outdir>/summary/:

	trials.csv          one row per trained network
	models.csv          the best trial of every architecture
	curves.csv          per epoch curves carrying the architecture columns
	classaccuracy.csv   per class accuracy of each trial's best epoch
	report.md           the tables below as markdown
	*.pdf               the graphs listed in the report

The tuner reports the test loss as sum(per batch mean NLL)/examples*100, which is
monotonic but scaled. The *NLL columns undo that scaling so they read as plain cross
entropy; the ranking itself does not depend on it.

If <outdir>/long/results exists (scripts/shootout_long.sh), the extended budget runs are read as
well and reported beside the ranking rather than inside it, since they are not comparable at equal
epoch counts. They add longtrials.csv, longcurves.csv, longclassaccuracy.csv, long_convergence.csv,
long_learning_curves.pdf, long_gain_after_<budget>.pdf, long_ranking_shift.pdf and
long_class_accuracy.pdf, all in the same summary directory. That convergence analysis is folded in
from scripts/shootoutlongplot.py, which asks whether the ranking was an artifact of the truncated
epoch budget.
"""

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler

# Everything has to survive being printed in black and white: the palette is grayscale and
# series differ in line style as well as in shade, heatmaps use an inverted gray colormap so
# that the dark cells are the good ones. The color cycler is on the right of the product so
# that it varies fastest, otherwise consecutive series share a color and only differ in line
# style, which bars ignore entirely.
GRAY_SHADES = [f"{v:.2f}" for v in np.linspace(0.0, 0.72, 8)]
LINE_STYLES = ("-", "--", "-.", ":")
plt.rcParams["axes.prop_cycle"] = cycler(linestyle=LINE_STYLES) * cycler(color=GRAY_SHADES)
plt.rcParams["image.cmap"] = "gray_r"

# Gray fills print as muddy smudges, so the first series is solid black and every later one is
# white with a black hatch pattern. Bars are narrower than the slot and outlined so they do not
# touch their neighbours.
BAR_STYLE = dict(width=0.6, edgecolor="black", linewidth=0.7)
# Ordered so the first few series differ in direction, not just density; the default hatch line
# width is too thin to survive printing, hence the rc bump below.
BAR_HATCHES = ("", "//", "\\\\", "xx", "..", "||", "--", "++")
plt.rcParams["hatch.linewidth"] = 1.8

# Figures are written as encapsulated PDFs for the thesis, so embed TrueType outlines instead of
# the Type 3 bitmaps matplotlib uses by default: same vector output, but scalable and searchable
# once included in the document.
plt.rcParams["pdf.fonttype"] = 42

BATCH_SIZE = 256

# A long run whose best epoch lies this close to its last epoch had not levelled off when
# training stopped, so its result is a lower bound and the shootout budget truncated it.
STILL_IMPROVING_WINDOW = 5


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


def readReport(path: Path):
	"""Pull the class names and counts out of a torchkissann_datasetinfo report."""
	if not path.is_file():
		return [], []
	lines = [line for line in path.read_text().splitlines() if line.strip()]
	for index, line in enumerate(lines):
		if [field.strip() for field in line.split(",")][:3] == ["0", "1", "2"]:
			names = [field.strip() for field in lines[index + 1].split(",")]
			counts = [int(float(field)) for field in lines[index + 2].split(",")]
			return names, counts
	return [], []


def readJobTable(outdir: Path, fileName: str, sub: str = "results") -> pd.DataFrame:
	"""Concatenate one csv file per job, keyed by job and trial index."""
	frames = []
	for table in sorted((outdir / sub).glob("*/" + fileName)):
		frame = pd.read_csv(table)
		if frame.empty:
			continue
		frame["job"] = table.parent.name
		frames.append(frame)
	if not frames:
		return pd.DataFrame()
	joined = pd.concat(frames, ignore_index=True)
	joined["key"] = joined["job"] + "/" + joined["trial"].astype(str)
	if "script" in joined:
		joined["script"] = joined["script"].fillna("")
	return joined


def describe(frame: pd.DataFrame) -> pd.DataFrame:
	"""Add the architecture columns that every table and plot is grouped by."""
	def family(row):
		if row["script"]:
			match = re.match(r"[a-zA-Z]+", row["script"])
			return match.group(0) if match else row["script"]
		return row["network"]

	frame["family"] = frame.apply(family, axis=1)
	frame["model"] = [f"{name} {'c2' if script.endswith('_c2.pt') else 'c1'}"
					  if script else network
					  for name, script, network in zip(frame["family"], frame["script"], frame["network"])]
	frame["channels"] = [2 if script.endswith("_c2.pt") else 1 for script in frame["script"]]
	frame["source"] = np.where(frame["script"] != "", "script", "native")
	return frame


def testCount(outdir: Path) -> int:
	_, counts = readReport(outdir / "dataset_report_test.txt")
	return int(sum(counts)) if counts else 0


def loadTrials(outdir: Path) -> pd.DataFrame:
	trials = readJobTable(outdir, "results.csv")
	if trials.empty:
		raise SystemExit(f"no results.csv below {outdir}/results, run scripts/shootout.sh first")
	trials["status"] = trials["status"].fillna("failed")
	trials = describe(trials)

	# The tuner divides the summed per batch losses by the number of examples instead
	# of the number of batches, undo that to get a readable cross entropy.
	examples = testCount(outdir)
	batches = max(1, math.ceil(examples / BATCH_SIZE))
	for column in ("loss", "bestLoss"):
		trials[column + "NLL"] = trials[column] / 100.0 * examples / batches
	return trials


def bestPerModel(trials: pd.DataFrame) -> pd.DataFrame:
	ok = trials[trials["status"] == "ok"]
	if ok.empty:
		return ok
	return ok.sort_values(["model", "bestAccuracy"], ascending=[True, False]).groupby("model").first().reset_index()


def plotAccuracyVsParams(trials: pd.DataFrame, summary: Path, longTrials: pd.DataFrame = None):
	"""Model size against test accuracy, the extended budget runs beside the shootout ones.

	Both series are the best test accuracy of the model's best hyperparameters, so a triangle
	sits above the circle of the same architecture when the longer budget bought something and
	the vertical distance between the two is the effect of the epoch count alone.
	"""
	best = bestPerModel(trials)
	if best.empty:
		return
	fig, ax = plt.subplots(figsize=(9.5, 6.5))
	for _, row in best.iterrows():
		ax.scatter(row["parameters"], row["bestAccuracy"], s=60, marker="o", color="0.0")
		ax.annotate(row["model"], (row["parameters"], row["bestAccuracy"]),
					textcoords="offset points", xytext=(6, 3), fontsize=8)
	# the extended budget runs are drawn apart, they are not comparable at equal epochs
	if longTrials is not None and not longTrials.empty:
		for _, row in bestPerModel(longTrials).iterrows():
			ax.scatter(row["parameters"], row["bestAccuracy"], s=90, marker="^", color="0.0", zorder=3)
			ax.annotate(f"{row['model']} ({int(row['epochs'])} ep)",
						(row["parameters"], row["bestAccuracy"]),
						textcoords="offset points", xytext=(6, -10), fontsize=8)
	ax.set_xscale("log")
	ax.set_xlabel("trainable parameters")
	ax.set_ylabel("best test accuracy")
	ax.set_title("Accuracy against model size, best hyperparameters per model\n"
					"(triangles are the runs with an extended epoch budget, same metric)")
	ax.grid(True, which="both", alpha=0.3)
	fig.tight_layout()
	fig.savefig(summary / "accuracy_vs_parameters.pdf", dpi=140)
	plt.close(fig)


def plotLearningRate(trials: pd.DataFrame, summary: Path):
	ok = trials[trials["status"] == "ok"]
	for decay, group in ok.groupby("weightDecay"):
		pivot = group.groupby(["model", "learningRate"])["bestAccuracy"].max().unstack()
		if pivot.empty:
			continue
		pivot = pivot.sort_values(pivot.columns[0], ascending=False)
		fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(pivot)), 5.5))
		pivot.plot(kind="bar", ax=ax, legend=False, **BAR_STYLE)
		ax.set_ylabel("best test accuracy")
		ax.set_xlabel("")
		ax.set_title(f"Learning rate sweep per architecture (weight decay {decay:g})")
		ax.tick_params(axis="x", rotation=60)
		ax.grid(True, axis="y", alpha=0.3)
		styleBars(ax, "learning rate")
		fig.tight_layout()
		fig.savefig(summary / f"accuracy_by_learningrate_wd{str(decay).replace('.', 'p')}.pdf", dpi=140)
		plt.close(fig)


def plotWeightDecay(trials: pd.DataFrame, summary: Path):
	ok = trials[trials["status"] == "ok"]
	decays = sorted(ok["weightDecay"].unique())
	if len(decays) < 2:
		return
	pivot = ok.groupby(["model", "weightDecay"])["bestAccuracy"].max().unstack().dropna(how="any")
	if pivot.empty:
		return
	pivot = pivot[decays].sort_values(decays[-1], ascending=False)
	fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(pivot)), 5.5))
	pivot.plot(kind="bar", ax=ax, legend=False, **BAR_STYLE)
	ax.set_ylabel("best test accuracy")
	ax.set_xlabel("")
	ax.set_title("Weight decay effect per architecture (best learning rate per bar)")
	ax.tick_params(axis="x", rotation=60)
	ax.grid(True, axis="y", alpha=0.3)
	styleBars(ax, "weight decay")
	fig.tight_layout()
	fig.savefig(summary / "weight_decay_effect.pdf", dpi=140)
	plt.close(fig)


def plotChannels(trials: pd.DataFrame, summary: Path):
	ok = trials[(trials["status"] == "ok") & (trials["script"] != "")]
	pivot = ok.groupby(["family", "channels"])["bestAccuracy"].max().unstack().dropna(how="any")
	if pivot.empty or pivot.shape[1] < 2:
		return
	pivot = pivot.sort_values(pivot.columns[-1], ascending=False)
	fig, ax = plt.subplots(figsize=(8, 5))
	pivot.plot(kind="bar", ax=ax, legend=False, **BAR_STYLE)
	ax.set_ylabel("best test accuracy")
	ax.set_xlabel("")
	ax.set_title("One channel (flat vector) against two channels (Re/Im split)")
	ax.tick_params(axis="x", rotation=45)
	ax.grid(True, axis="y", alpha=0.3)
	styleBars(ax, "input channels")
	fig.tight_layout()
	fig.savefig(summary / "channel_split_effect.pdf", dpi=140)
	plt.close(fig)


def plotNativeTaper(trials: pd.DataFrame, summary: Path):
	native = trials[(trials["status"] == "ok") & (trials["source"] == "native")]
	if native.empty:
		return
	networks = sorted(native["network"].unique())
	learnRates = sorted(native["learningRate"].unique())
	fig, axes = plt.subplots(len(networks), len(learnRates),
		figsize=(4.8 * len(learnRates), 4.2 * len(networks)), squeeze=False)
	for row, network in enumerate(networks):
		for column, rate in enumerate(learnRates):
			ax = axes[row][column]
			part = native[(native["network"] == network) & (native["learningRate"] == rate)]
			grid = part.pivot_table(index="downsampleSteps", columns="extraSteps",
									values="bestAccuracy", aggfunc="max")
			if grid.empty:
				ax.axis("off")
				continue
			image = ax.imshow(grid.values, aspect="auto")
			low, high = np.nanmin(grid.values), np.nanmax(grid.values)
			ax.set_xticks(range(len(grid.columns)), [str(value) for value in grid.columns])
			ax.set_yticks(range(len(grid.index)), [str(value) for value in grid.index])
			for i in range(grid.shape[0]):
				for j in range(grid.shape[1]):
					value = grid.values[i][j]
					if not np.isnan(value):
						ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8,
								color=textShade(value, low, high))
			ax.set_title(f"{network}, learning rate {rate:g}\nrows downsampleSteps, columns extraSteps",
						fontsize=10)
			fig.colorbar(image, ax=ax, fraction=0.046)
	fig.tight_layout()
	fig.savefig(summary / "native_taper_grid.pdf", dpi=140)
	plt.close(fig)


def plotNativeTaperLine(trials: pd.DataFrame, summary: Path):
	"""Performance of the native networks against their size, one line per network.

	The learning rate and weight decay are frozen at the values of each network's best trial,
	so the line traces the taper grid (downsampleSteps x extraSteps) alone.
	"""
	native = trials[(trials["status"] == "ok") & (trials["source"] == "native")]
	if native.empty:
		return
	fig, ax = plt.subplots(figsize=(8, 5))
	# as in the convergence plots: the two lines differ in style, not in shade
	ax.set_prop_cycle(cycler(linestyle=LINE_STYLES) * cycler(color=["black"]))
	for network in sorted(native["network"].unique()):
		part = native[native["network"] == network]
		best = part.loc[part["bestAccuracy"].idxmax()]
		fixed = part[(part["learningRate"] == best["learningRate"])
					 & (part["weightDecay"] == best["weightDecay"])].sort_values("parameters")
		if fixed.empty:
			continue
		ax.plot(fixed["parameters"], fixed["bestAccuracy"], marker="o",
				label=f"{network}, lr {best['learningRate']:g}, wd {best['weightDecay']:g}")
	ax.set_xscale("log")
	ax.set_xlabel("trainable parameters")
	ax.set_ylabel("best test accuracy")
	ax.set_title("Native networks against model size, one line per network\n"
				 "(learning rate and weight decay frozen at each network's best trial)")
	ax.grid(True, which="both", alpha=0.3)
	ax.legend(fontsize=8)
	fig.tight_layout()
	fig.savefig(summary / "native_taper_vs_parameters.pdf", dpi=140)
	plt.close(fig)


# Convergence is plotted one graph per family with the channel variants together, except for
# the two plain stacks, which are only interesting next to each other.
CURVE_GROUPS = {"conv": "conv_simple", "simple": "conv_simple",
				"convnet": "convnet_simplenet", "simplenet": "convnet_simplenet"}


def curveGroup(family: str) -> str:
	return CURVE_GROUPS.get(family, family)


def plotLearningCurves(curves: pd.DataFrame, trials: pd.DataFrame, summary: Path):
	"""One convergence graph per architecture family, best configuration per channel variant."""
	best = bestPerModel(trials)
	if curves.empty or best.empty:
		return
	for group, rows in best.groupby(best["family"].map(curveGroup)):
		fig, ax = plt.subplots(figsize=(7, 4.8))
		# no gray shades in the convergence plots: every line is black and the series
		# differ in line style only
		ax.set_prop_cycle(cycler(linestyle=LINE_STYLES) * cycler(color=["black"]))
		plotted = 0
		for _, row in rows.sort_values("bestAccuracy", ascending=False).iterrows():
			part = curves[curves["key"] == row["key"]].sort_values("epoch")
			if part.empty:
				continue
			ax.plot(part["epoch"], part["testAccuracy"],
					label=f"{row['model']} lr {row['learningRate']:g}")
			plotted += 1
		if not plotted:
			plt.close(fig)
			continue
		ax.set_xlabel("epoch")
		ax.set_ylabel("test accuracy")
		ax.set_title(f"Convergence of {group.replace('_', ' and ')}")
		ax.grid(True, alpha=0.3)
		ax.legend(fontsize=8)
		fig.tight_layout()
		fig.savefig(summary / f"learning_curves_{group}.pdf", dpi=140)
		plt.close(fig)


def plotClassAccuracy(accuracy: pd.DataFrame, trials: pd.DataFrame, summary: Path):
	if accuracy.empty:
		return
	models = [model for model in bestPerModel(trials).sort_values("bestAccuracy", ascending=False).head(6)["model"]
			  if model in set(accuracy["model"])]
	if not models:
		return
	pivot = accuracy[accuracy["model"].isin(models)].pivot_table(
		index="className", columns="model", values="accuracy", aggfunc="mean")
	pivot = pivot[[model for model in models if model in pivot.columns]]
	pivot = pivot.reindex(accuracy.groupby("className")["accuracy"].mean().sort_values(ascending=False).index)

	fig, ax = plt.subplots(figsize=(1.4 * len(pivot.columns) + 4, 0.34 * len(pivot) + 3))
	image = ax.imshow(pivot.values, vmin=0, vmax=1, aspect="auto")
	ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
	ax.set_yticks(range(len(pivot.index)), pivot.index)
	for i in range(pivot.shape[0]):
		for j in range(pivot.shape[1]):
			value = pivot.values[i][j]
			if not np.isnan(value):
				ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
						color=textShade(value, 0, 1))
	ax.set_title("Per class test accuracy of the six best architectures")
	fig.colorbar(image, ax=ax, fraction=0.046)
	fig.tight_layout()
	fig.savefig(summary / "class_accuracy.pdf", dpi=140)
	plt.close(fig)


def loadLong(outdir: Path, classNames: list = None) -> dict:
	"""Read the extended budget runs of scripts/shootout_long.sh, if they exist.

	Unlike the regression follow up there is no circuit dimension here: the classification
	shootout trains on a single dataset, so every long job is one architecture and one trial.
	"""
	empty = {"trials": pd.DataFrame(), "curves": pd.DataFrame(), "classAccuracy": pd.DataFrame()}
	if not (outdir / "long" / "results").is_dir():
		return empty
	long = {"trials": readJobTable(outdir, "results.csv", "long/results"),
			"curves": readJobTable(outdir, "curves.csv", "long/results"),
			"classAccuracy": readJobTable(outdir, "classaccuracy.csv", "long/results")}
	for frame in long.values():
		if not frame.empty:
			describe(frame)
	accuracy = long["classAccuracy"]
	if not accuracy.empty:
		names = classNames or []
		accuracy["className"] = [names[index] if index < len(names) else str(index)
								 for index in accuracy["class"]]
	if not long["trials"].empty:
		long["trials"]["status"] = long["trials"]["status"].fillna("failed")
		long["trials"]["epochs"] = (long["curves"].groupby("key")["epoch"].max() + 1).reindex(
			long["trials"]["key"]).values if not long["curves"].empty else np.nan
	return long


def longConvergence(long: dict, trials: pd.DataFrame, budget: int) -> pd.DataFrame:
	"""Where each extended run stood at the old budget and where it ended up.

	The question the follow up was trained for: a run still improving when the shootout stopped
	was ranked by an artifact of the truncated budget.
	"""
	curves = long["curves"]
	if curves.empty:
		return pd.DataFrame()
	rows = []
	for model, part in curves.groupby("model"):
		part = part[part["testAccuracy"].notna()].sort_values("epoch")
		if part.empty:
			continue
		first, best = part.iloc[0], part.loc[part["testAccuracy"].idxmax()]
		last = int(part["epoch"].max()) + 1
		bestEpoch = int(best["epoch"]) + 1
		# the epochs in the curves are zero based while the budget counts whole epochs, and a run
		# shorter than the budget reports its final value instead of a gap
		reached = part[part["epoch"] <= budget - 1]
		atBase = reached.iloc[-1]["testAccuracy"] if not reached.empty else np.nan
		rows.append({"model": model, "learningRate": first["learningRate"],
					 "weightDecay": first["weightDecay"], "epochs": last, "bestEpoch": bestEpoch,
					 "bestAcc": best["testAccuracy"], "finalAcc": part.iloc[-1]["testAccuracy"],
					 f"accAt{budget}": atBase, "gainAfterBase": best["testAccuracy"] - atBase,
					 "stillImproving": bestEpoch >= last - STILL_IMPROVING_WINDOW})
	if not rows:
		return pd.DataFrame()
	frame = pd.DataFrame(rows).sort_values("bestAcc", ascending=False).reset_index(drop=True)
	if not trials.empty:
		baseline = bestPerModel(trials)[["model", "bestAccuracy"]].rename(
			columns={"bestAccuracy": "shootoutAcc"})
		frame = frame.merge(baseline, on="model", how="left")
	return frame


def plotLongLearningCurves(long: dict, budget: int, summary: Path):
	"""Test accuracy against epoch for the extended runs, the old budget marked."""
	curves = long["curves"]
	if curves.empty:
		return
	fig, ax = plt.subplots(figsize=(8.5, 5.5))
	# no gray shades in the convergence plots: every line is black and the series
	# differ in line style only
	ax.set_prop_cycle(cycler(linestyle=LINE_STYLES) * cycler(color=["black"]))
	for model, part in curves.groupby("model"):
		part = part.sort_values("epoch")
		ax.plot(part["epoch"] + 1, part["testAccuracy"], label=model, lw=1.3)
	ax.axvline(budget, color="0.3", linestyle=":", linewidth=1.2)
	ax.set_xlabel("epoch")
	ax.set_ylabel("test accuracy")
	ax.set_title("Convergence of the extended runs, one line per architecture\n"
				 f"(dotted: the {budget} epoch shootout budget)")
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize=8, loc="lower right")
	fig.tight_layout()
	fig.savefig(summary / "long_learning_curves.pdf", dpi=140)
	plt.close(fig)


def plotLongGain(convergence: pd.DataFrame, budget: int, summary: Path):
	"""What the epochs beyond the shootout budget bought, per architecture."""
	if convergence.empty:
		return
	models = list(convergence["model"])
	fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(models)), 4.8))
	ax.bar(range(len(models)), convergence["gainAfterBase"].fillna(0), **BAR_STYLE)
	ax.set_xticks(range(len(models)), models, fontsize=8)
	ax.axhline(0, color="black", linewidth=0.8)
	ax.set_ylabel(f"test accuracy gained after epoch {budget}")
	ax.set_title("What the epochs beyond the shootout budget bought")
	ax.grid(True, axis="y", alpha=0.3)
	fig.tight_layout()
	fig.savefig(summary / f"long_gain_after_{budget}.pdf", dpi=140)
	plt.close(fig)


def plotLongRankingShift(convergence: pd.DataFrame, budget: int, summary: Path):
	"""The shootout ranking beside the extended one, to see whether the order survived."""
	if convergence.empty:
		return
	models = list(convergence["model"])
	fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(models)), 4.8))
	width = 0.38 if "shootoutAcc" in convergence else 0.6
	if "shootoutAcc" in convergence:
		ax.bar([position - width / 2 for position in range(len(models))],
			   convergence["shootoutAcc"].fillna(0), width, label=f"{budget} epochs (shootout)")
		ax.bar([position + width / 2 for position in range(len(models))],
			   convergence["bestAcc"].fillna(0), width, label="extended budget")
	else:
		ax.bar(range(len(models)), convergence["bestAcc"].fillna(0), width, label="extended budget")
	ax.set_xticks(range(len(models)), models, fontsize=8)
	ax.set_ylabel("best test accuracy")
	ax.set_title("Short budget against the extended run, per architecture")
	ax.grid(True, axis="y", alpha=0.3)
	styleBars(ax)
	fig.tight_layout()
	fig.savefig(summary / "long_ranking_shift.pdf", dpi=140)
	plt.close(fig)


def plotLongClassAccuracy(long: dict, summary: Path):
	"""Per class test accuracy of the extended runs, one column per architecture."""
	accuracy = long["classAccuracy"]
	if accuracy.empty or "className" not in accuracy:
		return
	models = sorted(accuracy["model"].unique())
	pivot = accuracy.pivot_table(index="className", columns="model", values="accuracy",
								 aggfunc="mean")
	pivot = pivot[[model for model in models if model in pivot.columns]]
	pivot = pivot.reindex(accuracy.groupby("className")["accuracy"].mean()
						  .sort_values(ascending=False).index)
	if pivot.empty:
		return
	fig, ax = plt.subplots(figsize=(2.2 * len(pivot.columns) + 4, 0.34 * len(pivot) + 3))
	image = ax.imshow(pivot.values, vmin=0, vmax=1, aspect="auto")
	ax.set_xticks(range(len(pivot.columns)), pivot.columns, fontsize=8)
	ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=7)
	for i in range(pivot.shape[0]):
		for j in range(pivot.shape[1]):
			value = pivot.values[i][j]
			if not np.isnan(value):
				ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
						color=textShade(value, 0, 1))
	ax.set_title("Per class test accuracy of the extended runs\n"
				 "(at each run's best epoch)")
	fig.colorbar(image, ax=ax, fraction=0.046)
	fig.tight_layout()
	fig.savefig(summary / "long_class_accuracy.pdf", dpi=140)
	plt.close(fig)


def markdownTable(frame: pd.DataFrame) -> str:
	header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
	separator = "|" + "|".join("---" for _ in frame.columns) + "|"
	rows = ["| " + " | ".join("" if pd.isna(value) else
			(f"{value:.4f}" if isinstance(value, float) else str(value)) for value in row) + " |"
			for row in frame.itertuples(index=False)]
	return "\n".join([header, separator] + rows)


def writeReport(outdir: Path, trials: pd.DataFrame, accuracy: pd.DataFrame, summary: Path,
				epochs: int, long: dict = None, longConvergenceTable: pd.DataFrame = None):
	ok = trials[trials["status"] == "ok"]
	names, _ = readReport(outdir / "dataset_report_test.txt")
	lines = ["# Classification shootout", "",
			 f"- test set: {testCount(outdir)} spectra in {len(names)} classes",
			 f"- {len(trials)} trials, {len(ok)} trained successfully, {epochs} epochs each, "
			 f"batch size {BATCH_SIZE}, class weights enabled",
			 "- ranked by best test accuracy over the epochs, `bestLossNLL` is the tuner loss",
			 "  converted back to plain cross entropy", ""]

	if not ok.empty:
		columns = ["model", "source", "parameters", "learningRate", "weightDecay",
				   "downsampleSteps", "extraSteps", "bestEpoch", "bestAccuracy", "bestLossNLL", "seconds"]
		lines += ["## All trials ranked by best test accuracy (top 30)", "",
				  markdownTable(ok.sort_values("bestAccuracy", ascending=False)[columns].head(30)), ""]
		lines += ["## Best configuration per architecture", "",
				  markdownTable(bestPerModel(trials).sort_values("bestAccuracy", ascending=False)[columns]), ""]

		native = ok[ok["source"] == "native"]
		if not native.empty:
			grid = native.pivot_table(index=["network", "learningRate", "downsampleSteps"],
				columns="extraSteps", values="bestAccuracy", aggfunc="max")
			lines += ["## Native taper grid, best test accuracy", "",
					  markdownTable(grid.reset_index().round(4)), ""]

		if not accuracy.empty:
			winner = bestPerModel(trials).sort_values("bestAccuracy", ascending=False).iloc[0]
			perClass = accuracy[accuracy["key"] == winner["key"]].groupby("className")["accuracy"].mean().sort_values()
			lines += [f"## Weakest classes of the winner ({winner['model']}, lr {winner['learningRate']:g})", "",
					  markdownTable(pd.DataFrame({"class": perClass.index,
												   "accuracy": perClass.round(4)}).head(8)), ""]

	if long is not None and longConvergenceTable is not None and not longConvergenceTable.empty:
		budget = epochs
		longEpochs = int(longConvergenceTable["epochs"].max())
		lines += ["## Extended training budget", "",
				  f"Retrained with {longEpochs} epochs instead of {budget} on the same dataset and the",
				  "same architectures, because several of them were still improving when the shootout",
				  "stopped. They are kept out of the ranking above, which compares runs of equal",
				  "length, and both columns of the comparison are best test accuracy.", ""]
		columns = [column for column in ["model", "learningRate", "weightDecay", "epochs",
										 "bestEpoch", "shootoutAcc", f"accAt{budget}", "bestAcc",
										 "gainAfterBase", "stillImproving"]
				   if column in longConvergenceTable]
		lines += [markdownTable(longConvergenceTable[columns].round(4)), ""]
		for _, row in longConvergenceTable.iterrows():
			reference = row["shootoutAcc"] if "shootoutAcc" in row else float("nan")
			known = not (reference is None or pd.isna(reference))
			lines += [f"- {row['model']}: "
					  + (f"{reference:.4f} within {budget} epochs, " if known else "")
					  + f"{row['bestAcc']:.4f} after {longEpochs} epochs"
					  + (f", {row['bestAcc'] - reference:+.4f} from the extra epochs"
						 if known else "")
					  + ("; still improving when it stopped" if row["stillImproving"] else "")]
		lines += ["", "![convergence of the extended runs](long_learning_curves.pdf)", "",
				  f"![gain after the shootout budget](long_gain_after_{budget}.pdf)", "",
				  "![short budget against the extended run](long_ranking_shift.pdf)", "",
				  "![per class accuracy of the extended runs](long_class_accuracy.pdf)", ""]

	(summary / "report.md").write_text("\n".join(lines))


def main():
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("outdir", type=Path, help="the shootout directory")
	parser.add_argument("--epochs", type=int, default=20, help="only used in the report text")
	args = parser.parse_args()

	summary = args.outdir / "summary"
	summary.mkdir(parents=True, exist_ok=True)

	trials = loadTrials(args.outdir)
	curves = describe(readJobTable(args.outdir, "curves.csv"))
	accuracy = readJobTable(args.outdir, "classaccuracy.csv")
	classNames, _ = readReport(args.outdir / "dataset_report_test.txt")
	if not accuracy.empty:
		accuracy = describe(accuracy)
		names, _ = readReport(args.outdir / "dataset_report_test.txt")
		accuracy["className"] = [names[index] if index < len(names) else str(index)
								 for index in accuracy["class"]]

	columns = ["job", "key", "model", "family", "source", "channels", "network", "script",
			   "parameters", "learningRate", "weightDecay", "downsampleSteps", "extraSteps",
			   "bestEpoch", "bestLoss", "bestLossNLL", "loss", "lossNLL",
			   "bestAccuracy", "finalAccuracy", "seconds", "status", "error"]
	trials[columns].sort_values("bestAccuracy", ascending=False).to_csv(summary / "trials.csv", index=False)
	bestPerModel(trials)[columns].to_csv(summary / "models.csv", index=False)
	if not curves.empty:
		curves.to_csv(summary / "curves.csv", index=False)
	if not accuracy.empty:
		accuracy.to_csv(summary / "classaccuracy.csv", index=False)

	# the extended budget runs of scripts/shootout_long.sh, folded in from
	# scripts/shootoutlongplot.py, reported beside the ranking rather than inside it
	long = loadLong(args.outdir, classNames)
	longConvergenceTable = longConvergence(long, trials, args.epochs)
	if not long["trials"].empty:
		long["trials"].to_csv(summary / "longtrials.csv", index=False)
	if not long["curves"].empty:
		long["curves"].to_csv(summary / "longcurves.csv", index=False)
	if not long["classAccuracy"].empty:
		long["classAccuracy"].to_csv(summary / "longclassaccuracy.csv", index=False)
	if not longConvergenceTable.empty:
		longConvergenceTable.to_csv(summary / "long_convergence.csv", index=False)

	plotAccuracyVsParams(trials, summary, long["trials"])
	plotLearningRate(trials, summary)
	plotWeightDecay(trials, summary)
	plotChannels(trials, summary)
	plotNativeTaper(trials, summary)
	plotNativeTaperLine(trials, summary)
	plotLearningCurves(curves, trials, summary)
	plotClassAccuracy(accuracy, trials, summary)
	plotLongLearningCurves(long, args.epochs, summary)
	plotLongGain(longConvergenceTable, args.epochs, summary)
	plotLongRankingShift(longConvergenceTable, args.epochs, summary)
	plotLongClassAccuracy(long, summary)
	writeReport(args.outdir, trials, accuracy, summary, args.epochs,
				long, longConvergenceTable)

	ok = trials[trials["status"] == "ok"]
	print(f"{len(trials)} trials, {len(ok)} succeeded")
	if not ok.empty:
		print(ok.sort_values("bestAccuracy", ascending=False)[
			["model", "parameters", "learningRate", "weightDecay", "bestAccuracy", "bestLossNLL"]]
			.head(15).to_string(index=False))
	if not longConvergenceTable.empty:
		print(f"\n{len(longConvergenceTable)} extended budget runs:")
		print(longConvergenceTable[["model", "epochs", "bestEpoch", "bestAcc",
									f"accAt{args.epochs}", "gainAfterBase", "stillImproving"]]
			  .to_string(index=False))
	print(f"wrote {summary}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
