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

"""Aggregate the regression shootout of scripts/shootout_reg.sh into tables and plots.

Reads <outdir>/results/*/results.csv, curves.csv and targetr2.csv, the circuit list in
<outdir>/datasets.tsv and the torchkissann_datasetinfo reports and writes <outdir>/summary/:

	trials.csv       one row per trained network and circuit
	configs.csv      one row per configuration with the R2 of every circuit and the mean
	models.csv       the best configuration of every architecture
	curves.csv       per epoch curves carrying the architecture columns
	targetr2.csv     the R2 of every circuit parameter at each trial's best epoch
	report.md        the tables below as markdown
	*.pdf            the graphs listed in the report

If <outdir>/long/results exists (scripts/shootout_reg_long.sh), the extended budget runs are
reported beside the main ranking rather than inside it, since they are not comparable at equal
epoch counts. They add longtrials.csv, longconfigs.csv, longcurves.csv, long_convergence.csv,
long_mean_convergence.csv, long_runs.pdf, long_learning_curves.pdf, long_gain_after_<budget>.pdf,
long_ranking_shift.pdf and long_target_r2.pdf, all in the same summary directory as the rest, and
they are drawn into r2_vs_parameters.pdf as triangles, there as the same mean over all circuits as
the shootout runs, so the distance to the circle of an architecture is the gain of the extra epochs.
That convergence analysis is folded in from scripts/shootoutreglongplot.py, which asks whether the
shootout ranking was an artifact of the truncated epoch budget.

Everything is ranked by the mean R2 over all circuits at the best epoch, so an architecture
has to recover the parameters of every circuit instead of only the easiest one. R2 is computed
in the normalized space the network works in (targets scaled onto [-1,1], wide targets
additionally log10), which is the same space the MSE loss is evaluated in, so on a single
circuit loss and R2 rank identically.
"""

import argparse
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

# The generator names the targets after the circuit component with a random suffix, e.g.
# rHaM_0 for the resistance or pgvd_1 for the second CPE parameter.
COMPONENT_NAMES = {"r": "R", "c": "C", "l": "L", "p": "CPE", "w": "Warburg"}

# A configuration whose mean R2 falls below this has diverged rather than learned; the value
# is an artifact of the run, not a real result. It would stretch the axis of the taper line
# plot and swamp the colormap of the taper grid, so both drop it.
TAPER_R2_FLOOR = -5.0

# An extended run whose best epoch lies this close to its last epoch had not levelled off when
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


def targetLabel(name: str) -> str:
	"""Turn a generated target name into the circuit parameter it stands for."""
	match = re.match(r"[a-zA-Z]", name.strip())
	if not match:
		return name
	component = COMPONENT_NAMES.get(match.group(0).lower(), match.group(0).upper())
	index = name.strip().rsplit("_", 1)[-1]
	if component == "CPE":
		return "CPE n" if index == "1" else "CPE Q"
	return component


def readDatasets(outdir: Path) -> pd.DataFrame:
	"""The circuits of a run: tag, circuit string, target count, train and test size."""
	frame = pd.DataFrame(columns=["tag", "circuit", "targets", "train", "test"])
	path = outdir / "datasets.tsv"
	if not path.is_file():
		return frame
	read = pd.read_csv(path, sep="\t", header=None,
					   names=["tag", "circuit", "targets", "train", "test"])
	return read[read["tag"].notna()].reset_index(drop=True)


def datasetOf(job: str, tags: list) -> str:
	"""The circuit a job belongs to, from the tag embedded in its directory name."""
	for field in job.split("_")[1:]:
		if field in tags:
			return field
	return ""


def readTargetNames(outdir: Path, tag: str = "") -> list:
	"""Pull the target names out of a torchkissann_datasetinfo regression report."""
	suffix = f"_{tag}" if tag else ""
	for report in (f"dataset_report{suffix}_train.txt", f"dataset_report{suffix}_test.txt"):
		path = outdir / report
		if not path.is_file():
			continue
		lines = [line for line in path.read_text().splitlines() if line.strip()]
		for index, line in enumerate(lines):
			if [field.strip() for field in line.split(",")][:3] == ["0", "1", "2"]:
				names = [field.strip() for field in lines[index + 1].split(",")]
				labels, seen = [], {}
				for name in names:
					label = targetLabel(name)
					seen[label] = seen.get(label, 0) + 1
					labels.append(label if seen[label] == 1 else f"{label} {seen[label]}")
				return labels
	return []


def readJobTable(outdir: Path, fileName: str, sub: str = "results", tags: list = None) -> pd.DataFrame:
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
	if tags:
		joined["dataset"] = [datasetOf(job, tags) for job in joined["job"]]
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
	# The configuration is what is compared between the circuits: the same architecture with
	# the same hyperparameters. The taper of a TorchScript model is not a degree of freedom,
	# so it is only part of the key of the native networks.
	frame["config"] = [
		f"{model} lr {rate:g} wd {decay:g}"
		+ ("" if script else f" ds {steps} ex {extra}")
		for model, script, rate, decay, steps, extra in zip(
			frame["model"], frame["script"], frame["learningRate"], frame["weightDecay"],
			frame.get("downsampleSteps", [""] * len(frame)), frame.get("extraSteps", [""] * len(frame)))]
	return frame


def crossCircuit(trials: pd.DataFrame, circuits: list) -> pd.DataFrame:
	"""One row per configuration with the R2 reached on every circuit and the mean over them.

	This is the figure of merit of the shootout: a configuration scores well by recovering the
	parameters of every circuit, not by winning the easiest one. Configurations that did not
	run on every circuit are still listed, with the number of circuits they do have.
	"""
	ok = trials[(trials["status"] == "ok") & trials["bestR2"].notna()]
	if ok.empty:
		return ok
	# the best trial of a configuration within one circuit, which is normally the only one
	best = ok.sort_values("bestR2", ascending=False).groupby(["config", "dataset"]).first()
	best = best.reset_index()
	perCircuit = best.pivot_table(index="config", columns="dataset", values="bestR2")
	ordered = [circuit for circuit in circuits if circuit in perCircuit.columns]
	ordered += [c for c in perCircuit.columns if c not in ordered]
	perCircuit = perCircuit[ordered]
	info = best.sort_values("bestR2", ascending=False).groupby("config").first().reset_index()
	combined = info.set_index("config").join(perCircuit)
	combined["meanR2"] = perCircuit.mean(axis=1, skipna=True)
	combined["worstR2"] = perCircuit.min(axis=1)
	combined["circuits"] = perCircuit.notna().sum(axis=1)
	if "runtimeMs" in best.columns:
		runtime = best.pivot_table(index="config", columns="dataset", values="runtimeMs")
		combined["meanRuntime"] = runtime.mean(axis=1, skipna=True)
	combined = combined.reset_index()
	return combined.sort_values(["meanR2", "worstR2"], ascending=False).reset_index(drop=True)


def bestPerModelConfigs(configs: pd.DataFrame) -> pd.DataFrame:
	"""The best configuration of every architecture, by the mean over all circuits."""
	if configs.empty:
		return configs
	return configs.sort_values("meanR2", ascending=False).groupby("model", sort=False) \
		.first().reset_index()


def loadTrials(outdir: Path, tags: list = None) -> pd.DataFrame:
	trials = readJobTable(outdir, "results.csv", tags=tags)
	if trials.empty:
		raise SystemExit(f"no results.csv below {outdir}/results, run scripts/shootout_reg.sh first")
	trials["status"] = trials["status"].fillna("failed")
	if "metric" not in trials:
		trials["metric"] = "r2"
	return describe(trials)


def loadLong(outdir: Path, names: dict = None, tags: list = None) -> dict:
	"""Read the extended-budget runs of scripts/shootout_reg_long.sh, if they exist.

	They live in a separate tree so that the main ranking keeps comparing runs that all
	got the same number of epochs; the long runs are reported beside it. The tags give every
	row its circuit, so the long runs can be ranked by the same mean over the circuits and the
	targets can be named the way the main tables name them.
	"""
	empty = {"trials": pd.DataFrame(), "curves": pd.DataFrame(), "targetR2": pd.DataFrame()}
	if not (outdir / "long" / "results").is_dir():
		return empty
	long = {"trials": readJobTable(outdir, "results.csv", "long/results", tags=tags),
			"curves": readJobTable(outdir, "curves.csv", "long/results", tags=tags),
			"targetR2": readJobTable(outdir, "targetr2.csv", "long/results", tags=tags)}
	for frame in long.values():
		if not frame.empty:
			describe(frame)
	if not long["targetR2"].empty:
		labels = names or {}

		def targetName(dataset: str, index: int) -> str:
			perCircuit = labels.get(dataset, [])
			return f"{dataset}/{perCircuit[index]}" if index < len(perCircuit) \
				else f"{dataset}/{index}"

		long["targetR2"]["targetName"] = [targetName(dataset, index)
										  for dataset, index in zip(long["targetR2"]["dataset"],
																   long["targetR2"]["target"])]
	if not long["trials"].empty:
		long["trials"]["status"] = long["trials"]["status"].fillna("failed")
		long["trials"]["epochs"] = (long["curves"].groupby("key")["epoch"].max() + 1).reindex(
			long["trials"]["key"]).values if not long["curves"].empty else np.nan
	return long


def milestones(curves: pd.DataFrame, points: list) -> pd.DataFrame:
	"""The R2 of every long run at a fixed set of epochs, for the convergence table."""
	if curves.empty:
		return pd.DataFrame()
	rows = []
	for key, part in curves.groupby("key"):
		part = part.sort_values("epoch")
		first = part.iloc[0]
		row = {"model": first["model"], "learningRate": first["learningRate"],
			   "weightDecay": first["weightDecay"]}
		for epoch in points:
			# the last epoch that is not later than the milestone, so a short run still
			# reports its final value instead of a gap
			reached = part[part["epoch"] <= epoch - 1]
			row[f"epoch{epoch}"] = reached.iloc[-1]["r2Mean"] if not reached.empty else np.nan
		best = part.loc[part["r2Mean"].idxmax()]
		row["bestR2"] = best["r2Mean"]
		row["bestEpoch"] = int(best["epoch"]) + 1
		rows.append(row)
	return pd.DataFrame(rows).sort_values(["model", "bestR2"], ascending=[True, False]) \
		.reset_index(drop=True)


def plotMeanR2(configs: pd.DataFrame, circuits: list, summary: Path):
	"""The headline view: mean R2 over all circuits per architecture, plus the spread per circuit."""
	best = bestPerModelConfigs(configs)
	if best.empty:
		return
	top = best.head(18)
	ordered = top.sort_values("meanR2")
	fig, ax = plt.subplots(figsize=(7, 0.5 * len(ordered) + 2.5))
	ax.barh(range(len(ordered)), ordered["meanR2"], color="0.15", edgecolor="black",
			linewidth=0.7)
	ax.set_yticks(range(len(ordered)), ordered["model"])
	ax.set_xlabel("mean R2 over all circuits")
	ax.set_title("Best configuration per architecture, averaged over the circuits")
	ax.grid(True, axis="x", alpha=0.3)
	fig.tight_layout()
	fig.savefig(summary / "mean_r2_by_model.pdf", dpi=140)
	plt.close(fig)

	used = [circuit for circuit in circuits if circuit in top.columns]
	if not used:
		return
	heat = top.set_index("model")[used]
	fig, ax = plt.subplots(figsize=(2.2 * len(used) + 3.5, 0.5 * len(heat) + 2.5))
	image = ax.imshow(heat.values, vmin=0, vmax=1, aspect="auto")
	ax.set_xticks(range(len(used)), used, rotation=45, ha="right")
	ax.set_yticks(range(len(heat)), heat.index)
	for i in range(heat.shape[0]):
		for j in range(heat.shape[1]):
			value = heat.values[i][j]
			if not np.isnan(value):
				ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8,
						color=textShade(value, 0, 1))
	ax.set_title("The same configurations per circuit")
	fig.colorbar(image, ax=ax, fraction=0.046)
	fig.tight_layout()
	fig.savefig(summary / "mean_r2_by_circuit.pdf", dpi=140)
	plt.close(fig)


def plotR2VsParams(configs: pd.DataFrame, summary: Path, longConfigs: pd.DataFrame = None):
	"""Model size against parameter recovery, the extended budget runs beside the shootout ones.

	Both series are the mean over all circuits, so a triangle sits above the circle of the same
	architecture when the longer budget bought something and the vertical distance between the
	two is the effect of the epoch count alone.
	"""
	best = bestPerModelConfigs(configs)
	if best.empty:
		return
	fig, ax = plt.subplots(figsize=(9.5, 6.5))
	for _, row in best.iterrows():
		ax.scatter(row["parameters"], row["meanR2"], s=60, marker="o", color="0.0")
		ax.annotate(row["model"], (row["parameters"], row["meanR2"]),
					textcoords="offset points", xytext=(6, 3), fontsize=8)
	# the extended budget runs are drawn apart, they are not comparable at equal epochs
	if longConfigs is not None and not longConfigs.empty:
		for _, row in bestPerModelConfigs(longConfigs).iterrows():
			ax.scatter(row["parameters"], row["meanR2"], s=90, marker="^", color="0.0", zorder=3)
			ax.annotate(f"{row['model']} ({int(row['epochs'])} ep)",
						(row["parameters"], row["meanR2"]),
						textcoords="offset points", xytext=(6, -10), fontsize=8)
	ax.set_xscale("log")
	ax.set_xlabel("trainable parameters")
	ax.set_ylabel("mean R2 over all circuits")
	ax.set_title("Parameter recovery against model size, best hyperparameters per model\n"
					"(triangles are the runs with an extended epoch budget, same mean)")
	ax.grid(True, which="both", alpha=0.3)
	fig.tight_layout()
	fig.savefig(summary / "r2_vs_parameters.pdf", dpi=140)
	plt.close(fig)


def loadRuntime(outdir: Path) -> pd.DataFrame:
	"""Read the per architecture compute cost written by the torchkissann_bench step."""
	path = outdir / "bench.csv"
	if not path.exists():
		return pd.DataFrame()
	frame = pd.read_csv(path)
	if frame.empty:
		return frame
	frame["script"] = frame["script"].fillna("")
	return frame


def runtimeKey(row, outputSize) -> tuple:
	"""The architecture identity that joins a bench row to a trial row.

	TorchScript models are identified by their file, whose name already carries the target
	count, so the file alone is unique. Native networks need the output size in the key, since
	the same taper costs different amounts of compute per circuit. The parameter count is
	deliberately not part of the key, since the whole point is that it does not predict runtime.
	"""
	script = row.get("script", "")
	if not isinstance(script, str) or pd.isna(script):
		script = ""
	if script:
		return ("s", script)

	def asInt(value):
		try:
			return int(value)
		except (TypeError, ValueError):
			return -1

	return ("n", row.get("network", ""), asInt(row.get("downsampleSteps")),
			asInt(row.get("extraSteps")), asInt(outputSize))


def addRuntime(trials: pd.DataFrame, runtimeMs: dict, targetsByDataset: dict) -> pd.DataFrame:
	"""Attach the median ms per iteration of each trial's architecture, NaN where not benched."""
	trials = trials.copy()

	def key(row):
		return runtimeKey(row, targetsByDataset.get(row.get("dataset", ""), None))

	trials["runtimeMs"] = [runtimeMs.get(key(row), float("nan")) for _, row in trials.iterrows()]
	return trials


def plotR2VsRuntime(configs: pd.DataFrame, summary: Path):
	"""Parameter recovery against compute cost, the point being that parameters do not predict it."""
	best = bestPerModelConfigs(configs)
	measured = best[best["meanRuntime"].notna()] if "meanRuntime" in best else best.iloc[0:0]
	if measured.empty:
		return
	fig, ax = plt.subplots(figsize=(9.5, 6.5))
	for _, row in measured.iterrows():
		ax.scatter(row["meanRuntime"], row["meanR2"], s=60, marker="o", color="0.0")
		ax.annotate(row["model"], (row["meanRuntime"], row["meanR2"]),
					textcoords="offset points", xytext=(6, 3), fontsize=8)
	ax.set_xscale("log")
	ax.set_xlabel("median ms per training iteration (forward + backward + step), mean over circuits")
	ax.set_ylabel("mean R2 over all circuits")
	ax.set_title("Parameter recovery against compute cost, best hyperparameters per model\n"
				 "(MI100, batch 256, random data)")
	ax.grid(True, which="both", alpha=0.3)
	fig.tight_layout()
	fig.savefig(summary / "r2_vs_runtime.pdf", dpi=140)
	plt.close(fig)


def plotLearningRate(configs: pd.DataFrame, summary: Path):
	for decay, group in configs.groupby("weightDecay"):
		pivot = group.groupby(["model", "learningRate"])["meanR2"].max().unstack()
		if pivot.empty:
			continue
		pivot = pivot.sort_values(pivot.columns[0], ascending=False)
		fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(pivot)), 5.5))
		pivot.plot(kind="bar", ax=ax, legend=False, **BAR_STYLE)
		ax.set_ylabel("mean R2 over all circuits")
		ax.set_xlabel("")
		ax.set_title(f"Learning rate sweep per architecture (weight decay {decay:g})")
		ax.tick_params(axis="x", rotation=60)
		ax.grid(True, axis="y", alpha=0.3)
		styleBars(ax, "learning rate")
		fig.tight_layout()
		fig.savefig(summary / f"r2_by_learningrate_wd{str(decay).replace('.', 'p')}.pdf", dpi=140)
		plt.close(fig)


def plotWeightDecay(configs: pd.DataFrame, summary: Path):
	decays = sorted(configs["weightDecay"].unique())
	if len(decays) < 2:
		return
	pivot = configs.groupby(["model", "weightDecay"])["meanR2"].max().unstack().dropna(how="any")
	if pivot.empty:
		return
	pivot = pivot[decays].sort_values(decays[-1], ascending=False)
	fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(pivot)), 5.5))
	pivot.plot(kind="bar", ax=ax, legend=False, **BAR_STYLE)
	ax.set_ylabel("mean R2 over all circuits")
	ax.set_xlabel("")
	ax.set_title("Weight decay effect per architecture (best learning rate per bar)")
	ax.tick_params(axis="x", rotation=60)
	ax.grid(True, axis="y", alpha=0.3)
	styleBars(ax, "weight decay")
	fig.tight_layout()
	fig.savefig(summary / "weight_decay_effect.pdf", dpi=140)
	plt.close(fig)


def plotChannels(configs: pd.DataFrame, summary: Path):
	scripts = configs[configs["script"] != ""]
	pivot = scripts.groupby(["family", "channels"])["meanR2"].max().unstack().dropna(how="any")
	if pivot.empty or pivot.shape[1] < 2:
		return
	pivot = pivot.sort_values(pivot.columns[-1], ascending=False)
	fig, ax = plt.subplots(figsize=(8, 5))
	pivot.plot(kind="bar", ax=ax, legend=False, **BAR_STYLE)
	ax.set_ylabel("mean R2 over all circuits")
	ax.set_xlabel("")
	ax.set_title("One channel (flat vector) against two channels (Re/Im split)")
	ax.tick_params(axis="x", rotation=45)
	ax.grid(True, axis="y", alpha=0.3)
	styleBars(ax, "input channels")
	fig.tight_layout()
	fig.savefig(summary / "channel_split_effect.pdf", dpi=140)
	plt.close(fig)


def plotNativeTaper(configs: pd.DataFrame, summary: Path):
	native = configs[(configs["source"] == "native") & (configs["meanR2"] >= TAPER_R2_FLOOR)]
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
									values="meanR2", aggfunc="max")
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


def plotNativeTaperLine(configs: pd.DataFrame, summary: Path):
	"""Performance of the native networks against their size, one line per network.

	The learning rate and weight decay are frozen at the values of each network's best
	configuration, so the line traces the taper grid (downsampleSteps x extraSteps) alone.
	"""
	native = configs[(configs["source"] == "native") & (configs["meanR2"] >= TAPER_R2_FLOOR)]
	if native.empty:
		return
	fig, ax = plt.subplots(figsize=(8, 5))
	# as in the convergence plots: the two lines differ in style, not in shade
	ax.set_prop_cycle(cycler(linestyle=LINE_STYLES) * cycler(color=["black"]))
	for network in sorted(native["network"].unique()):
		part = native[native["network"] == network]
		best = part.loc[part["meanR2"].idxmax()]
		fixed = part[(part["learningRate"] == best["learningRate"])
					 & (part["weightDecay"] == best["weightDecay"])].sort_values("parameters")
		if fixed.empty:
			continue
		ax.plot(fixed["parameters"], fixed["meanR2"], marker="o",
				label=f"{network}, lr {best['learningRate']:g}, wd {best['weightDecay']:g}")
	ax.set_xscale("log")
	ax.set_xlabel("trainable parameters")
	ax.set_ylabel("mean R2 over all circuits")
	ax.set_title("Native networks against model size, one line per network\n"
				 "(learning rate and weight decay frozen at each network's best configuration)")
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


def plotLearningCurves(curves: pd.DataFrame, configs: pd.DataFrame, circuits: list, summary: Path):
	"""Convergence per family: one row per circuit, one line per model's best configuration."""
	best = bestPerModelConfigs(configs)
	if curves.empty or best.empty or "dataset" not in curves:
		return
	used = [circuit for circuit in circuits if circuit in set(curves["dataset"])]
	if not used:
		return
	for group, rows in best.groupby(best["family"].map(curveGroup)):
		fig, axes = plt.subplots(len(used), 2, figsize=(11, 3.4 * len(used)), squeeze=False)
		# no gray shades in the convergence plots: every line is black and the series
		# differ in line style only
		for axis in axes.flat:
			axis.set_prop_cycle(cycler(linestyle=LINE_STYLES) * cycler(color=["black"]))
		plotted = 0
		for line, circuit in enumerate(used):
			for _, row in rows.sort_values("meanR2", ascending=False).iterrows():
				# the winning configuration of that model, traced on this circuit
				part = curves[(curves["config"] == row["config"]) & (curves["dataset"] == circuit)]
				part = part.sort_values("epoch")
				if part.empty:
					continue
				label = f"{row['model']} lr {row['learningRate']:g}"
				axes[line][0].plot(part["epoch"], part["testLoss"], label=label)
				axes[line][1].plot(part["epoch"], part["r2Mean"], label=label)
				plotted += 1
			axes[line][0].set_ylabel("test MSE (normalized)")
			axes[line][0].set_yscale("log")
			axes[line][0].set_title(f"{circuit}, test loss", fontsize=10)
			axes[line][0].grid(True, alpha=0.3)
			axes[line][1].set_ylabel("mean R2")
			axes[line][1].set_title(f"{circuit}, mean R2", fontsize=10)
			axes[line][1].grid(True, alpha=0.3)
			axes[line][1].axhline(0, color="black", linewidth=0.8)
			if line == 0:
				axes[line][1].legend(fontsize=8)
		if not plotted:
			plt.close(fig)
			continue
		axes[-1][0].set_xlabel("epoch")
		axes[-1][1].set_xlabel("epoch")
		fig.suptitle(f"Convergence of {group.replace('_', ' and ')}, best configuration per circuit")
		fig.tight_layout()
		fig.savefig(summary / f"learning_curves_{group}.pdf", dpi=140)
		plt.close(fig)


def plotLongRuns(long: dict, trials: pd.DataFrame, summary: Path):
	"""Convergence of the extended-budget runs against the shootout's epoch budget."""
	curves = long["curves"]
	if curves.empty:
		return
	models = sorted(curves["model"].unique())
	reference = trials[(trials["status"] == "ok") & trials["bestR2"].notna()]
	winnerR2 = reference["bestR2"].max() if not reference.empty else np.nan
	budget = int(reference["bestEpoch"].max()) if not reference.empty else 20

	fig, axes = plt.subplots(1, len(models), figsize=(7.5 * len(models), 5.5), squeeze=False)
	for axis, model in zip(axes[0], models):
		part = curves[curves["model"] == model]
		for rate, run in part.groupby("learningRate"):
			run = run.sort_values("epoch")
			axis.plot(run["epoch"] + 1, run["r2Mean"], label=f"learning rate {rate:g}")
			best = run.loc[run["r2Mean"].idxmax()]
			axis.scatter(best["epoch"] + 1, best["r2Mean"], s=55, zorder=3, marker="o",
						 edgecolor="black", linewidth=0.6)
		# where the same architecture ended up within the shootout budget
		shootout = reference[reference["model"] == model]
		if not shootout.empty:
			axis.scatter(budget, shootout["bestR2"].max(), s=90, marker="s", color="0.0",
						 zorder=4, label=f"best within {budget} epochs")
		axis.axvline(budget, color="0.0", linestyle=":", linewidth=1.2)
		if not np.isnan(winnerR2):
			axis.axhline(winnerR2, color="0.4", linestyle="--", linewidth=1.2)
			axis.annotate(f"shootout winner {winnerR2:.3f}", (axis.get_xlim()[1], winnerR2),
						  xytext=(-6, 5), textcoords="offset points", ha="right", fontsize=8,
						  color="0.3")
		axis.set_xlabel("epoch")
		axis.set_ylim(0, 1)
		axis.grid(True, alpha=0.3)
		axis.set_title(f"{model}, extended training budget")
		axis.legend(fontsize=8, loc="lower right")
	axes[0][0].set_ylabel("mean R2")
	fig.tight_layout()
	fig.savefig(summary / "long_runs.pdf", dpi=140)
	plt.close(fig)


def longConvergence(long: dict, budget: int) -> pd.DataFrame:
	"""Where every extended run stood at the old budget and where it ended up.

	This is the question the follow up was trained for: a run that was still improving when the
	shootout stopped was ranked by an artifact of the truncated budget, and the gain column says
	how much of its ranking it had not earned yet.
	"""
	curves = long["curves"]
	if curves.empty or "dataset" not in curves:
		return pd.DataFrame()
	rows = []
	for (model, dataset), part in curves.groupby(["model", "dataset"]):
		part = part[part["r2Mean"].notna()].sort_values("epoch")
		if part.empty:
			continue
		first, best = part.iloc[0], part.loc[part["r2Mean"].idxmax()]
		last = int(part["epoch"].max()) + 1
		bestEpoch = int(best["epoch"]) + 1
		# the epochs in the curves are zero based while the budget counts whole epochs, and a run
		# shorter than the budget reports its final value instead of a gap
		reached = part[part["epoch"] <= budget - 1]
		atBase = reached.iloc[-1]["r2Mean"] if not reached.empty else np.nan
		rows.append({"model": model, "dataset": dataset,
					 "learningRate": first["learningRate"], "weightDecay": first["weightDecay"],
					 "epochs": last, "bestEpoch": bestEpoch, "bestR2": best["r2Mean"],
					 "finalR2": part.iloc[-1]["r2Mean"], f"r2At{budget}": atBase,
					 "gainAfterBase": best["r2Mean"] - atBase,
					 "stillImproving": bestEpoch >= last - STILL_IMPROVING_WINDOW})
	if not rows:
		return pd.DataFrame()
	return pd.DataFrame(rows).sort_values(["model", "dataset"]).reset_index(drop=True)


def longMeanConvergence(convergence: pd.DataFrame, configs: pd.DataFrame) -> pd.DataFrame:
	"""The extended runs averaged over the circuits, beside the mean R2 of the shootout."""
	if convergence.empty:
		return pd.DataFrame()
	base = [column for column in convergence.columns if column.startswith("r2At")]
	aggregates = {"circuits": ("dataset", "nunique"), "meanBestR2": ("bestR2", "mean"),
				  "meanFinalR2": ("finalR2", "mean"), "meanBestEpoch": ("bestEpoch", "mean"),
				  "meanGainAfterBase": ("gainAfterBase", "mean"),
				  "stillImproving": ("stillImproving", "sum")}
	if base:
		column = base[0]
		aggregates[f"mean{column[0].upper()}{column[1:]}"] = (column, "mean")
	averaged = convergence.groupby("model", as_index=False).agg(**aggregates)
	if not configs.empty:
		baseline = bestPerModelConfigs(configs)[["model", "meanR2"]].rename(
			columns={"meanR2": "shootoutR2"})
		averaged = averaged.merge(baseline, on="model", how="left")
	return averaged.sort_values("meanBestR2", ascending=False).reset_index(drop=True)


def plotLongLearningCurves(long: dict, circuits: list, budget: int, summary: Path):
	"""One panel per circuit, one line per architecture, with the old budget marked."""
	curves = long["curves"]
	if curves.empty or "dataset" not in curves:
		return
	used = [circuit for circuit in circuits if circuit in set(curves["dataset"])]
	if not used:
		return
	columns = min(2, len(used))
	rows = (len(used) + columns - 1) // columns
	fig, axes = plt.subplots(rows, columns, figsize=(5.6 * columns, 4.0 * rows), squeeze=False)
	# as in the other convergence plots: black lines that differ in style, not in shade
	for axis in axes.flat:
		axis.set_prop_cycle(cycler(linestyle=LINE_STYLES) * cycler(color=["black"]))
	for index, circuit in enumerate(used):
		axis = axes[index // columns][index % columns]
		for model, part in curves[curves["dataset"] == circuit].groupby("model"):
			part = part.sort_values("epoch")
			axis.plot(part["epoch"] + 1, part["r2Mean"], label=model, lw=1.3)
		axis.axvline(budget, color="0.3", linestyle=":", linewidth=1.2)
		axis.set_xlabel("epoch")
		axis.set_ylabel("mean R2")
		axis.set_title(circuit, fontsize=10)
		axis.grid(True, alpha=0.3)
	for index in range(len(used), rows * columns):
		axes[index // columns][index % columns].axis("off")
	axes.flat[0].legend(fontsize=7, loc="lower right")
	fig.suptitle("Convergence of the extended runs, one panel per circuit\n"
				 f"(dotted: the {budget} epoch shootout budget)", fontsize=12)
	fig.tight_layout()
	fig.savefig(summary / "long_learning_curves.pdf", dpi=140)
	plt.close(fig)


def plotLongGain(convergence: pd.DataFrame, circuits: list, budget: int, summary: Path):
	"""What the epochs beyond the shootout budget bought, per architecture and circuit."""
	if convergence.empty:
		return
	used = [circuit for circuit in circuits if circuit in set(convergence["dataset"])]
	models = sorted(convergence["model"].unique())
	if not used or not models:
		return
	fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(models)), 4.8))
	width = 0.8 / len(used)
	for index, circuit in enumerate(used):
		part = convergence[convergence["dataset"] == circuit].set_index("model")
		values = [part["gainAfterBase"].get(model, np.nan) for model in models]
		ax.bar([position + index * width for position in range(len(models))],
			   [0.0 if pd.isna(value) else value for value in values], width, label=circuit)
	ax.set_xticks(range(len(models)), models, fontsize=8)
	ax.axhline(0, color="black", linewidth=0.8)
	ax.set_ylabel(f"mean R2 gained after epoch {budget}")
	ax.set_title("What the epochs beyond the shootout budget bought")
	ax.grid(True, axis="y", alpha=0.3)
	styleBars(ax, "circuit")
	fig.tight_layout()
	fig.savefig(summary / f"long_gain_after_{budget}.pdf", dpi=140)
	plt.close(fig)


def plotLongRankingShift(averaged: pd.DataFrame, budget: int, summary: Path):
	"""The shootout ranking beside the extended one, to see whether the order survived."""
	if averaged.empty:
		return
	models = list(averaged["model"])
	fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(models)), 4.8))
	width = 0.38 if "shootoutR2" in averaged else 0.6
	if "shootoutR2" in averaged:
		ax.bar([position - width / 2 for position in range(len(models))],
			   averaged["shootoutR2"].fillna(0), width, label=f"{budget} epochs (shootout)")
		ax.bar([position + width / 2 for position in range(len(models))],
			   averaged["meanBestR2"].fillna(0), width, label="extended budget")
	else:
		ax.bar(range(len(models)), averaged["meanBestR2"].fillna(0), width, label="extended budget")
	ax.set_xticks(range(len(models)), models, fontsize=8)
	ax.set_ylabel("mean R2 over all circuits")
	ax.set_title("Short budget against the extended run, per architecture")
	ax.grid(True, axis="y", alpha=0.3)
	styleBars(ax)
	fig.tight_layout()
	fig.savefig(summary / "long_ranking_shift.pdf", dpi=140)
	plt.close(fig)


def plotLongTargetR2(long: dict, summary: Path):
	"""R2 per circuit parameter of the extended runs, one column per architecture."""
	accuracy = long["targetR2"]
	if accuracy.empty or "targetName" not in accuracy:
		return
	models = sorted(accuracy["model"].unique())
	pivot = accuracy.pivot_table(index="targetName", columns="model", values="r2", aggfunc="mean")
	pivot = pivot[[model for model in models if model in pivot.columns]]
	# the rows keep the order in which the circuits and their parameters were read
	order = [name for name in dict.fromkeys(accuracy["targetName"]) if name in pivot.index]
	pivot = pivot.reindex(order)
	if pivot.empty:
		return
	fig, ax = plt.subplots(figsize=(2.6 * len(pivot.columns) + 4, 0.34 * len(pivot) + 3))
	image = ax.imshow(pivot.values, vmin=0, vmax=1, aspect="auto")
	ax.set_xticks(range(len(pivot.columns)), pivot.columns, fontsize=8)
	ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=7)
	for i in range(pivot.shape[0]):
		for j in range(pivot.shape[1]):
			value = pivot.values[i][j]
			if not np.isnan(value):
				ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
						color=textShade(value, 0, 1))
	ax.set_title("R2 per circuit parameter of the extended runs\n"
				 "(normalized target space, at each run's best epoch)")
	fig.colorbar(image, ax=ax, fraction=0.046)
	fig.tight_layout()
	fig.savefig(summary / "long_target_r2.pdf", dpi=140)
	plt.close(fig)


def plotTargetR2(accuracy: pd.DataFrame, configs: pd.DataFrame, summary: Path, count: int = 4):
	"""R2 per circuit parameter for the best configurations, one block of rows per circuit."""
	if accuracy.empty or configs.empty:
		return
	top = configs.head(count)
	columns = {}
	for _, row in top.iterrows():
		part = accuracy[accuracy["key"] == row["key"]]
		if part.empty:
			continue
		columns[f"{row['model']}\nlr {row['learningRate']:g}"] = part.set_index("targetName")["r2"]
	if not columns:
		return
	pivot = pd.DataFrame(columns)
	# the rows keep the order in which the circuits and their parameters were read, so the
	# parameters of one circuit stay together and in the order of the circuit string
	pivot = pivot.reindex([name for name in dict.fromkeys(accuracy["targetName"])
						   if name in pivot.index])

	fig, ax = plt.subplots(figsize=(3.2 * len(pivot.columns) + 4, 0.42 * len(pivot) + 3))
	image = ax.imshow(pivot.values, vmin=0, vmax=1, aspect="auto")
	ax.set_xticks(range(len(pivot.columns)), pivot.columns, fontsize=8)
	ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=8)
	for i in range(pivot.shape[0]):
		for j in range(pivot.shape[1]):
			value = pivot.values[i][j]
			if not np.isnan(value):
				ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
						color=textShade(value, 0, 1))
	ax.set_title("R2 per circuit parameter for the best configurations\n"
				 "(normalized target space, at each trial's best epoch)")
	fig.colorbar(image, ax=ax, fraction=0.046)
	fig.tight_layout()
	fig.savefig(summary / "target_r2.pdf", dpi=140)
	plt.close(fig)


def markdownTable(frame: pd.DataFrame) -> str:
	header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
	separator = "|" + "|".join("---" for _ in frame.columns) + "|"
	rows = ["| " + " | ".join("" if pd.isna(value) else
			(f"{value:.4f}" if isinstance(value, float) else str(value)) for value in row) + " |"
			for row in frame.itertuples(index=False)]
	return "\n".join([header, separator] + rows)


def writeReport(outdir: Path, trials: pd.DataFrame, configs: pd.DataFrame,
				targetR2: pd.DataFrame, summary: Path, epochs: int, circuits: pd.DataFrame,
				long: dict = None, longTable: pd.DataFrame = None,
				longMilestones: pd.DataFrame = None, longConvergenceTable: pd.DataFrame = None,
				longAveraged: pd.DataFrame = None):
	ok = trials[trials["status"] == "ok"]
	lines = ["# Regression shootout", ""]
	if circuits.empty:
		lines += ["- no datasets.tsv, the circuits of this run are unknown", ""]
	else:
		lines += [f"- {len(circuits)} circuits, every candidate trained on all of them:", ""]
		for _, row in circuits.iterrows():
			lines += [f"  - `{row['circuit']}` ({row['tag']}): {row['targets']} targets, "
					  f"{row['train']} train / {row['test']} test spectra"]
		lines += [""]
	lines += [f"- {len(trials)} trials, {len(ok)} trained successfully, {epochs} epochs each, "
			  "MSE loss in normalized target space",
			  "- ranked by the mean R2 over all circuits at the best epoch; R2 and the loss are",
			  "  both computed in the normalized target space, so on one circuit they rank",
			  "  identically, and the mean is what decides the shootout", ""]

	perCircuit = [row["tag"] for _, row in circuits.iterrows() if row["tag"] in getattr(configs, "columns", [])]

	if not configs.empty:
		columns = ["model", "source", "parameters", "learningRate", "weightDecay",
				   "downsampleSteps", "extraSteps"] + perCircuit + ["meanR2", "worstR2", "circuits"]
		columns = [column for column in columns if column in configs.columns]
		lines += ["## Configurations ranked by the mean R2 over all circuits (top 30)", "",
				  markdownTable(configs.head(30)[columns].round(4)), ""]
		best = bestPerModelConfigs(configs)
		lines += ["## Best configuration per architecture", "",
				  markdownTable(best[columns].round(4)), ""]
		lines += ["![mean R2 over all circuits per model](mean_r2_by_model.pdf)", ""]
		lines += ["![the same configurations per circuit](mean_r2_by_circuit.pdf)", ""]
		if len(best) > 1:
			lines += [f"The architectures span {(best['meanR2'].max() - best['meanR2'].min()):.4f} "
					  f"of mean R2: {best.iloc[0]['model']} at {best.iloc[0]['meanR2']:.4f} down to "
					  f"{best.iloc[-1]['model']} at {best.iloc[-1]['meanR2']:.4f}.", ""]

		if "meanRuntime" in best and best["meanRuntime"].notna().any():
			bench = best[best["meanRuntime"].notna()].sort_values("meanRuntime")
			lines += ["## Compute cost", "",
					  "Median wall clock of one forward + backward + optimiser step on the MI100 at",
					  "batch 256 over random data (`torchkissann_bench`), averaged over the circuits,",
					  "for the best configuration of every architecture. Total training time is this",
					  "times the iterations per epoch, so it is directly comparable across models.",
					  "Note how little it tracks the parameter count: the recurrent and transformer",
					  "models are launch bound and slow per step despite few parameters.", "",
					  markdownTable(bench[["model", "source", "parameters", "meanRuntime",
											"meanR2"]].round(4)), "",
					  "![R2 against compute cost](r2_vs_runtime.pdf)", ""]

		native = configs[configs["source"] == "native"]
		if not native.empty:
			grid = native.pivot_table(index=["network", "learningRate", "downsampleSteps"],
				columns="extraSteps", values="meanR2", aggfunc="max")
			lines += ["## Native taper grid, mean R2 over all circuits", "",
					  markdownTable(grid.reset_index().round(4)), ""]

		# whether the mean hides a model that is only good on one of the circuits
		if perCircuit:
			lines += ["## Ranking within each circuit (top 5 configurations)", ""]
			for circuit in perCircuit:
				top = configs.sort_values(circuit, ascending=False).head(5)
				lines += [f"### {circuit}", "",
						  markdownTable(top[["model", "learningRate", "weightDecay", circuit,
											 "meanR2"]].round(4)), ""]

		if not targetR2.empty:
			perTarget = targetR2.groupby("targetName")["r2"].mean().sort_values()
			lines += ["## Mean R2 per circuit parameter over all successful trials", "",
					  markdownTable(pd.DataFrame({"target": perTarget.index,
												  "r2": perTarget.round(4)})), ""]

			winner = configs.iloc[0]
			perClass = targetR2[targetR2["key"] == winner["key"]].groupby("targetName")["r2"].mean()
			order = [name for name in dict.fromkeys(targetR2["targetName"]) if name in perClass.index]
			perClass = perClass.reindex(order)
			lines += [f"## Winner ({winner['model']}, lr {winner['learningRate']:g}) per parameter", "",
					  markdownTable(pd.DataFrame({"target": perClass.index,
												  "r2": perClass.round(4)})), ""]
			lines += ["![R2 per parameter](target_r2.pdf)", ""]

	if not ok.empty:
		columns = ["dataset", "model", "source", "parameters", "learningRate", "weightDecay",
				   "downsampleSteps", "extraSteps", "bestEpoch", "bestR2", "bestLoss", "seconds"]
		columns = [column for column in columns if column in ok.columns]
		lines += ["## All trials ranked by best R2 on their own circuit (top 30)", "",
				  markdownTable(ok.sort_values("bestR2", ascending=False)[columns].head(30)), ""]

	if long and longTable is not None and not longTable.empty:
		budget = epochs
		longEpochs = int(longTable["epochs"].max())
		lines += ["## Extended training budget", "",
				  f"Retrained with {longEpochs} epochs instead of {budget} on the same datasets and the",
				  "same networks, because these architectures reached their best epoch in the last",
				  f"epochs of the {budget} epoch budget. They are deliberately kept out of the ranking",
				  "above, which compares runs of equal length.", ""]
		if longMilestones is not None and not longMilestones.empty:
			lines += [f"Mean R2 at fixed epoch counts (learning rate / weight decay as listed)", "",
					  markdownTable(longMilestones.round(4)), ""]
		columns = ["model", "parameters", "learningRate", "weightDecay", "epochs",
				   "bestEpoch", "bestR2", "bestLoss", "seconds"]
		lines += [f"Best configuration per architecture over {longEpochs} epochs", "",
				  markdownTable(longTable.sort_values("bestR2", ascending=False)
								[columns].round(4)), ""]
		# both sides of this comparison are the mean over all circuits: comparing a per circuit
		# best R2 against the shootout's mean made the gain of the extra epochs look far larger
		# than it is, because the two numbers measure different things
		if longAveraged is not None and not longAveraged.empty:
			columns = [column for column in ["model", "circuits", "shootoutR2",
											 f"meanR2At{budget}", "meanBestR2",
											 "meanGainAfterBase", "meanBestEpoch", "stillImproving"]
					   if column in longAveraged]
			lines += [f"Mean R2 over all circuits, {budget} epochs against {longEpochs}", "",
					  markdownTable(longAveraged[columns].round(4)), ""]
			lines += ["![short budget against the extended run](long_ranking_shift.pdf)", ""]
			for _, row in longAveraged.iterrows():
				reference = row["shootoutR2"] if "shootoutR2" in row else float("nan")
				known = not (reference is None or pd.isna(reference))
				lines += [f"- {row['model']}: "
						  + (f"{reference:.4f} within {budget} epochs, " if known else "")
						  + f"{row['meanBestR2']:.4f} after {longEpochs} epochs"
						  + (f", {row['meanBestR2'] - reference:+.4f} from the extra epochs" if known else "")
						  + f"; {int(row['stillImproving'])} of {int(row['circuits'])} circuits "
						  f"still improving when it stopped"]
			lines += [""]

		if longConvergenceTable is not None and not longConvergenceTable.empty:
			lines += ["Convergence per circuit, R2 in normalized target space", "",
					  markdownTable(longConvergenceTable.round(4)), ""]
			lines += ["![convergence of the extended runs](long_learning_curves.pdf)", "",
					  f"![gain after the shootout budget](long_gain_after_{budget}.pdf)", ""]

		perTarget = long.get("targetR2")
		if perTarget is not None and not perTarget.empty:
			top = longTable.sort_values("bestR2", ascending=False).iloc[0]
			perClass = perTarget[perTarget["key"] == top["key"]].groupby("targetName")["r2"].mean()
			lines += [f"Per parameter for the best extended run "
					  f"({top['model']}, lr {top['learningRate']:g}, epoch {int(top['bestEpoch'])})", "",
					  markdownTable(pd.DataFrame({"target": perClass.index,
												  "r2": perClass.round(4)})), ""]
			lines += ["![R2 per parameter of the extended runs](long_target_r2.pdf)", ""]

		lines += ["![extended runs](long_runs.pdf)", ""]

	(summary / "report.md").write_text("\n".join(lines))


def main():
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("outdir", type=Path, help="the shootout directory")
	parser.add_argument("--epochs", type=int, default=20, help="only used in the report text")
	args = parser.parse_args()

	summary = args.outdir / "summary"
	summary.mkdir(parents=True, exist_ok=True)

	circuits = readDatasets(args.outdir)
	tags = [str(tag) for tag in circuits["tag"]]
	names = {tag: readTargetNames(args.outdir, tag) for tag in tags}
	trials = loadTrials(args.outdir, tags)
	# the compute cost of every architecture, joined onto each trial by its circuit's target count
	targetsByDataset = {str(row["tag"]): row["targets"] for _, row in circuits.iterrows()}
	runtimeFrame = loadRuntime(args.outdir)
	runtimeMs = {runtimeKey(row, row.get("outputSize")): row["medianMs"]
				 for _, row in runtimeFrame.iterrows()} if not runtimeFrame.empty else {}
	if runtimeMs:
		trials = addRuntime(trials, runtimeMs, targetsByDataset)
	curves = describe(readJobTable(args.outdir, "curves.csv", tags=tags))
	targetR2 = readJobTable(args.outdir, "targetr2.csv", tags=tags)
	for frame in (curves, targetR2):
		if not frame.empty and "dataset" not in frame:
			frame["dataset"] = ""
	if not targetR2.empty:
		targetR2 = describe(targetR2)

		def targetName(dataset: str, index: int) -> str:
			# the parameters of two circuits are different parameters, so the circuit is part
			# of the name wherever the targets of more than one circuit meet in one table
			labels = names.get(dataset, [])
			return f"{dataset}/{labels[index]}" if index < len(labels) else f"{dataset}/{index}"

		targetR2["targetName"] = [targetName(dataset, index)
								  for dataset, index in zip(targetR2["dataset"], targetR2["target"])]
	long = loadLong(args.outdir, names, tags)

	# the ranking of the shootout: one row per configuration, the R2 of every circuit beside it
	configs = crossCircuit(trials, tags)
	# the same ranking for the extended budget runs, so both rest on the mean over the circuits
	longConfigs = crossCircuit(long["trials"], tags) if not long["trials"].empty \
		else pd.DataFrame()

	columns = ["job", "key", "dataset", "config", "model", "family", "source", "channels",
			   "network", "script", "parameters", "learningRate", "weightDecay",
			   "downsampleSteps", "extraSteps", "bestEpoch", "bestLoss", "loss", "bestR2",
			   "finalR2", "seconds", "metric", "status", "error"]
	columns = [column for column in columns if column in trials]
	trials[columns].sort_values("bestR2", ascending=False).to_csv(summary / "trials.csv", index=False)
	if not configs.empty:
		configs.to_csv(summary / "configs.csv", index=False)
		best = bestPerModelConfigs(configs)
		best.to_csv(summary / "models.csv", index=False)
		if "meanRuntime" in best:
			best[best["meanRuntime"].notna()][["model", "source", "parameters",
											   "meanRuntime", "meanR2"]].sort_values("meanRuntime") \
				.to_csv(summary / "runtime.csv", index=False)
	if not curves.empty:
		curves.to_csv(summary / "curves.csv", index=False)
	if not targetR2.empty:
		targetR2.to_csv(summary / "targetr2.csv", index=False)

	longTable = pd.DataFrame()
	longMilestones = pd.DataFrame()
	longOk = long["trials"][(long["trials"]["status"] == "ok") & long["trials"]["bestR2"].notna()] \
		if not long["trials"].empty else pd.DataFrame()
	if not long["trials"].empty:
		if not longOk.empty:
			longTable = longOk.sort_values(["model", "bestR2"], ascending=[True, False]) \
				.groupby("model").first().reset_index()
		longColumns = [column for column in columns if column in long["trials"]] + ["epochs"]
		long["trials"][longColumns].sort_values("bestR2", ascending=False) \
			.to_csv(summary / "longtrials.csv", index=False)
		if not longConfigs.empty:
			longConfigs.to_csv(summary / "longconfigs.csv", index=False)
	if not long["curves"].empty:
		long["curves"].to_csv(summary / "longcurves.csv", index=False)
		if not longOk.empty:
			top = int(long["curves"]["epoch"].max()) + 1
			points = sorted({args.epochs, *[e for e in (50, 100, 150) if e < top], top})
			longMilestones = milestones(long["curves"], points)

	# the convergence analysis folded in from scripts/shootoutreglongplot.py
	longConvergenceTable = longConvergence(long, args.epochs)
	longAveraged = longMeanConvergence(longConvergenceTable, configs)
	if not longConvergenceTable.empty:
		longConvergenceTable.to_csv(summary / "long_convergence.csv", index=False)
		longAveraged.to_csv(summary / "long_mean_convergence.csv", index=False)

	plotMeanR2(configs, tags, summary)
	plotR2VsParams(configs, summary, longConfigs)
	plotR2VsRuntime(configs, summary)
	plotLearningRate(configs, summary)
	plotWeightDecay(configs, summary)
	plotChannels(configs, summary)
	plotNativeTaper(configs, summary)
	plotNativeTaperLine(configs, summary)
	plotLearningCurves(curves, configs, tags, summary)
	plotTargetR2(targetR2, configs, summary)
	plotLongRuns(long, trials, summary)
	plotLongLearningCurves(long, tags, args.epochs, summary)
	plotLongGain(longConvergenceTable, tags, args.epochs, summary)
	plotLongRankingShift(longAveraged, args.epochs, summary)
	plotLongTargetR2(long, summary)
	writeReport(args.outdir, trials, configs, targetR2, summary, args.epochs, circuits,
				long, longTable, longMilestones, longConvergenceTable, longAveraged)

	ok = trials[trials["status"] == "ok"]
	print(f"{len(trials)} trials on {len(tags)} circuits, {len(ok)} succeeded")
	if not long["trials"].empty:
		longOk = long["trials"][long["trials"]["status"] == "ok"]
		print(f"{len(long['trials'])} extended budget trials, {len(longOk)} succeeded")
		if not longOk.empty:
			print(longOk.sort_values("bestR2", ascending=False)[
				["model", "epochs", "learningRate", "bestEpoch", "bestR2", "bestLoss"]]
				.head(10).to_string(index=False))
	if not configs.empty:
		summaryColumns = ["model", "parameters", "learningRate", "weightDecay",
						  *[tag for tag in tags if tag in configs.columns], "meanR2", "worstR2"]
		print(configs.head(15)[[column for column in summaryColumns if column in configs]]
			  .to_string(index=False))
		best = bestPerModelConfigs(configs)
		print(f"\nbest per architecture, by the mean over {len(tags)} circuits:")
		print(best.head(15)[[column for column in summaryColumns if column in best]]
			  .to_string(index=False))
	print(f"wrote {summary}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
