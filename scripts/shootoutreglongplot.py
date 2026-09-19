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
Aggregate the long shootout runs into convergence tables and plots.

	./scripts/shootoutreglongplot.py [-d shootout_reg/long] [-b shootout_reg]

Reads <dir>/results/<job>/ (job names NNN_<circuit-tag>_<architecture>) plus the
parent run's datasets.tsv for the circuit mapping and summary/configs.csv for the
40 epoch baseline. Writes into <dir>/summary/:

  convergence.csv        per architecture and circuit: epochs run, best epoch,
                         best/final R2, R2 at the old budget, gain after it,
                         stillImproving
  mean_convergence.csv   per architecture, averaged over the circuits
  learning_curves.pdf    one panel per circuit, test R2 vs epoch
  gain_after_40.pdf      how much R2 the epochs beyond the old budget bought
  ranking_shift.pdf      old budget vs long mean R2 per architecture
  target_r2.pdf          per parameter R2 at the best epoch, per circuit
  report.md              the tables above

The question this answers: most 40 epoch trials were still improving when they
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
from cycler import cycler

# same grayscale style as shootoutregplot.py
GRAY_SHADES = ["#000000", "#3d3d3d", "#666666", "#8c8c8c",
               "#a6a6a6", "#bdbdbd", "#d0d0d0", "#dedede"]
LINE_STYLES = ("-", "--", "-.", ":")
plt.rcParams["axes.prop_cycle"] = cycler(linestyle=LINE_STYLES) * cycler(color=GRAY_SHADES)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = ":"
plt.rcParams["grid.linewidth"] = 0.6
plt.rcParams["grid.alpha"] = 0.6
plt.rcParams["axes.axisbelow"] = True
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["image.cmap"] = "gray"
plt.rcParams["pdf.fonttype"] = 42

JOB_RE = re.compile(r"^\d+_(?P<rest>.+)$")


def readTsv(path):
	"""datasets.tsv, which has no header: tag, circuit, targets, train size, test size."""
	if not os.path.isfile(path):
		sys.exit(f"not found: {path}")
	fields = ("tag", "circuit", "targets", "train", "test")
	rows = []
	with open(path, newline="") as fh:
		for line in csv.reader(fh, delimiter="\t"):
			if not line or not line[0].strip():
				continue
			rows.append(dict(zip(fields, [f.strip() for f in line])))
	return rows


def splitJob(job, tags):
	"""jobs are NNN_<anything>_<tag>_<architecture>; architecture names have no underscores."""
	m = JOB_RE.match(job)
	if not m:
		return None, None
	rest = m.group("rest")
	for index, field in enumerate(rest.split("_")):
		if field in tags:
			return field, "_".join(rest.split("_")[index + 1:]) or field
	return None, None


def readCurves(resultsdir, tags):
	"""-> ({circuit: {arch: {epoch: r2}}}, longest epoch per circuit)."""
	curves = defaultdict(dict)
	longest = defaultdict(int)
	for job in sorted(os.listdir(resultsdir)):
		circuit, arch = splitJob(job, tags)
		if circuit is None:
			continue
		path = os.path.join(resultsdir, job, "curves.csv")
		if not os.path.isfile(path):
			continue
		series = {}
		with open(path, newline="") as fh:
			for row in csv.DictReader(fh):
				if row["trial"] != "1":
					continue
				try:
					series[int(row["epoch"])] = float(row["r2Mean"])
				except ValueError:
					continue
		if not series:
			continue
		curves[circuit][arch] = series
		longest[circuit] = max(longest[circuit], max(series))
	return curves, longest


def readTargetR2(resultsdir, tags):
	"""-> {circuit: {arch: {target: r2}}} at each run's best epoch."""
	out = defaultdict(dict)
	for job in sorted(os.listdir(resultsdir)):
		circuit, arch = splitJob(job, tags)
		if circuit is None:
			continue
		path = os.path.join(resultsdir, job, "targetr2.csv")
		if not os.path.isfile(path):
			continue
		byEpoch = defaultdict(dict)
		with open(path, newline="") as fh:
			for row in csv.DictReader(fh):
				if row["trial"] != "1":
					continue
				try:
					r2 = float(row["r2"])
				except ValueError:
					continue
				byEpoch[int(row["epoch"])][row["target"]] = r2
		if not byEpoch:
			continue
		best = max(byEpoch, key=lambda e: sum(byEpoch[e].values()) / len(byEpoch[e]))
		out[circuit][arch] = byEpoch[best]
	return out


def readBaseline(base):
	"""the short shootout: mean R2 per family."""
	path = os.path.join(base, "summary", "configs.csv")
	if not os.path.isfile(path):
		return {}
	values = defaultdict(list)
	with open(path, newline="") as fh:
		for row in csv.DictReader(fh):
			try:
				values[row["family"]].append(float(row["meanR2"]))
			except (ValueError, KeyError):
				continue
	return {f: sum(v) / len(v) for f, v in values.items()}


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
	parser = argparse.ArgumentParser(description="plot the long regression shootout")
	parser.add_argument("-d", "--dir", default="shootout_reg/long",
	                    help="the long run directory (default: shootout_reg/long)")
	parser.add_argument("-b", "--base", default=None,
	                    help="the short shootout run (default: parent of --dir)")
	parser.add_argument("--baseline-epoch", type=int, default=40,
	                    help="the old epoch budget (default: 40)")
	args = parser.parse_args()

	resultsdir = os.path.join(args.dir, "results")
	if not os.path.isdir(resultsdir):
		sys.exit(f"no results in {resultsdir} -- run scripts/shootout_reg_long.sh first")
	base = args.base or os.path.dirname(os.path.abspath(args.dir))

	rows = readTsv(os.path.join(base, "datasets.tsv"))
	tags = [r["tag"] for r in rows]
	circuitOf = {r["tag"]: r["circuit"] for r in rows}

	curves, longest = readCurves(resultsdir, tags)
	targets = readTargetR2(resultsdir, tags)
	baseline = readBaseline(base)

	archs = sorted({a for per in curves.values() for a in per})
	used = [t for t in tags if curves.get(t)]
	if not archs or not used:
		sys.exit("no parsable long runs found")

	summarydir = os.path.join(args.dir, "summary")
	os.makedirs(summarydir, exist_ok=True)
	baseKey = f"r2At{args.baseline_epoch}"

	# ---- convergence, per architecture and circuit ----
	conv = []
	for arch in archs:
		for tag in used:
			series = curves[tag].get(arch)
			if not series:
				continue
			epochs = sorted(series)
			bestEpoch = max(epochs, key=lambda e: series[e])
			atBase = series.get(args.baseline_epoch)
			conv.append({
				"architecture": arch,
				"circuit": circuitOf[tag],
				"epochs": epochs[-1],
				"bestEpoch": bestEpoch,
				"bestR2": series[bestEpoch],
				"finalR2": series[epochs[-1]],
				baseKey: atBase,
				"gainAfterBase": None if atBase is None else series[bestEpoch] - atBase,
				"stillImproving": bestEpoch >= epochs[-1] - 5,
			})
	with open(os.path.join(summarydir, "convergence.csv"), "w", newline="") as fh:
		writer = csv.DictWriter(fh, fieldnames=list(conv[0].keys()))
		writer.writeheader()
		writer.writerows(conv)

	# ---- averaged over circuits ----
	agg = []
	for arch in archs:
		sub = [r for r in conv if r["architecture"] == arch]
		agg.append({
			"architecture": arch,
			"circuits": len(sub),
			"meanBestR2": mean(r["bestR2"] for r in sub),
			"meanFinalR2": mean(r["finalR2"] for r in sub),
			f"mean{baseKey}": mean(r[baseKey] for r in sub),
			"meanGainAfterBase": mean(r["gainAfterBase"] for r in sub),
			"meanBestEpoch": mean(r["bestEpoch"] for r in sub),
			"stillImproving": sum(1 for r in sub if r["stillImproving"]),
			"baselineR2": baseline.get(arch),
		})
	agg.sort(key=lambda r: -(r["meanBestR2"] if r["meanBestR2"] is not None else -9))
	with open(os.path.join(summarydir, "mean_convergence.csv"), "w", newline="") as fh:
		writer = csv.DictWriter(fh, fieldnames=list(agg[0].keys()))
		writer.writeheader()
		writer.writerows(agg)

	# ---- learning curves, one panel per circuit ----
	fig, axes = plt.subplots(2, 2, figsize=(11, 8))
	for ax, tag in zip(axes.flat, used):
		for arch in archs:
			series = curves[tag].get(arch)
			if not series:
				continue
			epochs = sorted(series)
			ax.plot(epochs, [series[e] for e in epochs], label=arch, lw=1.4)
		ax.axvline(args.baseline_epoch, color="#888888", ls=":", lw=1.0)
		ax.set_title(f"{circuitOf[tag]}  ({tag})", fontsize=10)
		ax.set_xlabel("epoch")
		ax.set_ylabel("test R2")
	for ax in axes.flat[len(used):]:
		ax.axis("off")
	axes.flat[min(1, len(used) - 1)].legend(fontsize=8, loc="lower right")
	fig.suptitle(f"Test R2 over up to {max(longest.values())} epochs "
	             f"(dotted: the {args.baseline_epoch} epoch budget)", fontsize=12)
	fig.tight_layout()
	fig.savefig(os.path.join(summarydir, "learning_curves.pdf"))
	plt.close(fig)

	# ---- gain beyond the old budget ----
	fig, ax = plt.subplots(figsize=(8, 4.2))
	width = 0.8 / len(used)
	for i, tag in enumerate(used):
		gains = [next((r["gainAfterBase"] for r in conv
		              if r["architecture"] == a and r["circuit"] == circuitOf[tag]), None)
		         for a in archs]
		ax.bar([j + i * width for j in range(len(archs))],
		       [g if g is not None else 0 for g in gains], width,
		       label=f"{tag} ({circuitOf[tag]})")
	ax.set_xticks(range(len(archs)))
	ax.set_xticklabels(archs, fontsize=8)
	ax.axhline(0, color="#333333", lw=0.8)
	ax.set_ylabel(f"R2 gained after epoch {args.baseline_epoch}")
	ax.legend(fontsize=7)
	fig.tight_layout()
	fig.savefig(os.path.join(summarydir, "gain_after_40.pdf"))
	plt.close(fig)

	# ---- ranking shift ----
	fig, ax = plt.subplots(figsize=(7.5, 4.2))
	width = 0.38
	old = [baseline.get(a) for a in archs]
	new = [next((r["meanBestR2"] for r in agg if r["architecture"] == a), None) for a in archs]
	ax.bar([j - width / 2 for j in range(len(archs))], [o if o is not None else 0 for o in old],
	       width, label=f"{args.baseline_epoch} epochs (short shootout)")
	ax.bar([j + width / 2 for j in range(len(archs))], [n if n is not None else 0 for n in new],
	       width, label="long run")
	ax.set_xticks(range(len(archs)))
	ax.set_xticklabels(archs, fontsize=8)
	ax.set_ylabel("mean test R2 (all circuits)")
	ax.legend(fontsize=8)
	fig.tight_layout()
	fig.savefig(os.path.join(summarydir, "ranking_shift.pdf"))
	plt.close(fig)

	# ---- per parameter R2 ----
	if targets:
		fig, axes = plt.subplots(2, 2, figsize=(12, 8))
		for ax, tag in zip(axes.flat, used):
			cell = targets.get(tag, {})
			names = sorted({n for v in cell.values() for n in v})
			if not names:
				ax.axis("off")
				continue
			width = 0.8 / len(archs)
			for i, arch in enumerate(archs):
				vals = [cell.get(arch, {}).get(n) for n in names]
				ax.bar([j + i * width for j in range(len(names))],
				       [v if v is not None else 0 for v in vals], width, label=arch)
			ax.set_xticks(range(len(names)))
			ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
			ax.axhline(0, color="#333333", lw=0.8)
			ax.set_title(f"{circuitOf[tag]}  ({tag})", fontsize=10)
		for ax in axes.flat[len(used):]:
			ax.axis("off")
		axes.flat[0].set_ylabel("R2 at best epoch")
		axes.flat[0].legend(fontsize=7)
		fig.tight_layout()
		fig.savefig(os.path.join(summarydir, "target_r2.pdf"))
		plt.close(fig)

	# ---- report ----
	def cell(v):
		if isinstance(v, bool):
			return "yes" if v else "no"
		if isinstance(v, float):
			return fmt(v)
		return v

	lines = [f"# Long shootout: {len(archs)} architectures x {len(used)} circuits", ""]
	lines.append(markdownTable(
		["architecture", "circuits", "mean best R2", f"mean R2 @ {args.baseline_epoch}",
		 "mean gain", "mean best epoch", "still improving", f"short shootout ({args.baseline_epoch} ep)"],
		[[r["architecture"], r["circuits"], fmt(r["meanBestR2"]), fmt(r[f"mean{baseKey}"]),
		  fmt(r["meanGainAfterBase"]), fmt(r["meanBestEpoch"], 1),
		  f"{r['stillImproving']}/{r['circuits']}", fmt(r["baselineR2"])] for r in agg]))
	lines += ["", "## Per circuit", ""]
	lines.append(markdownTable(
		["arch", "circuit", "epochs", "best ep", "best R2", "final R2",
		 f"R2 @ {args.baseline_epoch}", "gain", "improving"],
		[[cell(r[k]) for k in ("architecture", "circuit", "epochs", "bestEpoch", "bestR2",
		                      "finalR2", baseKey, "gainAfterBase", "stillImproving")] for r in conv]))
	lines += ["", "Plots: `learning_curves.pdf`, `gain_after_40.pdf`, `ranking_shift.pdf`, "
	          "`target_r2.pdf`.", ""]
	with open(os.path.join(summarydir, "report.md"), "w") as fh:
		fh.write("\n".join(lines))

	print(f"wrote {summarydir}/report.md")
	print(f"{len(conv)} runs, architectures: {', '.join(archs)}")


if __name__ == "__main__":
	main()
