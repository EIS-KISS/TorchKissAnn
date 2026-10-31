#!/usr/bin/env python3

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


import argparse
import math
import os
import re
import sys
import tarfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

plt.rcParams["pdf.fonttype"] = 42

BARE_CLASSES = ("r", "c", "w", "p", "l")

SPECTRA_NORMS = ("none", "zmean", "zgeom", "zmax")

VIOLIN_STYLES = (
	{"linestyle": "solid", "facecolor": "white", "markerfacecolor": "black", "markersize": 6,
	"markerline": 0.0},
	{"linestyle": (0, (5, 2)), "facecolor": "none", "markerfacecolor": "none", "markersize": 12,
	"markerline": 0.8},
)


def class_of(model: str) -> str:
	out = []
	depth = 0
	for c in model:
		if c in "{[<":
			depth += 1
		if depth == 0:
			out.append(c)
		if c in "}]>":
			depth -= 1
	name = "".join(out)
	if len(name) < 2 and name not in BARE_CLASSES:
		name = "Union"
	return name


def parse_model(lines) -> str:
	model = ""
	for i, line in enumerate(lines):
		if line.startswith("EISF"):
			if i + 1 < len(lines):
				model = lines[i + 1]
			break
	else:
		if len(lines) > 1:
			model = lines[1]
	return model.strip().strip('"')


def parse_spectra(lines):
	hdr = None
	for i, line in enumerate(lines):
		if line.startswith("omega"):
			hdr = i
			break
	if hdr is None:
		return None
	rows = []
	for line in lines[hdr + 1:]:
		line = line.strip()
		if not line:
			continue
		fields = line.split(",")
		if len(fields) < 3:
			continue
		try:
			rows.append((float(fields[0]), float(fields[1]), float(fields[2])))
		except ValueError:
			break
	if not rows:
		return None
	arr = np.array(rows, dtype=float)
	order = np.argsort(arr[:, 0])
	arr = arr[order]
	return arr[:, 0], arr[:, 1], arr[:, 2]


def load_dataset(tarpath: str):
	kept = {}
	with tarfile.open(tarpath) as tf:
		for member in tf:
			if not member.isfile() or not member.name.endswith(".csv"):
				continue
			text = tf.extractfile(member).read().decode(errors="replace")
			lines = text.splitlines()
			if not lines:
				continue
			spec = parse_spectra(lines)
			if spec is None:
				continue
			kept.setdefault(class_of(parse_model(lines)), []).append(spec)
	return kept


def norm_reference(re, im, mode):
	mag = np.sqrt(re * re + im * im)
	mag = mag[np.isfinite(mag)]
	if mag.size == 0:
		return 0.0
	if mode == "zmean":
		return float(mag.mean())
	if mode == "zgeom":
		pos = mag[mag > 0]
		if pos.size == 0:
			return 0.0
		return float(10.0 ** np.mean(np.log10(pos)))
	if mode == "zmax":
		return float(mag.max())
	return 0.0


def parse_feature(spec: str, omega_axis: bool):
	if "@" not in spec:
		raise ValueError(f"feature '{spec}' is not of the form part@frequency")
	part, _, freq = spec.partition("@")
	part = part.strip().lower()
	if part in ("re", "real"):
		part = "Re"
	elif part in ("im", "imag", "imaginary"):
		part = "Im"
	else:
		raise ValueError(f"feature '{spec}' part must be re or im, not '{part}'")
	value = parse_frequency(freq)
	return part, (value if omega_axis else 2.0 * math.pi * value)


def parse_frequency(text: str) -> float:
	t = text.strip().lower()
	if not t:
		raise ValueError("empty frequency")
	factor = 1.0
	if t.endswith("hz"):
		t = t[:-2]
		if t.endswith("k"):
			factor, t = 1e3, t[:-1]
		elif t.endswith("m"):
			factor, t = 1e6, t[:-1]
		elif t.endswith("g"):
			factor, t = 1e9, t[:-1]
	elif t and t[-1] in "kmg":
		factor = {"k": 1e3, "m": 1e6, "g": 1e9}[t[-1]]
		t = t[:-1]
	try:
		value = float(t)
	except ValueError:
		raise ValueError(f"cannot parse frequency '{text}'")
	if not value > 0:
		raise ValueError(f"frequency '{text}' must be positive")
	return value * factor


def label(hz: float) -> str:
	for unit, div in (("Grad/s", 1e9), ("Mrad/s", 1e6), ("krad/s", 1e3), ("rad/s", 1)):
		if hz >= div:
			v = hz / div
			return "%g %s" % (round(v, 3) if v < 100 else round(v, 1), unit)
	return "%g mrad/s" % round(hz / 1e-3, 3)


def slug(hz: float) -> str:
	return label(hz).replace(" ", "").replace(".", "p").replace("/", "")


def natural_key(name: str):
	return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def feature_values(spectra, part, omega, norm):
	values = []
	skipped = 0
	for omega_axis, re, im in spectra:
		ref = norm_reference(re, im, norm)
		if norm != "none" and ref > 0 and math.isfinite(ref):
			re, im = re / ref, im / ref
		data = re if part == "Re" else im
		if omega < omega_axis[0] or omega > omega_axis[-1]:
			skipped += 1
			continue
		values.append(float(np.interp(math.log(omega), np.log(omega_axis), data)))
	return np.array(values), skipped


class AsinhLocator(ticker.Locator):

	def __init__(self, linthresh, nbins=6):
		super().__init__()
		self.linthresh = linthresh
		self._nice = ticker.MaxNLocator(nbins=nbins)

	def tick_values(self, vmin, vmax):
		lo = math.asinh(vmin / self.linthresh)
		hi = math.asinh(vmax / self.linthresh)
		return self.linthresh * np.sinh(self._nice.tick_values(lo, hi))

	def __call__(self):
		vmin, vmax = self.axis.get_view_interval()
		return self.tick_values(vmin, vmax)


def asinh_tick_label(value, _pos):
	if abs(value) < 1e-12:
		return "0"
	return "%.3g" % value


def draw_feature(ax, part, omega, series, classes, points, bw, xscale="linear", linthresh=1.0,
	show_ylabels=True, legend_labels=None):
	pos = {c: i + 1 for i, c in enumerate(classes)}
	all_rows = []
	all_medians = []
	for idx, by_class in enumerate(series):
		style = VIOLIN_STYLES[idx]
		avail = [c for c in classes if len(by_class.get(c, ())) >= 2]
		if not avail:
			continue
		positions = [pos[c] for c in avail]
		rows = [by_class[c] for c in avail]
		all_rows.extend(rows)
		medians = [float(np.median(r)) for r in rows]
		all_medians.extend(medians)
		parts = ax.violinplot(rows, positions=positions, orientation="horizontal", widths=0.86,
			showextrema=False, showmeans=False, showmedians=False, points=points, bw_method=bw)
		for body in parts["bodies"]:
			body.set_facecolor(style["facecolor"])
			body.set_edgecolor("black")
			body.set_linewidth(0.9)
			body.set_linestyle(style["linestyle"])
			body.set_alpha(1.0)
		ax.scatter(medians, positions, s=style["markersize"], marker="o",
			facecolors=style["markerfacecolor"], edgecolors="black",
			linewidths=style["markerline"], zorder=5)

	vals = np.concatenate(all_rows)
	if xscale == "asinh":
		ax.set_xscale("asinh", linthresh=linthresh)
		ax.xaxis.set_major_locator(AsinhLocator(linthresh))
		ax.xaxis.set_major_formatter(ticker.FuncFormatter(asinh_tick_label))
		ax.xaxis.set_minor_locator(ticker.NullLocator())
		ax.set_xlim(float(vals.min()), float(vals.max()))
	else:
		iqr = np.array([np.subtract(*np.percentile(r, [75, 25])) for r in all_rows])
		spread = max(float(np.median(iqr)), 1e-12)
		med = np.array(all_medians)
		ax.set_xlim(max(float(vals.min()), float(med.min()) - 8.0 * spread),
			min(float(vals.max()), float(med.max()) + 8.0 * spread))

	ax.set_yticks(list(pos.values()))
	ax.set_yticklabels(classes, fontsize=6.5)
	ax.tick_params(axis="y", left=show_ylabels, labelleft=show_ylabels)
	ax.invert_yaxis()
	ax.tick_params(axis="x", labelsize=6)
	ax.set_xlabel("%s(Z) at %s" % (part, label(omega)), fontsize=7)
	ax.grid(axis="x", linewidth=0.3, color="black", alpha=0.35, linestyle=":")
	ax.set_axisbelow(True)
	for s in ax.spines.values():
		s.set_linewidth(0.5)

	if legend_labels:
		handles = [
			Line2D([], [], color="black", linestyle=st["linestyle"], marker="o",
				markerfacecolor=st["markerfacecolor"], markeredgecolor="black",
				markeredgewidth=st["markerline"] if st["markerline"] else 0.0, label=lab)
			for st, lab in zip(VIOLIN_STYLES, legend_labels)]
		ax.legend(handles=handles, fontsize=5.5, loc="lower right", framealpha=0.9,
			handletextpad=0.4, borderpad=0.3)


def separability(values, labels):
	classes = sorted(set(labels), key=natural_key)
	med = {c: float(np.median([v for v, l in zip(values, labels) if l == c])) for c in classes}
	med_arr = np.array([med[c] for c in classes])
	assigned = [classes[int(np.abs(v - med_arr).argmin())] for v in values]
	return float(np.mean([a == l for a, l in zip(assigned, labels)]))


def main():
	ap = argparse.ArgumentParser(
		description="Per class violin plots of one or more spectral features of a dataset tar.")
	ap.add_argument("datasets", nargs="+", metavar="TAR",
		help="one dataset tar, or two to overlay the second on the first (dashed outline, open circle)")
	ap.add_argument("-F", "--feature", action="append", default=[], metavar="PART@FREQ",
		help="feature to plot, e.g. re@1000 or im@0.1 (repeatable, comma separated list allowed)")
	ap.add_argument("-o", "--output-dir", default="violins", help="directory for the plots")
	ap.add_argument("-f", "--format", default="pdf", help="comma separated output formats (pdf,png,svg)")
	ap.add_argument("--combined", action="store_true",
		help="write a single figure with every feature laid out horizontally instead of one file per feature")
	ap.add_argument("--spectra-norm", choices=SPECTRA_NORMS, default="none",
		help="divide each spectra by a reference impedance before reading the feature (match the "
		     "mode the network was trained with)")
	ap.add_argument("--bw", default="scott", help="violin bandwidth method (scott, silverman, or a number)")
	ap.add_argument("--points", type=int, default=60, help="KDE evaluation points per violin")
	ap.add_argument("--width", type=float, default=4.0, help="figure width in inches (per feature)")
	ap.add_argument("--xscale", choices=("linear", "asinh"), default="linear",
		help="x axis scale: linear (default) or asinh, a pseudo log that also shows zero and negative values")
	ap.add_argument("--linthresh", type=float, default=1.0, metavar="VALUE",
		help="with --xscale asinh, the magnitude below which the axis stays linear (the pseudo log offset)")
	args = ap.parse_args()

	if len(args.datasets) > 2:
		raise SystemExit("give one or two dataset tars")
	for path in args.datasets:
		if not os.path.isfile(path):
			raise SystemExit(f"no such tar: {path}")

	specs = []
	for chunk in args.feature:
		specs.extend(s for s in chunk.split(",") if s.strip())
	if not specs:
		raise SystemExit("give at least one feature with -F/--feature, e.g. -F re@1000")
	features = []
	for spec in specs:
		try:
			features.append(parse_feature(spec, True))
		except ValueError as e:
			raise SystemExit(str(e))

	formats = [f.strip().lstrip(".") for f in args.format.split(",") if f.strip()]
	if not formats:
		raise SystemExit("no output format given")
	os.makedirs(args.output_dir, exist_ok=True)

	kepts, ploteds, ds_labels = [], [], []
	for path in args.datasets:
		print(f"reading {path}")
		kept = load_dataset(path)
		if not kept:
			raise SystemExit(f"no spectra with a parseable data block found in {path}")
		classes = sorted(kept, key=natural_key)
		total = sum(len(v) for v in kept.values())
		print(f"  {total} spectra in {len(classes)} classes: {', '.join(classes)}")
		plotted = [c for c in classes if len(kept[c]) >= 2]
		thin = [c for c in classes if len(kept[c]) < 2]
		if thin:
			print(f"  skipping classes with <2 spectra: {', '.join(thin)}", file=sys.stderr)
		kepts.append(kept)
		ploteds.append(plotted)
		ds_labels.append(os.path.splitext(os.path.basename(path))[0])

	classes = sorted({c for plotted in ploteds for c in plotted}, key=natural_key)
	if not classes:
		raise SystemExit("no class has two or more spectra, nothing to plot")

	bw = args.bw
	try:
		bw = float(bw)
	except ValueError:
		pass

	legend_labels = ds_labels if len(kepts) > 1 else None
	labelw = max(len(l) for l in ds_labels)

	feats = []
	for part, omega in features:
		series, infos = [], []
		for kept, plotted in zip(kepts, ploteds):
			by_class = {}
			skipped = 0
			for c in plotted:
				v, s = feature_values(kept[c], part, omega, args.spectra_norm)
				skipped += s
				by_class[c] = v
			series.append(by_class)
			values = [x for c in plotted for x in by_class[c]]
			labels = [c for c in plotted for _ in range(len(by_class[c]))]
			k = sum(1 for c in plotted if len(by_class[c]) >= 2)
			single = separability(values, labels) if values else float("nan")
			infos.append((len(values), k, single, skipped))
		plottable = any(len(v) >= 2 for by_class in series for v in by_class.values())
		feats.append((part, omega, series, infos, plottable))

	for part, omega, series, infos, plottable in feats:
		for lab, (n, k, single, skipped) in zip(ds_labels, infos):
			note = " (%d out of band)" % skipped if skipped else ""
			if not plottable:
				note += " [skipped: no class has an in band sample]"
			print("%-3s %-14s [%-*s] n=%-6d classes=%-3d nearest median %.1f%%%s"
				% (part, label(omega), labelw, lab, n, k, 100 * single, note))

	written = 0
	if args.combined:
		drawable = [f for f in feats if f[4]]
		if drawable:
			n = len(drawable)
			fig, axes = plt.subplots(1, n, sharey=True, squeeze=False,
				figsize=(args.width * n, max(2.4, 0.34 * len(classes) + 1.4)))
			for i, (part, omega, series, _, _) in enumerate(drawable):
				draw_feature(axes[0][i], part, omega, series, classes, args.points, bw,
					args.xscale, args.linthresh, show_ylabels=(i == 0),
					legend_labels=legend_labels if i == 0 else None)
			fig.tight_layout(pad=0.4)
			for fmt in formats:
				fig.savefig(os.path.join(args.output_dir, "violins_combined.%s" % fmt))
			plt.close(fig)
			written = 1
	else:
		for part, omega, series, _, plottable in feats:
			if not plottable:
				continue
			fig, ax = plt.subplots(figsize=(args.width, max(2.4, 0.34 * len(classes) + 1.4)))
			draw_feature(ax, part, omega, series, classes, args.points, bw,
				args.xscale, args.linthresh, show_ylabels=True, legend_labels=legend_labels)
			fig.tight_layout(pad=0.4)
			for fmt in formats:
				fig.savefig(os.path.join(args.output_dir, "%s_%s.%s" % (part, slug(omega), fmt)))
			plt.close(fig)
			written += 1

	with open(os.path.join(args.output_dir, "features.csv"), "w") as f:
		f.write("dataset,part,frequency_rad_s,spectra,classes,nearest_median_accuracy\n")
		for part, omega, series, infos, _ in feats:
			for lab, (n, k, single, _) in zip(ds_labels, infos):
				f.write('%s,%s,%.6g,%d,%d,%.4f\n' % (lab, part, omega, n, k, single))
	mode = "combined figure" if args.combined else "feature plots"
	print(f"wrote {written} {mode} ({'/'.join(formats)}) and features.csv to {args.output_dir}")


if __name__ == "__main__":
	main()
