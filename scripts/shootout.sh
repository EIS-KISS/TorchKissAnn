#!/usr/bin/env bash
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
#
# Classification shootout between the TorchScript architectures produced by
# buildnet.py and the tunable native networks.
#
# Everything is written into the output directory (default ./shootout):
#
#   ds_curated_{train,test}.tar   the classification dataset, one class per circuit
#   nets_all/*.pt                 the TorchScript candidates
#   groups/gN/*.pt                the candidates split over the worker slots
#   jobs/NNN_name.sh              one tuner invocation per file
#   results/NNN_name/             results.csv, curves.csv, classaccuracy.csv, best/
#   logs/NNN_name.log             full tuner output of that job
#   failed.txt                    jobs that did not finish cleanly
#   summary/                      aggregated tables and plots (shootoutplot.py)
#
# Usage: scripts/shootout.sh [outputdir]
# Environment: EPOCHS, JOBS, MODELLIST, FREQS, TRAINSIZE, TESTPERCENT, LRS,
#              WEIGHTDECAYS, NATIVE_LRS, NATIVE_WD, DOWNSAMPLE_STEPS, EXTRA_STEPS

set -uo pipefail

OUT="${1:-$(pwd)/shootout}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TUNE="$REPO/build/src/tune/torchkissann_tune"
DATASETINFO="$REPO/build/src/datasetinfo/torchkissann_datasetinfo"
BENCH="$REPO/build/src/bench/torchkissann_bench"
MODELLIST="${MODELLIST:-$HOME/Programming/kiss/KissDatasetGenerator/modellists/curated.txt}"

EPOCHS="${EPOCHS:-20}"
JOBS="${JOBS:-6}"
FREQS="${FREQS:-200}"
TRAINSIZE="${TRAINSIZE:-12667}"
TESTPERCENT="${TESTPERCENT:-20}"
LRS="${LRS:-0.01,0.003,0.001}"
WEIGHTDECAYS="${WEIGHTDECAYS:-0.01,0.001}"
NATIVE_LRS="${NATIVE_LRS:-0.005,0.001}"
NATIVE_WD="${NATIVE_WD:-0.001}"
DOWNSAMPLE_STEPS="${DOWNSAMPLE_STEPS:-2,3,4,6}"
EXTRA_STEPS="${EXTRA_STEPS:-0,3,6}"
# Benchmark settings: batch matches the tuner default, the rest control the timing loop.
BENCH_BATCH="${BENCH_BATCH:-256}"
BENCH_WARMUP="${BENCH_WARMUP:-10}"
BENCH_ITERS="${BENCH_ITERS:-30}"
# Device 0 is normally the card that serves the desktop and everything else, use the
# free MI100 when the machine has one (its uuid is exported as MI100A).
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-${MI100A:-}}"

tag() { echo "$1" | tr '.' 'p'; }

[ -x "$TUNE" ] || { echo "$TUNE is missing, build the tools first" >&2; exit 1; }
command -v kissdatasetgenerator >/dev/null || { echo "kissdatasetgenerator not in PATH" >&2; exit 1; }

mkdir -p "$OUT/nets_all" "$OUT/results" "$OUT/logs"

echo "== dataset =="
if [ -s "$OUT/ds_curated_train.tar" ] && [ -s "$OUT/ds_curated_test.tar" ]; then
	echo "keeping the existing dataset in $OUT"
else
	kissdatasetgenerator -t gen -d "$MODELLIST" -o "$OUT/ds_curated" -p "$TESTPERCENT" \
		-s "size=$TRAINSIZE" -c "$FREQS" > "$OUT/logs/dataset.log" 2>&1 || {
		echo "dataset generation failed, see $OUT/logs/dataset.log" >&2; exit 1; }
fi
"$DATASETINFO" -d tar -f "$OUT/ds_curated_train.tar" -r "$OUT/dataset_report_train.txt" \
	> "$OUT/logs/datasetinfo.log" 2>&1 || echo "warning: datasetinfo failed"
"$DATASETINFO" -d tar -f "$OUT/ds_curated_test.tar" -r "$OUT/dataset_report_test.txt" \
	>> "$OUT/logs/datasetinfo.log" 2>&1 || echo "warning: datasetinfo failed"
head -2 "$OUT/dataset_report_train.txt" 2>/dev/null

echo "== torchscript architectures =="
# buildnet.py names its output after the architecture and the io sizes, so the two
# channel variants are built in separate directories and merged with a _cN suffix.
for channels in 1 2; do
	src="$OUT/nets_c$channels"
	mkdir -p "$src"
	for type in simple conv resnet upsamplenet inception tcn gru transformer seresnet; do
		# Only the nets that build a sequence of tokens know about channels.
		if [ "$channels" = 2 ]; then
			case "$type" in
				inception|tcn|gru|transformer|seresnet) ;;
				*) continue ;;
			esac
		fi
		ls "$src"/"$type"*.pt >/dev/null 2>&1 && continue
		(cd "$src" && python3 "$REPO/scripts/buildnet.py" -t "$type" -i "$FREQS" -o 19 \
			-c "$channels" > "$OUT/logs/buildnet_${type}_c$channels.log" 2>&1) ||
			echo "warning: could not build $type with $channels channels"
	done
	for file in "$src"/*.pt; do
		[ -e "$file" ] || continue
		cp "$file" "$OUT/nets_all/$(basename "$file" .pt)_c$channels.pt"
	done
done
# The resnet declares no input size in its metadata, name it after the real one.
for file in "$OUT"/nets_all/resnetANY-*_c1.pt; do
	[ -e "$file" ] || continue
	mv "$file" "${file/resnetANY-/resnet$FREQS-}"
done
ls "$OUT/nets_all" | sed 's/^/  /'

echo "== splitting the architectures over $JOBS workers =="
# Seconds per epoch on a MI100, measured, used to balance the worker slots.
declare -A COST=([seresnet]=27 [inception]=25 [resnet]=24 [convnet]=18 [tcn]=17
	[upsamplenet]=16 [gru]=13 [transformer]=6 [simplenet]=4)
rm -rf "$OUT/groups"
for i in $(seq 1 "$JOBS"); do
	mkdir -p "$OUT/groups/g$i"
	load[$((i - 1))]=0
done
for file in "$OUT"/nets_all/*.pt; do
	name="$(basename "$file")"
	cost=10
	for key in "${!COST[@]}"; do
		[[ "$name" == "$key"* ]] && cost="${COST[$key]}"
	done
	best=0
	for i in $(seq 1 "$JOBS"); do
		((load[$((i - 1))] < load[best])) && best=$((i - 1))
	done
	cp "$file" "$OUT/groups/g$((best + 1))/"
	load[best]=$((load[best] + cost))
done
for i in $(seq 1 "$JOBS"); do
	echo "  g$i (cost ${load[$((i - 1))]}): $(ls "$OUT/groups/g$i" | tr '\n' ' ')"
done

echo "== jobs =="
rm -rf "$OUT/jobs"
mkdir -p "$OUT/jobs"
number=0
add_job() {
	number=$((number + 1))
	local id
	id="$(printf '%03d' "$number")"
	printf '#!/usr/bin/env bash\n%s\n' "$2" > "$OUT/jobs/${id}_$1.sh"
	chmod +x "$OUT/jobs/${id}_$1.sh"
	echo "  ${id}_$1 -> results/${id}_$1"
}

# Phase A: every TorchScript architecture against the learning rate / weight decay grid.
for wd in ${WEIGHTDECAYS//,/ }; do
	for i in $(seq 1 "$JOBS"); do
		[ -n "$(ls -A "$OUT/groups/g$i" 2>/dev/null)" ] || continue
		name="A_wd$(tag "$wd")_g$i"
		id="$(printf '%03d' $((number + 1)))"
		add_job "$name" "$TUNE -m ann -d tar -f $OUT/ds_curated_train.tar -t $OUT/ds_curated_test.tar --scripts $OUT/groups/g$i -i $EPOCHS --learning-rates $LRS --weight-decays $wd -o $OUT/results/${id}_${name}"
	done
done

# Phase B: the taper of the native networks, crossed with the learning rate.
for net in conv ann; do
	for lr in ${NATIVE_LRS//,/ }; do
		for ds in ${DOWNSAMPLE_STEPS//,/ }; do
			name="B_${net}_ds${ds}_lr$(tag "$lr")"
			id="$(printf '%03d' $((number + 1)))"
			add_job "$name" "$TUNE -m $net -d tar -f $OUT/ds_curated_train.tar -t $OUT/ds_curated_test.tar -i $EPOCHS --downsample-steps $ds --extra-steps $EXTRA_STEPS --learning-rates $lr --weight-decays $NATIVE_WD -o $OUT/results/${id}_${name}"
		done
	done
done
echo "  $number jobs"

echo "== running (this is the slow part) =="
: > "$OUT/failed.txt"
start=$(date +%s)
# shellcheck disable=SC2016
find "$OUT/jobs" -name '*.sh' | sort |
	xargs -P "$JOBS" -I{} env OUT="$OUT" bash -c \
		'if ! bash "$1" > "$OUT/logs/$(basename "$1" .sh).log" 2>&1; then echo "$(basename "$1")" >> "$OUT/failed.txt"; fi' _ {}
echo "elapsed: $((($(date +%s) - start) / 60)) minutes"
echo "failed jobs: $(wc -l < "$OUT/failed.txt")"
sed 's/^/  /' "$OUT/failed.txt"

echo "== benchmark =="
# Measure the per iteration compute cost of every architecture on random data, so the summary
# can plot accuracy against runtime instead of against the parameter count, which barely
# predicts it. One row per architecture in bench.csv, joined to the results by the architecture
# identity: the TorchScript file, or the network and its taper.
if [ -x "$BENCH" ]; then
	BENCHCSV="$OUT/bench.csv"
	: > "$BENCHCSV"
	for file in "$OUT"/nets_all/*.pt; do
		[ -e "$file" ] || continue
		"$BENCH" -n script -s "$file" -i "$FREQS" -b "$BENCH_BATCH" -w "$BENCH_WARMUP" -x "$BENCH_ITERS" \
			-S "$TRAINSIZE" -o "$BENCHCSV" > "$OUT/logs/bench_$(basename "$file" .pt).log" 2>&1 \
			|| echo "warning: benchmark failed for $(basename "$file")"
	done
	for net in simple conv; do
		for ds in ${DOWNSAMPLE_STEPS//,/ }; do
			for extra in ${EXTRA_STEPS//,/ }; do
				"$BENCH" -n "$net" -i "$FREQS" -O 19 -D "$ds" -E "$extra" -b "$BENCH_BATCH" \
					-w "$BENCH_WARMUP" -x "$BENCH_ITERS" -S "$TRAINSIZE" -o "$BENCHCSV" \
					> "$OUT/logs/bench_${net}_ds${ds}_ex${extra}.log" 2>&1 \
					|| echo "warning: benchmark failed for $net ds$ds ex$extra"
			done
		done
	done
	echo "  $(($(wc -l < "$BENCHCSV") - 1)) architectures benchmarked -> bench.csv"
else
	echo "  $BENCH is missing, skipping the benchmark"
fi

echo "== aggregating =="
python3 "$REPO/scripts/shootoutplot.py" "$OUT" || echo "warning: aggregation failed"
