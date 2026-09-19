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
# Regression shootout between the TorchScript architectures produced by buildnet.py and the
# tunable native networks, on four equivalent circuits.
#
# One circuit does not separate the architectures: with 20 epochs on rp-w every model landed
# between 0.93 and 0.97 R2. Every candidate is therefore trained on four circuits with five to
# nine parameters and ranked by the mean R2 over all of them, so a model has to be good on
# every circuit instead of lucky on one, and the 40 epoch budget lets the slow architectures
# actually converge instead of being ranked by how fast they start.
#
# Everything is written into the output directory (default ./shootout_reg):
#
#   ds_<tag>_{train,test}.tar    one regression dataset per circuit
#   dataset_report_<tag>_*.txt   the datasetinfo report of each of them
#   datasets.tsv                 tag, circuit, target count, train and test size
#   nets_t<N>_cM/*.pt            the TorchScript candidates for a circuit with N targets
#   nets_all_t<N>/*.pt           the same, merged and suffixed with _cM
#   groups/<tag>/gNN/*.pt        the candidates split into one directory per job
#   jobs/NNN_name.sh             one tuner invocation per file, its gpu pinned inside
#   queue_N.txt                  the jobs assigned to one gpu
#   results/NNN_name/            results.csv, curves.csv, targetr2.csv, best/
#   logs/NNN_name.log            full tuner output of that job
#   failed.txt                   jobs that did not finish cleanly
#   summary/                     aggregated tables and plots (shootoutregplot.py)
#
# The jobs are spread over every card in $GPUS and each card runs $SLOTS tuner processes at
# the same time. The workload is small batch and launch bound, so sixteen concurrent jobs
# measured about four times the throughput of a single one on a MI100 and the gain beyond
# that is small. The cards are not equally fast though: at 16 slots the two MI100s did about
# 10.9 epochs per minute, the RX 7900 XTX 3.0 and the R9700 1.3, hence the uneven weights.
#
# The spectra loss (-s spectra) is deliberately not part of this: it aborts on the ROCm
# backend with "UNSUPPORTED DTYPE: complex", so it could only be compared on the cpu, where
# a run is an order of magnitude slower than the whole gpu shootout.
#
# Usage: scripts/shootout_reg.sh [outputdir]
# Environment: EPOCHS, SLOTS, GPUS, GPU_WEIGHTS, MODELS, FREQS, TRAINSIZE, TESTPERCENT,
#              LRS, WEIGHTDECAYS, NATIVE_LRS, NATIVE_WD, DOWNSAMPLE_STEPS, EXTRA_STEPS

set -uo pipefail

OUT="${1:-$(pwd)/shootout_reg}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TUNE="$REPO/build/src/tune/torchkissann_tune"
DATASETINFO="$REPO/build/src/datasetinfo/torchkissann_datasetinfo"

# The circuits to train on, all from KissDatasetGenerator/modellists/curated.txt. An entry may
# carry a short tag ("rpp=rp-p"), without one the tag is the circuit with every non
# alphanumeric character dropped, and the tag is what names every file below.
MODELS="${MODELS:-rp-p lr-rp-p rp-rp-p lr-rp-p(r-w)}"

EPOCHS="${EPOCHS:-40}"
FREQS="${FREQS:-200}"
TESTPERCENT="${TESTPERCENT:-20}"
# Requested spectra per circuit. The generator sweeps a parameter grid and uses
# round(size^(1/parameters)) steps per parameter, so the real size is that number raised back
# to the parameter count. The request is snapped to the closest grid that is still at least
# half of it, because an eight parameter circuit would otherwise be sampled with three values
# per parameter, which is a lookup table and not a regression task.
TRAINSIZE="${TRAINSIZE:-16384}"
LRS="${LRS:-0.01,0.003,0.001}"
WEIGHTDECAYS="${WEIGHTDECAYS:-0.01,0.001}"
NATIVE_LRS="${NATIVE_LRS:-0.005,0.001}"
NATIVE_WD="${NATIVE_WD:-0.001}"
# The taper of the native networks, up to 16 stages and 16 constant width extra layers.
DOWNSAMPLE_STEPS="${DOWNSAMPLE_STEPS:-2,4,8,16}"
EXTRA_STEPS="${EXTRA_STEPS:-0,4,8,16}"

# The cards to use, by rocm uuid. GPU_WEIGHTS is the relative throughput of each card in
# percent, only used to decide how much work each queue gets. The default is measured, not
# assumed: on this box these small networks are launch latency bound and the four cards do
# 10.9 / 10.9 / 3.0 / 1.3 epochs per minute at 16 slots each, so the MI100s get eight times
# the work of the R9700. Weights that do not match the hardware leave the fast cards idle
# for as long as the slow one needs its share, which here would have been three days.
# Re-measure with a short single-circuit run per card before trusting a new GPU mix.
GPUS="${GPUS:-${MI100A:-} ${MI100B:-} ${RX7900XTX:-} ${R9700:-}}"
GPU_WEIGHTS="${GPU_WEIGHTS:-100 100 27 12}"
SLOTS="${SLOTS:-16}"

tag() { echo "$1" | tr '.' 'p'; }

# The generator only produces grids, so ask for a size that round trips to a grid close to the
# request and never below half of it.
plan_size() {
	python3 -c '
import math, sys
parameters, want = int(sys.argv[1]), int(sys.argv[2])
def penalty(steps):
	size = steps ** parameters
	return abs(math.log(size / want)) + (10 if size < want / 2 else 0)
steps = min(range(2, 64), key=penalty)
# ask for slightly more than the exact grid: the generator rounds size^(1/parameters) itself
# and an exact power can fall on the wrong side of that rounding, silently dropping a whole
# step per parameter
print(int(steps ** parameters * 1.02) + 1)
' "$1" "$2"
}

# The number of targets is the number of circuit parameters and sizes every network built
# below; it is the third field of the second line of a regression datasetinfo report.
report_targets() { sed -n '2p' "$1" | tr ',' '\n' | sed -n '3p' | tr -d '[:space:]'; }
report_size() { sed -n '2p' "$1" | tr ',' '\n' | sed -n '2p' | tr -d '[:space:]'; }

[ -x "$TUNE" ] || { echo "$TUNE is missing, build the tools first" >&2; exit 1; }
command -v kissdatasetgenerator >/dev/null || { echo "kissdatasetgenerator not in PATH" >&2; exit 1; }

mkdir -p "$OUT/results" "$OUT/logs"

echo "== datasets =="
: > "$OUT/datasets.tsv"
TAGS=(); CIRCUITS=(); TARGETS=(); TRAINS=()
for entry in $MODELS; do
	if [[ "$entry" == *=* ]]; then
		circuit="${entry#*=}"; short="${entry%%=*}"
	else
		circuit="$entry"; short="$(printf '%s' "$entry" | tr -cd '[:alnum:]')"
	fi
	train="$OUT/ds_${short}_train.tar"
	testtar="$OUT/ds_${short}_test.tar"
	if [ -s "$train" ] && [ -s "$testtar" ]; then
		echo "  keeping the existing $short dataset"
	else
		# the grid size depends on the parameter count, so probe it at a negligible frequency
		# count first and only then generate the real dataset at the snapped size
		probe="$OUT/logs/probe_$short"
		kissdatasetgenerator -t regression -d "$circuit" -o "$probe" -p "$TESTPERCENT" \
			-s "size=$TRAINSIZE" -c 4 > "$OUT/logs/probe_$short.log" 2>&1 || {
			echo "dataset probe for $circuit failed, see $OUT/logs/probe_$short.log" >&2; exit 1; }
		"$DATASETINFO" -d tarreg -f "${probe}_train.tar" -r "$OUT/logs/probe_$short.txt" \
			>> "$OUT/logs/probe_$short.log" 2>&1 || {
			echo "datasetinfo on the probe for $circuit failed" >&2; exit 1; }
		steps="$(plan_size "$(report_targets "$OUT/logs/probe_$short.txt")" "$TRAINSIZE")"
		rm -f "${probe}"_*.tar
		echo "  generating $circuit ($steps spectra requested)"
		kissdatasetgenerator -t regression -d "$circuit" -o "$OUT/ds_${short}" -p "$TESTPERCENT" \
			-s "size=$steps" -c "$FREQS" > "$OUT/logs/dataset_$short.log" 2>&1 || {
			echo "dataset generation for $circuit failed, see $OUT/logs/dataset_$short.log" >&2
			exit 1; }
	fi
	"$DATASETINFO" -d tarreg -f "$train" -r "$OUT/dataset_report_${short}_train.txt" \
		> "$OUT/logs/datasetinfo_$short.log" 2>&1 || echo "warning: datasetinfo failed"
	"$DATASETINFO" -d tarreg -f "$testtar" -r "$OUT/dataset_report_${short}_test.txt" \
		>> "$OUT/logs/datasetinfo_$short.log" 2>&1 || echo "warning: datasetinfo failed"
	targets="$(report_targets "$OUT/dataset_report_${short}_train.txt")"
	trainSize="$(report_size "$OUT/dataset_report_${short}_train.txt")"
	testSize="$(report_size "$OUT/dataset_report_${short}_test.txt")"
	[ -n "$targets" ] || { echo "could not read the target count of $circuit" >&2; exit 1; }
	printf '%s\t%s\t%s\t%s\t%s\n' "$short" "$circuit" "$targets" "$trainSize" "$testSize" \
		>> "$OUT/datasets.tsv"
	echo "  $short: $circuit, $targets targets, $trainSize train / $testSize test spectra"
	TAGS+=("$short"); CIRCUITS+=("$circuit"); TARGETS+=("$targets"); TRAINS+=("$trainSize")
done

echo "== torchscript architectures =="
# buildnet.py names its output after the architecture and the io sizes, so the two channel
# variants are built in separate directories and merged with a _cN suffix. The networks depend
# on the target count, which differs per circuit, so they are built once per target count.
declare -A SEEN_TARGETS
for targets in "${TARGETS[@]}"; do
	[ -z "${SEEN_TARGETS[$targets]:-}" ] || continue
	SEEN_TARGETS[$targets]=1
	all="$OUT/nets_all_t$targets"
	mkdir -p "$all"
	for channels in 1 2; do
		src="$OUT/nets_t${targets}_c$channels"
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
			(cd "$src" && python3 "$REPO/scripts/buildnet.py" -t "$type" -i "$FREQS" -o "$targets" \
				-c "$channels" > "$OUT/logs/buildnet_t${targets}_${type}_c$channels.log" 2>&1) ||
				echo "warning: could not build $type with $channels channels"
		done
		for file in "$src"/*.pt; do
			[ -e "$file" ] || continue
			cp "$file" "$all/$(basename "$file" .pt)_c$channels.pt"
		done
	done
	# The resnet declares no input size in its metadata, name it after the real one.
	for file in "$all"/resnetANY-*_c1.pt; do
		[ -e "$file" ] || continue
		mv "$file" "${file/resnetANY-/resnet$FREQS-}"
	done
	echo "  t$targets: $(ls "$all" | tr '\n' ' ')"
done

echo "== jobs =="
# Rough seconds per epoch on one MI100 with the original 16.5k spectra dataset, only used to
# hand the cards comparable amounts of work. The native nets get one more full width layer per
# extra step, so their cost grows with the average of the extra steps in the job.
declare -A COST=([seresnet]=41 [inception]=38 [resnet]=36 [tcn]=31 [gru]=28
	[convnet]=27 [upsamplenet]=24 [transformer]=9 [simplenet]=6 [simple]=6 [conv]=27)
REFSIZE=16546
count_field() { echo "${1//,/ }" | wc -w; }
# One cost unit is one epoch on the reference dataset of the first shootout. The 1000x scaling
# keeps the numbers usable when the datasets are tiny, as they are in a dry run.
cost_units() { # seconds per epoch, trials in the job, train spectra
	echo $(( ($1 * $2 * EPOCHS * $3 * 1000 + REFSIZE * 10) / (REFSIZE * 20) ))
}
avg_extra=0
for e in ${EXTRA_STEPS//,/ }; do avg_extra=$((avg_extra + e)); done
extra_list=$(count_field "$EXTRA_STEPS")
avg_extra=$((avg_extra / extra_list))

rm -rf "$OUT/jobs" "$OUT/groups"
mkdir -p "$OUT/jobs"
number=0
total=0
JOB_NAME=(); JOB_COST=(); JOB_CMD=()
add_job() { # name cost command
	number=$((number + 1))
	total=$((total + $2))
	JOB_NAME[$number]="$1"; JOB_COST[$number]="$2"; JOB_CMD[$number]="$3"
}

for index in "${!TAGS[@]}"; do
	short="${TAGS[$index]}"; targets="${TARGETS[$index]}"; train="${TRAINS[$index]}"
	train=$((train + 0))

	# Phase A: every TorchScript architecture against the learning rate / weight decay grid.
	# One job per architecture, so the scheduler has enough pieces to keep four cards busy.
	g=0
	for file in "$OUT"/nets_all_t"$targets"/*.pt; do
		[ -e "$file" ] || continue
		g=$((g + 1))
		dir="$OUT/groups/$short/$(printf 'g%02d' "$g")"
		mkdir -p "$dir"
		cp "$file" "$dir/"
		name="$(basename "$file" .pt)"
		cost=${COST[${name%%[0-9]*}]:-20}
		trials=$(( $(count_field "$LRS") * $(count_field "$WEIGHTDECAYS") ))
		add_job "A_${short}_${name}" "$(cost_units "$cost" "$trials" "$train")" \
			"$TUNE -m regression -d tarreg -f $OUT/ds_${short}_train.tar -t $OUT/ds_${short}_test.tar --scripts $dir -i $EPOCHS --learning-rates $LRS --weight-decays $WEIGHTDECAYS -o @RESULTS@"
	done

	# Phase B: the taper of the native networks, crossed with the learning rate. The network is
	# selected with --network here, -m picks the task (regression) and not the architecture.
	for net in simple conv; do
		base=${COST[$net]}
		for lr in ${NATIVE_LRS//,/ }; do
			for ds in ${DOWNSAMPLE_STEPS//,/ }; do
				cost=$(( base * (8 + avg_extra) / 8 ))
				add_job "B_${short}_${net}_ds${ds}_lr$(tag "$lr")" \
					"$(cost_units "$cost" "$extra_list" "$train")" \
					"$TUNE -m regression --network $net -d tarreg -f $OUT/ds_${short}_train.tar -t $OUT/ds_${short}_test.tar -i $EPOCHS --downsample-steps $ds --extra-steps $EXTRA_STEPS --learning-rates $lr --weight-decays $NATIVE_WD -o @RESULTS@"
			done
		done
	done
done
echo "  $number jobs, $total cost units"

# Longest processing time first, then hand every job to the card with the least work per unit
# of measured speed. Each card gets its own queue so that no card can ever run more than
# $SLOTS tuner processes at once, and the device is pinned inside the job file.
read -r -a DEVICES <<< "$GPUS"
read -r -a WEIGHTS <<< "$GPU_WEIGHTS"
if [ "${#DEVICES[@]}" -eq 0 ]; then
	echo "  no gpu uuids found, running unpinned on the default device"
	DEVICES=(""); WEIGHTS=(100)
fi
for d in "${!DEVICES[@]}"; do : > "$OUT/queue_$d.txt"; load[$d]=0; done
order=$(for i in $(seq 1 "$number"); do echo "${JOB_COST[$i]} $i"; done | sort -k1,1nr -k2,2n)
skipped=0
while read -r cost i; do
	[ -n "$i" ] || continue
	best=0
	for d in "${!DEVICES[@]}"; do
		weighted_d=$((load[best] * 100))
		weighted_this=$((load[$d] * 100))
		# compare load / weight without leaving the integers
		(( weighted_this * WEIGHTS[best] < weighted_d * WEIGHTS[$d] )) && best=$d
	done
	id="$(printf '%03d' "$i")"
	name="${JOB_NAME[$i]}"
	file="$OUT/jobs/${id}_${name}.sh"
	# the tuner only writes results.csv once a job is through, so an existing file means the
	# job is done: an interrupted shootout is resumed by simply running it again
	if [ -s "$OUT/results/${id}_${name}/results.csv" ] && [ "${FORCE:-0}" != 1 ]; then
		rm -f "$file"
		skipped=$((skipped + 1))
		continue
	fi
	{
		printf '#!/usr/bin/env bash\n'
		if [ -n "${DEVICES[$best]}" ]; then
			printf 'export HIP_VISIBLE_DEVICES=%s ROCR_VISIBLE_DEVICES=%s\n' \
				"${DEVICES[$best]}" "${DEVICES[$best]}"
		fi
		printf 'set -uo pipefail\n'
		printf '%s\n' "${JOB_CMD[$i]//@RESULTS@/$OUT/results/${id}_${name}}"
	} > "$file"
	chmod +x "$file"
	echo "$file" >> "$OUT/queue_$best.txt"
	load[$best]=$((load[best] + JOB_COST[i]))
done
for d in "${!DEVICES[@]}"; do
	echo "  queue_$d (${DEVICES[$d]:-default}, weight ${WEIGHTS[$d]}): $(wc -l < "$OUT/queue_$d.txt") jobs, cost ${load[$d]}"
done
[ "$skipped" -eq 0 ] || echo "  $skipped jobs kept from an earlier run (FORCE=1 to redo them)"

echo "== running (this is the slow part) =="
: > "$OUT/failed.txt"
start=$(date +%s)
pids=()
for d in "${!DEVICES[@]}"; do
	[ -s "$OUT/queue_$d.txt" ] || continue
	# shellcheck disable=SC2016
	xargs -P "$SLOTS" -I{} env OUT="$OUT" bash -c \
		'if ! bash "$1" > "$OUT/logs/$(basename "$1" .sh).log" 2>&1; then echo "$(basename "$1")" >> "$OUT/failed.txt"; fi' _ {} \
		< "$OUT/queue_$d.txt" &
	pids+=($!)
done
wait "${pids[@]}"
echo "elapsed: $((($(date +%s) - start) / 60)) minutes"
echo "failed jobs: $(wc -l < "$OUT/failed.txt")"
sed 's/^/  /' "$OUT/failed.txt"

echo "== aggregating =="
python3 "$REPO/scripts/shootoutregplot.py" "$OUT" --epochs "$EPOCHS" || echo "warning: aggregation failed"
