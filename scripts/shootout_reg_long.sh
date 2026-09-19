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
# Follow up to scripts/shootout_reg.sh: retrain the winning architectures with a much larger
# epoch budget and find out whether the 40 epoch ranking was an artifact of the truncation.
#
# 58 % of all shootout trials and 70 % of the top 60 reached their best epoch in the last five
# epochs of the 40 epoch budget, so the ranking may say as much about the budget as about the
# architectures. This takes the best four distinct architectures of the shootout, freezes one
# hyperparameter set each (the one that won), and trains every one of them on every circuit of
# the original run for $EPOCHS epochs. One run per architecture and circuit, no grid.
#
# The four are derived from <shootout>/summary/configs.csv by grouping on the architecture
# family and keeping the best mean R2, which collapses the c1 and c2 channel variants into one
# architecture and picks the better of the two, as intended. Override with
# WINNERS="seresnet:0.003:0.001:2 tcn:0.003:0.01:2" if you want a specific set.
#
# Everything goes into <shootout>/long, next to the original run, so that the main ranking
# keeps comparing runs that all got the same budget:
#
#   groups/<tag>/<family>/*.pt    the one frozen network of a family for that circuit
#   jobs/NNN_e<EPOCHS>_<tag>_<family>.sh  one tuner invocation, its gpu pinned inside
#   queue_N.txt                   the jobs assigned to one gpu
#   results/NNN_e<EPOCHS>_<tag>_<family>/  results.csv, curves.csv, targetr2.csv, best/
#   logs/                         full tuner output per job
#   failed.txt                    jobs that did not finish cleanly
#   summary/                      convergence tables and plots (shootoutreglongplot.py)
#
# shootoutregplot.py picks the long tree up on its own and folds it into the main report, so
# both plotters run at the end.
#
# Usage: scripts/shootout_reg_long.sh [shootoutdir]
# Environment: EPOCHS, TOP, WINNERS, GPUS, GPU_WEIGHTS, SLOTS, FORCE, PLAN
# PLAN=1 prints the winners and the job list and exits without training.
#
# The run is resumable: a job whose results/results.csv already holds a passing trial is
# skipped, so an interrupted run is continued by starting it again with the same command. The
# epoch budget is part of the job name, so changing $EPOCHS does not silently reuse the results
# of a shorter run. Interrupt with Ctrl-C twice (xargs needs the second one) and check with
# pgrep -f torchkissann_tune that no tuner is left holding a card.

set -uo pipefail

SHOOTOUT="${1:-$(pwd)/shootout_reg}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TUNE="$REPO/build/src/tune/torchkissann_tune"

EPOCHS="${EPOCHS:-150}"
TOP="${TOP:-4}"

# The two MI100s. These small networks are launch latency bound rather than compute bound, so
# several tuner processes share a card well; eight slots per card measured about the same
# epochs per minute as sixteen did in the shootout, at half the memory pressure.
GPUS="${GPUS:-${MI100A:-} ${MI100B:-}}"
GPU_WEIGHTS="${GPU_WEIGHTS:-100 100}"
SLOTS="${SLOTS:-8}"

[ -x "$TUNE" ] || { echo "$TUNE is missing, build the tools first" >&2; exit 1; }
[ -s "$SHOOTOUT/datasets.tsv" ] || { echo "$SHOOTOUT/datasets.tsv is missing, run scripts/shootout_reg.sh first" >&2; exit 1; }
[ -s "$SHOOTOUT/summary/configs.csv" ] || [ -n "${WINNERS:-}" ] || {
	echo "$SHOOTOUT/summary/configs.csv is missing and WINNERS is not set" >&2; exit 1; }

OUT="$SHOOTOUT/long"
mkdir -p "$OUT/results" "$OUT/logs"

tag() { echo "$1" | tr '.' 'p'; }

echo "== winners =="
# One line per architecture: family, learning rate, weight decay, channel count. The file name
# of a circuit is rebuilt from those below, because buildnet.py names it after the target count
# and the frequency count, which differ between the circuits.
if [ -n "${WINNERS:-}" ]; then
	selected="$WINNERS"
else
	selected=$(python3 - "$SHOOTOUT/summary/configs.csv" "$TOP" <<'PY'
import csv, re, sys
best = {}
with open(sys.argv[1], newline="") as fh:
	for row in csv.DictReader(fh):
		if row.get("status") != "ok" or row.get("family", "") == "":
			continue
		try:
			mean = float(row["meanR2"])
		except (ValueError, KeyError):
			continue
		family = row["family"]
		# on a tie the smaller network wins, so the ranking is not just a capacity ranking
		key = (mean, -float(row.get("parameters") or 0))
		if family not in best or key > best[family][0]:
			best[family] = (key, row)
top = sorted(best.values(), key=lambda item: -item[0][0])[:int(sys.argv[2])]
for _, row in top:
	channels = re.search(r"_c(\d+)\.pt$", row["script"])
	print(f"{row['family']}:{row['learningRate']}:{row['weightDecay']}:"
	      f"{channels.group(1) if channels else 1}")
PY
) || { echo "could not read the winners out of $SHOOTOUT/summary/configs.csv" >&2; exit 1; }
fi
[ -n "$selected" ] || { echo "no winners selected" >&2; exit 1; }

declare -a TAGS TARGETS train_sizes
while IFS=$'\t' read -r short circuit targets train test; do
	[ -n "$short" ] || continue
	TAGS+=("$short"); TARGETS+=("$targets"); train_sizes+=("$((train + 0))")
done < "$SHOOTOUT/datasets.tsv"
[ "${#TAGS[@]}" -gt 0 ] || { echo "no circuits in $SHOOTOUT/datasets.tsv" >&2; exit 1; }

for winner in $selected; do
	family="${winner%%:*}"
	rest="${winner#*:}"; lr="${rest%%:*}"; rest="${rest#*:}"
	wd="${rest%%:*}"; channels="${rest##*:}"
	[ "$channels" = "$wd" ] && channels=1
	echo "  $family  lr $lr  wd $wd  channels $channels"
done
echo "  circuits: ${TAGS[*]}"

echo "== jobs ($EPOCHS epochs each) =="
# Rough seconds per epoch on one MI100 at shootout batch size, measured during the shootout and
# only used to hand the two cards comparable amounts of work. Scaled by the dataset size below.
declare -A COST=([seresnet]=41 [inception]=38 [resnet]=36 [tcn]=31 [gru]=28
	[convnet]=27 [conv]=27 [upsamplenet]=24 [transformer]=9 [simplenet]=6 [simple]=6)
REFSIZE=16546

rm -rf "$OUT/jobs" "$OUT/groups"
mkdir -p "$OUT/jobs"
number=0
total=0
declare -A JOB_NAME JOB_COST JOB_CMD
for winner in $selected; do
	family="${winner%%:*}"
	rest="${winner#*:}"; lr="${rest%%:*}"; rest="${rest#*:}"
	wd="${rest%%:*}"; channels="${rest##*:}"
	[ "$channels" = "$wd" ] && channels=1
	cost=${COST[$family]:-30}
	for index in "${!TAGS[@]}"; do
		short="${TAGS[$index]}"; targets="${TARGETS[$index]}"
		# the frozen network of this family for this circuit. the shootout built one file per
		# architecture, target count and channel count, so the winner transfers exactly.
		file=$(ls "$SHOOTOUT"/nets_all_t"$targets"/"$family"*"-${targets}_c$channels.pt" 2>/dev/null | head -1)
		[ -n "$file" ] || { echo "  no $family c$channels network for $targets targets, skipping" >&2; continue; }
		number=$((number + 1))
		dir="$OUT/groups/$short/$family"
		mkdir -p "$dir"
		cp "$file" "$dir/"
		# the budget is part of the name, so a shorter earlier run is not silently kept when
		# $EPOCHS changes; the hyperparameters are not, they live in the group directory and
		# in the log, so redo with FORCE=1 after changing WINNERS
		name="e${EPOCHS}_${short}_${family}"
		JOB_NAME[$number]="$name"
		JOB_COST[$number]=$(( (cost * EPOCHS * train_sizes[$index] + REFSIZE) / REFSIZE ))
		total=$((total + JOB_COST[$number]))
		JOB_CMD[$number]="$TUNE -m regression -d tarreg -f $SHOOTOUT/ds_${short}_train.tar -t $SHOOTOUT/ds_${short}_test.tar --scripts $dir -i $EPOCHS --learning-rates $lr --weight-decays $wd -o @RESULTS@"
	done
done
echo "  $number jobs, $total cost units"

# Longest processing time first, then hand every job to the card with the least work per unit of
# measured speed. One queue per card so a card can never run more than $SLOTS tuners at once.
read -r -a DEVICES <<< "$GPUS"
read -r -a WEIGHTS <<< "$GPU_WEIGHTS"
if [ "${#DEVICES[@]}" -eq 0 ]; then
	echo "  no gpu uuids found, running unpinned on the default device"
	DEVICES=(""); WEIGHTS=(100)
fi
# a GPU_WEIGHTS list that is shorter than GPUS means: all equally fast
for d in "${!DEVICES[@]}"; do WEIGHTS[$d]="${WEIGHTS[$d]:-100}"; done
for d in "${!DEVICES[@]}"; do : > "$OUT/queue_$d.txt"; load[$d]=0; done
order=$(for i in $(seq 1 "$number"); do echo "${JOB_COST[$i]} $i"; done | sort -k1,1nr -k2,2n)
skipped=0
while read -r cost i; do
	[ -n "$i" ] || continue
	best=0
	for d in "${!DEVICES[@]}"; do
		(( load[$d] * WEIGHTS[best] < load[best] * WEIGHTS[$d] )) && best=$d
	done
	id="$(printf '%03d' "$i")"
	name="${JOB_NAME[$i]}"
	file="$OUT/jobs/${id}_${name}.sh"
	if [ -s "$OUT/results/${id}_${name}/results.csv" ] && [ "${FORCE:-0}" != 1 ]; then
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
	load[$best]=$((load[best] + cost))
done <<< "$order"
for d in "${!DEVICES[@]}"; do
	echo "  queue_$d (${DEVICES[$d]:-default}, weight ${WEIGHTS[$d]}): $(wc -l < "$OUT/queue_$d.txt") jobs, cost ${load[$d]}"
done
[ "$skipped" -eq 0 ] || echo "  $skipped jobs kept from an earlier run (FORCE=1 to redo them)"

if [ "${PLAN:-0}" = 1 ]; then
	echo "PLAN=1, not running anything"
	exit 0
fi

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
python3 "$REPO/scripts/shootoutreglongplot.py" -d "$OUT" -b "$SHOOTOUT" || echo "warning: long aggregation failed"
python3 "$REPO/scripts/shootoutregplot.py" "$SHOOTOUT" || echo "warning: main aggregation failed"
