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
# Follow up to scripts/shootout.sh: retrain the winning architectures with a much larger
# epoch budget and find out whether the 20 epoch ranking was an artifact of the truncation.
#
# This takes the best four distinct architectures of the shootout, freezes one hyperparameter
# set each (the one that won), and trains every one of them on the classification dataset for
# $EPOCHS epochs. One run per architecture, no grid.
#
# The four are derived from <shootout>/summary/trials.csv by grouping on the architecture
# family and keeping the best test accuracy, which collapses the c1 and c2 channel variants
# into one architecture and picks the better of the two, as intended. Override with
# WINNERS="tcn:0.003:0.001:1 seresnet:0.001:0.01:1" if you want a specific set.
#
# Unlike the regression follow up there is no per-circuit loop: the classification shootout
# trains on a single dataset, so each architecture gets exactly one run. There is also no long
# plotter yet, so the summary is refreshed with the main plotter only and the extended runs
# stay in <shootout>/long for later analysis.
#
# Everything goes into <shootout>/long, next to the original run, so that the main ranking
# keeps comparing runs that all got the same budget:
#
#   groups/<family>/*.pt    the one frozen network of a family
#   jobs/NNN_e<EPOCHS>_<family>.sh  one tuner invocation, its gpu pinned inside
#   queue_N.txt                   the jobs assigned to one gpu
#   results/NNN_e<EPOCHS>_<family>/  results.csv, curves.csv, classaccuracy.csv, best/
#   logs/                         full tuner output per job
#   failed.txt                    jobs that did not finish cleanly
#
# Usage: scripts/shootout_long.sh [shootoutdir]
# Environment: EPOCHS, TOP, WINNERS, GPUS, GPU_WEIGHTS, SLOTS, FORCE, PLAN
# PLAN=1 prints the winners and the job list and exits without training.
#
# The run is resumable: a job whose results/results.csv already exists is skipped, so an
# interrupted run is continued by starting it again with the same command. The epoch budget
# is part of the job name, so changing $EPOCHS does not silently reuse the results of a
# shorter run. Interrupt with Ctrl-C twice (xargs needs the second one) and check with
# pgrep -f torchkissann_tune that no tuner is left holding a card.

set -uo pipefail

SHOOTOUT="${1:-$(pwd)/shootout}"
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
[ -s "$SHOOTOUT/ds_curated_train.tar" ] || { echo "$SHOOTOUT/ds_curated_train.tar is missing, run scripts/shootout.sh first" >&2; exit 1; }
[ -s "$SHOOTOUT/ds_curated_test.tar" ] || { echo "$SHOOTOUT/ds_curated_test.tar is missing, run scripts/shootout.sh first" >&2; exit 1; }
[ -s "$SHOOTOUT/summary/trials.csv" ] || [ -n "${WINNERS:-}" ] || {
	echo "$SHOOTOUT/summary/trials.csv is missing and WINNERS is not set" >&2; exit 1; }

OUT="$SHOOTOUT/long"
mkdir -p "$OUT/results" "$OUT/logs"

echo "== winners =="
# One line per architecture: family, learning rate, weight decay, channel count. The file name
# of the network is rebuilt from those below, because buildnet.py names it after the frequency
# count and the output size.
if [ -n "${WINNERS:-}" ]; then
	selected="$WINNERS"
else
	selected=$(python3 - "$SHOOTOUT/summary/trials.csv" "$TOP" <<'PY'
import csv, re, sys
best = {}
with open(sys.argv[1], newline="") as fh:
	for row in csv.DictReader(fh):
		if row.get("status") != "ok" or row.get("family", "") == "":
			continue
		try:
			acc = float(row["bestAccuracy"])
		except (ValueError, KeyError):
			continue
		family = row["family"]
		# on a tie the smaller network wins, so the ranking is not just a capacity ranking
		key = (acc, -float(row.get("parameters") or 0))
		if family not in best or key > best[family][0]:
			best[family] = (key, row)
top = sorted(best.values(), key=lambda item: -item[0][0])[:int(sys.argv[2])]
for _, row in top:
	channels = re.search(r"_c(\d+)\.pt$", row["script"])
	print(f"{row['family']}:{row['learningRate']}:{row['weightDecay']}:"
	      f"{channels.group(1) if channels else 1}")
PY
) || { echo "could not read the winners out of $SHOOTOUT/summary/trials.csv" >&2; exit 1; }
fi
[ -n "$selected" ] || { echo "no winners selected" >&2; exit 1; }

for winner in $selected; do
	family="${winner%%:*}"
	rest="${winner#*:}"; lr="${rest%%:*}"; rest="${rest#*:}"
	wd="${rest%%:*}"; channels="${rest##*:}"
	[ "$channels" = "$wd" ] && channels=1
	echo "  $family  lr $lr  wd $wd  channels $channels"
done

echo "== jobs ($EPOCHS epochs each) =="
# Rough seconds per epoch on one MI100 at shootout batch size, measured during the shootout and
# only used to hand the cards comparable amounts of work. Every job trains on the same dataset,
# so the cost is the per epoch figure times the budget.
declare -A COST=([seresnet]=27 [inception]=25 [resnet]=24 [convnet]=18 [tcn]=17
	[upsamplenet]=16 [gru]=13 [transformer]=6 [simplenet]=4 [simple]=4 [conv]=18)

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
	# the frozen network of this family. the shootout built one file per architecture and
	# channel count, so the winner transfers exactly. the digit after the family name is the
	# frequency count, which keeps a short name from matching a longer one (conv vs convnet).
	file=$(ls "$SHOOTOUT"/nets_all/"$family"[0-9]*"_c$channels.pt" 2>/dev/null | head -1)
	[ -n "$file" ] || { echo "  no $family c$channels network in $SHOOTOUT/nets_all, skipping" >&2; continue; }
	number=$((number + 1))
	dir="$OUT/groups/$family"
	mkdir -p "$dir"
	cp "$file" "$dir/"
	# the budget is part of the name, so a shorter earlier run is not silently kept when
	# $EPOCHS changes; the hyperparameters are not, they live in the group directory and
	# in the log, so redo with FORCE=1 after changing WINNERS
	name="e${EPOCHS}_${family}"
	JOB_NAME[$number]="$name"
	JOB_COST[$number]=$(( cost * EPOCHS ))
	total=$((total + JOB_COST[$number]))
	JOB_CMD[$number]="$TUNE -m ann -d tar -f $SHOOTOUT/ds_curated_train.tar -t $SHOOTOUT/ds_curated_test.tar --scripts $dir -i $EPOCHS --learning-rates $lr --weight-decays $wd -o @RESULTS@"
done
[ "$number" -gt 0 ] || { echo "no jobs to run" >&2; exit 1; }
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
	load[$best]=$((load[$best] + cost))
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
# There is no classification long plotter yet, so this only refreshes the main summary; the
# extended runs stay in $OUT for a dedicated plotter to pick up later.
python3 "$REPO/scripts/shootoutplot.py" "$SHOOTOUT" || echo "warning: main aggregation failed"
