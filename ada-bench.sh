#!/usr/bin/bash

#PBS	-N matmul-bench
#PBS	-j oe
#PBS	-l select=2:ncpus=48:mpiprocs=48:mem=4gb
#PBS	-l walltime=01:00:00
#PBS	-q paralela

# Benchmark serial (baseline), mpi and mpiblas on a few square matrix orders
# and, for the MPI programs, a few block sizes. Every case runs REPS times.
#
# Every run uses the same seed, so the dumped result matrices of different
# programs can be compared. Results go to results/<job id>/:
#   results.csv   one row per run
#   logs/         full output of every run
#   *.mat         result matrices, from the first repetition only
#
# Override settings at submit time, e.g.: qsub -v SEED=7,REPS=5 ada-bench.sh
# (lists are space separated: qsub -v "ORDERS=1024 2048" ...)

set -u  # no -e: a failed or timed out run must not end the job

cd "${PBS_O_WORKDIR:-.}"

module load gcc/12.2
module load openmpi/5.0.6-gcc-12.2.0
module load openblas/0.3.21-gcc-12.2.0

# keep in sync with the select line above
NODES=${NODES:-2}
PROC_PER_NODE=${PROC_PER_NODE:-48}
MEM_PER_NODE_MB=${MEM_PER_NODE_MB:-4096}
PROC_TOTAL=$((NODES * PROC_PER_NODE))

# Below 512 the MPI programs only measure communication overhead; serial
# takes ~4 min at 4096 and does not finish 8192 in reasonable time.
ORDERS=${ORDERS:-"512 1024 2048 4096"}
# Block sizes below 64 were 10-100x slower in every case.
BLOCK_SIZES=${BLOCK_SIZES:-"64 128 256 512"}
REPS=${REPS:-3}
TIMEOUT_SERIAL=${TIMEOUT_SERIAL:-10m}
TIMEOUT_MPI=${TIMEOUT_MPI:-2m}  # the slowest successful MPI run took ~12 s
SEED=${SEED:-42}

# One MPI rank per core already uses every core. Without this, the OpenMP
# build of OpenBLAS starts a team of threads in every rank, oversubscribing
# the cores.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

OUT_DIR="results/${PBS_JOBID:-local-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR/logs"
CSV="$OUT_DIR/results.csv"
echo "program,order,bs,procs,seed,rep,warmup_s,elapsed_s,status" > "$CSV"

echo "job ${PBS_JOBID:-local} on $(uname -n), $PROC_TOTAL procs, seed $SEED"
echo "results in $OUT_DIR"

# run PROG ORDER BS REP
# Runs one case, appends a row to the CSV and succeeds only if the run did.
run() {
	local prog=$1 order=$2 bs=$3 rep=$4
	local name="$prog-n$order-bs$bs"
	local log="$OUT_DIR/logs/$name-r$rep.log"

	# Peak memory on the node hosting rank 0, in floats: A, B and C, plus for
	# the MPI programs a task buffer (block row of A, block column of B) and
	# a result block per rank.
	local floats=$((3 * order * order))
	local procs=1
	local limit=$TIMEOUT_SERIAL
	local cmd=(./build/src/serial -n "$order")
	if [ "$prog" != serial ]; then
		limit=$TIMEOUT_MPI
		floats=$((floats + PROC_PER_NODE * (2 * order * bs + bs * bs)))
		procs=$PROC_TOTAL
		cmd=(mpirun --np "$procs" --map-by "ppr:$PROC_PER_NODE:node"
		     "./build/src/$prog" -n "$order" -s "$bs")
	fi

	# repetitions give the same matrix: dump it once
	if [ "$rep" -eq 1 ]; then
		cmd+=(-o "$OUT_DIR/$name.mat")
	fi

	local state warmup= elapsed=
	if [ $((floats * 4 / 1024 / 1024)) -gt $((MEM_PER_NODE_MB * 9 / 10)) ]; then
		# exceeding the PBS memory limit kills the whole job, not one run
		state=skipped_mem
	else
		timeout -k 30s "$limit" "${cmd[@]}" -r "$SEED" > "$log" 2>&1
		local status=$?
		case $status in
		0)       state=ok ;;
		124|137) state=timeout ;;  # 137: killed after -k
		*)       state="error($status)" ;;
		esac
		warmup=$(grep -o 'warmup=[0-9.]*' "$log" | cut -d= -f2)
		elapsed=$(grep -o 'elapsed=[0-9.]*' "$log" | cut -d= -f2)
	fi

	echo "$prog,$order,$bs,$procs,$SEED,$rep,$warmup,$elapsed,$state" >> "$CSV"
	echo "$name rep $rep: $state ${elapsed:+${elapsed}s}"
	[ "$state" = ok ]
}

echo -e "\n==> serial"
for order in $ORDERS; do
	for ((rep = 1; rep <= REPS; rep++)); do
		# a failed run would fail again: skip its remaining repetitions
		run serial "$order" 0 "$rep" || break
	done
done

for prog in mpi mpiblas; do
	echo -e "\n==> $prog"
	for order in $ORDERS; do
		for bs in $BLOCK_SIZES; do
			# bs must divide the order, and leave at least 16 tasks (q >= 4)
			[ $((order % bs)) -eq 0 ] && [ $((order / bs)) -ge 4 ] || continue
			for ((rep = 1; rep <= REPS; rep++)); do
				run "$prog" "$order" "$bs" "$rep" || break
			done
		done
	done
done

echo -e "\ndone, results in $CSV"
