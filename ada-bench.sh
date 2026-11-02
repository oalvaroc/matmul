#!/usr/bin/bash

#PBS	-N matmul-bench
#PBS	-j oe
#PBS	-l select=2:ncpus=48:mpiprocs=48:mem=32gb
#PBS	-l walltime=02:00:00
#PBS	-q paralela

# Benchmarks serial, mpi and mpiblas. Experiments:
#
#   serial  the reference times for speedups
#   blocks  both nodes, 24 ranks per node, varying order and block size;
#           mpiblas runs 2 OpenBLAS threads per rank, on the other cores
#   strong  fixed order, 2 to 96 processes
#   nodes   16, 32 and 48 processes on one node or split across two
#   weak    order grows with the number of workers
#   hybrid  mpiblas on every core, as ranks x OpenBLAS threads
#
# Every case runs REPS times, one row per run in results/<job id>/results.csv
# and the output of each run in logs/. The first repetition of every mpi and
# mpiblas case also checks its result
# against the serial product (-c); the check is timed apart from elapsed, and
# a failed check is recorded as status "wrong".
#
# Run some experiments: qsub -v "EXPERIMENTS=serial blocks" ada-bench.sh
# Quick test:           qsub -l walltime=00:20:00 -v REPS=1 ada-bench.sh

set -u  # no -e: a failed run must not end the job

cd "${PBS_O_WORKDIR:-.}"

module load gcc/12.2
module load openmpi/5.0.6-gcc-12.2.0
module load openblas/0.3.21-gcc-12.2.0

CORES_PER_NODE=48  # keep in sync with the select line above
EXPERIMENTS=${EXPERIMENTS:-"serial blocks strong nodes weak hybrid"}
REPS=${REPS:-5}
SEED=${SEED:-42}
TIMEOUT=${TIMEOUT:-5m}

# The UD transport of UCX timed out inside MPI_Send between nodes (jobs
# 1033827 and 1034182). Reliable connections did not (job 1034517).
export UCX_TLS=rc,sm,self

OUT="results/${PBS_JOBID:-local-$(date +%Y%m%d-%H%M%S)}"
CSV="$OUT/results.csv"
mkdir -p "$OUT/logs"
echo "experiment,program,order,bs,procs,nodes,threads,rep,warmup_s,elapsed_s,status,check,max_rel_err,check_s" > "$CSV"
# hardware and compiler flags, for the report
lscpu > "$OUT/lscpu.txt"
grep -E '^CMAKE_(BUILD_TYPE|C_FLAGS|C_COMPILER):' build/CMakeCache.txt > "$OUT/build.txt"

echo "job ${PBS_JOBID:-local}: $EXPERIMENTS; seed $SEED; results in $OUT"

# run EXPERIMENT PROGRAM ORDER [BS PROCS NODES THREADS]
# Runs the REPS repetitions of one case. Stops at the first failure: the
# remaining repetitions would fail too.
run() {
	local exp=$1 prog=$2 order=$3 bs=${4:-0} procs=${5:-1} nodes=${6:-1} threads=${7:-1}
	local name="$exp-$prog-n$order-b$bs-p$procs-N$nodes-t$threads"

	local cmd=("./build/src/$prog" -n "$order" -r "$SEED")
	if [ "$prog" != serial ]; then
		# processes split evenly across the nodes; no core binding, so the
		# OpenBLAS threads of a rank can use any free core of its socket
		local per_node=$(( (procs + nodes - 1) / nodes ))
		cmd=(mpirun --np "$procs" --map-by "ppr:$per_node:node"
		     -x OMP_NUM_THREADS -x UCX_TLS "${cmd[@]}" -s "$bs")
	fi

	local rep
	for ((rep = 1; rep <= REPS; rep++)); do
		local log="$OUT/logs/$name-r$rep.log"
		local args=()
		# checking costs a serial multiplication, too slow for every run
		if [ "$rep" -eq 1 ] && [ "$prog" != serial ]; then
			args=(-c)
		fi

		OMP_NUM_THREADS=$threads timeout -k 30s "$TIMEOUT" \
			"${cmd[@]}" "${args[@]}" > "$log" 2>&1
		local state
		case $? in
		0)       state=ok ;;
		124|137) state=timeout ;;  # 137: killed after -k
		*)       state=error ;;
		esac
		# value of key=value in the log, empty if missing
		field() { grep -o "$1=[^ s]*" "$log" | head -n 1 | cut -d= -f2; }
		local warmup elapsed check
		warmup=$(field warmup)
		elapsed=$(field elapsed)
		check=$(field check)
		[ "$check" = fail ] && state=wrong

		echo "$exp,$prog,$order,$bs,$procs,$nodes,$threads,$rep,$warmup,$elapsed,$state,$check,$(field max_rel_err),$(field check_time)" >> "$CSV"
		echo "$name rep $rep: $state ${elapsed:+${elapsed}s}${check:+ check=$check}"
		[ "$state" = ok ] || return
	done
}

# nodes needed for P processes, one per core
nodes_for() {
	echo $(( $1 > CORES_PER_NODE ? 2 : 1 ))
}

# P processes are 1 master, which only hands out tasks, and P - 1 workers.
# These orders and block sizes give 256 tasks, enough for 95 workers.
STRONG_PROCS="2 3 5 9 17 33 48 65 96"

for exp in $EXPERIMENTS; do
	echo -e "\n==> $exp"
	case $exp in
	serial)
		for order in 512 1024 2048 3072 4096; do
			run serial serial "$order"
		done
		;;
	blocks)
		for prog in mpi mpiblas; do
			threads=1
			[ "$prog" = mpiblas ] && threads=2
			for order in 512 1024 2048 4096; do
				for bs in 64 128 256 512; do
					# bs must divide the order and leave at least 16 tasks
					[ $((order % bs)) -eq 0 ] && [ $((order / bs)) -ge 4 ] || continue
					run blocks "$prog" "$order" "$bs" 48 2 "$threads"
				done
			done
		done
		;;
	strong)
		for p in $STRONG_PROCS; do
			run strong mpi 2048 128 "$p" "$(nodes_for "$p")"
			run strong mpiblas 2048 128 "$p" "$(nodes_for "$p")"
			run strong mpiblas 4096 256 "$p" "$(nodes_for "$p")"
		done
		;;
	nodes)
		for prog in mpi mpiblas; do
			for p in 16 32 48; do
				run nodes "$prog" 2048 128 "$p" 1
				run nodes "$prog" 2048 128 "$p" 2
			done
		done
		;;
	weak)
		# W = P - 1 workers get order 1024 * cbrt(W): 2 * 1024^3 flops each
		for prog in mpi mpiblas; do
			run weak "$prog" 1024 256 2 1
			run weak "$prog" 2048 256 9 1
			run weak "$prog" 3072 256 28 1
			run weak "$prog" 4096 256 65 2
		done
		;;
	hybrid)
		# ranks x threads per node, on both nodes
		for rt in "48 1" "24 2" "16 3" "8 6" "4 12" "2 24"; do
			set -- $rt
			run hybrid mpiblas 4096 256 $(($1 * 2)) 2 "$2"
		done
		;;
	*)
		echo "unknown experiment: $exp"
		;;
	esac
done

echo -e "\ndone, results in $CSV"
