#!/usr/bin/bash

#PBS	-N matmul-bench
#PBS	-j oe
#PBS	-l select=2:ncpus=48:mpiprocs=48:mem=4gb
#PBS	-l walltime=01:00:00
#PBS	-q paralela

# Benchmark serial (baseline), mpi and mpiblas on square matrices of growing
# order 2^p, with every block size 2^b <= order for the MPI programs.
#
# Every run uses the same seed, so the dumped result matrices of different
# programs can be compared. Results go to results/<job id>/:
#   results.csv   one row per run
#   logs/         full output of every run
#   *.mat         result matrices (see mat_dump in src/matrix.c)
#
# Override settings at submit time, e.g.: qsub -v SEED=7,MAX_ORDER=4096 ada-bench.sh

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

MAX_ORDER=${MAX_ORDER:-8192} # a dump is order^2 * 4 bytes: 256 MiB at 8192
MAX_BS=${MAX_BS:-1024}
TIMEOUT=${TIMEOUT:-5m}
SEED=${SEED:-42}

# One MPI rank per core already uses every core. Without this, the OpenMP
# build of OpenBLAS starts a team of threads in every rank, oversubscribing
# the cores.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

OUT_DIR="results/${PBS_JOBID:-local-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR/logs"
CSV="$OUT_DIR/results.csv"
echo "program,order,bs,procs,seed,elapsed_s,status" > "$CSV"

echo "job ${PBS_JOBID:-local} on $(uname -n), $PROC_TOTAL procs, seed $SEED"
echo "results in $OUT_DIR"

# run PROG ORDER BS
# Runs one case, appends a row to the CSV and returns 124 if it timed out.
run() {
	local prog=$1 order=$2 bs=$3
	local name="$prog-n$order-bs$bs"
	local log="$OUT_DIR/logs/$name.log"

	# Peak memory on the node hosting rank 0, in floats: A, B and C, plus for
	# the MPI programs a task buffer (block row of A, block column of B) and
	# a result block per rank.
	local floats=$((3 * order * order))
	local procs=1
	local cmd=(./build/src/serial -n "$order")
	if [ "$prog" != serial ]; then
		floats=$((floats + PROC_PER_NODE * (2 * order * bs + bs * bs)))
		procs=$PROC_TOTAL
		cmd=(mpirun --np "$procs" --map-by "ppr:$PROC_PER_NODE:node"
		     "./build/src/$prog" -n "$order" -s "$bs")
	fi

	local state elapsed= status=0
	if [ $((floats * 4 / 1024 / 1024)) -gt $((MEM_PER_NODE_MB * 9 / 10)) ]; then
		# exceeding the PBS memory limit kills the whole job, not one run
		state=skipped_mem
	else
		timeout -k 30s "$TIMEOUT" "${cmd[@]}" -r "$SEED" \
			-o "$OUT_DIR/$name.mat" > "$log" 2>&1
		status=$?
		case $status in
		0)       state=ok ;;
		124|137) state=timeout; status=124 ;;  # 137: killed after -k
		*)       state="error($status)" ;;
		esac
		elapsed=$(grep -o 'elapsed=[0-9.]*' "$log" | cut -d= -f2)
	fi

	echo "$prog,$order,$bs,$procs,$SEED,$elapsed,$state" >> "$CSV"
	echo "$name: $state ${elapsed:+${elapsed}s}"
	return $status
}

echo -e "\n==> serial"
for ((order = 2; order <= MAX_ORDER; order *= 2)); do
	run serial "$order" 0
	# a larger order would only take longer
	[ $? -eq 124 ] && break
done

for prog in mpi mpiblas; do
	echo -e "\n==> $prog"
	timed_out=()
	for ((order = 2; order <= MAX_ORDER; order *= 2)); do
		# both are powers of 2, so bs divides order
		for ((bs = 1; bs <= order && bs <= MAX_BS; bs *= 2)); do
			# timed out at a smaller order, so it would here too
			[ -n "${timed_out[bs]:-}" ] && continue
			run "$prog" "$order" "$bs"
			[ $? -eq 124 ] && timed_out[bs]=1
		done
	done
done

echo -e "\ndone, results in $CSV"
