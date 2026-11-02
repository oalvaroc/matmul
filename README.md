# matmul

Single-precision square matrix multiplication, `C = A × B`, in three
versions, built to compare their performance on a cluster:

- `serial`: a plain triple loop, the reference for speedups and results.
- `mpi`: master–worker with MPI. C is split into `b × b` blocks; for each
  block the master (rank 0) packs a row of blocks of A and a column of
  blocks of B into one message, sends it to a free worker and receives the
  block of C back.
- `mpiblas`: the same scheme, with the workers' block product done by
  OpenBLAS (`cblas_sgemm`).

The results, measured on the ada cluster, are in the report in
[`report/`](report/) (in Portuguese).

## Build

Requirements: CMake ≥ 3.18, a C11 compiler, an MPI implementation, OpenBLAS
(found through `pkg-config`) and OpenMP.

```sh
cmake -B build -S .
cmake --build build
```

The build type defaults to `Release` (`-O3`). The programs are written to
`build/src/`.

## Run

All three generate A and B from a seed, so the same `-n` and `-r` give the
same matrices in every program.

```sh
./build/src/serial -n 1024 -r 42
mpirun -n 5 ./build/src/mpi -n 1024 -s 128 -r 42 -c
OMP_NUM_THREADS=1 mpirun -n 5 ./build/src/mpiblas -n 1024 -s 128 -r 42 -c
```

| Option | Meaning |
| --- | --- |
| `-n`, `--order` | matrix order (required) |
| `-s`, `--bs` | block size, must divide the order (`mpi`, `mpiblas`; default 4) |
| `-r`, `--seed` | seed of the random matrices (default 1) |
| `-c`, `--check` | check the result against the serial product (`mpi`, `mpiblas`) |
| `-h`, `--help` | show the options |

The MPI programs need at least 2 processes: one master and one or more
workers. `OMP_NUM_THREADS` sets the OpenBLAS threads of each `mpiblas`
rank.

Each program prints the time of the multiplication only (`elapsed`), without
generating the matrices. The MPI programs also print how many blocks each
worker computed and `warmup`, the time spent opening the MPI connections
before the timer starts. With `-c`, they then print whether the result
matches the serial product within a tolerance of `10 · sqrt(n) · FLT_EPSILON`,
the largest relative error and the time of the check, which is not part
of `elapsed`. The program exits with an error if the check fails.

```
order=1024 bs=128 procs=5 warmup=0.000055s elapsed=0.046100s
check=ok max_rel_err=0.000e+00 check_time=0.539041s
```

## Benchmark

[`ada-bench.sh`](ada-bench.sh) is a PBS job for the ada cluster that runs
every experiment of the report (5 repetitions each) and writes one CSV row
per run to `results/<job id>/results.csv`, with the output of each run in
`logs/`. Build first, then submit it from the repository root:

```sh
qsub ada-bench.sh
qsub -v "EXPERIMENTS=serial strong" ada-bench.sh  # only some experiments
```

## Report

[`report/`](report/) builds `relatorio.pdf` from the results of one job. It
needs Python with `pandas`, `numpy` and `matplotlib` (in `.venv/` by default),
pandoc and tectonic.

```sh
cd report
make                                # job 1035570
make JOB=../results/<job id>        # another job
```
